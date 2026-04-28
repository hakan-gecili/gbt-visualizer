from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.core.session_store import session_store
from app.domain.session_types import SessionState
from app.services.counterfactual_service import generate_counterfactual_for_session
from app.services.dataset_service import apply_dataset_ranges, load_dataset_from_path, summarize_dataset
from app.services.feature_schema_service import build_feature_metadata
from app.services.feature_schema_service import parse_feature_schema_json
from app.services.model_loader import load_ensemble_model_from_path
from app.services.prediction_service import predict_model


FEATURE_NAMES = ["age", "income", "debt"]
REPO_ROOT = Path(__file__).resolve().parent.parent


def _train_tiny_binary_booster() -> xgb.Booster:
    data = np.array(
        [
            [22.0, 35.0, 4.0],
            [45.0, 82.0, 15.0],
            [31.0, np.nan, 8.0],
            [53.0, 120.0, np.nan],
            [28.0, 42.0, 2.0],
            [61.0, 99.0, 21.0],
        ],
        dtype=float,
    )
    labels = np.array([0, 1, 0, 1, 0, 1], dtype=float)
    matrix = xgb.DMatrix(data, label=labels, feature_names=FEATURE_NAMES)
    return xgb.train(
        {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": 0.8,
            "max_depth": 2,
            "min_child_weight": 0,
            "lambda": 0,
            "base_score": 0.5,
            "seed": 7,
            "nthread": 1,
        },
        matrix,
        num_boost_round=4,
    )


def _feature_vector(row: np.ndarray) -> dict[str, float | None]:
    return {
        feature_name: None if math.isnan(float(value)) else float(value)
        for feature_name, value in zip(FEATURE_NAMES, row)
    }


def _normalize_feature_key(feature_name: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in str(feature_name)).strip("_").lower()


def _align_schema_and_dataframe(schema_overrides: list[dict], dataframe, feature_names: list[str]):
    normalized_names = {_normalize_feature_key(feature_name): feature_name for feature_name in feature_names}
    rename_columns = {
        str(column): normalized_names[_normalize_feature_key(column)]
        for column in dataframe.columns
        if _normalize_feature_key(column) in normalized_names and str(column) != normalized_names[_normalize_feature_key(column)]
    }
    aligned_schema = []
    for feature in schema_overrides:
        next_feature = dict(feature)
        next_feature["name"] = normalized_names.get(_normalize_feature_key(feature.get("name", "")), feature.get("name"))
        aligned_schema.append(next_feature)
    if rename_columns:
        dataframe = dataframe.rename(columns=rename_columns)
    return aligned_schema, dataframe


def _assert_counterfactual_replays(
    *,
    session_id: str,
    model,
    predictor,
    feature_metadata,
    original_vector,
    result,
    threshold: float,
    target_class: int,
) -> None:
    counterfactuals = result.get("counterfactuals", [])
    if not counterfactuals:
        return

    feature_by_name = {str(feature["name"]): feature for feature in feature_metadata}
    replay_vector = dict(original_vector)
    counterfactual = counterfactuals[0]
    for change in counterfactual.get("changes", []):
        feature = feature_by_name.get(str(change["feature"]))
        if feature is None:
            raise AssertionError(f"{session_id} returned unknown feature {change['feature']!r}.")
        if str(feature.get("type")) in {"categorical", "binary"}:
            allowed_values = {str(option["value"]) for option in feature.get("options", [])}
            if str(change["new_value"]) not in allowed_values:
                raise AssertionError(
                    f"{session_id} returned invalid categorical value {change['new_value']!r} "
                    f"for {change['feature']!r}; expected one of {sorted(allowed_values)}."
                )
        if _is_integer_like_feature(feature) and isinstance(change["new_value"], (int, float)):
            if not float(change["new_value"]).is_integer():
                raise AssertionError(f"{session_id} returned fractional integer-like value {change!r}.")
        replay_vector[str(change["feature"])] = change["new_value"]

    _, replay_prediction = predict_model(model, predictor, feature_metadata, replay_vector)
    replay_probability = float(replay_prediction.probability)
    replay_label = int(replay_probability >= threshold)
    if replay_label != target_class:
        raise AssertionError(f"{session_id} counterfactual did not reach target class {target_class}.")
    if replay_label != int(counterfactual["new_prediction"]):
        raise AssertionError(
            f"{session_id} replayed label {replay_label} did not match returned label {counterfactual['new_prediction']}."
        )
    if not math.isclose(replay_probability, float(counterfactual["new_probability"]), rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError(
            f"{session_id} replayed probability {replay_probability} did not match returned "
            f"{counterfactual['new_probability']}."
        )


def _is_integer_like_feature(feature: dict) -> bool:
    name = str(feature.get("name", "")).lower()
    if name.endswith("-num") or name.endswith("_num") or name.endswith(" count") or name.endswith("_count"):
        return True
    bounds = [feature.get("min_value"), feature.get("max_value")]
    present_bounds = [value for value in bounds if value is not None]
    return len(present_bounds) == 2 and all(float(value).is_integer() for value in present_bounds)


def _load_xgboost_example_session(example_name: str):
    example_dir = REPO_ROOT / "examples" / example_name / "xgboost"
    model, predictor = load_ensemble_model_from_path(example_dir / "model.json", model_family="xgboost")
    dataframe = load_dataset_from_path(str(example_dir / "dataset.csv"))
    schema_overrides = parse_feature_schema_json((example_dir / "feature_schema.json").read_text(encoding="utf-8"))
    schema_overrides, dataframe = _align_schema_and_dataframe(schema_overrides, dataframe, model.feature_names)
    feature_metadata = apply_dataset_ranges(build_feature_metadata(model.feature_names, schema_overrides), dataframe)
    session_id = f"xgboost-smoke-{example_name}"
    session_store.save(
        SessionState(
            session_id=session_id,
            model=model,
            predictor=predictor,
            feature_metadata=feature_metadata,
            dataset_frame=dataframe,
            dataset_summary=summarize_dataset(dataframe, model.feature_names),
        )
    )
    return session_id, model, predictor, feature_metadata, dataframe


def _assert_xgboost_example_counterfactual(example_name: str, row_index: int) -> None:
    session_id, model, predictor, feature_metadata, dataframe = _load_xgboost_example_session(example_name)
    row = dataframe.iloc[row_index]
    original_vector = {}
    for feature in feature_metadata:
        feature_name = str(feature["name"])
        value = row[feature_name]
        original_vector[feature_name] = None if pd.isna(value) else value.item() if hasattr(value, "item") else value
    _, original_prediction = predict_model(model, predictor, feature_metadata, original_vector)
    threshold = 0.5
    target_class = 1 - int(float(original_prediction.probability) >= threshold)
    result = generate_counterfactual_for_session(
        session_id,
        row_index=row_index,
        threshold=threshold,
        target_class=target_class,
        max_steps=3,
    )
    _assert_counterfactual_replays(
        session_id=session_id,
        model=model,
        predictor=predictor,
        feature_metadata=feature_metadata,
        original_vector=original_vector,
        result=result,
        threshold=threshold,
        target_class=target_class,
    )
    print(f"{session_id}_counterfactuals", len(result.get("counterfactuals", [])))


def main() -> None:
    booster = _train_tiny_binary_booster()
    test_rows = np.array(
        [
            [24.0, 38.0, 3.0],
            [56.0, np.nan, 18.0],
            [48.0, 76.0, np.nan],
        ],
        dtype=float,
    )

    with tempfile.NamedTemporaryFile(suffix=".json") as model_file:
        booster.save_model(model_file.name)
        model, predictor = load_ensemble_model_from_path(model_file.name)

    if model.model_family != "xgboost":
        raise AssertionError(f"Expected xgboost model family, received {model.model_family}.")
    if model.num_trees != 4:
        raise AssertionError(f"Expected 4 normalized trees, received {model.num_trees}.")

    feature_metadata = build_feature_metadata(model.feature_names)
    native_matrix = xgb.DMatrix(test_rows, feature_names=FEATURE_NAMES)
    native_margins = booster.predict(native_matrix, output_margin=True)
    native_probabilities = booster.predict(native_matrix)
    native_leaf_ids = booster.predict(native_matrix, pred_leaf=True)

    for row_index, row in enumerate(test_rows):
        _, app_prediction = predict_model(model, predictor, feature_metadata, _feature_vector(row))
        if len(app_prediction.tree_results) != model.num_trees:
            raise AssertionError("Every normalized XGBoost tree should produce one traversal result.")

        if not math.isclose(app_prediction.margin, float(native_margins[row_index]), rel_tol=1e-6, abs_tol=1e-6):
            raise AssertionError(
                f"Margin mismatch on row {row_index}: app={app_prediction.margin}, "
                f"native={native_margins[row_index]}"
            )
        if not math.isclose(
            app_prediction.probability,
            float(native_probabilities[row_index]),
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise AssertionError(
                f"Probability mismatch on row {row_index}: app={app_prediction.probability}, "
                f"native={native_probabilities[row_index]}"
            )

        app_leaf_ids = [result.selected_leaf_id for result in app_prediction.tree_results]
        expected_leaf_ids = [int(leaf_id) for leaf_id in np.atleast_1d(native_leaf_ids[row_index])]
        if app_leaf_ids != expected_leaf_ids:
            raise AssertionError(
                f"Leaf traversal mismatch on row {row_index}: app={app_leaf_ids}, native={expected_leaf_ids}"
            )
        if not all(result.path_node_ids for result in app_prediction.tree_results):
            raise AssertionError(f"Expected non-empty active paths for row {row_index}.")

        session_id = f"xgboost-missing-smoke-{row_index}"
        dataframe = pd.DataFrame(test_rows, columns=FEATURE_NAMES)
        session_store.save(
            SessionState(
                session_id=session_id,
                model=model,
                predictor=predictor,
                feature_metadata=feature_metadata,
                dataset_frame=dataframe,
                dataset_summary=summarize_dataset(dataframe, model.feature_names),
            )
        )
        target_class = 1 - int(app_prediction.probability >= 0.5)
        counterfactual_result = generate_counterfactual_for_session(
            session_id,
            row_index=row_index,
            threshold=0.5,
            target_class=target_class,
            max_steps=3,
        )
        _assert_counterfactual_replays(
            session_id=session_id,
            model=model,
            predictor=predictor,
            feature_metadata=feature_metadata,
            original_vector=_feature_vector(row),
            result=counterfactual_result,
            threshold=0.5,
            target_class=target_class,
        )

    _assert_xgboost_example_counterfactual("breast_cancer", 0)
    _assert_xgboost_example_counterfactual("titanic", 0)
    _assert_xgboost_example_counterfactual("bank_marketing", 2)

    print("XGBoost integration smoke test passed.")


if __name__ == "__main__":
    main()
