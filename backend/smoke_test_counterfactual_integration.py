from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_ROOT.parent

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.session_store import session_store
from app.domain.session_types import SessionState
from app.services.counterfactual_service import generate_counterfactual_for_session, get_session_counterfactual_engine
from app.services.cf_engine.counterfactual_service import prune_counterfactual_changes
from app.services.dataset_service import apply_dataset_ranges, load_dataset_from_path, summarize_dataset
from app.services.feature_schema_service import build_feature_metadata, parse_feature_schema_json
from app.services.model_loader import load_ensemble_model_from_path


BREAST_CANCER_OBSERVED_ROW_INDEX = 1
BREAST_CANCER_OBSERVED_THRESHOLD = 0.75


def _load_example_session(example_name: str) -> SessionState:
    example_dir = REPO_ROOT / "examples" / example_name
    model_files = sorted(example_dir.glob("*.txt"))
    dataset_files = sorted(example_dir.glob("*.csv"))
    if len(model_files) != 1 or len(dataset_files) != 1:
        raise FileNotFoundError(f"Expected one model .txt and one dataset .csv for {example_name!r} in {example_dir}")
    model_path = model_files[0]
    dataset_path = dataset_files[0]
    schema_path = example_dir / "feature_schema.json"

    model, predictor = load_ensemble_model_from_path(model_path)
    dataframe = load_dataset_from_path(str(dataset_path))

    schema_overrides = None
    if schema_path.exists():
        schema_overrides = parse_feature_schema_json(schema_path.read_text(encoding="utf-8"))

    feature_metadata = build_feature_metadata(model.feature_names, schema_overrides)
    feature_metadata = apply_dataset_ranges(feature_metadata, dataframe)
    dataset_summary = summarize_dataset(dataframe, model.feature_names)

    session = SessionState(
        session_id=f"smoke-{example_name}",
        model=model,
        predictor=predictor,
        feature_metadata=feature_metadata,
        dataset_frame=dataframe,
        dataset_summary=dataset_summary,
    )
    session_store.save(session)
    return session


def _find_flipping_counterfactual(session_id: str, threshold: float) -> tuple[int, dict]:
    session = session_store.get(session_id)
    assert session.dataset_frame is not None

    for row_index in range(len(session.dataset_frame)):
        engine = get_session_counterfactual_engine(session_id)
        current_probability = float(engine.model.predict_proba(engine.dataset.iloc[[row_index]])[0, 1])
        current_prediction = int(current_probability >= threshold)
        target_class = 1 - current_prediction
        result = generate_counterfactual_for_session(
            session_id,
            row_index=row_index,
            threshold=threshold,
            target_class=target_class,
            max_steps=3,
        )
        counterfactuals = result.get("counterfactuals", [])
        if not counterfactuals:
            continue
        if counterfactuals[0]["new_prediction"] != target_class:
            continue
        if result["current_prediction"] == counterfactuals[0]["new_prediction"]:
            continue
        return row_index, result

    raise RuntimeError(f"No flipping counterfactual found for session {session_id!r} at threshold {threshold}.")


def _prediction_for_changes(engine, row_index: int, changes: list[dict], threshold: float) -> int:
    x_tmp = engine.dataset.iloc[[row_index]].copy()
    row_label = x_tmp.index[0]
    for change in changes:
        try:
            new_value = float(change["new_value"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Cannot replay decoded non-numeric change for {change['feature']!r}") from exc
        x_tmp.loc[row_label, change["feature"]] = new_value
    probability = float(engine.model.predict_proba(x_tmp)[0, 1])
    return int(probability >= threshold)


def _assert_returned_changes_are_minimal(session_id: str, row_index: int, result: dict) -> None:
    engine = get_session_counterfactual_engine(session_id)
    threshold = float(result["threshold"])
    counterfactual = result["counterfactuals"][0]
    target_class = int(counterfactual["new_prediction"])
    changes = list(counterfactual.get("changes", []))
    checked = 0
    for change in changes:
        remaining = [candidate for candidate in changes if candidate is not change]
        try:
            prediction = _prediction_for_changes(engine, row_index, remaining, threshold)
        except ValueError:
            continue
        checked += 1
        if prediction == target_class:
            raise AssertionError(
                f"Counterfactual is not minimal; removing {change['feature']!r} still reaches target {target_class}."
            )
    print("generated_minimality_checked_changes", checked)


def _assert_observed_breast_cancer_case_prunes() -> None:
    session = _load_example_session("breast_cancer")
    engine = get_session_counterfactual_engine(session.session_id)
    row_index = BREAST_CANCER_OBSERVED_ROW_INDEX
    threshold = BREAST_CANCER_OBSERVED_THRESHOLD
    original = engine.dataset.iloc[[row_index]].copy()
    original_probability = float(engine.model.predict_proba(original)[0, 1])
    target_class = 1 - int(original_probability >= threshold)

    changes = [
        {"feature": "mean_texture", "old_value": 16.67, "new_value": 17.195},
        {"feature": "area_error", "old_value": 34.37, "new_value": 35.08},
        {"feature": "worst_radius", "old_value": 13.33, "new_value": 17.51},
    ]
    _, pruned_changes, removed_changes, is_minimal = prune_counterfactual_changes(
        original_row=original,
        changes=changes,
        target_class=target_class,
        decision_threshold=threshold,
        predict_fn=lambda row: float(engine.model.predict_proba(row)[0, 1]),
        projection_layer=engine.projection,
    )

    pruned_features = {change["feature"] for change in pruned_changes}
    removed_features = {change["feature"] for change in removed_changes}
    if pruned_features != {"worst_radius"}:
        raise AssertionError(f"Expected only 'worst radius' to remain, got {sorted(pruned_features)}.")
    if not {"mean_texture", "area_error"}.issubset(removed_features):
        raise AssertionError(f"Expected redundant smaller changes to be removed, got {sorted(removed_features)}.")
    if not is_minimal:
        raise AssertionError("Expected breast cancer observed case to be minimal after pruning.")

    print("breast_cancer_observed_before_changes", len(changes))
    print("breast_cancer_observed_after_changes", len(pruned_changes))


def _assert_generated_breast_cancer_case_is_pruned() -> None:
    session = _load_example_session("breast_cancer")
    engine = get_session_counterfactual_engine(session.session_id)
    row_index = BREAST_CANCER_OBSERVED_ROW_INDEX
    threshold = BREAST_CANCER_OBSERVED_THRESHOLD
    original = engine.dataset.iloc[[row_index]].copy()
    original_probability = float(engine.model.predict_proba(original)[0, 1])
    target_class = 1 - int(original_probability >= threshold)

    result = generate_counterfactual_for_session(
        session.session_id,
        row_index=row_index,
        threshold=threshold,
        target_class=target_class,
        max_steps=3,
    )
    counterfactuals = result.get("counterfactuals", [])
    if not counterfactuals:
        raise AssertionError("Expected a generated breast cancer counterfactual.")

    counterfactual = counterfactuals[0]
    final_features = {change["feature"] for change in counterfactual["changes"]}
    removed_features = {change["feature"] for change in counterfactual.get("removed_changes", [])}

    if final_features != {"worst_radius"}:
        raise AssertionError(f"Expected generated final changes to contain only 'worst_radius', got {sorted(final_features)}.")
    if "area_error" not in removed_features and "mean_texture" not in removed_features:
        raise AssertionError(f"Expected at least one redundant preliminary change to be pruned, got {sorted(removed_features)}.")
    if counterfactual.get("pruned_num_changes") != 1:
        raise AssertionError(f"Expected pruned_num_changes=1, got {counterfactual.get('pruned_num_changes')}.")
    if counterfactual.get("is_minimal_after_pruning") is not True:
        raise AssertionError("Generated breast cancer counterfactual was not marked minimal after pruning.")

    worst_radius_change = counterfactual["changes"][0]
    if not float(worst_radius_change["new_value"]) > 17.51:
        raise AssertionError(f"Expected threshold-safe worst_radius above 17.51, got {worst_radius_change['new_value']!r}.")

    print("breast_cancer_generated_original_changes", counterfactual.get("original_num_changes"))
    print("breast_cancer_generated_pruned_changes", counterfactual.get("pruned_num_changes"))


def main() -> None:
    session = _load_example_session("adult_income")
    threshold = 0.6

    engine = get_session_counterfactual_engine(session.session_id)
    cached_engine = get_session_counterfactual_engine(session.session_id)
    if engine is not cached_engine:
        raise AssertionError("Counterfactual engine cache did not return the same engine instance.")

    row_index, result = _find_flipping_counterfactual(session.session_id, threshold)
    counterfactual = result["counterfactuals"][0]

    if result["threshold"] != threshold:
        raise AssertionError(f"Expected threshold {threshold}, got {result['threshold']}.")
    if result["current_prediction"] == counterfactual["new_prediction"]:
        raise AssertionError("Counterfactual did not flip the prediction.")
    _assert_returned_changes_are_minimal(session.session_id, row_index, result)
    _assert_observed_breast_cancer_case_prunes()
    _assert_generated_breast_cancer_case_is_pruned()

    print("engine_build_ok", True)
    print("cache_hit_ok", True)
    print("session_id", session.session_id)
    print("row_index", row_index)
    print("threshold", result["threshold"])
    print("current_prediction", result["current_prediction"])
    print("new_prediction", counterfactual["new_prediction"])
    print("counterfactuals", len(result["counterfactuals"]))
    print("original_num_changes", counterfactual.get("original_num_changes"))
    print("pruned_num_changes", counterfactual.get("pruned_num_changes"))


if __name__ == "__main__":
    main()
