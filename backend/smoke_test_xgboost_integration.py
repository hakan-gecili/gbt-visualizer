from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.services.feature_schema_service import build_feature_metadata
from app.services.model_loader import load_ensemble_model_from_path
from app.services.prediction_service import predict_model


FEATURE_NAMES = ["age", "income", "debt"]


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

    print("XGBoost integration smoke test passed.")


if __name__ == "__main__":
    main()
