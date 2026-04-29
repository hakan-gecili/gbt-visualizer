from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from smoke_test_counterfactual_parity import _load_lightgbm_example_session
from smoke_test_xgboost_integration import _load_xgboost_example_session

from app.domain.model_types import EnsembleModel
from app.domain.session_types import SessionState
from app.services.cf_engine.fast_prediction import (
    AppPredictModelEvaluator,
    LightGBMBoosterFastEvaluator,
    XGBoostBoosterFastEvaluator,
)
from app.services.cf_engine.unified_counterfactual_engine import _generate_candidates
from app.services.dataset_service import extract_feature_vector_from_row
from app.services.feature_schema_service import FeatureValue, encode_feature_vector
from app.services.prediction_service import predict_model


DATASETS = ("breast_cancer", "adult_income", "bank_marketing", "titanic")
THRESHOLD = 0.5


@dataclass(frozen=True)
class ComparisonSummary:
    model_family: str
    dataset: str
    rows: int
    max_margin_diff: float
    avg_margin_diff: float
    max_probability_diff: float
    avg_probability_diff: float
    label_mismatches: int


def _candidate_rows_from_session(
    session: SessionState,
    row_indices: list[int],
    limit: int = 100,
) -> list[dict[str, FeatureValue]]:
    model = session.model
    feature_metadata = session.feature_metadata
    feature_by_name = {str(feature["name"]): feature for feature in feature_metadata}
    rows: list[dict[str, FeatureValue]] = []

    for row_index in row_indices:
        raw_vector = extract_feature_vector_from_row(session.dataset_frame, row_index, feature_metadata)
        prepared_vector, _ = predict_model(model, session.predictor, feature_metadata, raw_vector)
        encoded_vector = encode_feature_vector(feature_metadata, prepared_vector)
        seen: set[tuple[tuple[str, str], ...]] = set()
        for candidate in _generate_candidates(model, feature_by_name, prepared_vector, encoded_vector):
            trial_vector = dict(prepared_vector)
            trial_vector[candidate.feature] = candidate.new_value
            key = tuple((feature_name, repr(trial_vector.get(feature_name))) for feature_name in model.feature_names)
            if key in seen:
                continue
            seen.add(key)
            rows.append(trial_vector)
            if len(rows) >= limit:
                return rows

    if not rows:
        raise AssertionError(f"No candidate rows generated for {session.session_id}.")
    return rows


def _comparison(
    *,
    model_family: str,
    dataset: str,
    model: EnsembleModel,
    predictor: object,
    feature_metadata: list[dict[str, Any]],
    rows: list[dict[str, FeatureValue]],
) -> ComparisonSummary:
    app_evaluator = AppPredictModelEvaluator(model=model, predictor=predictor, feature_metadata=feature_metadata)
    if model_family == "lightgbm":
        fast_evaluator = LightGBMBoosterFastEvaluator(
            model=model,
            booster=predictor,
            feature_metadata=feature_metadata,
        )
    elif model_family == "xgboost":
        fast_evaluator = XGBoostBoosterFastEvaluator(
            model=model,
            booster=predictor,
            feature_metadata=feature_metadata,
        )
    else:
        raise AssertionError(f"Unsupported model family: {model_family}")

    app_prediction = app_evaluator.predict_batch(rows, threshold=THRESHOLD)
    fast_prediction = fast_evaluator.predict_batch(rows, threshold=THRESHOLD)

    margin_diff = np.abs(app_prediction.margins - fast_prediction.margins)
    probability_diff = np.abs(app_prediction.probabilities - fast_prediction.probabilities)
    label_mismatches = int(np.sum(app_prediction.labels != fast_prediction.labels))
    summary = ComparisonSummary(
        model_family=model_family,
        dataset=dataset,
        rows=len(rows),
        max_margin_diff=float(np.max(margin_diff)),
        avg_margin_diff=float(np.mean(margin_diff)),
        max_probability_diff=float(np.max(probability_diff)),
        avg_probability_diff=float(np.mean(probability_diff)),
        label_mismatches=label_mismatches,
    )
    print(
        "comparison",
        model_family,
        dataset,
        f"rows={summary.rows}",
        f"max_margin_diff={summary.max_margin_diff:.12g}",
        f"avg_margin_diff={summary.avg_margin_diff:.12g}",
        f"max_probability_diff={summary.max_probability_diff:.12g}",
        f"avg_probability_diff={summary.avg_probability_diff:.12g}",
        f"label_mismatches={summary.label_mismatches}",
    )
    return summary


def _cycle_rows(rows: list[dict[str, FeatureValue]], count: int) -> list[dict[str, FeatureValue]]:
    return [rows[index % len(rows)] for index in range(count)]


def _time_call(callback) -> float:
    started = time.perf_counter()
    callback()
    return time.perf_counter() - started


def _benchmark(
    *,
    model_family: str,
    dataset: str,
    model: EnsembleModel,
    predictor: object,
    feature_metadata: list[dict[str, Any]],
    rows: list[dict[str, FeatureValue]],
) -> None:
    app_evaluator = AppPredictModelEvaluator(model=model, predictor=predictor, feature_metadata=feature_metadata)
    if model_family == "lightgbm":
        fast_evaluator = LightGBMBoosterFastEvaluator(
            model=model,
            booster=predictor,
            feature_metadata=feature_metadata,
        )
    else:
        fast_evaluator = XGBoostBoosterFastEvaluator(
            model=model,
            booster=predictor,
            feature_metadata=feature_metadata,
        )

    for count in (100, 500):
        benchmark_rows = _cycle_rows(rows, count)
        app_seconds = _time_call(lambda: app_evaluator.predict_batch(benchmark_rows, threshold=THRESHOLD))
        fast_single_seconds = _time_call(
            lambda: [fast_evaluator.predict_batch([row], threshold=THRESHOLD) for row in benchmark_rows]
        )
        fast_batch_seconds = _time_call(lambda: fast_evaluator.predict_batch(benchmark_rows, threshold=THRESHOLD))
        print(
            "benchmark",
            model_family,
            dataset,
            f"n={count}",
            f"predict_model_ms={1000.0 * app_seconds:.3f}",
            f"predict_model_per_row_ms={1000.0 * app_seconds / count:.4f}",
            f"fast_single_ms={1000.0 * fast_single_seconds:.3f}",
            f"fast_single_per_row_ms={1000.0 * fast_single_seconds / count:.4f}",
            f"fast_batch_ms={1000.0 * fast_batch_seconds:.3f}",
            f"fast_batch_per_row_ms={1000.0 * fast_batch_seconds / count:.4f}",
            f"speedup_batch={app_seconds / max(fast_batch_seconds, 1e-12):.2f}x",
        )


def _assert_lightgbm_summary(summary: ComparisonSummary) -> None:
    if summary.label_mismatches:
        raise AssertionError(f"{summary.dataset} LightGBM had {summary.label_mismatches} label mismatches.")
    if summary.max_margin_diff > 1e-9 or summary.max_probability_diff > 1e-12:
        raise AssertionError(
            f"{summary.dataset} LightGBM fast prediction drifted from predict_model: "
            f"max_margin_diff={summary.max_margin_diff}, max_probability_diff={summary.max_probability_diff}"
        )


def _assert_xgboost_summary(summary: ComparisonSummary) -> None:
    mismatch_rate = summary.label_mismatches / max(summary.rows, 1)
    if mismatch_rate > 0.05:
        raise AssertionError(
            f"{summary.dataset} XGBoost fast prediction had severe label mismatch rate: "
            f"{summary.label_mismatches}/{summary.rows}."
        )
    if not math.isfinite(summary.max_margin_diff) or not math.isfinite(summary.max_probability_diff):
        raise AssertionError(f"{summary.dataset} XGBoost produced non-finite prediction differences.")


def main() -> None:
    lightgbm_summaries: list[ComparisonSummary] = []
    xgboost_summaries: list[ComparisonSummary] = []

    for dataset in DATASETS:
        lightgbm_session = _load_lightgbm_example_session(dataset)
        lightgbm_rows = _candidate_rows_from_session(lightgbm_session, [0, 1, 2, 3, 8, 12, 29], limit=100)
        lightgbm_summary = _comparison(
            model_family="lightgbm",
            dataset=dataset,
            model=lightgbm_session.model,
            predictor=lightgbm_session.predictor,
            feature_metadata=lightgbm_session.feature_metadata,
            rows=lightgbm_rows,
        )
        _assert_lightgbm_summary(lightgbm_summary)
        lightgbm_summaries.append(lightgbm_summary)

        xgb_session_id, xgb_model, xgb_predictor, xgb_feature_metadata, xgb_dataframe = _load_xgboost_example_session(
            dataset
        )
        xgb_session = SessionState(
            session_id=xgb_session_id,
            model=xgb_model,
            predictor=xgb_predictor,
            feature_metadata=xgb_feature_metadata,
            dataset_frame=xgb_dataframe,
        )
        xgb_rows = _candidate_rows_from_session(xgb_session, [0, 1, 2, 3, 8, 12, 29], limit=100)
        xgb_summary = _comparison(
            model_family="xgboost",
            dataset=dataset,
            model=xgb_model,
            predictor=xgb_predictor,
            feature_metadata=xgb_feature_metadata,
            rows=xgb_rows,
        )
        _assert_xgboost_summary(xgb_summary)
        xgboost_summaries.append(xgb_summary)

    adult_lightgbm = _load_lightgbm_example_session("adult_income")
    adult_lightgbm_rows = _candidate_rows_from_session(adult_lightgbm, [8, 1, 12], limit=160)
    _benchmark(
        model_family="lightgbm",
        dataset="adult_income",
        model=adult_lightgbm.model,
        predictor=adult_lightgbm.predictor,
        feature_metadata=adult_lightgbm.feature_metadata,
        rows=adult_lightgbm_rows,
    )

    _, adult_xgb_model, adult_xgb_predictor, adult_xgb_feature_metadata, adult_xgb_dataframe = (
        _load_xgboost_example_session("adult_income")
    )
    adult_xgb_session = SessionState(
        session_id="fast-prediction-xgboost-adult-income",
        model=adult_xgb_model,
        predictor=adult_xgb_predictor,
        feature_metadata=adult_xgb_feature_metadata,
        dataset_frame=adult_xgb_dataframe,
    )
    adult_xgb_rows = _candidate_rows_from_session(adult_xgb_session, [8, 1, 12], limit=160)
    _benchmark(
        model_family="xgboost",
        dataset="adult_income",
        model=adult_xgb_model,
        predictor=adult_xgb_predictor,
        feature_metadata=adult_xgb_feature_metadata,
        rows=adult_xgb_rows,
    )

    print(
        "fast_prediction_test_passed",
        f"lightgbm_datasets={len(lightgbm_summaries)}",
        f"xgboost_datasets={len(xgboost_summaries)}",
    )


if __name__ == "__main__":
    main()
