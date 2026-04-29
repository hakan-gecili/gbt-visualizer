from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_ROOT.parent

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.session_store import session_store
from app.domain.model_types import EnsembleModel
from app.domain.session_types import SessionState
from app.services.counterfactual_service import generate_counterfactual_for_session
from app.services.dataset_service import apply_dataset_ranges, extract_feature_vector_from_row, load_dataset_from_path, summarize_dataset
from app.services.feature_schema_service import FeatureValue, build_feature_metadata, parse_feature_schema_json
from app.services.model_loader import load_ensemble_model_from_path
from app.services.prediction_service import predict_model


MAX_STEPS = 3
ROWS_PER_DATASET = 12
THRESHOLD = 0.5
DATASETS = ["breast_cancer", "titanic", "adult_income", "bank_marketing"]


@dataclass
class ReplaySummary:
    probability: float | None
    margin: float | None
    label: int | None
    num_changes: int = 0


@dataclass(frozen=True)
class RowCase:
    row_index: int
    probability: float
    label: int
    tags: tuple[str, ...]


@dataclass
class RowResult:
    example_name: str
    row_index: int
    tags: tuple[str, ...]
    target_class: int
    legacy_found: bool
    unified_found: bool
    legacy_runtime_ms: float
    unified_runtime_ms: float
    unified_candidate_count: int | None
    unified_replay_count: int | None
    unified_fallback_used: bool | None
    parity_gap: str | None


def _normalize_feature_key(feature_name: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in str(feature_name)).strip("_").lower()


def _align_schema_and_dataframe(
    schema_overrides: list[dict[str, Any]] | None,
    dataframe: pd.DataFrame,
    feature_names: list[str],
) -> tuple[list[dict[str, Any]] | None, pd.DataFrame]:
    normalized_names = {_normalize_feature_key(feature_name): feature_name for feature_name in feature_names}
    rename_columns: dict[str, str] = {}
    aligned_schema: list[dict[str, Any]] = []

    for feature in schema_overrides or []:
        next_feature = dict(feature)
        feature_name = str(next_feature.get("name", ""))
        model_feature_name = normalized_names.get(_normalize_feature_key(feature_name), feature_name)
        if model_feature_name != feature_name:
            rename_columns[feature_name] = model_feature_name
            next_feature["name"] = model_feature_name
        aligned_schema.append(next_feature)

    for column in dataframe.columns:
        model_feature_name = normalized_names.get(_normalize_feature_key(str(column)))
        if model_feature_name is not None and str(column) != model_feature_name:
            rename_columns[str(column)] = model_feature_name

    if rename_columns:
        dataframe = dataframe.rename(columns=rename_columns)
    return (aligned_schema if schema_overrides is not None else None), dataframe


def _load_lightgbm_example_session(example_name: str) -> SessionState:
    example_dir = REPO_ROOT / "examples" / example_name / "lightgbm"
    model_path = example_dir / "model.txt"
    dataset_path = example_dir / "dataset.csv"
    schema_path = example_dir / "feature_schema.json"

    model, predictor = load_ensemble_model_from_path(model_path)
    dataframe = load_dataset_from_path(str(dataset_path))
    schema_overrides = parse_feature_schema_json(schema_path.read_text(encoding="utf-8")) if schema_path.exists() else None
    schema_overrides, dataframe = _align_schema_and_dataframe(schema_overrides, dataframe, model.feature_names)
    feature_metadata = apply_dataset_ranges(build_feature_metadata(model.feature_names, schema_overrides), dataframe)
    session = SessionState(
        session_id=f"parity-lightgbm-{example_name}",
        model=model,
        predictor=predictor,
        feature_metadata=feature_metadata,
        dataset_frame=dataframe,
        dataset_summary=summarize_dataset(dataframe, model.feature_names),
    )
    session_store.save(session)
    return session


def _assert_result_replays(
    *,
    label: str,
    model: EnsembleModel,
    predictor: object | None,
    feature_metadata: list[dict[str, Any]],
    original_vector: dict[str, FeatureValue],
    result: dict[str, Any],
    target_class: int,
    threshold: float,
    max_changes: int,
) -> ReplaySummary:
    counterfactuals = result.get("counterfactuals", [])
    if not counterfactuals:
        return ReplaySummary(probability=None, margin=None, label=None)

    counterfactual = counterfactuals[0]
    changes = list(counterfactual.get("changes", []))
    if len(changes) > int(max_changes):
        raise AssertionError(
            f"{label} returned {len(changes)} feature changes, exceeding max_changes={int(max_changes)}."
        )

    replay_vector = dict(original_vector)
    for change in changes:
        _assert_feature_value_is_valid(label, feature_metadata, change)
        replay_vector[str(change["feature"])] = change["new_value"]

    _, replay_prediction = predict_model(model, predictor, feature_metadata, replay_vector)
    probability = float(replay_prediction.probability)
    margin = float(replay_prediction.margin)
    replay_label = int(probability >= threshold)
    if replay_label != target_class:
        raise AssertionError(f"{label} returned non-flipping counterfactual: got {replay_label}, expected {target_class}.")
    if replay_label != int(counterfactual["new_prediction"]):
        raise AssertionError(f"{label} replay label does not match returned label.")
    if "new_probability" in counterfactual and not math.isclose(
        probability,
        float(counterfactual["new_probability"]),
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise AssertionError(f"{label} replay probability does not match returned probability.")
    if "new_margin" in counterfactual and not math.isclose(
        margin,
        float(counterfactual["new_margin"]),
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise AssertionError(f"{label} replay margin does not match returned margin.")
    return ReplaySummary(probability=probability, margin=margin, label=replay_label, num_changes=len(changes))


def _assert_feature_value_is_valid(
    label: str,
    feature_metadata: list[dict[str, Any]],
    change: dict[str, Any],
) -> None:
    feature_by_name = {str(feature["name"]): feature for feature in feature_metadata}
    feature_name = str(change["feature"])
    feature = feature_by_name.get(feature_name)
    if feature is None:
        raise AssertionError(f"{label} returned unknown feature {feature_name!r}.")

    if str(feature.get("type")) in {"categorical", "binary"}:
        allowed_values = {str(option["value"]) for option in feature.get("options", [])}
        if str(change["new_value"]) not in allowed_values:
            raise AssertionError(
                f"{label} returned invalid categorical value {change['new_value']!r} "
                f"for {feature_name!r}; expected one of {sorted(allowed_values)}."
            )
    if _is_integer_like_feature(feature) and isinstance(change["new_value"], (int, float)):
        if not float(change["new_value"]).is_integer():
            raise AssertionError(f"{label} returned fractional integer-like value {change!r}.")
    if _is_missing_value(change.get("old_value")) and not _is_missing_value(change.get("new_value")):
        raise AssertionError(f"{label} changed missing value for {feature_name!r}; missing values should be preserved.")


def _changed_features(result: dict[str, Any]) -> list[str]:
    counterfactuals = result.get("counterfactuals", [])
    if not counterfactuals:
        return []
    return [str(change["feature"]) for change in counterfactuals[0].get("changes", [])]


def _select_row_cases(session: SessionState, *, max_rows: int = ROWS_PER_DATASET) -> list[RowCase]:
    assert session.dataset_frame is not None
    scored: list[dict[str, Any]] = []
    categorical_features = {
        str(feature["name"])
        for feature in session.feature_metadata
        if str(feature.get("type")) in {"categorical", "binary"}
    }

    for row_index in range(len(session.dataset_frame)):
        raw_vector = extract_feature_vector_from_row(session.dataset_frame, row_index, session.feature_metadata)
        _, prediction = predict_model(
            session.model,
            session.predictor,
            session.feature_metadata,
            raw_vector,
        )
        source_row = session.dataset_frame.iloc[row_index]
        has_missing = any(pd.isna(source_row.get(feature_name)) for feature_name in session.model.feature_names)
        categorical_values = tuple(
            str(source_row.get(feature_name))
            for feature_name in sorted(categorical_features)
            if feature_name in source_row
        )
        scored.append(
            {
                "row_index": row_index,
                "probability": float(prediction.probability),
                "label": int(float(prediction.probability) >= THRESHOLD),
                "has_missing": has_missing,
                "categorical_values": categorical_values,
            }
        )

    selected: dict[int, set[str]] = {}

    def _add(row: dict[str, Any], tag: str) -> None:
        selected.setdefault(int(row["row_index"]), set()).add(tag)

    for row in sorted(scored, key=lambda item: abs(float(item["probability"]) - THRESHOLD))[:4]:
        _add(row, "near_boundary")
    for row in sorted(scored, key=lambda item: float(item["probability"]))[:3]:
        _add(row, "far_low")
    for row in sorted(scored, key=lambda item: float(item["probability"]), reverse=True)[:3]:
        _add(row, "far_high")
    for row in sorted((item for item in scored if item["has_missing"]), key=lambda item: abs(float(item["probability"]) - THRESHOLD))[:3]:
        _add(row, "missing")

    seen_categories: set[tuple[str, ...]] = set()
    for row in scored:
        categories = row["categorical_values"]
        if categories and categories not in seen_categories:
            _add(row, "categorical")
            seen_categories.add(categories)
        if len(seen_categories) >= 4:
            break

    if len(selected) < max_rows and scored:
        denominator = max(max_rows - 1, 1)
        for index in range(max_rows):
            row = scored[round(index * (len(scored) - 1) / denominator)]
            _add(row, "spread")
            if len(selected) >= max_rows:
                break

    scored_by_index = {int(row["row_index"]): row for row in scored}
    cases = [
        RowCase(
            row_index=row_index,
            probability=float(scored_by_index[row_index]["probability"]),
            label=int(scored_by_index[row_index]["label"]),
            tags=tuple(sorted(tags)),
        )
        for row_index, tags in selected.items()
    ]
    return sorted(cases, key=lambda case: (min(_tag_priority(tag) for tag in case.tags), case.row_index))[:max_rows]


def _tag_priority(tag: str) -> int:
    priorities = {
        "near_boundary": 0,
        "missing": 1,
        "categorical": 2,
        "far_low": 3,
        "far_high": 4,
        "spread": 5,
    }
    return priorities.get(tag, 99)


def _run_dataset_row(session: SessionState, example_name: str, row_case: RowCase) -> RowResult:
    assert session.dataset_frame is not None
    row_index = row_case.row_index

    raw_vector = extract_feature_vector_from_row(session.dataset_frame, row_index, session.feature_metadata)
    original_vector, original_prediction = predict_model(
        session.model,
        session.predictor,
        session.feature_metadata,
        raw_vector,
    )
    target_class = 1 - int(float(original_prediction.probability) >= THRESHOLD)

    legacy_started = time.perf_counter()
    legacy_result = generate_counterfactual_for_session(
        session.session_id,
        row_index=row_index,
        threshold=THRESHOLD,
        target_class=target_class,
        max_steps=MAX_STEPS,
    )
    legacy_runtime_ms = 1000.0 * (time.perf_counter() - legacy_started)

    unified_started = time.perf_counter()
    unified_result = _generate_unified_lightgbm_with_flag(
        session_id=session.session_id,
        row_index=row_index,
        threshold=THRESHOLD,
        target_class=target_class,
        max_steps=MAX_STEPS,
    )
    unified_runtime_ms = 1000.0 * (time.perf_counter() - unified_started)

    legacy_replay = _assert_result_replays(
        label=f"{example_name}[{row_index}] legacy",
        model=session.model,
        predictor=session.predictor,
        feature_metadata=session.feature_metadata,
        original_vector=original_vector,
        result=legacy_result,
        target_class=target_class,
        threshold=THRESHOLD,
        max_changes=MAX_STEPS,
    )
    unified_replay = _assert_result_replays(
        label=f"{example_name}[{row_index}] unified",
        model=session.model,
        predictor=session.predictor,
        feature_metadata=session.feature_metadata,
        original_vector=original_vector,
        result=unified_result,
        target_class=target_class,
        threshold=THRESHOLD,
        max_changes=MAX_STEPS,
    )

    legacy_found = bool(legacy_result.get("counterfactuals"))
    unified_found = bool(unified_result.get("counterfactuals"))
    legacy_features = _changed_features(legacy_result)
    unified_features = _changed_features(unified_result)
    diagnostics = unified_result.get("diagnostics", {})
    top_ranked_features = diagnostics.get("top_ranked_features", [])
    ranking_overlap = sorted(set(legacy_features) & set(top_ranked_features))
    parity_gap = None
    if legacy_found and not unified_found:
        parity_gap = "unified_missing_counterfactual"
    if unified_found and not legacy_found:
        parity_gap = "legacy_missing_counterfactual"

    print(
        f"dataset={example_name} row={row_index} target={target_class} "
        f"original_probability={float(original_prediction.probability):.6f} tags={list(row_case.tags)}"
    )
    print(f"  legacy_found={legacy_found} unified_found={unified_found}")
    print(f"  legacy_max_steps={MAX_STEPS} unified_max_steps={MAX_STEPS}")
    print(f"  legacy_max_changes={MAX_STEPS} unified_max_changes={MAX_STEPS}")
    print(f"  legacy_changed_features={legacy_features}")
    print(f"  unified_changed_features={unified_features}")
    print(f"  legacy_num_changes={legacy_replay.num_changes} unified_num_changes={unified_replay.num_changes}")
    print(f"  unified_top_ranked_features={top_ranked_features}")
    print(f"  candidate_ranking_overlap={ranking_overlap}")
    print(
        f"  legacy_replay_probability={legacy_replay.probability} "
        f"legacy_replay_margin={legacy_replay.margin} legacy_replay_label={legacy_replay.label}"
    )
    print(
        f"  unified_replay_probability={unified_replay.probability} "
        f"unified_replay_margin={unified_replay.margin} unified_replay_label={unified_replay.label}"
    )
    print(f"  legacy_runtime_ms={legacy_runtime_ms:.3f} unified_runtime_ms={unified_runtime_ms:.3f}")
    print(
        f"  unified_candidate_count={diagnostics.get('candidate_count')} "
        f"unified_replay_count={diagnostics.get('replay_count')} "
        f"unified_fallback_used={diagnostics.get('fallback_used')}"
    )
    if _runtime_gap_is_notable(legacy_runtime_ms, unified_runtime_ms):
        print("  runtime_gap=significant")
    if parity_gap:
        print(f"  parity_gap={parity_gap}")

    return RowResult(
        example_name=example_name,
        row_index=row_index,
        tags=row_case.tags,
        target_class=target_class,
        legacy_found=legacy_found,
        unified_found=unified_found,
        legacy_runtime_ms=legacy_runtime_ms,
        unified_runtime_ms=unified_runtime_ms,
        unified_candidate_count=diagnostics.get("candidate_count"),
        unified_replay_count=diagnostics.get("replay_count"),
        unified_fallback_used=diagnostics.get("fallback_used"),
        parity_gap=parity_gap,
    )


def _generate_unified_lightgbm_with_flag(
    *,
    session_id: str,
    row_index: int,
    threshold: float,
    target_class: int,
    max_steps: int,
) -> dict[str, Any]:
    previous = os.environ.get("USE_UNIFIED_CF_FOR_LIGHTGBM")
    os.environ["USE_UNIFIED_CF_FOR_LIGHTGBM"] = "1"
    try:
        return generate_counterfactual_for_session(
            session_id,
            row_index=row_index,
            threshold=threshold,
            target_class=target_class,
            max_steps=max_steps,
        )
    finally:
        if previous is None:
            os.environ.pop("USE_UNIFIED_CF_FOR_LIGHTGBM", None)
        else:
            os.environ["USE_UNIFIED_CF_FOR_LIGHTGBM"] = previous


def _runtime_gap_is_notable(legacy_ms: float, unified_ms: float) -> bool:
    faster = min(float(legacy_ms), float(unified_ms))
    slower = max(float(legacy_ms), float(unified_ms))
    return slower >= 250.0 and slower / max(faster, 1e-9) >= 3.0


def _is_integer_like_feature(feature: dict[str, Any]) -> bool:
    name = str(feature.get("name", "")).lower()
    return name.endswith("-num") or name.endswith("_num") or name.endswith(" count") or name.endswith("_count")


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def main() -> None:
    results: list[RowResult] = []
    correctness_failures: list[str] = []

    for example_name in DATASETS:
        session = _load_lightgbm_example_session(example_name)
        row_cases = _select_row_cases(session)
        print(f"dataset={example_name} selected_rows={[case.row_index for case in row_cases]}")
        for row_case in row_cases:
            try:
                results.append(_run_dataset_row(session, example_name, row_case))
            except Exception as exc:
                message = f"{example_name}[{row_case.row_index}] {type(exc).__name__}: {exc}"
                correctness_failures.append(message)
                print(f"correctness_failure={message}")

    parity_gaps = [result for result in results if result.parity_gap]
    runtime_gaps = [
        result
        for result in results
        if _runtime_gap_is_notable(result.legacy_runtime_ms, result.unified_runtime_ms)
    ]
    legacy_avg = sum(result.legacy_runtime_ms for result in results) / max(len(results), 1)
    unified_avg = sum(result.unified_runtime_ms for result in results) / max(len(results), 1)
    unified_fallbacks = sum(1 for result in results if result.unified_fallback_used)

    print("parity_rows_tested", len(results))
    print("parity_correctness_failures", len(correctness_failures))
    print("parity_gaps_reported", len(parity_gaps))
    print("parity_runtime_gaps_reported", len(runtime_gaps))
    print("legacy_average_runtime_ms", round(legacy_avg, 3))
    print("unified_average_runtime_ms", round(unified_avg, 3))
    print("unified_fallback_rows", unified_fallbacks)
    print(
        "notable_edge_cases",
        [
            {
                "dataset": result.example_name,
                "row": result.row_index,
                "gap": result.parity_gap,
                "tags": list(result.tags),
            }
            for result in parity_gaps[:10]
        ],
    )
    if correctness_failures:
        print("correctness_failure_details", correctness_failures)
        raise AssertionError(f"{len(correctness_failures)} parity correctness failures detected.")
    if parity_gaps or runtime_gaps:
        print(
            "recommendation",
            "Keep unified LightGBM behind USE_UNIFIED_CF_FOR_LIGHTGBM=1 until parity gaps and fallback-heavy rows are reviewed.",
        )
    else:
        print("recommendation", "Unified LightGBM is a candidate for default rollout after a final manual UI pass.")
    print("counterfactual_parity_smoke_ok", True)


if __name__ == "__main__":
    main()
