"""Clustering quality metrics for reviewed memberships and matched intervals.

Event-partition metrics require a label for every event in the evaluated scope. Delivered
IZPADI rows do not provide those labels and must not be passed off as membership truth.
Temporal metrics operate on an already matched predicted/reference interval table; candidate
generation and one-to-one matching remain separate policy decisions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

_METRIC_NAMES: Final[tuple[str, ...]] = (
    "adjusted_rand_index",
    "bcubed_precision",
    "bcubed_recall",
    "bcubed_f1",
    "mean_matched_cluster_iou",
    "matched_incident_precision",
    "matched_incident_recall",
    "matched_incident_f1",
    "merge_rate",
    "fragmentation_rate",
)
_MATCH_COLUMNS: Final[tuple[str, ...]] = (
    "reference_label",
    "predicted_label",
    "intersection_event_count",
    "reference_event_count",
    "predicted_event_count",
    "cluster_iou",
    "cluster_f1",
)
_DEFAULT_BOUNDARY_TOLERANCE: Final[pd.Timedelta] = pd.Timedelta(minutes=15)
_DEFAULT_SEARCH_BUFFER: Final[pd.Timedelta] = pd.Timedelta(minutes=60)


def add_minute_precision_reference_bounds(
    references: pd.DataFrame,
    *,
    start_column: str = "reference_start",
    end_column: str = "reference_end",
) -> pd.DataFrame:
    """Add effective bounds for minute-granularity interval annotations.

    A non-zero interval includes the complete recorded end minute and is
    represented as a half-open interval ending one minute later. A zero-duration
    reference remains a point observation and receives no artificial duration.
    Recorded boundaries are retained unchanged for audit and comparison.
    """
    required = {start_column, end_column}
    missing = sorted(required.difference(references.columns))
    if missing:
        raise ValueError(f"reference frame is missing columns: {missing}")

    result = references.copy()
    result[start_column] = pd.to_datetime(result[start_column], errors="raise")
    result[end_column] = pd.to_datetime(result[end_column], errors="raise")
    if result[[start_column, end_column]].isna().any().any():
        raise ValueError("reference boundaries cannot be missing")
    if result[end_column].lt(result[start_column]).any():
        raise ValueError("reference interval end precedes its start")

    is_point = result[start_column].eq(result[end_column])
    result["reference_end_recorded"] = result[end_column]
    result["reference_end_effective"] = result[end_column].where(
        is_point,
        result[end_column].add(pd.Timedelta(minutes=1)),
    )
    result["reference_time_kind"] = pd.Series(
        np.where(is_point, "point", "minute_granularity_interval"),
        index=result.index,
        dtype="string",
    )
    return result


def build_exact_asset_candidates(
    clusters: pd.DataFrame,
    references: pd.DataFrame,
    *,
    predicted_end_column: str = "cluster_end",
    boundary_tolerance: pd.Timedelta = _DEFAULT_BOUNDARY_TOLERANCE,
    search_buffer: pd.Timedelta = _DEFAULT_SEARCH_BUFFER,
) -> pd.DataFrame:
    """Return exact-asset candidate pairs under the frozen proxy protocol."""
    cluster_columns = {"cluster_id", "SifraSredstva", "cluster_start", predicted_end_column}
    reference_columns = {"reference_id", "anchor_asset", "reference_start", "reference_end"}
    missing_clusters = sorted(cluster_columns.difference(clusters.columns))
    missing_references = sorted(reference_columns.difference(references.columns))
    if missing_clusters:
        raise ValueError(f"cluster frame is missing columns: {missing_clusters}")
    if missing_references:
        raise ValueError(f"reference frame is missing columns: {missing_references}")
    if boundary_tolerance < pd.Timedelta(0) or search_buffer < pd.Timedelta(0):
        raise ValueError("candidate timing tolerances cannot be negative")

    cluster_frame = clusters.loc[
        :,
        ["cluster_id", "SifraSredstva", "cluster_start", predicted_end_column],
    ].dropna()
    cluster_frame = cluster_frame.rename(columns={predicted_end_column: "cluster_end"})
    cluster_frame["anchor_asset"] = cluster_frame.pop("SifraSredstva").astype("string")
    reference_frame = references.loc[
        :,
        ["reference_id", "anchor_asset", "reference_start", "reference_end"],
    ].dropna()
    reference_frame["anchor_asset"] = reference_frame["anchor_asset"].astype("string")

    pairs = cluster_frame.merge(reference_frame, on="anchor_asset", how="inner", validate="many_to_many")
    if pairs.empty:
        return pairs.assign(
            start_error_seconds=pd.Series(dtype="float64"),
            end_error_seconds=pd.Series(dtype="float64"),
            boundary_gap_seconds=pd.Series(dtype="float64"),
            matched_boundary=pd.Series(dtype="string"),
        )

    search_compatible = pairs["cluster_start"].le(pairs["reference_end"].add(search_buffer)) & pairs["cluster_end"].ge(
        pairs["reference_start"].sub(search_buffer)
    )
    pairs["start_error_seconds"] = pairs["cluster_start"].sub(pairs["reference_start"]).abs().dt.total_seconds()
    pairs["end_error_seconds"] = pairs["cluster_end"].sub(pairs["reference_end"]).abs().dt.total_seconds()
    pairs["boundary_gap_seconds"] = pairs[["start_error_seconds", "end_error_seconds"]].min(axis=1)
    pairs["matched_boundary"] = np.where(
        pairs["start_error_seconds"].le(pairs["end_error_seconds"]),
        "start",
        "end",
    )
    within_tolerance = pairs["boundary_gap_seconds"].le(boundary_tolerance.total_seconds())
    return (
        pairs.loc[search_compatible & within_tolerance]
        .sort_values(["reference_start", "reference_id", "boundary_gap_seconds", "cluster_id"])
        .reset_index(drop=True)
    )


def optimal_one_to_one_interval_matches(
    candidates: pd.DataFrame,
    references: pd.DataFrame,
    *,
    boundary_tolerance: pd.Timedelta = _DEFAULT_BOUNDARY_TOLERANCE,
) -> pd.DataFrame:
    """Maximize interval match count, then minimize boundary gap deterministically."""
    if candidates.empty:
        return candidates.copy()
    if boundary_tolerance < pd.Timedelta(0):
        raise ValueError("boundary tolerance cannot be negative")

    reference_ids = (
        references.sort_values(["reference_start", "reference_id"])["reference_id"].drop_duplicates().tolist()
    )
    cluster_ids = sorted(candidates["cluster_id"].unique().tolist())
    reference_position = {value: position for position, value in enumerate(reference_ids)}
    cluster_position = {value: position for position, value in enumerate(cluster_ids)}
    tolerance_seconds = boundary_tolerance.total_seconds()
    dummy_penalty = (len(reference_ids) + 1) * (tolerance_seconds + 1)
    costs = np.full(
        (len(reference_ids), len(cluster_ids) + len(reference_ids)),
        2 * dummy_penalty,
        dtype="float64",
    )
    for position in range(len(reference_ids)):
        costs[position, len(cluster_ids) + position] = dummy_penalty
    for candidate in candidates.itertuples(index=False):
        left = reference_position[candidate.reference_id]
        right = cluster_position[candidate.cluster_id]
        costs[left, right] = min(costs[left, right], candidate.boundary_gap_seconds)

    assigned_references, assigned_columns = linear_sum_assignment(costs)
    assigned_pairs = [
        (reference_ids[left], cluster_ids[right])
        for left, right in zip(assigned_references, assigned_columns, strict=True)
        if right < len(cluster_ids) and costs[left, right] <= tolerance_seconds
    ]
    assignment = pd.DataFrame(assigned_pairs, columns=["reference_id", "cluster_id"])
    return assignment.merge(
        candidates,
        on=["reference_id", "cluster_id"],
        how="left",
        validate="one_to_one",
    )


def evaluate_episode_clusters(
    clusters: pd.DataFrame,
    references: pd.DataFrame,
    *,
    predicted_end_column: str = "cluster_end",
    boundary_tolerance: pd.Timedelta = _DEFAULT_BOUNDARY_TOLERANCE,
    search_buffer: pd.Timedelta = _DEFAULT_SEARCH_BUFFER,
) -> tuple[dict[str, float | int], pd.DataFrame, pd.DataFrame]:
    """Evaluate episode intervals against fuzzy references without membership claims."""
    candidates = build_exact_asset_candidates(
        clusters,
        references,
        predicted_end_column=predicted_end_column,
        boundary_tolerance=boundary_tolerance,
        search_buffer=search_buffer,
    )
    matches = optimal_one_to_one_interval_matches(
        candidates,
        references,
        boundary_tolerance=boundary_tolerance,
    )
    reference_candidate_counts = candidates.groupby("reference_id")["cluster_id"].nunique()
    cluster_candidate_counts = candidates.groupby("cluster_id")["reference_id"].nunique()
    reference_count = len(references)
    cluster_count = len(clusters)
    candidate_cluster_count = candidates["cluster_id"].nunique()
    match_count = len(matches)
    recall_proxy = match_count / reference_count if reference_count else float("nan")
    candidate_alignment = match_count / candidate_cluster_count if candidate_cluster_count else float("nan")
    proxy_f1_denominator = recall_proxy + candidate_alignment
    metrics: dict[str, float | int] = {
        "reference_count": reference_count,
        "cluster_count": cluster_count,
        "candidate_pair_count": len(candidates),
        "candidate_cluster_count": candidate_cluster_count,
        "matched_reference_count": match_count,
        "reference_recall_proxy": recall_proxy,
        "candidate_cluster_alignment_proxy": candidate_alignment,
        "all_cluster_reference_alignment": match_count / cluster_count if cluster_count else float("nan"),
        "proxy_f1_candidate_scope": (
            2 * recall_proxy * candidate_alignment / proxy_f1_denominator if proxy_f1_denominator else float("nan")
        ),
        "split_candidate_rate_references": (
            reference_candidate_counts.gt(1).sum() / reference_count if reference_count else float("nan")
        ),
        "merge_candidate_rate_clusters": (
            cluster_candidate_counts.gt(1).sum() / candidate_cluster_count if candidate_cluster_count else float("nan")
        ),
        "median_boundary_gap_seconds": matches["boundary_gap_seconds"].median() if match_count else float("nan"),
        "p90_boundary_gap_seconds": (matches["boundary_gap_seconds"].quantile(0.9) if match_count else float("nan")),
        "median_start_error_seconds": matches["start_error_seconds"].median() if match_count else float("nan"),
        "median_end_error_seconds": matches["end_error_seconds"].median() if match_count else float("nan"),
    }
    return metrics, candidates, matches


def harmonic_mean(precision: float, recall: float) -> float:
    """Return the harmonic mean, or NaN when either input is undefined."""
    if np.isnan(precision) or np.isnan(recall):
        return float("nan")
    denominator = precision + recall
    return 2 * precision * recall / denominator if denominator else 0.0


def _validated_label_codes(
    reference_labels: Sequence[object],
    predicted_labels: Sequence[object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reference = pd.Series(reference_labels, dtype="object")
    predicted = pd.Series(predicted_labels, dtype="object")
    if len(reference) != len(predicted):
        raise ValueError("reference and predicted labels must have equal length")
    if reference.isna().any() or predicted.isna().any():
        raise ValueError("every evaluated event must have both a reference and predicted label")

    reference_codes, reference_values = pd.factorize(reference, sort=False)
    predicted_codes, predicted_values = pd.factorize(predicted, sort=False)
    return reference_codes, predicted_codes, reference_values, predicted_values


def _partition_counts(
    reference_codes: np.ndarray,
    predicted_codes: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if len(reference_codes) == 0:
        return (
            pd.DataFrame(columns=["reference_code", "predicted_code", "intersection_event_count"]),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )

    frame = pd.DataFrame(
        {
            "reference_code": reference_codes,
            "predicted_code": predicted_codes,
        }
    )
    intersections = (
        frame.groupby(["reference_code", "predicted_code"], sort=False, observed=True)
        .size()
        .rename("intersection_event_count")
        .reset_index()
    )
    reference_sizes = np.bincount(reference_codes)
    predicted_sizes = np.bincount(predicted_codes)
    return intersections, reference_sizes, predicted_sizes


def bcubed_scores(
    reference_labels: Sequence[object],
    predicted_labels: Sequence[object],
) -> dict[str, float]:
    """Compute event-averaged B-cubed precision, recall and F1."""
    reference_codes, predicted_codes, _, _ = _validated_label_codes(reference_labels, predicted_labels)
    if len(reference_codes) == 0:
        return {
            "bcubed_precision": float("nan"),
            "bcubed_recall": float("nan"),
            "bcubed_f1": float("nan"),
        }

    intersections, reference_sizes, predicted_sizes = _partition_counts(reference_codes, predicted_codes)
    cell_counts = intersections.set_index(["reference_code", "predicted_code"])["intersection_event_count"]
    event_cells = pd.MultiIndex.from_arrays([reference_codes, predicted_codes])
    shared = cell_counts.reindex(event_cells).to_numpy(dtype="float64")
    precision = float(np.mean(shared / predicted_sizes[predicted_codes]))
    recall = float(np.mean(shared / reference_sizes[reference_codes]))
    return {
        "bcubed_precision": precision,
        "bcubed_recall": recall,
        "bcubed_f1": harmonic_mean(precision, recall),
    }


def match_clusters_by_membership(
    reference_labels: Sequence[object],
    predicted_labels: Sequence[object],
    *,
    max_dense_cells: int = 5_000_000,
) -> pd.DataFrame:
    """Return maximum-total-IoU one-to-one matches between nonempty clusters."""
    reference_codes, predicted_codes, reference_values, predicted_values = _validated_label_codes(
        reference_labels,
        predicted_labels,
    )
    intersections, reference_sizes, predicted_sizes = _partition_counts(reference_codes, predicted_codes)
    if intersections.empty:
        return pd.DataFrame(columns=_MATCH_COLUMNS)

    cell_count = len(reference_sizes) * len(predicted_sizes)
    if cell_count > max_dense_cells:
        raise ValueError(
            "cluster matching would require "
            f"{cell_count:,} dense IoU cells; reduce the evaluated scope or raise max_dense_cells explicitly"
        )

    iou = np.zeros((len(reference_sizes), len(predicted_sizes)), dtype="float64")
    for row in intersections.itertuples(index=False):
        union = (
            reference_sizes[row.reference_code] + predicted_sizes[row.predicted_code] - row.intersection_event_count
        )
        iou[row.reference_code, row.predicted_code] = row.intersection_event_count / union

    matched_reference, matched_prediction = linear_sum_assignment(iou, maximize=True)
    rows: list[dict[str, object]] = []
    intersection_lookup = intersections.set_index(["reference_code", "predicted_code"])["intersection_event_count"]
    for reference_code, predicted_code in zip(matched_reference, matched_prediction, strict=True):
        intersection = int(intersection_lookup.get((reference_code, predicted_code), 0))
        if intersection == 0:
            continue
        reference_size = int(reference_sizes[reference_code])
        predicted_size = int(predicted_sizes[predicted_code])
        rows.append(
            {
                "reference_label": reference_values[reference_code],
                "predicted_label": predicted_values[predicted_code],
                "intersection_event_count": intersection,
                "reference_event_count": reference_size,
                "predicted_event_count": predicted_size,
                "cluster_iou": intersection / (reference_size + predicted_size - intersection),
                "cluster_f1": 2 * intersection / (reference_size + predicted_size),
            }
        )
    return pd.DataFrame.from_records(rows, columns=_MATCH_COLUMNS)


def evaluate_event_partition(
    reference_labels: Sequence[object],
    predicted_labels: Sequence[object],
    *,
    incident_iou_threshold: float = 0.5,
    max_dense_cells: int = 5_000_000,
) -> dict[str, float | int]:
    """Evaluate one predicted event partition against reviewed incident membership."""
    if not 0 < incident_iou_threshold <= 1:
        raise ValueError("incident_iou_threshold must be in (0, 1]")

    reference_codes, predicted_codes, _, _ = _validated_label_codes(reference_labels, predicted_labels)
    event_count = len(reference_codes)
    if event_count == 0:
        return {
            "event_count": 0,
            "reference_incident_count": 0,
            "predicted_cluster_count": 0,
            **{name: float("nan") for name in _METRIC_NAMES},
        }

    intersections, reference_sizes, predicted_sizes = _partition_counts(reference_codes, predicted_codes)
    scores = bcubed_scores(reference_labels, predicted_labels)
    matches = match_clusters_by_membership(
        reference_labels,
        predicted_labels,
        max_dense_cells=max_dense_cells,
    )

    reference_incident_count = len(reference_sizes)
    predicted_cluster_count = len(predicted_sizes)
    qualifying_matches = int(matches["cluster_iou"].ge(incident_iou_threshold).sum())
    incident_precision = qualifying_matches / predicted_cluster_count
    incident_recall = qualifying_matches / reference_incident_count
    merge_rate = (
        intersections.groupby("predicted_code", observed=True)["reference_code"].nunique().gt(1).sum()
        / predicted_cluster_count
    )
    fragmentation_rate = (
        intersections.groupby("reference_code", observed=True)["predicted_code"].nunique().gt(1).sum()
        / reference_incident_count
    )

    return {
        "event_count": event_count,
        "reference_incident_count": reference_incident_count,
        "predicted_cluster_count": predicted_cluster_count,
        "adjusted_rand_index": float(adjusted_rand_score(reference_codes, predicted_codes)),
        **scores,
        "mean_matched_cluster_iou": float(matches["cluster_iou"].mean()),
        "matched_incident_precision": incident_precision,
        "matched_incident_recall": incident_recall,
        "matched_incident_f1": harmonic_mean(incident_precision, incident_recall),
        "merge_rate": float(merge_rate),
        "fragmentation_rate": float(fragmentation_rate),
    }


def evaluate_event_partition_by_group(
    frame: pd.DataFrame,
    *,
    group_column: str,
    reference_column: str,
    predicted_column: str,
    minimum_events: int = 2,
    incident_iou_threshold: float = 0.5,
    max_dense_cells: int = 5_000_000,
) -> pd.DataFrame:
    """Compute within-group diagnostics, for example one row per SCADA device.

    Metrics are recomputed after restricting to each group. They reveal locally weak devices,
    but cannot by themselves expose a predicted cluster that incorrectly joins different groups.
    """
    required = {group_column, reference_column, predicted_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"metric frame is missing columns: {missing}")
    if minimum_events < 1:
        raise ValueError("minimum_events must be positive")

    rows: list[dict[str, object]] = []
    for group, group_frame in frame.groupby(group_column, sort=False, dropna=False):
        if len(group_frame) < minimum_events:
            continue
        rows.append(
            {
                group_column: group,
                **evaluate_event_partition(
                    group_frame[reference_column],
                    group_frame[predicted_column],
                    incident_iou_threshold=incident_iou_threshold,
                    max_dense_cells=max_dense_cells,
                ),
            }
        )
    columns = [
        group_column,
        "event_count",
        "reference_incident_count",
        "predicted_cluster_count",
        *_METRIC_NAMES,
    ]
    return pd.DataFrame.from_records(rows, columns=columns)


def add_temporal_overlap_diagnostics(
    matches: pd.DataFrame,
    *,
    predicted_start_column: str = "cluster_start",
    predicted_end_column: str = "cluster_end",
    reference_start_column: str = "reference_start",
    reference_end_column: str = "reference_end",
) -> pd.DataFrame:
    """Add temporal IoU and absolute start/end boundary errors to matched intervals."""
    required = {
        predicted_start_column,
        predicted_end_column,
        reference_start_column,
        reference_end_column,
    }
    missing = sorted(required.difference(matches.columns))
    if missing:
        raise ValueError(f"matched interval frame is missing columns: {missing}")

    result = matches.copy()
    columns = [
        predicted_start_column,
        predicted_end_column,
        reference_start_column,
        reference_end_column,
    ]
    for column in columns:
        result[column] = pd.to_datetime(result[column], errors="raise")
    if result[columns].isna().any().any():
        raise ValueError("matched intervals cannot contain missing boundaries")
    if result[predicted_end_column].lt(result[predicted_start_column]).any():
        raise ValueError("predicted interval end precedes its start")
    if result[reference_end_column].lt(result[reference_start_column]).any():
        raise ValueError("reference interval end precedes its start")

    overlap_start = result[[predicted_start_column, reference_start_column]].max(axis=1)
    overlap_end = result[[predicted_end_column, reference_end_column]].min(axis=1)
    union_start = result[[predicted_start_column, reference_start_column]].min(axis=1)
    union_end = result[[predicted_end_column, reference_end_column]].max(axis=1)
    result["overlap_seconds"] = overlap_end.sub(overlap_start).dt.total_seconds().clip(lower=0)
    result["union_seconds"] = union_end.sub(union_start).dt.total_seconds().clip(lower=0)
    same_point = (
        result[predicted_start_column].eq(result[predicted_end_column])
        & result[reference_start_column].eq(result[reference_end_column])
        & result[predicted_start_column].eq(result[reference_start_column])
    )
    result["temporal_iou"] = np.where(
        result["union_seconds"].gt(0),
        result["overlap_seconds"] / result["union_seconds"],
        same_point.astype(float),
    )
    result["start_boundary_error_seconds"] = (
        result[predicted_start_column].sub(result[reference_start_column]).abs().dt.total_seconds()
    )
    result["end_boundary_error_seconds"] = (
        result[predicted_end_column].sub(result[reference_end_column]).abs().dt.total_seconds()
    )
    result["mean_boundary_error_seconds"] = result[
        ["start_boundary_error_seconds", "end_boundary_error_seconds"]
    ].mean(axis=1)
    return result


def summarize_temporal_matches(
    matches: pd.DataFrame,
    **column_names: str,
) -> dict[str, float | int]:
    """Summarize temporal agreement for an already matched interval table."""
    diagnostics = add_temporal_overlap_diagnostics(matches, **column_names)
    return {
        "matched_interval_count": len(diagnostics),
        "mean_temporal_iou": float(diagnostics["temporal_iou"].mean()),
        "median_temporal_iou": float(diagnostics["temporal_iou"].median()),
        "median_boundary_error_seconds": float(diagnostics["mean_boundary_error_seconds"].median()),
        "p90_boundary_error_seconds": float(diagnostics["mean_boundary_error_seconds"].quantile(0.9)),
    }


def summarize_temporal_matches_by_group(
    matches: pd.DataFrame,
    *,
    group_column: str,
    minimum_matches: int = 1,
    **column_names: str,
) -> pd.DataFrame:
    """Summarize temporal agreement by asset/device for already matched intervals."""
    if group_column not in matches:
        raise ValueError(f"matched interval frame is missing group column: {group_column!r}")
    if minimum_matches < 1:
        raise ValueError("minimum_matches must be positive")

    rows: list[dict[str, object]] = []
    for group, group_frame in matches.groupby(group_column, sort=False, dropna=False):
        if len(group_frame) < minimum_matches:
            continue
        rows.append(
            {
                group_column: group,
                **summarize_temporal_matches(group_frame, **column_names),
            }
        )
    return pd.DataFrame.from_records(rows)


def summarize_candidate_matches_by_group(
    cluster_groups: pd.DataFrame,
    candidates: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    group_column: str,
    cluster_id_column: str = "cluster_id",
    reference_id_column: str = "reference_id",
) -> pd.DataFrame:
    """Summarize fuzzy candidate association for every cluster-member group.

    A cluster contributes to every group it contains. The result therefore describes
    association of a device with clusters near an outage reference; it is not causal device
    attribution, and rows must not be summed across groups.
    """
    cluster_group_columns = {cluster_id_column, group_column}
    pair_columns = {cluster_id_column, reference_id_column}
    missing_cluster_groups = sorted(cluster_group_columns.difference(cluster_groups.columns))
    missing_candidates = sorted(pair_columns.difference(candidates.columns))
    missing_matches = sorted(pair_columns.difference(matches.columns))
    if missing_cluster_groups:
        raise ValueError(f"cluster-group frame is missing columns: {missing_cluster_groups}")
    if missing_candidates:
        raise ValueError(f"candidate frame is missing columns: {missing_candidates}")
    if missing_matches:
        raise ValueError(f"match frame is missing columns: {missing_matches}")
    if matches[cluster_id_column].duplicated().any() or matches[reference_id_column].duplicated().any():
        raise ValueError("matches must be one-to-one by cluster and reference")

    groups = cluster_groups[[cluster_id_column, group_column]].dropna(subset=[group_column]).drop_duplicates()
    candidate_pairs = candidates.drop_duplicates([cluster_id_column, reference_id_column])
    match_pairs = matches.drop_duplicates([cluster_id_column, reference_id_column])
    candidates_by_group = candidate_pairs.merge(
        groups,
        on=cluster_id_column,
        how="inner",
        validate="many_to_many",
    )
    matches_by_group = match_pairs.merge(
        groups,
        on=cluster_id_column,
        how="inner",
        validate="many_to_many",
    )

    rows: list[dict[str, object]] = []
    for group, group_clusters in groups.groupby(group_column, sort=False, dropna=False):
        cluster_ids = group_clusters[cluster_id_column]
        group_candidates = candidates_by_group.loc[candidates_by_group[group_column].eq(group)]
        group_matches = matches_by_group.loc[matches_by_group[group_column].eq(group)]
        cluster_count = cluster_ids.nunique()
        candidate_cluster_count = group_candidates[cluster_id_column].nunique()
        matched_cluster_count = group_matches[cluster_id_column].nunique()
        candidate_reference_count = group_candidates[reference_id_column].nunique()
        matched_reference_count = group_matches[reference_id_column].nunique()

        fragmentation_rate = (
            group_candidates.groupby(reference_id_column)[cluster_id_column].nunique().gt(1).mean()
            if candidate_reference_count
            else float("nan")
        )
        merge_rate = (
            group_candidates.groupby(cluster_id_column)[reference_id_column].nunique().gt(1).mean()
            if candidate_cluster_count
            else float("nan")
        )
        temporal = (
            summarize_temporal_matches(group_matches)
            if not group_matches.empty
            else {
                "matched_interval_count": 0,
                "mean_temporal_iou": float("nan"),
                "median_temporal_iou": float("nan"),
                "median_boundary_error_seconds": float("nan"),
                "p90_boundary_error_seconds": float("nan"),
            }
        )
        rows.append(
            {
                group_column: group,
                "cluster_count": cluster_count,
                "candidate_cluster_count": candidate_cluster_count,
                "matched_cluster_count": matched_cluster_count,
                "candidate_reference_count": candidate_reference_count,
                "matched_reference_count": matched_reference_count,
                "candidate_cluster_alignment_proxy": (
                    matched_cluster_count / candidate_cluster_count if candidate_cluster_count else float("nan")
                ),
                "candidate_reference_alignment_proxy": (
                    matched_reference_count / candidate_reference_count if candidate_reference_count else float("nan")
                ),
                "all_cluster_reference_alignment_proxy": (
                    matched_cluster_count / cluster_count if cluster_count else float("nan")
                ),
                "fragmentation_candidate_rate_proxy": float(fragmentation_rate),
                "merge_candidate_rate_proxy": float(merge_rate),
                **temporal,
            }
        )
    return pd.DataFrame.from_records(rows)


def clustering_metric_catalog() -> pd.DataFrame:
    """Describe the compact metric set and its evidence requirements."""
    rows = [
        ("Overall partition similarity", "adjusted_rand_index", "reviewed event membership"),
        ("Are unrelated events being merged?", "bcubed_precision", "reviewed event membership"),
        ("Are related events being split?", "bcubed_recall", "reviewed event membership"),
        ("Overall event grouping", "bcubed_f1", "reviewed event membership"),
        ("Did we reconstruct whole incidents?", "mean_matched_cluster_iou", "reviewed event membership"),
        ("Did enough whole incidents match?", "matched_incident_f1", "reviewed event membership"),
        ("What kind of errors occur?", "merge_rate / fragmentation_rate", "reviewed event membership"),
        ("Are temporal boundaries correct?", "temporal_iou / boundary_error", "matched incident intervals"),
    ]
    return pd.DataFrame(rows, columns=["question", "metric", "requires"])
