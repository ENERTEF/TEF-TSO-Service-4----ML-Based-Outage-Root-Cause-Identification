"""Hand-computed tests for Service 4 clustering evaluation metrics."""

from __future__ import annotations

import math
import unittest

import pandas as pd
from service4.evaluation import (
    add_minute_precision_reference_bounds,
    add_temporal_overlap_diagnostics,
    build_exact_asset_candidates,
    clustering_metric_catalog,
    evaluate_episode_clusters,
    evaluate_event_partition,
    evaluate_event_partition_by_group,
    match_clusters_by_membership,
    summarize_candidate_matches_by_group,
    summarize_temporal_matches,
    summarize_temporal_matches_by_group,
)


class EventPartitionMetricsTest(unittest.TestCase):
    def assert_close(self, value: float, expected: float) -> None:
        self.assertTrue(math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12), (value, expected))

    def test_perfect_partition_scores_one(self) -> None:
        reference = ["A", "A", "A", "B", "B"]
        predicted = [10, 10, 10, 20, 20]

        scores = evaluate_event_partition(reference, predicted)

        for name in [
            "adjusted_rand_index",
            "bcubed_precision",
            "bcubed_recall",
            "bcubed_f1",
            "mean_matched_cluster_iou",
            "matched_incident_precision",
            "matched_incident_recall",
            "matched_incident_f1",
        ]:
            self.assert_close(float(scores[name]), 1.0)
        self.assert_close(float(scores["merge_rate"]), 0.0)
        self.assert_close(float(scores["fragmentation_rate"]), 0.0)

    def test_total_fragmentation_has_high_precision_and_low_recall(self) -> None:
        reference = ["A", "A", "A", "B", "B"]
        predicted = [1, 2, 3, 4, 5]

        scores = evaluate_event_partition(reference, predicted)

        self.assert_close(float(scores["bcubed_precision"]), 1.0)
        self.assert_close(float(scores["bcubed_recall"]), 0.4)
        self.assert_close(float(scores["mean_matched_cluster_iou"]), 5 / 12)
        self.assert_close(float(scores["matched_incident_precision"]), 1 / 5)
        self.assert_close(float(scores["matched_incident_recall"]), 1 / 2)
        self.assert_close(float(scores["matched_incident_f1"]), 2 / 7)
        self.assert_close(float(scores["merge_rate"]), 0.0)
        self.assert_close(float(scores["fragmentation_rate"]), 1.0)

    def test_total_merging_has_low_precision_and_high_recall(self) -> None:
        reference = ["A", "A", "A", "B", "B"]
        predicted = [1, 1, 1, 1, 1]

        scores = evaluate_event_partition(reference, predicted)

        self.assert_close(float(scores["bcubed_precision"]), 13 / 25)
        self.assert_close(float(scores["bcubed_recall"]), 1.0)
        self.assert_close(float(scores["mean_matched_cluster_iou"]), 3 / 5)
        self.assert_close(float(scores["matched_incident_precision"]), 1.0)
        self.assert_close(float(scores["matched_incident_recall"]), 1 / 2)
        self.assert_close(float(scores["matched_incident_f1"]), 2 / 3)
        self.assert_close(float(scores["merge_rate"]), 1.0)
        self.assert_close(float(scores["fragmentation_rate"]), 0.0)

    def test_iou_matching_returns_event_counts(self) -> None:
        matches = match_clusters_by_membership(
            ["A", "A", "A", "B", "B"],
            [10, 10, 20, 20, 20],
        )

        self.assertEqual(len(matches), 2)
        match_a = matches.loc[matches["reference_label"].eq("A")].iloc[0]
        self.assertEqual(match_a["intersection_event_count"], 2)
        self.assertEqual(match_a["reference_event_count"], 3)
        self.assertEqual(match_a["predicted_event_count"], 2)
        self.assert_close(float(match_a["cluster_iou"]), 2 / 3)

    def test_per_group_report_recomputes_metrics_inside_device(self) -> None:
        frame = pd.DataFrame(
            {
                "device": ["d1", "d1", "d1", "d1", "d2", "d2"],
                "truth": ["A", "A", "B", "B", "C", "C"],
                "prediction": [1, 1, 2, 2, 3, 4],
            }
        )

        report = evaluate_event_partition_by_group(
            frame,
            group_column="device",
            reference_column="truth",
            predicted_column="prediction",
        ).set_index("device")

        self.assert_close(float(report.loc["d1", "bcubed_f1"]), 1.0)
        self.assert_close(float(report.loc["d2", "fragmentation_rate"]), 1.0)
        self.assertEqual(report.loc["d2", "event_count"], 2)

    def test_fuzzy_reference_summary_by_device_keeps_support_counts(self) -> None:
        base = pd.Timestamp("2026-01-01 00:00:00")
        cluster_groups = pd.DataFrame(
            {
                "cluster_id": ["c1", "c1", "c2", "c3", "c4"],
                "DevKey": ["d1", "d2", "d1", "d2", "d3"],
            }
        )
        candidates = pd.DataFrame(
            {
                "cluster_id": ["c1", "c1", "c2", "c3"],
                "reference_id": ["r1", "r2", "r1", "r3"],
            }
        )
        matches = pd.DataFrame(
            {
                "cluster_id": ["c1", "c2", "c3"],
                "reference_id": ["r2", "r1", "r3"],
                "cluster_start": [base, base, base],
                "cluster_end": [base, base, base],
                "reference_start": [base, base, base],
                "reference_end": [base, base, base],
            }
        )

        report = summarize_candidate_matches_by_group(
            cluster_groups,
            candidates,
            matches,
            group_column="DevKey",
        ).set_index("DevKey")

        self.assertEqual(report.loc["d1", "cluster_count"], 2)
        self.assertEqual(report.loc["d1", "candidate_reference_count"], 2)
        self.assert_close(float(report.loc["d1", "fragmentation_candidate_rate_proxy"]), 0.5)
        self.assert_close(float(report.loc["d1", "merge_candidate_rate_proxy"]), 0.5)
        self.assert_close(float(report.loc["d2", "candidate_reference_alignment_proxy"]), 2 / 3)
        self.assertEqual(report.loc["d3", "candidate_cluster_count"], 0)
        self.assertTrue(math.isnan(float(report.loc["d3", "candidate_cluster_alignment_proxy"])))

    def test_invalid_labels_and_dense_match_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal length"):
            evaluate_event_partition(["A"], [1, 2])
        with self.assertRaisesRegex(ValueError, "both a reference and predicted"):
            evaluate_event_partition(["A", None], [1, 2])
        with self.assertRaisesRegex(ValueError, "dense IoU cells"):
            evaluate_event_partition(["A", "B"], [1, 2], max_dense_cells=3)


class TemporalMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        base = pd.Timestamp("2026-01-01 00:00:00")
        self.matches = pd.DataFrame(
            {
                "asset": ["x", "y"],
                "cluster_start": [base, base],
                "cluster_end": [base + pd.Timedelta(seconds=10), base],
                "reference_start": [base + pd.Timedelta(seconds=2), base],
                "reference_end": [base + pd.Timedelta(seconds=12), base],
            }
        )

    def assert_close(self, value: float, expected: float) -> None:
        self.assertTrue(math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12), (value, expected))

    def test_temporal_iou_and_boundary_error(self) -> None:
        diagnostics = add_temporal_overlap_diagnostics(self.matches)

        self.assert_close(float(diagnostics.loc[0, "temporal_iou"]), 2 / 3)
        self.assert_close(float(diagnostics.loc[0, "mean_boundary_error_seconds"]), 2.0)
        self.assert_close(float(diagnostics.loc[1, "temporal_iou"]), 1.0)

        summary = summarize_temporal_matches(self.matches)
        self.assert_close(float(summary["mean_temporal_iou"]), 5 / 6)
        self.assert_close(float(summary["median_boundary_error_seconds"]), 1.0)

    def test_temporal_report_groups_by_asset(self) -> None:
        report = summarize_temporal_matches_by_group(
            self.matches,
            group_column="asset",
        ).set_index("asset")

        self.assertEqual(report.loc["x", "matched_interval_count"], 1)
        self.assert_close(float(report.loc["y", "mean_temporal_iou"]), 1.0)

    def test_reversed_interval_is_rejected(self) -> None:
        invalid = self.matches.iloc[[0]].copy()
        invalid["cluster_start"], invalid["cluster_end"] = (
            invalid["cluster_end"].copy(),
            invalid["cluster_start"].copy(),
        )
        with self.assertRaisesRegex(ValueError, "predicted interval end precedes"):
            add_temporal_overlap_diagnostics(invalid)

    def test_minute_precision_extends_only_nonzero_intervals(self) -> None:
        base = pd.Timestamp("2025-05-05 15:01:00")
        references = pd.DataFrame(
            {
                "reference_start": [base, base],
                "reference_end": [base + pd.Timedelta(minutes=2), base],
            }
        )

        result = add_minute_precision_reference_bounds(references)

        self.assertEqual(result.loc[0, "reference_end_recorded"], base + pd.Timedelta(minutes=2))
        self.assertEqual(result.loc[0, "reference_end_effective"], base + pd.Timedelta(minutes=3))
        self.assertEqual(result.loc[0, "reference_time_kind"], "minute_granularity_interval")
        self.assertEqual(result.loc[1, "reference_end_effective"], base)
        self.assertEqual(result.loc[1, "reference_time_kind"], "point")

    def test_catalog_separates_membership_and_interval_evidence(self) -> None:
        catalog = clustering_metric_catalog()
        self.assertEqual(len(catalog), 8)
        self.assertEqual(
            set(catalog["requires"]),
            {"reviewed event membership", "matched incident intervals"},
        )


class ProxyIntervalEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        base = pd.Timestamp("2025-01-01 00:00:00")
        self.clusters = pd.DataFrame(
            {
                "cluster_id": [1, 2],
                "SifraSredstva": ["asset-a", "asset-b"],
                "cluster_start": [base, base],
                "cluster_end": [base + pd.Timedelta(seconds=10), base + pd.Timedelta(seconds=10)],
                "closed_at": [base + pd.Timedelta(seconds=100), base + pd.Timedelta(seconds=10)],
            }
        )
        self.references = pd.DataFrame(
            {
                "reference_id": [10, 20],
                "anchor_asset": ["asset-a", "asset-c"],
                "reference_start": [base + pd.Timedelta(seconds=95), base],
                "reference_end": [base + pd.Timedelta(seconds=100), base + pd.Timedelta(seconds=10)],
            }
        )

    def test_candidate_builder_can_use_lifecycle_end(self) -> None:
        activity = build_exact_asset_candidates(
            self.clusters,
            self.references,
            boundary_tolerance=pd.Timedelta(seconds=15),
        )
        lifecycle = build_exact_asset_candidates(
            self.clusters,
            self.references,
            predicted_end_column="closed_at",
            boundary_tolerance=pd.Timedelta(seconds=15),
        )

        self.assertTrue(activity.empty)
        self.assertEqual(lifecycle[["cluster_id", "reference_id"]].values.tolist(), [[1, 10]])

    def test_proxy_summary_keeps_reference_and_cluster_denominators(self) -> None:
        metrics, candidates, matches = evaluate_episode_clusters(
            self.clusters,
            self.references,
            predicted_end_column="closed_at",
            boundary_tolerance=pd.Timedelta(seconds=15),
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(len(matches), 1)
        self.assertEqual(metrics["reference_count"], 2)
        self.assertEqual(metrics["cluster_count"], 2)
        self.assertEqual(metrics["reference_recall_proxy"], 0.5)
        self.assertEqual(metrics["candidate_cluster_alignment_proxy"], 1.0)


if __name__ == "__main__":
    unittest.main()
