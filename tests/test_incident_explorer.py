from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd
from apps import incident_explorer


class CachedAssetDisplayLabelsTest(unittest.TestCase):
    def test_reuses_labels_for_the_same_artifact_manifest(self) -> None:
        events = pd.DataFrame(
            {
                "SifraSredstva": ["10", "10"],
                "lokacija": ["Station A", "Station A"],
                "point_description": ["Point 1", "Point 2"],
                "aoj": ["AREA", "AREA"],
            }
        )
        incident_explorer.build_cached_asset_display_labels.clear()

        with patch.object(
            incident_explorer,
            "build_asset_display_labels",
            wraps=incident_explorer.build_asset_display_labels,
        ) as builder:
            first = incident_explorer.build_cached_asset_display_labels(events, "manifest-a")
            second = incident_explorer.build_cached_asset_display_labels(events, "manifest-a")

        self.assertEqual(first, {"10": "10 · Station A"})
        self.assertEqual(second, first)
        self.assertEqual(builder.call_count, 1)


class InterestingSamplesTest(unittest.TestCase):
    @staticmethod
    def _tables_for_curated_samples() -> dict[str, dict[str, pd.DataFrame]]:
        rows_by_model: dict[str, list[dict[str, object]]] = {model: [] for model in incident_explorer.MODEL_ORDER}
        for index, spec in enumerate(incident_explorer.CURATED_SAMPLE_SPECS):
            sample = spec.sample
            rows_by_model[sample.source_model].append(
                {
                    "asset_episode_id": index,
                    "SifraSredstva": sample.asset_id,
                    "cluster_start": sample.anchor_time,
                    "closed_at": sample.anchor_time + pd.Timedelta(minutes=1),
                }
            )
        return {model: {"episodes": pd.DataFrame(rows)} for model, rows in rows_by_model.items()}

    def test_returns_audited_unique_targets_in_declared_order(self) -> None:
        samples = incident_explorer.build_interesting_samples(self._tables_for_curated_samples())

        expected_keys = [
            *(spec.key for spec in incident_explorer.CURATED_SAMPLE_SPECS),
            "browse",
        ]
        targets = [(sample.asset_id, sample.anchor_time) for key, sample in samples.items() if key != "browse"]
        self.assertEqual(list(samples), expected_keys)
        self.assertEqual(len(targets), len(set(targets)))

    def test_hides_comparisons_when_a_required_model_is_not_selected(self) -> None:
        tables = self._tables_for_curated_samples()
        samples = incident_explorer.build_interesting_samples({"v2.0": tables["v2.0"]})

        self.assertEqual(
            list(samples),
            [
                "planned_work_test_chatter",
                "test_mode_retry_false_positive",
                "browse",
            ],
        )

    def test_fails_if_a_frozen_artifact_loses_an_audited_target(self) -> None:
        tables = self._tables_for_curated_samples()
        tables["v1.1"]["episodes"] = tables["v1.1"]["episodes"].iloc[0:0]

        with self.assertRaisesRegex(ValueError, "audited interesting sample"):
            incident_explorer.build_interesting_samples(tables)


class ComparisonFigureTest(unittest.TestCase):
    def test_draws_versions_on_separate_episode_lanes(self) -> None:
        start = pd.Timestamp("2025-01-01 00:00:00")
        primary_episodes = pd.DataFrame(
            {
                "asset_episode_id": [11],
                "cluster_start": [start],
                "closed_at": [start + pd.Timedelta(minutes=2)],
                "closure_reason": ["control_confirmed_restoration"],
                "event_count": [2],
            }
        )
        comparison_episodes = pd.DataFrame(
            {
                "asset_episode_id": [10],
                "cluster_start": [start],
                "closed_at": [start + pd.Timedelta(hours=1)],
                "closure_reason": ["maximum_duration"],
                "event_count": [2],
            }
        )
        v2_episodes = comparison_episodes.assign(asset_episode_id=9)
        empty_events = pd.DataFrame(columns=["DevKey"])
        window = incident_explorer.TimelineWindow(
            events=empty_events,
            episodes=primary_episodes,
            context=empty_events,
            unassigned_events=empty_events,
        )

        figure = incident_explorer.build_figure(
            window,
            unplanned=pd.DataFrame(),
            planned=pd.DataFrame(),
            focus_episode_id=11,
            episode_layers=(
                ("V1.0", comparison_episodes, "#1f77b4"),
                ("V1.1", primary_episodes, "#9467bd"),
                ("V2.0", v2_episodes, "#2ca02c"),
            ),
        )

        lanes = {trace.y[0] for trace in figure.data if len(trace.y)}
        self.assertIn("V1.1 episodes", lanes)
        self.assertIn("V1.0 episodes", lanes)
        self.assertIn("V2.0 episodes", lanes)


if __name__ == "__main__":
    unittest.main()
