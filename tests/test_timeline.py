from __future__ import annotations

import unittest

import pandas as pd
from service4.timeline import (
    TimelineWindowTooLarge,
    build_asset_display_labels,
    select_reference_intervals,
    select_timeline_window,
)

START = pd.Timestamp("2025-01-01 00:00:00")


class BuildAssetDisplayLabelsTest(unittest.TestCase):
    def test_uses_most_frequent_location_and_deterministic_tie_break(self) -> None:
        events = pd.DataFrame(
            {
                "SifraSredstva": ["10", "10", "10", "20", "20", "30"],
                "lokacija": ["  Station   A  ", "Station A", "Station B", "", None, ""],
                "point_description": ["Point 1", "Point 2", "Point 3", "Zulu", "Alpha", ""],
                "aoj": ["TEST", "TEST", "REAL", "TEST", "REAL", ""],
            }
        )

        labels = build_asset_display_labels(events)

        self.assertEqual(labels["10"], "10 · Station A")
        self.assertEqual(labels["20"], "20 · Alpha")
        self.assertEqual(labels["30"], "30")


class SelectTimelineWindowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.events = pd.DataFrame(
            {
                "event_index": [1, 2, 3],
                "date_time": [
                    START,
                    START + pd.Timedelta(minutes=2),
                    START + pd.Timedelta(minutes=3),
                ],
                "SifraSredstva": ["asset-a", "asset-a", "asset-a"],
                "DevKey": ["point-a", "point-b", "point-c"],
                "episode_event_role": [
                    "breaker_trip",
                    "protection_assertion",
                    "support",
                ],
            }
        )
        self.episodes = pd.DataFrame(
            {
                "asset_episode_id": [10],
                "SifraSredstva": ["asset-a"],
                "cluster_start": [START],
                "cluster_end": [START + pd.Timedelta(minutes=2)],
                "closed_at": [START + pd.Timedelta(minutes=4)],
            }
        )
        self.membership = pd.DataFrame(
            {
                "event_index": [1, 2],
                "asset_episode_id": [10, 10],
                "episode_event_role": ["breaker_trip", "protection_assertion"],
            }
        )
        self.context = pd.DataFrame(
            {
                "event_index": [20],
                "date_time": [START + pd.Timedelta(minutes=1)],
                "SifraSredstva": ["asset-a"],
                "asset_episode_id": [10],
            }
        )

    def test_selects_overlapping_episode_and_visible_rows(self) -> None:
        window = select_timeline_window(
            self.events,
            self.episodes,
            self.membership,
            self.context,
            start=START + pd.Timedelta(seconds=30),
            end=START + pd.Timedelta(minutes=3),
            assets=["asset-a"],
            include_unassigned=True,
        )

        self.assertEqual(window.episodes["asset_episode_id"].tolist(), [10])
        self.assertEqual(window.events["event_index"].tolist(), [2])
        self.assertEqual(window.context["event_index"].tolist(), [20])
        self.assertEqual(window.unassigned_events["event_index"].tolist(), [3])

    def test_empty_asset_selection_returns_empty_tables(self) -> None:
        window = select_timeline_window(
            self.events,
            self.episodes,
            self.membership,
            self.context,
            start=START,
            end=START + pd.Timedelta(minutes=3),
            assets=[],
        )
        self.assertTrue(window.events.empty)
        self.assertTrue(window.episodes.empty)

    def test_rejects_oversized_window(self) -> None:
        with self.assertRaises(TimelineWindowTooLarge):
            select_timeline_window(
                self.events,
                self.episodes,
                self.membership,
                self.context,
                start=START,
                end=START + pd.Timedelta(minutes=4),
                assets=["asset-a"],
                include_unassigned=True,
                max_events=2,
            )


class SelectReferenceIntervalsTest(unittest.TestCase):
    def test_normalizes_both_asset_columns_without_duplicates(self) -> None:
        references = pd.DataFrame(
            {
                "DCVDogodekOutageID": [1, 2],
                "DatumUraCETOd": [START, START + pd.Timedelta(hours=2)],
                "DatumUraCETDo": [
                    START + pd.Timedelta(minutes=30),
                    START + pd.Timedelta(hours=3),
                ],
                "DCVNapravaSID_SifraSredstva": [123, 999],
                "DCVNapravaSIDIzklopna_SifraSredstva": [123, 123],
                "Opis": ["first", "second"],
                "Povod": ["cause-a", "cause-b"],
            }
        )

        selected = select_reference_intervals(
            references,
            start=START,
            end=START + pd.Timedelta(hours=4),
            assets=["123"],
        )

        self.assertEqual(selected["reference_id"].tolist(), [1, 2])
        self.assertEqual(selected["asset_id"].tolist(), ["123", "123"])

    def test_extends_nonzero_end_minute_but_keeps_point_reference(self) -> None:
        references = pd.DataFrame(
            {
                "DCVDogodekOutageID": [1, 2],
                "DatumUraCETOd": [START, START + pd.Timedelta(minutes=2)],
                "DatumUraCETDo": [START + pd.Timedelta(minutes=1), START + pd.Timedelta(minutes=2)],
                "DCVNapravaSID_SifraSredstva": [123, 123],
                "DCVNapravaSIDIzklopna_SifraSredstva": [123, 123],
                "Opis": ["interval", "point"],
                "Povod": ["cause-a", "cause-b"],
            }
        )

        selected = select_reference_intervals(
            references,
            start=START,
            end=START + pd.Timedelta(minutes=3),
            assets=["123"],
            nonzero_end_extension=pd.Timedelta(minutes=1),
        ).set_index("reference_id")

        self.assertEqual(selected.loc[1, "reference_end"], START + pd.Timedelta(minutes=2))
        self.assertEqual(selected.loc[1, "reference_end_recorded"], START + pd.Timedelta(minutes=1))
        self.assertEqual(selected.loc[2, "reference_end"], START + pd.Timedelta(minutes=2))
        self.assertEqual(selected.loc[2, "reference_time_kind"], "point")

    def test_uses_stable_envelope_for_repeated_reference_rows(self) -> None:
        references = pd.DataFrame(
            {
                "DCVDogodekOutageID": [1, 1],
                "DatumUraCETOd": [START + pd.Timedelta(minutes=2), START],
                "DatumUraCETDo": [
                    START + pd.Timedelta(minutes=3),
                    START + pd.Timedelta(minutes=5),
                ],
                "DCVNapravaSID_SifraSredstva": [123, 123],
                "DCVNapravaSIDIzklopna_SifraSredstva": [999, 123],
                "Opis": ["first row", "second row"],
                "Povod": ["cause-a", "cause-b"],
            }
        )

        selected = select_reference_intervals(
            references,
            start=START + pd.Timedelta(minutes=4),
            end=START + pd.Timedelta(minutes=6),
            assets=["123"],
            nonzero_end_extension=pd.Timedelta(minutes=1),
        ).iloc[0]

        self.assertEqual(selected["reference_start"], START)
        self.assertEqual(selected["reference_end_recorded"], START + pd.Timedelta(minutes=5))
        self.assertEqual(selected["reference_end"], START + pd.Timedelta(minutes=6))
        self.assertEqual(selected["source_row_count"], 2)
        self.assertEqual(selected["reported_boundary_count"], 2)
        self.assertEqual(selected["asset_link_role"], "affected_asset+switching_anchor")

    def test_half_open_interval_ending_at_window_start_is_not_selected(self) -> None:
        references = pd.DataFrame(
            {
                "DCVDogodekOutageID": [1],
                "DatumUraCETOd": [START],
                "DatumUraCETDo": [START + pd.Timedelta(minutes=1)],
                "DCVNapravaSID_SifraSredstva": [123],
                "DCVNapravaSIDIzklopna_SifraSredstva": [999],
                "Opis": ["interval"],
                "Povod": ["cause"],
            }
        )

        selected = select_reference_intervals(
            references,
            start=START + pd.Timedelta(minutes=2),
            end=START + pd.Timedelta(minutes=3),
            assets=["123"],
            nonzero_end_extension=pd.Timedelta(minutes=1),
        )

        self.assertTrue(selected.empty)


if __name__ == "__main__":
    unittest.main()
