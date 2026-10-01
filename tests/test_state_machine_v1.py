from __future__ import annotations

import unittest

import pandas as pd
from service4.state_machine_v1 import (
    build_state_machine_v1_episodes,
    prepare_state_machine_v1_events,
    run_state_machine_v1,
)

START = pd.Timestamp("2025-01-01 00:00:00")


def prepared_events(rows: list[tuple[object, ...]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "event_index",
            "date_time",
            "SifraSredstva",
            "DevKey",
            "episode_event_role",
            "episode_start_trigger",
            "from_state",
            "to_state",
            "reported_state",
        ],
    )


def handoff_events(rows: list[dict[str, object]]) -> pd.DataFrame:
    defaults: dict[str, object] = {
        "DevKey": "point-a",
        "SifraSredstva": "asset-a",
        "signal_description": "GENERIC",
        "event_family": "automatic_state",
        "point_family": "protection",
        "can_start_episode": True,
        "from_state": "KONEC",
        "to_state": "ZAČETEK",
        "reported_state": "ZAČETEK",
        "command_target_state": pd.NA,
    }
    records = [defaults | row for row in rows]
    return pd.DataFrame.from_records(records)


class PrepareStateMachineV1EventsTest(unittest.TestCase):
    def test_assigns_roles_and_filters_scope(self) -> None:
        frame = handoff_events(
            [
                {"event_index": 0, "date_time": START},
                {
                    "event_index": 1,
                    "date_time": START + pd.Timedelta(seconds=1),
                    "from_state": "ZAČETEK",
                    "to_state": "KONEC",
                    "reported_state": "KONEC",
                },
                {
                    "event_index": 2,
                    "date_time": START + pd.Timedelta(seconds=2),
                    "DevKey": "breaker-a",
                    "point_family": "switchgear",
                    "signal_description": "ODKLOPNIK",
                    "from_state": "VKLOP",
                    "to_state": "IZKLOP",
                    "reported_state": "IZKLOP",
                },
                {
                    "event_index": 3,
                    "date_time": START + pd.Timedelta(seconds=3),
                    "DevKey": "breaker-a",
                    "point_family": "switchgear",
                    "signal_description": "ODKLOPNIK",
                    "from_state": "IZKLOP",
                    "to_state": "VKLOP",
                    "reported_state": "VKLOP",
                },
                {
                    "event_index": 4,
                    "date_time": START + pd.Timedelta(seconds=4),
                    "point_family": "switchgear",
                    "signal_description": "LOČILNIK",
                    "from_state": "VKLOP",
                    "to_state": "IZKLOP",
                    "reported_state": "IZKLOP",
                },
                {
                    "event_index": 5,
                    "date_time": START + pd.Timedelta(seconds=5),
                    "can_start_episode": False,
                    "event_family": "control_command",
                },
                {
                    "event_index": 6,
                    "date_time": START + pd.Timedelta(seconds=6),
                    "SifraSredstva": pd.NA,
                },
            ]
        )

        events = prepare_state_machine_v1_events(frame)

        self.assertEqual(events["event_index"].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(
            events["episode_event_role"].tolist(),
            [
                "protection_assertion",
                "protection_clear",
                "breaker_trip",
                "breaker_restoration",
                "support",
            ],
        )
        self.assertEqual(events["episode_start_trigger"].tolist(), [True, False, True, False, False])

    def test_rejects_duplicate_event_identity(self) -> None:
        frame = handoff_events(
            [
                {"event_index": 1, "date_time": START},
                {"event_index": 1, "date_time": START + pd.Timedelta(seconds=1)},
            ]
        )
        with self.assertRaisesRegex(ValueError, "must be unique"):
            prepare_state_machine_v1_events(frame)

    def test_rejects_missing_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            prepare_state_machine_v1_events(pd.DataFrame({"event_index": [1]}))


class BuildStateMachineV1EpisodesTest(unittest.TestCase):
    def test_original_artificial_fixture(self) -> None:
        events = prepared_events(
            [
                (0, START, "asset-a", "protection-a", "protection_assertion", True, "KONEC", "ZAČETEK", "ZAČETEK"),
                (
                    1,
                    START + pd.Timedelta(seconds=1),
                    "asset-a",
                    "protection-a",
                    "protection_clear",
                    False,
                    "ZAČETEK",
                    "KONEC",
                    "KONEC",
                ),
                (
                    2,
                    START + pd.Timedelta(seconds=5),
                    "asset-a",
                    "protection-b",
                    "protection_assertion",
                    True,
                    "IZPAD",
                    "IZPAD",
                    "IZPAD",
                ),
                (
                    3,
                    START + pd.Timedelta(seconds=35),
                    "asset-a",
                    "protection-b",
                    "protection_assertion",
                    True,
                    "IZPAD",
                    "IZPAD",
                    "IZPAD",
                ),
                (
                    4,
                    START + pd.Timedelta(seconds=36),
                    "asset-a",
                    "breaker-a",
                    "breaker_trip",
                    True,
                    "VKLOP",
                    "IZKLOP",
                    "IZKLOP",
                ),
                (
                    5,
                    START + pd.Timedelta(minutes=5),
                    "asset-a",
                    "breaker-a",
                    "breaker_restoration",
                    False,
                    "IZKLOP",
                    "VKLOP",
                    "VKLOP",
                ),
                (
                    6,
                    START + pd.Timedelta(minutes=70),
                    "asset-a",
                    "isolator-a",
                    "support",
                    False,
                    "VKLOP",
                    "IZKLOP",
                    "IZKLOP",
                ),
                (
                    7,
                    START + pd.Timedelta(minutes=120),
                    "asset-a",
                    "protection-c",
                    "protection_assertion",
                    True,
                    "ALARM",
                    "ALARM",
                    "ALARM",
                ),
                (
                    8,
                    START + pd.Timedelta(minutes=190),
                    "asset-a",
                    "isolator-a",
                    "support",
                    False,
                    "IZKLOP",
                    "VKLOP",
                    "VKLOP",
                ),
                (9, START, "asset-b", "protection-d", "protection_assertion", True, "KONEC", "ZAČETEK", "ZAČETEK"),
                (
                    10,
                    START + pd.Timedelta(seconds=5),
                    "asset-b",
                    "protection-e",
                    "protection_assertion",
                    True,
                    "ALARM",
                    "ALARM",
                    "ALARM",
                ),
                (
                    11,
                    START + pd.Timedelta(seconds=60),
                    "asset-b",
                    "protection-d",
                    "protection_clear",
                    False,
                    "ZAČETEK",
                    "KONEC",
                    "KONEC",
                ),
                (
                    12,
                    START + pd.Timedelta(seconds=120),
                    "asset-b",
                    "isolator-b",
                    "support",
                    False,
                    "VKLOP",
                    "IZKLOP",
                    "IZKLOP",
                ),
            ]
        )

        episodes, membership = build_state_machine_v1_episodes(events)

        self.assertEqual(len(episodes), 3)
        self.assertEqual(episodes.iloc[0]["closure_reason"], "breaker_restoration")
        self.assertTrue(episodes.iloc[0]["two_stage_retry_pattern"])
        self.assertEqual(episodes.iloc[1]["closure_reason"], "maximum_duration")
        self.assertEqual(episodes.iloc[2]["closure_reason"], "quiet_after_recovery")
        self.assertNotIn(6, set(membership["event_index"]))
        self.assertIn(11, set(membership["event_index"]))
        self.assertNotIn(12, set(membership["event_index"]))

    def test_preserves_maximum_before_pending_close_precedence(self) -> None:
        events = prepared_events(
            [
                (0, START, "asset-a", "pulse-a", "protection_assertion", True, "ALARM", "ALARM", "ALARM"),
                (
                    1,
                    START + pd.Timedelta(minutes=61),
                    "asset-a",
                    "support-a",
                    "support",
                    False,
                    "NORMAL",
                    "NORMAL",
                    "NORMAL",
                ),
            ]
        )
        episodes, membership = build_state_machine_v1_episodes(events)

        self.assertEqual(episodes.iloc[0]["closure_reason"], "maximum_duration")
        self.assertFalse(episodes.iloc[0]["episode_complete"])
        self.assertEqual(membership["event_index"].tolist(), [0])

    def test_keeps_event_exactly_on_quiet_deadline(self) -> None:
        events = prepared_events(
            [
                (0, START, "asset-a", "pulse-a", "protection_assertion", True, "ALARM", "ALARM", "ALARM"),
                (
                    1,
                    START + pd.Timedelta(seconds=45),
                    "asset-a",
                    "support-a",
                    "support",
                    False,
                    "NORMAL",
                    "NORMAL",
                    "NORMAL",
                ),
            ]
        )
        episodes, membership = build_state_machine_v1_episodes(events)

        self.assertEqual(membership["event_index"].tolist(), [0, 1])
        self.assertEqual(episodes.iloc[0]["closure_reason"], "quiet_after_recovery")
        self.assertEqual(episodes.iloc[0]["closed_at"], START + pd.Timedelta(seconds=45))

    def test_restoration_closes_before_retrip(self) -> None:
        events = prepared_events(
            [
                (0, START, "asset-a", "breaker-a", "breaker_trip", True, "VKLOP", "IZKLOP", "IZKLOP"),
                (
                    1,
                    START + pd.Timedelta(seconds=10),
                    "asset-a",
                    "breaker-a",
                    "breaker_restoration",
                    False,
                    "IZKLOP",
                    "VKLOP",
                    "VKLOP",
                ),
                (
                    2,
                    START + pd.Timedelta(seconds=15),
                    "asset-a",
                    "breaker-a",
                    "breaker_trip",
                    True,
                    "VKLOP",
                    "IZKLOP",
                    "IZKLOP",
                ),
            ]
        )
        episodes, _ = build_state_machine_v1_episodes(events)

        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes.iloc[0]["closure_reason"], "breaker_restoration")
        self.assertEqual(episodes.iloc[1]["start_event_role"], "breaker_trip")

    def test_retry_diagnostic_is_asset_wide_in_v1(self) -> None:
        events = prepared_events(
            [
                (0, START, "asset-a", "point-a", "protection_assertion", True, "ALARM", "ALARM", "ALARM"),
                (
                    1,
                    START + pd.Timedelta(seconds=5),
                    "asset-a",
                    "point-b",
                    "protection_assertion",
                    True,
                    "ALARM",
                    "ALARM",
                    "ALARM",
                ),
                (
                    2,
                    START + pd.Timedelta(seconds=35),
                    "asset-a",
                    "point-c",
                    "protection_assertion",
                    True,
                    "ALARM",
                    "ALARM",
                    "ALARM",
                ),
            ]
        )
        episodes, _ = build_state_machine_v1_episodes(events)

        self.assertTrue(episodes.iloc[0]["two_stage_retry_pattern"])

    def test_result_is_independent_of_input_order(self) -> None:
        events = prepared_events(
            [
                (0, START, "asset-b", "point-b", "protection_assertion", True, "ALARM", "ALARM", "ALARM"),
                (1, START, "asset-a", "point-a", "protection_assertion", True, "ALARM", "ALARM", "ALARM"),
                (
                    2,
                    START + pd.Timedelta(seconds=46),
                    "asset-a",
                    "point-a",
                    "support",
                    False,
                    "NORMAL",
                    "NORMAL",
                    "NORMAL",
                ),
            ]
        )
        expected = build_state_machine_v1_episodes(events)
        actual = build_state_machine_v1_episodes(events.sample(frac=1, random_state=7))

        pd.testing.assert_frame_equal(actual[0], expected[0])
        pd.testing.assert_frame_equal(actual[1], expected[1])


class RunStateMachineV1Test(unittest.TestCase):
    def test_attaches_only_context_inside_episode_interval(self) -> None:
        frame = handoff_events(
            [
                {
                    "event_index": 0,
                    "date_time": START,
                    "reported_state": "ALARM",
                    "from_state": "ALARM",
                    "to_state": "ALARM",
                },
                {
                    "event_index": 1,
                    "date_time": START + pd.Timedelta(seconds=10),
                    "can_start_episode": False,
                    "event_family": "control_command",
                    "command_target_state": "VKLOP",
                },
                {
                    "event_index": 2,
                    "date_time": START + pd.Timedelta(seconds=46),
                    "point_family": "switchgear",
                    "signal_description": "LOČILNIK",
                    "reported_state": "NORMAL",
                    "from_state": "NORMAL",
                    "to_state": "NORMAL",
                },
                {
                    "event_index": 3,
                    "date_time": START + pd.Timedelta(seconds=70),
                    "can_start_episode": False,
                    "event_family": "control_feedback",
                },
            ]
        )

        result = run_state_machine_v1(frame)

        self.assertEqual(result.episodes.iloc[0]["control_context_count"], 1)
        self.assertEqual(result.episodes.iloc[0]["restoration_command_context_count"], 1)
        self.assertEqual(result.context_links["event_index"].tolist(), [1])
        self.assertEqual(result.membership["event_index"].tolist(), [0])


if __name__ == "__main__":
    unittest.main()
