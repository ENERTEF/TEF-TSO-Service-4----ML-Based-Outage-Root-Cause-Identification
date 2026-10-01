from __future__ import annotations

import unittest

import pandas as pd
from service4.state_machine_v1_1 import (
    run_state_machine_v1_1,
    select_control_confirmed_restorations,
)

START = pd.Timestamp("2025-01-01 00:00:00")


def handoff_events(rows: list[dict[str, object]]) -> pd.DataFrame:
    defaults: dict[str, object] = {
        "DevKey": "breaker-a",
        "SifraSredstva": "asset-a",
        "signal_description": "ODKLOPNIK",
        "event_family": "automatic_state_transition",
        "point_family": "switchgear",
        "can_start_episode": True,
        "from_state": "VKLOP",
        "to_state": "IZKLOP",
        "reported_state": "IZKLOP",
        "command_target_state": pd.NA,
        "classification_rule": "automatic_device_report",
        "classification_confidence": "medium",
    }
    return pd.DataFrame.from_records([defaults | row for row in rows])


class SelectControlConfirmedRestorationsTest(unittest.TestCase):
    def test_selects_only_direct_breaker_restoration_feedback(self) -> None:
        frame = handoff_events(
            [
                {
                    "event_index": 0,
                    "date_time": START,
                    "event_family": "control_feedback",
                    "can_start_episode": False,
                    "from_state": "IZKLOP",
                    "to_state": "VKLOP",
                    "reported_state": "VKLOP",
                    "classification_rule": "command_consequence_text",
                },
                {
                    "event_index": 1,
                    "date_time": START + pd.Timedelta(seconds=1),
                    "event_family": "control_feedback",
                    "can_start_episode": False,
                    "from_state": "IZKLOP",
                    "to_state": "VKLOP",
                    "reported_state": "VKLOP",
                    "classification_rule": "entered_command_feedback",
                },
                {
                    "event_index": 2,
                    "date_time": START + pd.Timedelta(seconds=2),
                    "event_family": "control_feedback",
                    "can_start_episode": False,
                    "signal_description": "LOČILNIK",
                    "from_state": "IZKLOP",
                    "to_state": "VKLOP",
                    "reported_state": "VKLOP",
                    "classification_rule": "command_consequence_text",
                },
            ]
        )

        selected = select_control_confirmed_restorations(frame)

        self.assertEqual(selected["event_index"].tolist(), [0])
        self.assertEqual(selected.iloc[0]["episode_event_role"], "control_confirmed_restoration")


class RunStateMachineV1_1Test(unittest.TestCase):
    def test_control_feedback_closes_same_breaker_without_becoming_member(self) -> None:
        frame = handoff_events(
            [
                {"event_index": 0, "date_time": START},
                {
                    "event_index": 1,
                    "date_time": START + pd.Timedelta(seconds=14),
                    "event_family": "control_command",
                    "can_start_episode": False,
                    "from_state": "IZKLOP",
                    "to_state": "VKLOP",
                    "reported_state": "VKLOP",
                    "command_target_state": "VKLOP",
                    "classification_rule": "control_action_text",
                    "classification_confidence": "high",
                },
                {
                    "event_index": 2,
                    "date_time": START + pd.Timedelta(seconds=25, milliseconds=900),
                    "event_family": "control_feedback",
                    "can_start_episode": False,
                    "from_state": "IZKLOP",
                    "to_state": "VKLOP",
                    "reported_state": "VKLOP",
                    "classification_rule": "command_consequence_text",
                },
                {
                    "event_index": 3,
                    "date_time": START + pd.Timedelta(hours=2),
                    "DevKey": "protection-b",
                    "SifraSredstva": "asset-b",
                    "signal_description": "ZAŠČITA",
                    "point_family": "protection",
                    "from_state": "ALARM",
                    "to_state": "ALARM",
                    "reported_state": "ALARM",
                },
            ]
        )

        result = run_state_machine_v1_1(frame)
        episode = result.episodes.loc[result.episodes["SifraSredstva"].eq("asset-a")].iloc[0]

        self.assertEqual(episode["closed_at"], START + pd.Timedelta(seconds=25, milliseconds=900))
        self.assertEqual(episode["closure_reason"], "control_confirmed_restoration")
        self.assertEqual(episode["closure_event_index"], 2)
        self.assertEqual(episode["closure_confidence"], "medium")
        self.assertTrue(episode["episode_complete"])
        self.assertEqual(episode["point_count"], 1)
        self.assertNotIn("device_count", result.episodes.columns)
        self.assertEqual(
            result.membership.loc[result.membership["asset_episode_id"].eq(0), "event_index"].tolist(), [0]
        )
        self.assertEqual(
            result.context_links.loc[result.context_links["asset_episode_id"].eq(0), "event_index"].tolist(), [1, 2]
        )

    def test_feedback_for_another_point_does_not_close_open_breaker(self) -> None:
        frame = handoff_events(
            [
                {"event_index": 0, "date_time": START},
                {
                    "event_index": 1,
                    "date_time": START + pd.Timedelta(seconds=25),
                    "DevKey": "breaker-b",
                    "event_family": "control_feedback",
                    "can_start_episode": False,
                    "from_state": "IZKLOP",
                    "to_state": "VKLOP",
                    "reported_state": "VKLOP",
                    "classification_rule": "command_consequence_text",
                },
                {
                    "event_index": 2,
                    "date_time": START + pd.Timedelta(hours=2),
                    "DevKey": "protection-b",
                    "SifraSredstva": "asset-b",
                    "signal_description": "ZAŠČITA",
                    "point_family": "protection",
                    "from_state": "ALARM",
                    "to_state": "ALARM",
                    "reported_state": "ALARM",
                },
            ]
        )

        result = run_state_machine_v1_1(frame)
        episode = result.episodes.loc[result.episodes["SifraSredstva"].eq("asset-a")].iloc[0]

        self.assertEqual(episode["closure_reason"], "maximum_duration")
        self.assertFalse(episode["episode_complete"])

    def test_unrelated_feedback_does_not_advance_episode_timers(self) -> None:
        frame = handoff_events(
            [
                {
                    "event_index": 0,
                    "date_time": START,
                    "DevKey": "protection-a",
                    "signal_description": "ZAŠČITA",
                    "point_family": "protection",
                    "from_state": "NORMAL",
                    "to_state": "IZPAD",
                    "reported_state": "IZPAD",
                },
                {
                    "event_index": 1,
                    "date_time": START + pd.Timedelta(seconds=100),
                    "DevKey": "breaker-b",
                    "event_family": "control_feedback",
                    "can_start_episode": False,
                    "from_state": "IZKLOP",
                    "to_state": "VKLOP",
                    "reported_state": "VKLOP",
                    "classification_rule": "command_consequence_text",
                },
                {
                    "event_index": 2,
                    "date_time": START + pd.Timedelta(hours=2),
                    "DevKey": "protection-b",
                    "SifraSredstva": "asset-b",
                    "signal_description": "ZAŠČITA",
                    "point_family": "protection",
                    "from_state": "NORMAL",
                    "to_state": "IZPAD",
                    "reported_state": "IZPAD",
                },
            ]
        )

        result = run_state_machine_v1_1(frame)
        episode = result.episodes.loc[result.episodes["SifraSredstva"].eq("asset-a")].iloc[0]

        self.assertEqual(episode["closure_reason"], "quiet_after_recovery")
        self.assertEqual(episode["closed_at"], START + pd.Timedelta(seconds=45))
        self.assertTrue(episode["episode_complete"])

    def test_equal_timestamp_feedback_remains_linked_to_episode_it_closed(self) -> None:
        frame = handoff_events(
            [
                {"event_index": 0, "date_time": START},
                {
                    "event_index": 1,
                    "date_time": START + pd.Timedelta(seconds=10),
                    "event_family": "control_feedback",
                    "can_start_episode": False,
                    "from_state": "IZKLOP",
                    "to_state": "VKLOP",
                    "reported_state": "VKLOP",
                    "classification_rule": "command_consequence_text",
                },
                {
                    "event_index": 2,
                    "date_time": START + pd.Timedelta(seconds=10),
                },
                {
                    "event_index": 3,
                    "date_time": START + pd.Timedelta(hours=2),
                    "DevKey": "protection-b",
                    "SifraSredstva": "asset-b",
                    "signal_description": "ZAŠČITA",
                    "point_family": "protection",
                    "from_state": "NORMAL",
                    "to_state": "IZPAD",
                    "reported_state": "IZPAD",
                },
            ]
        )

        result = run_state_machine_v1_1(frame)
        asset_episodes = result.episodes.loc[
            result.episodes["SifraSredstva"].eq("asset-a")
        ].sort_values("asset_episode_id")
        first_episode_id, second_episode_id = asset_episodes["asset_episode_id"].tolist()
        feedback_link = result.context_links.loc[result.context_links["event_index"].eq(1)].iloc[0]

        self.assertEqual(feedback_link["asset_episode_id"], first_episode_id)
        self.assertNotEqual(feedback_link["asset_episode_id"], second_episode_id)
        self.assertEqual(
            asset_episodes.set_index("asset_episode_id").loc[first_episode_id, "control_context_count"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
