from __future__ import annotations

import unittest

import pandas as pd
from service4.state_machine_v2 import build_state_machine_v2_episodes

START = pd.Timestamp("2025-01-01 00:00:00")


def prepared_events(rows: list[dict[str, object]]) -> pd.DataFrame:
    defaults: dict[str, object] = {
        "SifraSredstva": "asset-a",
        "DevKey": "protection-a",
        "episode_event_role": "protection_assertion",
        "episode_start_trigger": True,
        "reported_state": "IZPAD",
        "canonical_self_transition": True,
    }
    return pd.DataFrame.from_records([defaults | row for row in rows])


class BuildStateMachineV2Test(unittest.TestCase):
    def test_uses_earliest_pending_deadline_before_maximum(self) -> None:
        events = prepared_events(
            [
                {"event_index": 0, "date_time": START},
                {
                    "event_index": 1,
                    "date_time": START + pd.Timedelta(hours=2),
                    "SifraSredstva": "asset-b",
                },
            ]
        )

        episodes, _ = build_state_machine_v2_episodes(events)
        episode = episodes.loc[episodes["SifraSredstva"].eq("asset-a")].iloc[0]

        self.assertEqual(episode["closed_at"], START + pd.Timedelta(seconds=45))
        self.assertEqual(episode["closure_reason"], "quiet_after_protection_pulse")
        self.assertTrue(episode["episode_complete"])

    def test_breaker_restoration_is_provisional_until_quiet(self) -> None:
        events = prepared_events(
            [
                {
                    "event_index": 0,
                    "date_time": START,
                    "DevKey": "breaker-a",
                    "episode_event_role": "breaker_trip",
                    "reported_state": "IZKLOP",
                },
                {
                    "event_index": 1,
                    "date_time": START + pd.Timedelta(seconds=1),
                    "DevKey": "breaker-a",
                    "episode_event_role": "breaker_restoration",
                    "episode_start_trigger": False,
                    "reported_state": "VKLOP",
                },
                {
                    "event_index": 2,
                    "date_time": START + pd.Timedelta(seconds=5),
                    "DevKey": "breaker-a",
                    "episode_event_role": "breaker_trip",
                    "reported_state": "IZKLOP",
                },
                {
                    "event_index": 3,
                    "date_time": START + pd.Timedelta(seconds=10),
                    "DevKey": "breaker-a",
                    "episode_event_role": "breaker_restoration",
                    "episode_start_trigger": False,
                    "reported_state": "VKLOP",
                },
                {
                    "event_index": 4,
                    "date_time": START + pd.Timedelta(seconds=120),
                    "DevKey": "isolator-a",
                    "episode_event_role": "support",
                    "episode_start_trigger": False,
                    "reported_state": "IZKLOP",
                },
            ]
        )

        episodes, membership = build_state_machine_v2_episodes(events)

        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes.iloc[0]["closed_at"], START + pd.Timedelta(seconds=55))
        self.assertEqual(episodes.iloc[0]["closure_reason"], "quiet_after_breaker_restoration")
        self.assertEqual(membership["event_index"].tolist(), [0, 1, 2, 3])

    def test_retry_diagnostics_are_point_local(self) -> None:
        events = prepared_events(
            [
                {"event_index": 0, "date_time": START},
                {
                    "event_index": 1,
                    "date_time": START + pd.Timedelta(seconds=5),
                    "DevKey": "protection-b",
                },
                {"event_index": 2, "date_time": START + pd.Timedelta(seconds=5)},
                {"event_index": 3, "date_time": START + pd.Timedelta(seconds=35)},
                {
                    "event_index": 4,
                    "date_time": START + pd.Timedelta(seconds=100),
                    "DevKey": "isolator-a",
                    "episode_event_role": "support",
                    "episode_start_trigger": False,
                    "reported_state": "IZKLOP",
                },
            ]
        )

        episodes, _ = build_state_machine_v2_episodes(events)
        episode = episodes.iloc[0]

        self.assertEqual(episode["fast_retry_count"], 1)
        self.assertEqual(episode["delayed_retry_count"], 1)
        self.assertTrue(episode["two_stage_retry_pattern"])


if __name__ == "__main__":
    unittest.main()
