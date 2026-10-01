"""Frozen state-machine v2.0 parent-episode construction.

This module promotes the already-declared v2 parent algorithm from the
clustering notebook without changing its semantics. V2 uses the frozen-v1
automatic event scope, earliest-deadline closure, provisional recovery and
point-local retry diagnostics. Outage references do not enter construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from service4.state_machine_v1 import (
    PROTECTION_PULSE_STATES,
    attach_v1_control_context,
    prepare_state_machine_v1_events,
)

ASSET_EPISODE_GRAMMAR_VERSION: Final[str] = "2.0"
RETRY_GRACE: Final[pd.Timedelta] = pd.Timedelta(seconds=45)
MAX_EPISODE_DURATION: Final[pd.Timedelta] = pd.Timedelta(minutes=60)

_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "event_index",
        "date_time",
        "DevKey",
        "SifraSredstva",
        "episode_event_role",
        "episode_start_trigger",
        "reported_state",
    }
)
_EPISODE_COLUMNS: Final[tuple[str, ...]] = (
    "asset_episode_id",
    "SifraSredstva",
    "cluster_start",
    "cluster_end",
    "closed_at",
    "maximum_deadline",
    "closure_reason",
    "episode_complete",
    "start_event_role",
    "last_event_role",
    "active_protection_count_at_close",
    "open_breaker_count_at_close",
    "pending_close_at",
    "pending_close_reason",
    "recovery_pending_at_close",
    "event_count",
    "device_count",
    "first_event_index",
    "protection_trigger_count",
    "protection_self_report_count",
    "breaker_trip_count",
    "breaker_restoration_count",
    "fast_retry_count",
    "delayed_retry_count",
    "two_stage_retry_pattern",
)
_MEMBERSHIP_COLUMNS: Final[tuple[str, ...]] = (
    "event_index",
    "asset_episode_id",
    "episode_event_role",
)


@dataclass(frozen=True)
class StateMachineV2Result:
    """Tables produced by the frozen v2.0 parent construction."""

    events: pd.DataFrame
    episodes: pd.DataFrame
    membership: pd.DataFrame
    context_links: pd.DataFrame


def _validate_events(events: pd.DataFrame) -> None:
    missing = sorted(_REQUIRED_COLUMNS.difference(events.columns))
    if missing:
        raise ValueError(f"events is missing required columns: {missing}")
    if events["event_index"].isna().any() or events["event_index"].duplicated().any():
        raise ValueError("events.event_index must be non-null and unique")
    if not is_datetime64_any_dtype(events["date_time"].dtype):
        raise TypeError("events.date_time must have a pandas datetime dtype")
    if events["date_time"].isna().any():
        raise ValueError("events.date_time contains null values")


def build_state_machine_v2_episodes(
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct deterministic v2.0 parent episodes without outage references."""
    _validate_events(events)
    if events.empty:
        return (
            pd.DataFrame(columns=[*_EPISODE_COLUMNS, "observed_duration"]),
            pd.DataFrame(columns=_MEMBERSHIP_COLUMNS),
        )

    episode_records: list[dict[str, object]] = []
    membership_records: list[dict[str, object]] = []
    next_episode_id = 0
    data_end = events["date_time"].max()

    def arm_pending_close(
        current: dict[str, object],
        event_time: pd.Timestamp,
        reason: str,
    ) -> None:
        current["pending_close"] = event_time + RETRY_GRACE
        current["pending_close_reason"] = reason

    def cancel_pending_close(current: dict[str, object]) -> None:
        current["pending_close"] = None
        current["pending_close_reason"] = None

    def earliest_deadline(
        current: dict[str, object],
    ) -> tuple[pd.Timestamp, str, bool]:
        maximum_close = current["cluster_start"] + MAX_EPISODE_DURATION
        pending_close = current["pending_close"]
        if pending_close is not None and pending_close <= maximum_close:
            return pending_close, str(current["pending_close_reason"]), True
        return maximum_close, "maximum_duration", False

    def finish_episode(
        current: dict[str, object],
        closure_reason: str,
        closed_at: pd.Timestamp,
        complete: bool,
        active_protection_count: int,
        open_breaker_count: int,
    ) -> None:
        episode_records.append(
            {
                "asset_episode_id": current["asset_episode_id"],
                "SifraSredstva": current["SifraSredstva"],
                "cluster_start": current["cluster_start"],
                "cluster_end": current["last_event_time"],
                "closed_at": closed_at,
                "maximum_deadline": current["cluster_start"] + MAX_EPISODE_DURATION,
                "closure_reason": closure_reason,
                "episode_complete": complete,
                "start_event_role": current["start_event_role"],
                "last_event_role": current["last_event_role"],
                "active_protection_count_at_close": active_protection_count,
                "open_breaker_count_at_close": open_breaker_count,
                "pending_close_at": current["pending_close"],
                "pending_close_reason": current["pending_close_reason"],
                "recovery_pending_at_close": current["pending_close"] is not None,
                "event_count": current["event_count"],
                "device_count": len(current["devices"]),
                "first_event_index": current["first_event_index"],
                "protection_trigger_count": current["protection_trigger_count"],
                "protection_self_report_count": current["protection_self_report_count"],
                "breaker_trip_count": current["breaker_trip_count"],
                "breaker_restoration_count": current["breaker_restoration_count"],
                "fast_retry_count": current["fast_retry_count"],
                "delayed_retry_count": current["delayed_retry_count"],
                "two_stage_retry_pattern": current["two_stage_retry_pattern"],
            }
        )

    ordered = events.sort_values(["SifraSredstva", "date_time", "event_index"])
    for asset, asset_events in ordered.groupby("SifraSredstva", sort=True):
        current: dict[str, object] | None = None
        active_protection_points: set[str] = set()
        open_breakers: set[str] = set()

        for event in asset_events.itertuples(index=False):
            event_time = event.date_time
            if current is not None:
                deadline, deadline_reason, deadline_complete = earliest_deadline(current)
                if event_time > deadline:
                    finish_episode(
                        current,
                        deadline_reason,
                        deadline,
                        deadline_complete,
                        len(active_protection_points),
                        len(open_breakers),
                    )
                    current = None
                    active_protection_points = set()
                    open_breakers = set()

            if current is None:
                if not event.episode_start_trigger:
                    continue
                current = {
                    "asset_episode_id": next_episode_id,
                    "SifraSredstva": asset,
                    "cluster_start": event_time,
                    "last_event_time": event_time,
                    "pending_close": None,
                    "pending_close_reason": None,
                    "event_count": 0,
                    "devices": set(),
                    "first_event_index": event.event_index,
                    "start_event_role": event.episode_event_role,
                    "last_event_role": event.episode_event_role,
                    "protection_trigger_count": 0,
                    "protection_self_report_count": 0,
                    "breaker_trip_count": 0,
                    "breaker_restoration_count": 0,
                    "fast_retry_count": 0,
                    "delayed_retry_count": 0,
                    "two_stage_retry_pattern": False,
                    "protection_retry_state": {},
                }
                next_episode_id += 1

            membership_records.append(
                {
                    "event_index": event.event_index,
                    "asset_episode_id": current["asset_episode_id"],
                    "episode_event_role": event.episode_event_role,
                }
            )
            current["last_event_time"] = event_time
            current["last_event_role"] = event.episode_event_role
            current["event_count"] += 1
            current["devices"].add(event.DevKey)

            if event.episode_event_role == "protection_assertion":
                current["protection_trigger_count"] += 1
                canonical_self_transition = getattr(event, "canonical_self_transition", False)
                if pd.notna(canonical_self_transition) and bool(canonical_self_transition):
                    current["protection_self_report_count"] += 1

                point_retry_state = current["protection_retry_state"].setdefault(
                    event.DevKey,
                    {
                        "previous_trigger_time": None,
                        "previous_interval_seconds": None,
                        "cleared_since_trigger": False,
                    },
                )
                previous_trigger_time = point_retry_state["previous_trigger_time"]
                recurrence_is_eligible = event.reported_state in PROTECTION_PULSE_STATES or (
                    event.reported_state == "ZAČETEK"
                    and point_retry_state["cleared_since_trigger"]
                )
                if previous_trigger_time is None or recurrence_is_eligible:
                    if previous_trigger_time is not None:
                        interval_seconds = (event_time - previous_trigger_time).total_seconds()
                        if 0 < interval_seconds <= 10:
                            current["fast_retry_count"] += 1
                        if 20 <= interval_seconds <= 45:
                            current["delayed_retry_count"] += 1
                            previous_interval = point_retry_state["previous_interval_seconds"]
                            if previous_interval is not None and 0 < previous_interval <= 10:
                                current["two_stage_retry_pattern"] = True
                        point_retry_state["previous_interval_seconds"] = interval_seconds
                    point_retry_state["previous_trigger_time"] = event_time

                if event.reported_state == "ZAČETEK":
                    active_protection_points.add(event.DevKey)
                    point_retry_state["cleared_since_trigger"] = False
                if (
                    event.reported_state in PROTECTION_PULSE_STATES
                    and not active_protection_points
                    and not open_breakers
                ):
                    arm_pending_close(current, event_time, "quiet_after_protection_pulse")
                else:
                    cancel_pending_close(current)

            elif event.episode_event_role == "protection_clear":
                active_protection_points.discard(event.DevKey)
                point_retry_state = current["protection_retry_state"].setdefault(
                    event.DevKey,
                    {
                        "previous_trigger_time": None,
                        "previous_interval_seconds": None,
                        "cleared_since_trigger": False,
                    },
                )
                point_retry_state["cleared_since_trigger"] = True
                if not active_protection_points and not open_breakers:
                    arm_pending_close(current, event_time, "quiet_after_protection_recovery")

            elif event.episode_event_role == "breaker_trip":
                current["breaker_trip_count"] += 1
                open_breakers.add(event.DevKey)
                cancel_pending_close(current)

            elif event.episode_event_role == "breaker_restoration":
                current["breaker_restoration_count"] += 1
                open_breakers.discard(event.DevKey)
                if not active_protection_points and not open_breakers:
                    arm_pending_close(current, event_time, "quiet_after_breaker_restoration")

        if current is not None:
            deadline, deadline_reason, deadline_complete = earliest_deadline(current)
            if deadline <= data_end:
                finish_episode(
                    current,
                    deadline_reason,
                    deadline,
                    deadline_complete,
                    len(active_protection_points),
                    len(open_breakers),
                )
            else:
                finish_episode(
                    current,
                    "open_at_data_end",
                    data_end,
                    False,
                    len(active_protection_points),
                    len(open_breakers),
                )

    episodes = pd.DataFrame.from_records(episode_records, columns=_EPISODE_COLUMNS)
    episodes["observed_duration"] = episodes["cluster_end"].sub(episodes["cluster_start"])
    memberships = pd.DataFrame.from_records(membership_records, columns=_MEMBERSHIP_COLUMNS)
    return episodes, memberships


def run_state_machine_v2(handoff: pd.DataFrame) -> StateMachineV2Result:
    """Run frozen automatic selection, v2.0 construction and control linking."""
    events = prepare_state_machine_v1_events(handoff)
    episodes, membership = build_state_machine_v2_episodes(events)
    episodes_with_context, context_links = attach_v1_control_context(handoff, episodes)
    return StateMachineV2Result(
        events=events,
        episodes=episodes_with_context,
        membership=membership,
        context_links=context_links,
    )
