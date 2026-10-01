"""State-machine v1.1 with control-confirmed breaker restoration.

Version 1.1 is a narrow revision of frozen v1. Automatic event eligibility and
start rules are unchanged; episode membership may change when an earlier close
separates later activity. A direct ``Na objektu posledica komande`` breaker
restoration may update the state of an already-open breaker and close its episode,
but remains control context rather than automatic incident evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from service4.state_machine_v1 import (
    MAX_EPISODE_DURATION,
    PROTECTION_PULSE_STATES,
    RETRY_GRACE,
    attach_v1_control_context,
    prepare_state_machine_v1_events,
)

ASSET_EPISODE_GRAMMAR_VERSION: Final[str] = "1.1"
CONTROL_RESTORATION_RULE: Final[str] = "command_consequence_text"

_RESTORATION_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "event_index",
        "date_time",
        "DevKey",
        "SifraSredstva",
        "event_family",
        "point_family",
        "signal_description",
        "from_state",
        "to_state",
        "classification_rule",
        "classification_confidence",
    }
)
_PREPARED_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
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
    "closure_reason",
    "closure_event_index",
    "closure_evidence",
    "closure_confidence",
    "episode_complete",
    "start_event_role",
    "last_event_role",
    "active_protection_count_at_close",
    "open_breaker_count_at_close",
    "pending_close_at",
    "recovery_pending_at_close",
    "event_count",
    "point_count",
    "first_event_index",
    "protection_trigger_count",
    "breaker_trip_count",
    "breaker_restoration_count",
    "control_confirmed_restoration_count",
    "fast_retry_count",
    "delayed_retry_count",
    "two_stage_retry_pattern",
)
_MEMBERSHIP_COLUMNS: Final[tuple[str, ...]] = (
    "event_index",
    "asset_episode_id",
    "episode_event_role",
)
_STATE_UPDATE_OWNER_COLUMNS: Final[tuple[str, ...]] = (
    "event_index",
    "asset_episode_id",
)


@dataclass(frozen=True)
class StateMachineV1_1Result:
    """Tables produced by the complete v1.1 construction."""

    events: pd.DataFrame
    episodes: pd.DataFrame
    membership: pd.DataFrame
    context_links: pd.DataFrame


def _require_columns(frame: pd.DataFrame, required: frozenset[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _validate_event_identity(frame: pd.DataFrame, label: str) -> None:
    if frame["event_index"].isna().any():
        raise ValueError(f"{label}.event_index contains null values")
    if frame["event_index"].duplicated().any():
        raise ValueError(f"{label}.event_index must be unique")
    if not is_datetime64_any_dtype(frame["date_time"].dtype):
        raise TypeError(f"{label}.date_time must have a pandas datetime dtype")
    if frame["date_time"].isna().any():
        raise ValueError(f"{label}.date_time contains null values")


def select_control_confirmed_restorations(handoff: pd.DataFrame) -> pd.DataFrame:
    """Select direct command-consequence reports eligible to update breaker state."""
    _require_columns(handoff, _RESTORATION_REQUIRED_COLUMNS, "handoff")
    _validate_event_identity(handoff, "handoff")

    is_breaker = handoff["point_family"].eq("switchgear") & handoff["signal_description"].str.contains(
        r"\bODKLOPNIK\b",
        case=False,
        na=False,
    )
    restoration = handoff.loc[
        handoff["SifraSredstva"].notna()
        & handoff["event_family"].eq("control_feedback")
        & handoff["classification_rule"].eq(CONTROL_RESTORATION_RULE)
        & is_breaker
        & handoff["from_state"].eq("IZKLOP")
        & handoff["to_state"].eq("VKLOP"),
        [
            "event_index",
            "date_time",
            "DevKey",
            "SifraSredstva",
            "classification_confidence",
        ],
    ].copy()
    restoration["SifraSredstva"] = restoration["SifraSredstva"].astype("string")
    restoration["episode_event_role"] = "control_confirmed_restoration"
    restoration["episode_start_trigger"] = False
    restoration["reported_state"] = "VKLOP"
    restoration["stream_kind"] = "state_update_context"
    return restoration.sort_values(["SifraSredstva", "date_time", "event_index"])


def _construct_state_machine_v1_1_episodes(
    events: pd.DataFrame,
    control_restorations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construct episodes and preserve ownership of consumed state updates."""
    _require_columns(events, _PREPARED_REQUIRED_COLUMNS, "events")
    _validate_event_identity(events, "events")
    _require_columns(control_restorations, _PREPARED_REQUIRED_COLUMNS, "control restorations")
    _validate_event_identity(control_restorations, "control restorations")

    if events.empty:
        return (
            pd.DataFrame(columns=[*_EPISODE_COLUMNS, "observed_duration"]),
            pd.DataFrame(columns=_MEMBERSHIP_COLUMNS),
            pd.DataFrame(columns=_STATE_UPDATE_OWNER_COLUMNS),
        )

    automatic = events.copy()
    automatic["stream_kind"] = "automatic_event"
    if "classification_confidence" not in automatic:
        automatic["classification_confidence"] = "not_recorded"

    stream_columns = [
        "event_index",
        "date_time",
        "DevKey",
        "SifraSredstva",
        "episode_event_role",
        "episode_start_trigger",
        "reported_state",
        "classification_confidence",
        "stream_kind",
    ]
    ordered = pd.concat(
        [automatic.loc[:, stream_columns], control_restorations.loc[:, stream_columns]],
        ignore_index=True,
    ).sort_values(["SifraSredstva", "date_time", "event_index"])

    episode_records: list[dict[str, object]] = []
    membership_records: list[dict[str, object]] = []
    state_update_owner_records: list[dict[str, object]] = []
    next_episode_id = 0
    data_end = events["date_time"].max()

    def finish_episode(
        current: dict[str, object],
        *,
        closure_reason: str,
        closed_at: pd.Timestamp,
        complete: bool,
        active_protection_count: int,
        open_breaker_count: int,
        closure_event_index: int | None = None,
        closure_evidence: str,
        closure_confidence: str,
    ) -> None:
        episode_records.append(
            {
                "asset_episode_id": current["asset_episode_id"],
                "SifraSredstva": current["SifraSredstva"],
                "cluster_start": current["cluster_start"],
                "cluster_end": current["last_event_time"],
                "closed_at": closed_at,
                "closure_reason": closure_reason,
                "closure_event_index": closure_event_index,
                "closure_evidence": closure_evidence,
                "closure_confidence": closure_confidence,
                "episode_complete": complete,
                "start_event_role": current["start_event_role"],
                "last_event_role": current["last_event_role"],
                "active_protection_count_at_close": active_protection_count,
                "open_breaker_count_at_close": open_breaker_count,
                "pending_close_at": current["pending_close"],
                "recovery_pending_at_close": current["pending_close"] is not None,
                "event_count": current["event_count"],
                "point_count": len(current["points"]),
                "first_event_index": current["first_event_index"],
                "protection_trigger_count": current["protection_trigger_count"],
                "breaker_trip_count": current["breaker_trip_count"],
                "breaker_restoration_count": current["breaker_restoration_count"],
                "control_confirmed_restoration_count": current["control_confirmed_restoration_count"],
                "fast_retry_count": current["fast_retry_count"],
                "delayed_retry_count": current["delayed_retry_count"],
                "two_stage_retry_pattern": current["two_stage_retry_pattern"],
            }
        )

    for asset, asset_stream in ordered.groupby("SifraSredstva", sort=True):
        current: dict[str, object] | None = None
        active_protection_points: set[str] = set()
        open_breakers: set[str] = set()

        for event in asset_stream.itertuples(index=False):
            event_time = event.date_time
            is_state_update = event.stream_kind == "state_update_context"
            if is_state_update and (current is None or event.DevKey not in open_breakers):
                # Context for an unrelated point must not advance this episode's
                # timers or otherwise affect frozen-v1 episode construction.
                continue

            if current is not None:
                maximum_close = current["cluster_start"] + MAX_EPISODE_DURATION
                pending_close = current["pending_close"]
                if event_time > maximum_close:
                    finish_episode(
                        current,
                        closure_reason="maximum_duration",
                        closed_at=maximum_close,
                        complete=False,
                        active_protection_count=len(active_protection_points),
                        open_breaker_count=len(open_breakers),
                        closure_evidence="maximum_duration",
                        closure_confidence="not_observed",
                    )
                    current = None
                    active_protection_points = set()
                    open_breakers = set()
                elif pending_close is not None and event_time > pending_close:
                    finish_episode(
                        current,
                        closure_reason="quiet_after_recovery",
                        closed_at=pending_close,
                        complete=True,
                        active_protection_count=len(active_protection_points),
                        open_breaker_count=len(open_breakers),
                        closure_evidence="timer_rule",
                        closure_confidence="rule_based",
                    )
                    current = None
                    active_protection_points = set()
                    open_breakers = set()

            if is_state_update:
                if current is not None and event.DevKey in open_breakers:
                    state_update_owner_records.append(
                        {
                            "event_index": event.event_index,
                            "asset_episode_id": current["asset_episode_id"],
                        }
                    )
                    current["control_confirmed_restoration_count"] += 1
                    open_breakers.discard(event.DevKey)
                    if not active_protection_points and not open_breakers:
                        finish_episode(
                            current,
                            closure_reason="control_confirmed_restoration",
                            closed_at=event_time,
                            complete=True,
                            active_protection_count=0,
                            open_breaker_count=0,
                            closure_event_index=event.event_index,
                            closure_evidence="control_feedback",
                            closure_confidence=str(event.classification_confidence),
                        )
                        current = None
                continue

            if current is None:
                if not event.episode_start_trigger:
                    continue
                current = {
                    "asset_episode_id": next_episode_id,
                    "SifraSredstva": asset,
                    "cluster_start": event_time,
                    "last_event_time": event_time,
                    "pending_close": None,
                    "event_count": 0,
                    "points": set(),
                    "first_event_index": event.event_index,
                    "start_event_role": event.episode_event_role,
                    "last_event_role": event.episode_event_role,
                    "protection_trigger_count": 0,
                    "breaker_trip_count": 0,
                    "breaker_restoration_count": 0,
                    "control_confirmed_restoration_count": 0,
                    "fast_retry_count": 0,
                    "delayed_retry_count": 0,
                    "two_stage_retry_pattern": False,
                    "previous_protection_trigger_time": None,
                    "previous_protection_interval_seconds": None,
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
            current["points"].add(event.DevKey)

            if event.episode_event_role == "protection_assertion":
                current["protection_trigger_count"] += 1
                previous_trigger_time = current["previous_protection_trigger_time"]
                previous_interval = current["previous_protection_interval_seconds"]
                if previous_trigger_time is not None:
                    interval_seconds = (event_time - previous_trigger_time).total_seconds()
                    if interval_seconds <= 10:
                        current["fast_retry_count"] += 1
                    if 20 <= interval_seconds <= 45:
                        current["delayed_retry_count"] += 1
                        if previous_interval is not None and previous_interval <= 10:
                            current["two_stage_retry_pattern"] = True
                    current["previous_protection_interval_seconds"] = interval_seconds
                current["previous_protection_trigger_time"] = event_time

                if event.reported_state == "ZAČETEK":
                    active_protection_points.add(event.DevKey)
                if (
                    event.reported_state in PROTECTION_PULSE_STATES
                    and not active_protection_points
                    and not open_breakers
                ):
                    current["pending_close"] = event_time + RETRY_GRACE
                else:
                    current["pending_close"] = None

            elif event.episode_event_role == "protection_clear":
                active_protection_points.discard(event.DevKey)
                if not active_protection_points and not open_breakers:
                    current["pending_close"] = event_time + RETRY_GRACE

            elif event.episode_event_role == "breaker_trip":
                current["breaker_trip_count"] += 1
                open_breakers.add(event.DevKey)
                current["pending_close"] = None

            elif event.episode_event_role == "breaker_restoration":
                current["breaker_restoration_count"] += 1
                open_breakers.discard(event.DevKey)
                if not active_protection_points and not open_breakers:
                    finish_episode(
                        current,
                        closure_reason="breaker_restoration",
                        closed_at=event_time,
                        complete=True,
                        active_protection_count=0,
                        open_breaker_count=0,
                        closure_event_index=event.event_index,
                        closure_evidence="automatic_state_transition",
                        closure_confidence=str(event.classification_confidence),
                    )
                    current = None

        if current is not None:
            maximum_close = current["cluster_start"] + MAX_EPISODE_DURATION
            pending_close = current["pending_close"]
            if pending_close is not None and pending_close <= data_end:
                finish_episode(
                    current,
                    closure_reason="quiet_after_recovery",
                    closed_at=pending_close,
                    complete=True,
                    active_protection_count=len(active_protection_points),
                    open_breaker_count=len(open_breakers),
                    closure_evidence="timer_rule",
                    closure_confidence="rule_based",
                )
            elif maximum_close <= data_end:
                finish_episode(
                    current,
                    closure_reason="maximum_duration",
                    closed_at=maximum_close,
                    complete=False,
                    active_protection_count=len(active_protection_points),
                    open_breaker_count=len(open_breakers),
                    closure_evidence="maximum_duration",
                    closure_confidence="not_observed",
                )
            else:
                finish_episode(
                    current,
                    closure_reason="open_at_data_end",
                    closed_at=data_end,
                    complete=False,
                    active_protection_count=len(active_protection_points),
                    open_breaker_count=len(open_breakers),
                    closure_evidence="data_boundary",
                    closure_confidence="not_observed",
                )

    episodes = pd.DataFrame.from_records(episode_records, columns=_EPISODE_COLUMNS)
    episodes["closure_event_index"] = episodes["closure_event_index"].astype("Int64")
    episodes["observed_duration"] = episodes["cluster_end"].sub(episodes["cluster_start"])
    memberships = pd.DataFrame.from_records(membership_records, columns=_MEMBERSHIP_COLUMNS)
    state_update_owners = pd.DataFrame.from_records(
        state_update_owner_records,
        columns=_STATE_UPDATE_OWNER_COLUMNS,
    )
    return episodes, memberships, state_update_owners


def build_state_machine_v1_1_episodes(
    events: pd.DataFrame,
    control_restorations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct v1.1 episodes from automatic events and non-member state updates."""
    episodes, memberships, _ = _construct_state_machine_v1_1_episodes(
        events,
        control_restorations,
    )
    return episodes, memberships


def _attach_v1_1_control_context(
    handoff: pd.DataFrame,
    episodes: pd.DataFrame,
    state_update_owners: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach context while preserving constructor ownership of state updates."""
    _, links = attach_v1_control_context(handoff, episodes)

    if not links.empty and not state_update_owners.empty:
        episode_lookup = episodes.loc[
            :, ["asset_episode_id", "cluster_start", "closed_at"]
        ].rename(
            columns={
                "asset_episode_id": "owner_asset_episode_id",
                "cluster_start": "owner_cluster_start",
                "closed_at": "owner_closed_at",
            }
        )
        owners = state_update_owners.merge(
            episode_lookup,
            left_on="asset_episode_id",
            right_on="owner_asset_episode_id",
            validate="many_to_one",
        ).drop(columns="asset_episode_id")
        links = links.merge(owners, on="event_index", how="left", validate="one_to_one")
        owned = links["owner_asset_episode_id"].notna()
        links.loc[owned, "asset_episode_id"] = links.loc[owned, "owner_asset_episode_id"]
        links.loc[owned, "cluster_start"] = links.loc[owned, "owner_cluster_start"]
        links.loc[owned, "closed_at"] = links.loc[owned, "owner_closed_at"]
        links = links.drop(
            columns=["owner_asset_episode_id", "owner_cluster_start", "owner_closed_at"]
        )

    enhanced = episodes.copy()
    counts = links.groupby("asset_episode_id").agg(
        control_context_count=("event_index", "size"),
        restoration_command_context_count=("is_restoration_command_context", "sum"),
    )
    enhanced = enhanced.join(counts, on="asset_episode_id")
    for column in ("control_context_count", "restoration_command_context_count"):
        enhanced[column] = enhanced[column].fillna(0).astype("int64")
    enhanced["has_control_context"] = enhanced["control_context_count"].gt(0)
    enhanced["has_restoration_command_context"] = enhanced[
        "restoration_command_context_count"
    ].gt(0)
    return enhanced, links


def run_state_machine_v1_1(handoff: pd.DataFrame) -> StateMachineV1_1Result:
    """Run automatic selection, v1.1 construction and context linking."""
    events = prepare_state_machine_v1_events(handoff)
    restorations = select_control_confirmed_restorations(handoff)
    episodes, membership, state_update_owners = _construct_state_machine_v1_1_episodes(
        events,
        restorations,
    )
    episodes_with_context, context_links = _attach_v1_1_control_context(
        handoff,
        episodes,
        state_update_owners,
    )
    return StateMachineV1_1Result(
        events=events,
        episodes=episodes_with_context,
        membership=membership,
        context_links=context_links,
    )
