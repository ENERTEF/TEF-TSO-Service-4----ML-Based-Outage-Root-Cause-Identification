"""Frozen asset-level incident state machine, version 1.

The implementation intentionally preserves the recorded version-1 design
decisions. In particular, maximum-duration closure precedes
pending quiet closure and breaker restoration closes an episode immediately.
Those behaviours differ from the later research prototype and must not be
changed without declaring another version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

ASSET_EPISODE_GRAMMAR_VERSION: Final[int] = 1
RETRY_GRACE: Final[pd.Timedelta] = pd.Timedelta(seconds=45)
MAX_EPISODE_DURATION: Final[pd.Timedelta] = pd.Timedelta(minutes=60)

ASSET_SEQUENCE_POINT_FAMILIES: Final[frozenset[str]] = frozenset({"protection", "switchgear"})
PROTECTION_ASSERTED_STATES: Final[frozenset[str]] = frozenset({"ZAČETEK", "ALARM", "IZPAD", "DELOVAL"})
PROTECTION_PULSE_STATES: Final[frozenset[str]] = frozenset({"ALARM", "IZPAD", "DELOVAL"})
EPISODE_CONTROL_CONTEXT_FAMILIES: Final[frozenset[str]] = frozenset(
    {"control_command", "control_feedback", "manual_value_action"}
)

_HANDOFF_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "event_index",
        "date_time",
        "DevKey",
        "SifraSredstva",
        "signal_description",
        "event_family",
        "point_family",
        "can_start_episode",
        "from_state",
        "to_state",
        "reported_state",
        "command_target_state",
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
    "episode_complete",
    "start_event_role",
    "last_event_role",
    "active_protection_count_at_close",
    "open_breaker_count_at_close",
    "pending_close_at",
    "recovery_pending_at_close",
    "event_count",
    "device_count",
    "first_event_index",
    "protection_trigger_count",
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
_CONTEXT_COUNT_COLUMNS: Final[tuple[str, ...]] = (
    "control_context_count",
    "restoration_command_context_count",
)


@dataclass(frozen=True)
class StateMachineV1Result:
    """Tables produced by the complete frozen v1 construction."""

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
    duplicate_mask = frame["event_index"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicates = frame.loc[duplicate_mask, "event_index"].drop_duplicates().head(5).tolist()
        raise ValueError(f"{label}.event_index must be unique; examples: {duplicates}")
    if not is_datetime64_any_dtype(frame["date_time"].dtype):
        raise TypeError(f"{label}.date_time must have a pandas datetime dtype")
    if frame["date_time"].isna().any():
        raise ValueError(f"{label}.date_time contains null values")


def prepare_state_machine_v1_events(handoff: pd.DataFrame) -> pd.DataFrame:
    """Select and classify the frozen v1 automatic construction scope.

    can_start_episode is the historical handoff name for automatic episode
    evidence. The more restrictive episode_start_trigger generated here
    controls which protection and breaker observations may actually open a v1
    episode.
    """
    _require_columns(handoff, _HANDOFF_REQUIRED_COLUMNS, "handoff")
    _validate_event_identity(handoff, "handoff")

    automatic_scope = handoff["can_start_episode"].fillna(False)
    events = handoff.loc[
        automatic_scope
        & handoff["SifraSredstva"].notna()
        & handoff["point_family"].isin(ASSET_SEQUENCE_POINT_FAMILIES)
    ].copy()

    if events.empty:
        events["episode_event_role"] = pd.Series(dtype="string")
        events["episode_start_trigger"] = pd.Series(dtype="bool")
        return events

    if events[["DevKey", "SifraSredstva"]].isna().any(axis=None):
        raise ValueError("selected v1 events require non-null DevKey and SifraSredstva")

    is_protection = events["point_family"].eq("protection")
    is_switchgear = events["point_family"].eq("switchgear")
    is_breaker = is_switchgear & events["signal_description"].str.contains(
        r"\bODKLOPNIK\b",
        case=False,
        na=False,
    )
    is_protection_assertion = is_protection & events["reported_state"].isin(PROTECTION_ASSERTED_STATES)
    is_protection_clear = is_protection & events["reported_state"].eq("KONEC")
    is_breaker_trip = is_breaker & events["from_state"].eq("VKLOP") & events["to_state"].eq("IZKLOP")
    is_breaker_restoration = is_breaker & events["from_state"].eq("IZKLOP") & events["to_state"].eq("VKLOP")

    events["episode_event_role"] = "support"
    events.loc[is_protection_assertion, "episode_event_role"] = "protection_assertion"
    events.loc[is_protection_clear, "episode_event_role"] = "protection_clear"
    events.loc[is_breaker_trip, "episode_event_role"] = "breaker_trip"
    events.loc[is_breaker_restoration, "episode_event_role"] = "breaker_restoration"
    events["episode_start_trigger"] = is_protection_assertion | is_breaker_trip

    return events.sort_values(["SifraSredstva", "date_time", "event_index"]).copy()


def build_state_machine_v1_episodes(
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct deterministic local v1 episodes without reference information."""
    _require_columns(events, _PREPARED_REQUIRED_COLUMNS, "events")
    _validate_event_identity(events, "events")

    if events.empty:
        return (
            pd.DataFrame(columns=[*_EPISODE_COLUMNS, "observed_duration"]),
            pd.DataFrame(columns=_MEMBERSHIP_COLUMNS),
        )

    episode_records: list[dict[str, object]] = []
    membership_records: list[dict[str, object]] = []
    next_episode_id = 0
    data_end = events["date_time"].max()

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
                "closure_reason": closure_reason,
                "episode_complete": complete,
                "start_event_role": current["start_event_role"],
                "last_event_role": current["last_event_role"],
                "active_protection_count_at_close": active_protection_count,
                "open_breaker_count_at_close": open_breaker_count,
                "pending_close_at": current["pending_close"],
                "recovery_pending_at_close": current["pending_close"] is not None,
                "event_count": current["event_count"],
                "device_count": len(current["devices"]),
                "first_event_index": current["first_event_index"],
                "protection_trigger_count": current["protection_trigger_count"],
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
                maximum_close = current["cluster_start"] + MAX_EPISODE_DURATION
                pending_close = current["pending_close"]
                if event_time > maximum_close:
                    finish_episode(
                        current,
                        "maximum_duration",
                        maximum_close,
                        False,
                        len(active_protection_points),
                        len(open_breakers),
                    )
                    current = None
                    active_protection_points = set()
                    open_breakers = set()
                elif pending_close is not None and event_time > pending_close:
                    finish_episode(
                        current,
                        "quiet_after_recovery",
                        pending_close,
                        True,
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
                    "event_count": 0,
                    "devices": set(),
                    "first_event_index": event.event_index,
                    "start_event_role": event.episode_event_role,
                    "last_event_role": event.episode_event_role,
                    "protection_trigger_count": 0,
                    "breaker_trip_count": 0,
                    "breaker_restoration_count": 0,
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
            current["devices"].add(event.DevKey)

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
                        "breaker_restoration",
                        event_time,
                        True,
                        len(active_protection_points),
                        len(open_breakers),
                    )
                    current = None

        if current is not None:
            maximum_close = current["cluster_start"] + MAX_EPISODE_DURATION
            pending_close = current["pending_close"]
            if pending_close is not None and pending_close <= data_end:
                finish_episode(
                    current,
                    "quiet_after_recovery",
                    pending_close,
                    True,
                    len(active_protection_points),
                    len(open_breakers),
                )
            elif maximum_close <= data_end:
                finish_episode(
                    current,
                    "maximum_duration",
                    maximum_close,
                    False,
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
    memberships = pd.DataFrame.from_records(membership_records, columns=_MEMBERSHIP_COLUMNS)
    episodes["observed_duration"] = episodes["cluster_end"].sub(episodes["cluster_start"])
    return episodes, memberships


def select_v1_control_context(handoff: pd.DataFrame) -> pd.DataFrame:
    """Select non-starting control rows eligible for same-asset attachment."""
    _require_columns(handoff, _HANDOFF_REQUIRED_COLUMNS, "handoff")
    _validate_event_identity(handoff, "handoff")
    context = handoff.loc[
        ~handoff["can_start_episode"].fillna(False)
        & handoff["SifraSredstva"].notna()
        & handoff["event_family"].isin(EPISODE_CONTROL_CONTEXT_FAMILIES),
        [
            "event_index",
            "date_time",
            "SifraSredstva",
            "event_family",
            "command_target_state",
        ],
    ].copy()
    context["SifraSredstva"] = context["SifraSredstva"].astype("string")
    return context


def attach_v1_control_context(
    handoff: pd.DataFrame,
    episodes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach same-asset control context without changing episode membership."""
    context = select_v1_control_context(handoff)
    required_episode_columns = frozenset({"asset_episode_id", "SifraSredstva", "cluster_start", "closed_at"})
    _require_columns(episodes, required_episode_columns, "episodes")

    enhanced = episodes.copy()
    for column in _CONTEXT_COUNT_COLUMNS:
        enhanced[column] = 0
    enhanced["has_control_context"] = False
    enhanced["has_restoration_command_context"] = False

    if enhanced.empty or context.empty:
        links = context.assign(
            asset_episode_id=pd.Series(index=context.index, dtype="Int64"),
            cluster_start=pd.Series(index=context.index, dtype="datetime64[ns]"),
            closed_at=pd.Series(index=context.index, dtype="datetime64[ns]"),
            is_restoration_command_context=pd.Series(index=context.index, dtype="bool"),
        )
        return enhanced, links.iloc[0:0].copy()

    lookup = enhanced.loc[:, ["asset_episode_id", "SifraSredstva", "cluster_start", "closed_at"]].copy()
    lookup["SifraSredstva"] = lookup["SifraSredstva"].astype("string")

    links = pd.merge_asof(
        context.sort_values(["date_time", "SifraSredstva", "event_index"]),
        lookup.sort_values(["cluster_start", "SifraSredstva", "asset_episode_id"]),
        left_on="date_time",
        right_on="cluster_start",
        by="SifraSredstva",
        direction="backward",
        tolerance=MAX_EPISODE_DURATION,
    )
    links = links.loc[links["asset_episode_id"].notna() & links["date_time"].le(links["closed_at"])].copy()
    links["is_restoration_command_context"] = links["event_family"].eq("control_command") & links[
        "command_target_state"
    ].eq("VKLOP")

    counts = links.groupby("asset_episode_id").agg(
        control_context_count=("event_index", "size"),
        restoration_command_context_count=("is_restoration_command_context", "sum"),
    )
    enhanced = episodes.join(counts, on="asset_episode_id")
    for column in _CONTEXT_COUNT_COLUMNS:
        enhanced[column] = enhanced[column].fillna(0).astype("int64")
    enhanced["has_control_context"] = enhanced["control_context_count"].gt(0)
    enhanced["has_restoration_command_context"] = enhanced["restoration_command_context_count"].gt(0)

    return enhanced, links


def run_state_machine_v1(handoff: pd.DataFrame) -> StateMachineV1Result:
    """Run selection, role classification, v1 construction and context linking."""
    events = prepare_state_machine_v1_events(handoff)
    episodes, membership = build_state_machine_v1_episodes(events)
    episodes_with_context, context_links = attach_v1_control_context(handoff, episodes)
    return StateMachineV1Result(
        events=events,
        episodes=episodes_with_context,
        membership=membership,
        context_links=context_links,
    )
