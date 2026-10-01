"""Pure dataframe selection helpers for the incident timeline viewer."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import pandas as pd


class TimelineWindowTooLarge(ValueError):
    """Raised when a requested browser view would contain too many events."""


@dataclass(frozen=True)
class TimelineWindow:
    """Filtered tables needed to render one timeline window."""

    events: pd.DataFrame
    episodes: pd.DataFrame
    context: pd.DataFrame
    unassigned_events: pd.DataFrame


def build_asset_display_labels(
    events: pd.DataFrame,
    *,
    asset_column: str = "SifraSredstva",
    descriptor_columns: tuple[str, ...] = ("lokacija", "point_description", "aoj"),
) -> dict[str, str]:
    """Return stable human-readable labels without changing join identifiers."""
    required = {asset_column, *descriptor_columns}
    missing = sorted(required.difference(events.columns))
    if missing:
        raise ValueError(f"asset label frame is missing columns: {missing}")

    frame = events[[asset_column, *descriptor_columns]].copy()
    frame = frame.loc[frame[asset_column].notna()]
    frame[asset_column] = frame[asset_column].astype("string").str.strip()
    frame = frame.loc[frame[asset_column].ne("")]

    descriptors = frame.loc[:, descriptor_columns].astype("string")
    descriptors = descriptors.apply(lambda column: column.str.replace(r"\s+", " ", regex=True).str.strip())
    descriptors = descriptors.mask(descriptors.eq(""))
    # ``DataFrame.bfill(axis=1)`` is disproportionately expensive for pandas
    # StringDtype columns. Coalescing the small, ordered set of descriptor
    # columns preserves the same preference order without constructing a
    # row-wise backfilled frame.
    asset_descriptor = descriptors[descriptor_columns[0]]
    for column in descriptor_columns[1:]:
        asset_descriptor = asset_descriptor.fillna(descriptors[column])
    frame["asset_descriptor"] = asset_descriptor
    described = frame.dropna(subset=["asset_descriptor"])
    if described.empty:
        return {str(asset): str(asset) for asset in frame[asset_column].drop_duplicates()}

    counts = (
        described.groupby([asset_column, "asset_descriptor"], observed=True)
        .size()
        .rename("observation_count")
        .reset_index()
        .sort_values(
            [asset_column, "observation_count", "asset_descriptor"],
            ascending=[True, False, True],
        )
    )
    selected = counts.drop_duplicates(asset_column).set_index(asset_column)["asset_descriptor"]
    return {
        str(asset): f"{asset} · {selected.get(asset, '')}".removesuffix(" · ")
        for asset in frame[asset_column].drop_duplicates()
    }


def select_timeline_window(
    events: pd.DataFrame,
    episodes: pd.DataFrame,
    membership: pd.DataFrame,
    context_links: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    assets: Collection[str],
    include_unassigned: bool = False,
    max_events: int = 20_000,
) -> TimelineWindow:
    """Select a bounded display window without changing episode construction."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if start > end:
        raise ValueError("timeline start must not be after its end")
    if max_events < 1:
        raise ValueError("max_events must be positive")

    asset_values = {str(asset) for asset in assets}
    if not asset_values:
        empty_events = events.iloc[0:0].copy()
        return TimelineWindow(
            events=empty_events,
            episodes=episodes.iloc[0:0].copy(),
            context=context_links.iloc[0:0].copy(),
            unassigned_events=empty_events.copy(),
        )

    selected_episodes = episodes.loc[
        episodes["SifraSredstva"].astype("string").isin(asset_values)
        & episodes["cluster_start"].le(end)
        & episodes["closed_at"].ge(start)
    ].copy()
    episode_ids = set(selected_episodes["asset_episode_id"])

    selected_membership = membership.loc[membership["asset_episode_id"].isin(episode_ids)].copy()
    selected_events = selected_membership.merge(
        events,
        on=["event_index", "episode_event_role"],
        how="left",
        validate="one_to_one",
    )
    selected_events = selected_events.loc[
        selected_events["date_time"].between(start, end, inclusive="both")
    ].sort_values(["date_time", "event_index"])

    selected_context = context_links.loc[
        context_links["asset_episode_id"].isin(episode_ids)
        & context_links["date_time"].between(start, end, inclusive="both")
    ].sort_values(["date_time", "event_index"])

    unassigned = events.iloc[0:0].copy()
    if include_unassigned:
        assigned_ids = set(membership["event_index"])
        unassigned = events.loc[
            ~events["event_index"].isin(assigned_ids)
            & events["SifraSredstva"].astype("string").isin(asset_values)
            & events["date_time"].between(start, end, inclusive="both")
        ].sort_values(["date_time", "event_index"])

    display_event_count = len(selected_events) + len(selected_context) + len(unassigned)
    if display_event_count > max_events:
        raise TimelineWindowTooLarge(f"window contains {display_event_count:,} events; narrow it below {max_events:,}")

    return TimelineWindow(
        events=selected_events,
        episodes=selected_episodes.sort_values(["cluster_start", "asset_episode_id"]),
        context=selected_context,
        unassigned_events=unassigned,
    )


def select_reference_intervals(
    references: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    assets: Collection[str],
    nonzero_end_extension: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Normalize reference intervals linked to either delivered asset field."""
    required = {
        "DCVDogodekOutageID",
        "DatumUraCETOd",
        "DatumUraCETDo",
        "DCVNapravaSID_SifraSredstva",
        "DCVNapravaSIDIzklopna_SifraSredstva",
        "Opis",
        "Povod",
    }
    missing = sorted(required.difference(references.columns))
    if missing:
        raise ValueError(f"reference table is missing required columns: {missing}")

    asset_values = {str(asset) for asset in assets}
    if not asset_values:
        return pd.DataFrame(
            columns=[
                "reference_id",
                "reference_start",
                "reference_end",
                "reference_end_recorded",
                "reference_time_kind",
                "asset_id",
                "asset_link_role",
                "source_row_count",
                "reported_boundary_count",
                "description",
                "cause",
            ]
        )

    extension = pd.Timedelta(0) if nonzero_end_extension is None else pd.Timedelta(nonzero_end_extension)
    if extension < pd.Timedelta(0):
        raise ValueError("reference end extension cannot be negative")

    source = references.copy()
    source["_source_row_id"] = range(len(source))
    source["_reported_boundary"] = list(
        zip(source["DatumUraCETOd"], source["DatumUraCETDo"], strict=True)
    )
    normalized_frames: list[pd.DataFrame] = []
    for column, role in (
        ("DCVNapravaSID_SifraSredstva", "affected_asset"),
        ("DCVNapravaSIDIzklopna_SifraSredstva", "switching_anchor"),
    ):
        frame = source.loc[source[column].astype("string").isin(asset_values)].copy()
        frame["asset_id"] = frame[column].astype("string")
        frame["asset_link_role"] = role
        normalized_frames.append(frame)

    normalized = pd.concat(normalized_frames, ignore_index=True)
    if normalized.empty:
        return pd.DataFrame(
            columns=[
                "reference_id",
                "reference_start",
                "reference_end",
                "reference_end_recorded",
                "reference_time_kind",
                "asset_id",
                "asset_link_role",
                "source_row_count",
                "reported_boundary_count",
                "description",
                "cause",
            ]
        )

    # Delivered IZPADI can repeat one outage/asset with differing boundaries.
    # Establish a stable envelope before filtering the requested display window.
    normalized = (
        normalized.groupby(["DCVDogodekOutageID", "asset_id"], as_index=False, dropna=False)
        .agg(
            reference_start=("DatumUraCETOd", "min"),
            reference_end_recorded=("DatumUraCETDo", "max"),
            asset_link_role=(
                "asset_link_role",
                lambda values: "+".join(sorted(set(values))),
            ),
            source_row_count=("_source_row_id", "nunique"),
            reported_boundary_count=("_reported_boundary", "nunique"),
            description=("Opis", "first"),
            cause=("Povod", "first"),
        )
        .rename(columns={"DCVDogodekOutageID": "reference_id"})
    )
    is_point = normalized["reference_start"].eq(normalized["reference_end_recorded"])
    normalized["reference_end"] = normalized["reference_end_recorded"].where(
        is_point,
        normalized["reference_end_recorded"].add(extension),
    )
    normalized["reference_time_kind"] = pd.array(
        is_point.map({True: "point", False: "minute_granularity_interval"}),
        dtype="string",
    )
    starts_before_window_end = normalized["reference_start"].le(pd.Timestamp(end))
    ends_after_window_start = normalized["reference_end"].gt(pd.Timestamp(start))
    point_in_window = is_point & normalized["reference_end"].ge(pd.Timestamp(start))
    overlap = normalized.loc[
        starts_before_window_end & (ends_after_window_start | point_in_window)
    ]
    return overlap.sort_values(["reference_start", "reference_id"])
