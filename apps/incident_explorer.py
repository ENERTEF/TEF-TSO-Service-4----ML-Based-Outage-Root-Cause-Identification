"""Read-only interactive timeline for Service 4 state-machine outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Final

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from service4.artifacts import file_sha256, find_project_root, load_v1_artifacts
from service4.artifacts_v1_1 import load_v1_1_artifacts
from service4.artifacts_v2 import load_v2_artifacts
from service4.timeline import (
    TimelineWindow,
    TimelineWindowTooLarge,
    build_asset_display_labels,
    select_reference_intervals,
    select_timeline_window,
)

PROJECT_ROOT: Final[Path] = find_project_root()
V1_DIR: Final[Path] = PROJECT_ROOT / "data/processed/service4-state-machine-v1"
V1_1_DIR: Final[Path] = PROJECT_ROOT / "data/processed/service4-state-machine-v1-1"
V2_DIR: Final[Path] = PROJECT_ROOT / "data/processed/service4-state-machine-v2"
UNPLANNED_PATH: Final[Path] = PROJECT_ROOT / "data/processed/izpadi.parquet"
PLANNED_PATH: Final[Path] = PROJECT_ROOT / "data/processed/izklopi.parquet"
EVIDENCE_DIR: Final[Path] = PROJECT_ROOT / "data/processed/service4-bounded-lookahead-v1"
MAX_BROWSER_EVENTS: Final[int] = 20_000
DEFAULT_ASSET_ID: Final[str] = "6103964"
DEFAULT_EPISODE_BY_METHOD: Final[dict[str, int]] = {
    "v1.0": 18_101,
    "v1.1": 18_604,
    "v2.0": 13_539,
}
MODEL_ORDER: Final[tuple[str, ...]] = ("v1.0", "v1.1", "v2.0")
MODEL_COLORS: Final[dict[str, str]] = {
    "v1.0": "#1f77b4",
    "v1.1": "#9467bd",
    "v2.0": "#2ca02c",
}
MODEL_DIRECTORIES: Final[dict[str, Path]] = {
    "v1.0": V1_DIR,
    "v1.1": V1_1_DIR,
    "v2.0": V2_DIR,
}
MODEL_BUILD_COMMANDS: Final[dict[str, str]] = {
    "v1.0": "service4-build-v1",
    "v1.1": "service4-build-v1-1",
    "v2.0": "service4-build-v2",
}

ROLE_STYLE: Final[dict[str, tuple[str, str]]] = {
    "breaker_trip": ("#d62728", "triangle-down"),
    "breaker_restoration": ("#2ca02c", "triangle-up"),
    "protection_assertion": ("#ff7f0e", "diamond"),
    "protection_clear": ("#1f77b4", "diamond-open"),
    "support": ("#7f7f7f", "circle-open"),
}


@dataclass(frozen=True)
class InterestingSample:
    """An audited jump target for versioned episode artifacts."""

    label: str
    category: str
    asset_id: str
    anchor_time: pd.Timestamp | None
    source_model: str
    note: str


@dataclass(frozen=True)
class CuratedSampleSpec:
    """An audited sample plus the models required to interpret it."""

    key: str
    sample: InterestingSample
    required_models: frozenset[str]


CURATED_SAMPLE_SPECS: Final[tuple[CuratedSampleSpec, ...]] = (
    CuratedSampleSpec(
        key="incident_control_restoration",
        sample=InterestingSample(
            label="IZPADI-positive · control-confirmed restoration",
            category="Known actual incident",
            asset_id="6103964",
            anchor_time=pd.Timestamp("2025-05-05 15:01:00"),
            source_model="v1.1",
            note=(
                "IZPADI marks a real incident. V1.1 closes on exact-point control feedback at "
                "15:03:25.900; v1.0 and v2.0 remain open to the 60-minute limit."
            ),
        ),
        required_models=frozenset({"v1.1"}),
    ),
    CuratedSampleSpec(
        key="incident_single_breaker_trip",
        sample=InterestingSample(
            label="IZPADI-positive · one automatic breaker trip",
            category="Known actual incident",
            asset_id="6187976",
            anchor_time=pd.Timestamp("2022-01-05 21:15:27.346"),
            source_model="v1.1",
            note=(
                "A real IZPADI incident with only one automatic member. V1.1 observes restoration "
                "through later exact-point control feedback, so this singleton must not be dropped."
            ),
        ),
        required_models=frozenset({"v1.1"}),
    ),
    CuratedSampleSpec(
        key="fragmentation_after_restoration",
        sample=InterestingSample(
            label="Model failure · v1 fragments a six-event incident",
            category="Lifecycle-boundary diagnostic",
            asset_id="6220784",
            anchor_time=pd.Timestamp("2022-01-01 19:11:45.201"),
            source_model="v2.0",
            note=(
                "IZPADI marks the minute as a real incident. V1.0/v1.1 close immediately on breaker "
                "restoration and split a protection pulse ten seconds later; v2.0 keeps all six "
                "automatic events together through its recovery grace."
            ),
        ),
        required_models=frozenset({"v1.0", "v1.1", "v2.0"}),
    ),
    CuratedSampleSpec(
        key="timestamp_replay_batch",
        sample=InterestingSample(
            label="Data quality · adjusted timestamps collapsed",
            category="Timestamp-replay diagnostic",
            asset_id="6333294",
            anchor_time=pd.Timestamp("2024-12-16 08:31:59"),
            source_model="v2.0",
            note=(
                "All 115 v2.0 members share one receipt timestamp, while their messages embed 92 "
                "adjusted source times spanning multiple days. Model boundaries are not physically "
                "interpretable until receipt and effective event time are separated."
            ),
        ),
        required_models=frozenset({"v1.0", "v2.0"}),
    ),
    CuratedSampleSpec(
        key="planned_work_test_chatter",
        sample=InterestingSample(
            label="Maintenance · announced work and test chatter",
            category="Planned-work contamination",
            asset_id="6317767",
            anchor_time=pd.Timestamp("2026-05-13 08:09:34.476"),
            source_model="v2.0",
            note=(
                "This 1,007-event episode overlaps an IZKLOPI window, followed by test-mode and DAP "
                "chatter-suppression context. It is a contamination diagnostic, not a clean "
                "provisional-restoration benchmark."
            ),
        ),
        required_models=frozenset({"v2.0"}),
    ),
    CuratedSampleSpec(
        key="test_mode_retry_false_positive",
        sample=InterestingSample(
            label="Diagnostic error · test chatter resembles retry",
            category="Test-mode contamination",
            asset_id="6330297",
            anchor_time=pd.Timestamp("2025-05-27 09:00:59.710"),
            source_model="v2.0",
            note=(
                "Test mode was enabled five minutes before this 1,953-event burst. Paired state "
                "reports about 0.07–0.12 seconds apart plus 20–45-second chatter satisfy the literal "
                "retry timing rule but do not demonstrate a physical breaker retry."
            ),
        ),
        required_models=frozenset({"v2.0"}),
    ),
)


def build_interesting_samples(
    tables_by_model: dict[str, dict[str, pd.DataFrame]],
) -> dict[str, InterestingSample]:
    """Return audited, unique samples supported by the selected frozen artifacts."""
    samples: dict[str, InterestingSample] = {}
    selected_models = frozenset(tables_by_model)
    seen_targets: set[tuple[str, pd.Timestamp]] = set()

    for spec in CURATED_SAMPLE_SPECS:
        if not spec.required_models.issubset(selected_models):
            continue
        sample = spec.sample
        if sample.anchor_time is None:
            raise ValueError(f"audited interesting sample {spec.key!r} has no anchor")
        episodes = tables_by_model[sample.source_model]["episodes"]
        target_exists = (
            episodes["SifraSredstva"].astype("string").eq(sample.asset_id)
            & episodes["cluster_start"].le(sample.anchor_time)
            & episodes["closed_at"].ge(sample.anchor_time)
        ).any()
        if not target_exists:
            raise ValueError(f"audited interesting sample {spec.key!r} is absent from {sample.source_model} artifacts")

        target = (sample.asset_id, sample.anchor_time)
        if target in seen_targets:
            raise ValueError(f"duplicate audited interesting-sample target: {target}")
        seen_targets.add(target)
        samples[spec.key] = sample

    samples["browse"] = InterestingSample(
        label="Browse manually",
        category="Manual navigation",
        asset_id=DEFAULT_ASSET_ID,
        anchor_time=None,
        source_model="none",
        note="Use the asset and focus-episode controls directly.",
    )
    return samples


@st.cache_data(show_spinner="Loading verified v1 artifacts…")
def load_cached_v1(directory: str, manifest_sha256: str) -> dict[str, pd.DataFrame]:
    """Load artifacts with the manifest digest included in the cache key."""
    del manifest_sha256
    return load_v1_artifacts(Path(directory))


@st.cache_data(show_spinner="Loading verified v1.1 artifacts…")
def load_cached_v1_1(directory: str, manifest_sha256: str) -> dict[str, pd.DataFrame]:
    """Load v1.1 artifacts with the manifest digest included in the cache key."""
    del manifest_sha256
    return load_v1_1_artifacts(Path(directory))


@st.cache_data(show_spinner="Loading verified v2.0 artifacts…")
def load_cached_v2(directory: str, manifest_sha256: str) -> dict[str, pd.DataFrame]:
    """Load v2.0 artifacts with the manifest digest included in the cache key."""
    del manifest_sha256
    return load_v2_artifacts(Path(directory))


def load_model_artifacts(
    model: str,
    directory: Path,
    manifest_sha256: str,
) -> dict[str, pd.DataFrame]:
    """Dispatch to the checksum-validating loader for one model version."""
    if model == "v1.0":
        return load_cached_v1(str(directory), manifest_sha256)
    if model == "v1.1":
        return load_cached_v1_1(str(directory), manifest_sha256)
    if model == "v2.0":
        return load_cached_v2(str(directory), manifest_sha256)
    raise ValueError(f"unsupported episode model: {model}")


@st.cache_data(show_spinner=False)
def load_cached_parquet(path: str, sha256: str) -> pd.DataFrame:
    """Load a Parquet table with its digest included in the cache key."""
    del sha256
    return pd.read_parquet(path)


@st.cache_data(show_spinner="Preparing asset labels…")
def build_cached_asset_display_labels(
    _events: pd.DataFrame,
    manifest_sha256: str,
) -> dict[str, str]:
    """Build labels once per immutable v1 manifest, not once per UI rerun."""
    del manifest_sha256
    return build_asset_display_labels(_events)


def load_optional_evidence_status(
    expected_input_sha256: str,
) -> pd.DataFrame | None:
    """Load compatible bounded-look-ahead status when its checksum is valid."""
    manifest_path = EVIDENCE_DIR / "manifest.json"
    status_path = EVIDENCE_DIR / "episode-status.parquet"
    if not manifest_path.exists() or not status_path.exists():
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("clustering_input_sha256") != expected_input_sha256:
        st.warning("The optional evidence status was built from a different handoff.")
        return None

    metadata = manifest.get("artifacts", {}).get("episode-status.parquet")
    if metadata is None or file_sha256(status_path) != metadata.get("sha256"):
        st.warning("The optional evidence-status artifact failed checksum validation.")
        return None
    return load_cached_parquet(str(status_path), metadata["sha256"])


def add_interval_traces(
    figure: go.Figure,
    frame: pd.DataFrame,
    *,
    start_column: str,
    end_column: str,
    id_column: str,
    lane: str,
    name: str,
    color: str,
    hover_columns: tuple[str, ...] = (),
    annotation_prefix: str | None = None,
) -> None:
    """Add individually hoverable interval segments on one categorical lane."""
    for number, row in enumerate(frame.itertuples(index=False)):
        row_values = row._asdict()
        identifier = row_values[id_column]
        details = "<br>".join(
            f"{column}: {row_values.get(column, '')}" for column in hover_columns if pd.notna(row_values.get(column))
        )
        suffix = f"<br>{details}" if details else ""
        start_value = row_values[start_column]
        end_value = row_values[end_column]
        is_point = start_value == end_value
        style = (
            {"marker": {"color": color, "symbol": "diamond", "size": 12}}
            if is_point
            else {"line": {"color": color, "width": 10}}
        )
        figure.add_trace(
            go.Scatter(
                x=[row_values[start_column], row_values[end_column]],
                y=[lane, lane],
                mode="markers" if is_point else "lines",
                name=name,
                legendgroup=name,
                showlegend=number == 0,
                customdata=[[f"interval:{identifier}"], [f"interval:{identifier}"]],
                hovertemplate=(
                    f"{name}: {identifier}<br>start: {start_value}<br>end: {end_value}{suffix}<extra></extra>"
                ),
                **style,
            )
        )
        if annotation_prefix is not None:
            midpoint = row_values[start_column] + (row_values[end_column] - row_values[start_column]) / 2
            figure.add_annotation(
                x=midpoint,
                y=lane,
                text=f"{annotation_prefix} {identifier}",
                showarrow=False,
                yshift=13,
                font={"color": color, "size": 10},
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=color,
                borderpad=2,
            )


def add_event_traces(
    figure: go.Figure,
    events: pd.DataFrame,
    *,
    role_column: str,
    opacity: float = 1.0,
    name_prefix: str = "",
) -> None:
    """Add one marker trace per event role."""
    if events.empty:
        return

    for role, frame in events.groupby(role_column, sort=True):
        color, symbol = ROLE_STYLE.get(str(role), ("#9467bd", "circle"))
        hover = frame.loc[
            :,
            [
                "event_index",
                "asset_episode_id",
                "signal_description",
                "reported_state",
                "sporocilo",
            ],
        ].copy()
        hover = hover.fillna("").astype("string")
        figure.add_trace(
            go.Scattergl(
                x=frame["date_time"],
                y=frame["DevKey"].fillna("unknown point"),
                mode="markers",
                name=f"{name_prefix}{role}",
                marker={"color": color, "symbol": symbol, "size": 10, "opacity": opacity},
                customdata=hover.to_numpy(),
                hovertemplate=(
                    "event: %{customdata[0]}<br>"
                    "episode: %{customdata[1]}<br>"
                    "time: %{x|%Y-%m-%d %H:%M:%S.%L}<br>"
                    "point: %{y}<br>"
                    "signal: %{customdata[2]}<br>"
                    "state: %{customdata[3]}<br>"
                    "message: %{customdata[4]}<extra></extra>"
                ),
            )
        )


def build_figure(
    window: TimelineWindow,
    *,
    unplanned: pd.DataFrame,
    planned: pd.DataFrame,
    focus_episode_id: int,
    episode_layers: tuple[tuple[str, pd.DataFrame, str], ...],
) -> go.Figure:
    """Build the interactive event and interval timeline."""
    figure = go.Figure()

    add_interval_traces(
        figure,
        unplanned,
        start_column="reference_start",
        end_column="reference_end",
        id_column="reference_id",
        lane="IZPADI",
        name="IZPAD — actual outage",
        color="#d62728",
        hover_columns=("reference_end_recorded", "reference_time_kind", "description", "cause"),
        annotation_prefix="IZPAD",
    )
    add_interval_traces(
        figure,
        planned,
        start_column="reference_start",
        end_column="reference_end",
        id_column="reference_id",
        lane="IZKLOPI",
        name="IZKLOP — announced work",
        color="#17becf",
        hover_columns=("description", "cause"),
        annotation_prefix="IZKLOP",
    )
    for label, episodes, color in episode_layers:
        add_interval_traces(
            figure,
            episodes,
            start_column="cluster_start",
            end_column="closed_at",
            id_column="asset_episode_id",
            lane=f"{label} episodes",
            name=f"{label} episode",
            color=color,
            hover_columns=("closure_reason", "event_count"),
        )
    add_event_traces(figure, window.events, role_column="episode_event_role")

    if not window.context.empty:
        context = window.context.copy()
        context["asset_episode_id"] = context["asset_episode_id"].astype("Int64")
        context["episode_event_role"] = "control_context"
        observed_state = context.get(
            "reported_state",
            pd.Series(pd.NA, index=context.index, dtype="string"),
        )
        context["reported_state"] = context["command_target_state"].fillna(observed_state)
        add_event_traces(
            figure,
            context,
            role_column="episode_event_role",
            opacity=0.75,
        )

    if not window.unassigned_events.empty:
        unassigned = window.unassigned_events.copy()
        unassigned["asset_episode_id"] = pd.NA
        add_event_traces(
            figure,
            unassigned,
            role_column="episode_event_role",
            opacity=0.4,
            name_prefix="unassigned ",
        )

    focus = window.episodes.loc[window.episodes["asset_episode_id"].eq(focus_episode_id)]
    if not focus.empty:
        figure.add_vline(
            x=focus.iloc[0]["cluster_start"],
            line_dash="dot",
            line_color="#222222",
            opacity=0.8,
        )

    device_lanes = sorted(
        set(window.events["DevKey"].dropna().astype(str))
        | set(window.context.get("DevKey", pd.Series(dtype="string")).dropna().astype(str))
        | set(window.unassigned_events.get("DevKey", pd.Series(dtype="string")).dropna().astype(str))
    )
    category_order = [
        "IZPADI",
        "IZKLOPI",
        *(f"{label} episodes" for label, _, _ in episode_layers),
        *device_lanes,
    ]
    figure.update_layout(
        height=max(520, min(1_200, 300 + 28 * len(category_order))),
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        hovermode="closest",
        dragmode="select",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        xaxis={
            "title": "Time",
            "rangeslider": {"visible": True, "thickness": 0.08},
            "type": "date",
        },
        yaxis={
            "title": "Reference / episode / SCADA point",
            "categoryorder": "array",
            "categoryarray": list(reversed(category_order)),
        },
    )
    return figure


def selected_event_ids(selection: object) -> list[int]:
    """Extract event identifiers from a Streamlit Plotly selection."""
    if not selection:
        return []
    points = getattr(getattr(selection, "selection", None), "points", [])
    identifiers: list[int] = []
    for point in points:
        custom = point.get("customdata", [])
        if not custom:
            continue
        try:
            identifiers.append(int(custom[0]))
        except (TypeError, ValueError):
            continue
    return sorted(set(identifiers))


def main() -> None:
    """Render the read-only incident explorer."""
    st.set_page_config(page_title="Service 4 incident explorer", layout="wide")
    st.title("Service 4 incident explorer")
    st.sidebar.markdown("#### Episode models")
    selected_models = [
        model for model in MODEL_ORDER if st.sidebar.checkbox(model, value=True, key=f"show-model-{model}")
    ]
    if not selected_models:
        st.warning("Select at least one episode model.")
        st.stop()

    primary_method = next(model for model in ("v1.1", "v2.0", "v1.0") if model in selected_models)
    st.caption(
        f"Read-only view of {', '.join(selected_models)} on separate episode lanes. "
        "Timeline controls filter precomputed results and never reconstruct episodes."
    )
    st.caption(
        "IZPADI are actual unplanned-outage references. IZKLOPI are announced "
        "work periods and do not prove that switching was executed."
    )

    tables_by_model: dict[str, dict[str, pd.DataFrame]] = {}
    manifests_by_model: dict[str, dict[str, object]] = {}
    manifest_hashes: dict[str, str] = {}
    for model in selected_models:
        directory = MODEL_DIRECTORIES[model]
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            st.error(f"{model.upper()} artifacts are missing. Run: ../.venv/bin/{MODEL_BUILD_COMMANDS[model]}")
            st.stop()
        manifest_hash = file_sha256(manifest_path)
        manifest_hashes[model] = manifest_hash
        manifests_by_model[model] = json.loads(manifest_path.read_text(encoding="utf-8"))
        tables_by_model[model] = load_model_artifacts(model, directory, manifest_hash)

    tables = tables_by_model[primary_method]
    artifact_manifest = manifests_by_model[primary_method]
    manifest_sha256 = manifest_hashes[primary_method]
    events = tables["events"]
    episodes = tables["episodes"]
    membership = tables["membership"]
    context_links = tables["context_links"]

    evidence_status = (
        load_optional_evidence_status(artifact_manifest["input"]["sha256"]) if primary_method == "v1.0" else None
    )
    if evidence_status is not None:
        status_columns = [
            "asset_episode_id",
            "output_stream",
            "automatic_evidence_status",
            "resolution_status",
        ]
        episodes = episodes.merge(
            evidence_status.loc[:, status_columns],
            on="asset_episode_id",
            how="left",
            validate="one_to_one",
        )

    assets = sorted(
        set().union(
            *(
                set(model_tables["episodes"]["SifraSredstva"].dropna().astype(str).unique())
                for model_tables in tables_by_model.values()
            )
        )
    )
    if not assets:
        st.warning("No episodes are available for the selected models.")
        st.stop()

    asset_display_labels = build_cached_asset_display_labels(events, manifest_sha256)
    interesting_samples = build_interesting_samples(tables_by_model)
    selected_sample_key = st.sidebar.selectbox(
        "Interesting sample",
        list(interesting_samples),
        format_func=lambda key: interesting_samples[key].label,
    )
    selected_sample = interesting_samples[selected_sample_key]
    if selected_sample.anchor_time is not None:
        st.sidebar.caption(
            f"{selected_sample.category} · {selected_sample.source_model} · "
            f"{asset_display_labels.get(selected_sample.asset_id, selected_sample.asset_id)} · "
            f"{selected_sample.anchor_time}\n\n{selected_sample.note}"
        )

    default_asset_id = selected_sample.asset_id
    default_asset_index = assets.index(default_asset_id) if default_asset_id in assets else 0
    selected_asset = st.sidebar.selectbox(
        "Asset",
        assets,
        index=default_asset_index,
        format_func=lambda asset: asset_display_labels.get(str(asset), str(asset)),
        key=f"asset-{selected_sample_key}-{'-'.join(selected_models)}",
    )
    asset_episodes = episodes.loc[episodes["SifraSredstva"].astype("string").eq(selected_asset)].sort_values(
        "cluster_start", ascending=False
    )
    asset_episodes_by_model = {
        model: model_tables["episodes"].loc[
            model_tables["episodes"]["SifraSredstva"].astype("string").eq(selected_asset)
        ]
        for model, model_tables in tables_by_model.items()
    }

    episode_start_by_id = asset_episodes.set_index("asset_episode_id")["cluster_start"].to_dict()
    episode_ids = asset_episodes["asset_episode_id"].tolist()
    default_episode_id = DEFAULT_EPISODE_BY_METHOD[primary_method]
    if selected_sample.anchor_time is not None:
        containing = asset_episodes.loc[
            asset_episodes["cluster_start"].le(selected_sample.anchor_time)
            & asset_episodes["closed_at"].ge(selected_sample.anchor_time)
        ].copy()
        if containing.empty:
            containing = asset_episodes.assign(
                _sample_distance=asset_episodes["cluster_start"].sub(selected_sample.anchor_time).abs()
            ).sort_values(["_sample_distance", "cluster_start"])
        else:
            containing = containing.assign(
                _sample_distance=selected_sample.anchor_time - containing["cluster_start"]
            ).sort_values(["_sample_distance", "cluster_start"])
        if not containing.empty:
            default_episode_id = int(containing.iloc[0]["asset_episode_id"])
    default_episode_index = episode_ids.index(default_episode_id) if default_episode_id in episode_ids else 0
    selected_episode_id = st.sidebar.selectbox(
        "Focus episode" if len(selected_models) == 1 else f"Focus episode ({primary_method})",
        episode_ids,
        index=default_episode_index,
        format_func=lambda episode_id: f"{int(episode_id)} · {episode_start_by_id[episode_id]}",
        key=f"focus-{selected_sample_key}-{primary_method}",
    )
    focus = asset_episodes.loc[asset_episodes["asset_episode_id"].eq(selected_episode_id)].iloc[0]

    timeline_episode_bounds = pd.concat(
        list(asset_episodes_by_model.values()),
        ignore_index=True,
    )
    asset_min = timeline_episode_bounds["cluster_start"].min().to_pydatetime()
    asset_max = timeline_episode_bounds["closed_at"].max().to_pydatetime()
    slider_min = asset_min
    slider_max = asset_max
    if slider_min == slider_max:
        slider_min -= timedelta(minutes=1)
        slider_max += timedelta(minutes=1)

    default_start = max(
        slider_min,
        (focus["cluster_start"] - pd.Timedelta(minutes=10)).to_pydatetime(),
    )
    default_end = min(
        slider_max,
        (focus["closed_at"] + pd.Timedelta(minutes=10)).to_pydatetime(),
    )
    if default_start == default_end:
        default_end = min(slider_max, default_start + timedelta(minutes=1))

    start, end = st.slider(
        "Displayed time window",
        min_value=slider_min,
        max_value=slider_max,
        value=(default_start, default_end),
        step=timedelta(seconds=1),
        format="YYYY-MM-DD HH:mm:ss",
        key=f"window-{selected_asset}-{selected_episode_id}",
    )

    show_context = st.sidebar.checkbox("Control context", value=True)
    show_unassigned = st.sidebar.checkbox("Unassigned state observations", value=False)
    show_unplanned = st.sidebar.checkbox("IZPADI — actual outages", value=True)
    show_planned = st.sidebar.checkbox("IZKLOPI — announced work", value=True)

    display_episodes = episodes
    if "output_stream" in episodes.columns:
        status_options = sorted(episodes["output_stream"].dropna().unique())
        selected_statuses = st.sidebar.multiselect(
            "Evidence stream",
            status_options,
            default=status_options,
        )
        display_episodes = episodes.loc[episodes["output_stream"].isin(selected_statuses)]

    try:
        window = select_timeline_window(
            events,
            display_episodes,
            membership,
            context_links if show_context else context_links.iloc[0:0],
            start=pd.Timestamp(start),
            end=pd.Timestamp(end),
            assets=[selected_asset],
            include_unassigned=show_unassigned,
            max_events=MAX_BROWSER_EVENTS,
        )
    except TimelineWindowTooLarge as error:
        st.error(str(error))
        st.stop()

    windows_by_model: dict[str, TimelineWindow] = {primary_method: window}
    for model in selected_models:
        if model == primary_method:
            continue
        model_tables = tables_by_model[model]
        try:
            windows_by_model[model] = select_timeline_window(
                model_tables["events"],
                model_tables["episodes"],
                model_tables["membership"],
                model_tables["context_links"].iloc[0:0],
                start=pd.Timestamp(start),
                end=pd.Timestamp(end),
                assets=[selected_asset],
                include_unassigned=False,
                max_events=MAX_BROWSER_EVENTS,
            )
        except TimelineWindowTooLarge as error:
            st.error(f"{model.upper()} comparison: {error}")
            st.stop()

    unplanned = pd.DataFrame()
    if show_unplanned:
        unplanned_table = load_cached_parquet(
            str(UNPLANNED_PATH),
            file_sha256(UNPLANNED_PATH),
        )
        unplanned = select_reference_intervals(
            unplanned_table,
            start=pd.Timestamp(start),
            end=pd.Timestamp(end),
            assets=[selected_asset],
            nonzero_end_extension=(pd.Timedelta(minutes=1) if "v1.1" in selected_models else None),
        )

    planned = pd.DataFrame()
    if show_planned:
        planned_table = load_cached_parquet(
            str(PLANNED_PATH),
            file_sha256(PLANNED_PATH),
        )
        planned = select_reference_intervals(
            planned_table,
            start=pd.Timestamp(start),
            end=pd.Timestamp(end),
            assets=[selected_asset],
        )

    episode_layers = tuple(
        (
            model.upper(),
            windows_by_model[model].episodes,
            MODEL_COLORS[model],
        )
        for model in selected_models
    )
    figure = build_figure(
        window,
        unplanned=unplanned,
        planned=planned,
        focus_episode_id=int(selected_episode_id),
        episode_layers=episode_layers,
    )
    selection = st.plotly_chart(
        figure,
        width="stretch",
        key="incident-timeline",
        on_select="rerun",
        selection_mode=("points", "box"),
    )

    event_ids = selected_event_ids(selection)
    if event_ids:
        st.subheader("Selected events")
        selected_rows = pd.concat(
            [
                window.events.loc[window.events["event_index"].isin(event_ids)],
                window.context.loc[window.context["event_index"].isin(event_ids)],
                window.unassigned_events.loc[window.unassigned_events["event_index"].isin(event_ids)],
            ],
            ignore_index=True,
        )
        st.dataframe(selected_rows, width="stretch", hide_index=True)

    episode_tab, event_tab, context_tab, reference_tab = st.tabs(
        ["Episodes", "Events", "Control context", "References"]
    )
    with episode_tab:
        episode_frames = [windows_by_model[model].episodes.assign(model=model) for model in selected_models]
        st.dataframe(
            pd.concat(episode_frames, ignore_index=True).sort_values(["cluster_start", "model"]),
            width="stretch",
            hide_index=True,
        )
    with event_tab:
        st.dataframe(window.events, width="stretch", hide_index=True)
    with context_tab:
        st.dataframe(window.context, width="stretch", hide_index=True)
    with reference_tab:
        st.markdown("### IZPADI — actual unplanned outages")
        st.dataframe(unplanned, width="stretch", hide_index=True)
        st.markdown("### IZKLOPI — announced/planned work")
        st.dataframe(planned, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
