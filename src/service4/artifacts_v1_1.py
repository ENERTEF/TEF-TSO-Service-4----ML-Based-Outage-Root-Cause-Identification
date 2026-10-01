"""Versioned artifact I/O for Service 4 state-machine v1.1."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Final

import pandas as pd

from service4.artifacts import file_sha256, find_project_root, load_verified_handoff
from service4.state_machine_v1 import MAX_EPISODE_DURATION, RETRY_GRACE
from service4.state_machine_v1_1 import (
    ASSET_EPISODE_GRAMMAR_VERSION,
    CONTROL_RESTORATION_RULE,
    run_state_machine_v1_1,
)

V1_1_ARTIFACT_SCHEMA_VERSION: Final[int] = 1
V1_1_ARTIFACT_DIRECTORY: Final[str] = "service4-state-machine-v1-1"
V1_1_ARTIFACT_FILENAMES: Final[tuple[str, ...]] = (
    "events.parquet",
    "episodes.parquet",
    "membership.parquet",
    "context-links.parquet",
)
_EVENT_EXPORT_COLUMNS: Final[tuple[str, ...]] = (
    "event_index",
    "date_time",
    "DevKey",
    "SifraSredstva",
    "aoj",
    "lokacija",
    "point_description",
    "signal_description",
    "prioriteta",
    "reported_state",
    "from_state",
    "to_state",
    "event_family",
    "point_family",
    "classification_rule",
    "classification_confidence",
    "sporocilo",
    "episode_event_role",
    "episode_start_trigger",
)
_CONTEXT_DETAIL_COLUMNS: Final[tuple[str, ...]] = (
    "event_index",
    "DevKey",
    "point_description",
    "signal_description",
    "prioriteta",
    "classification_rule",
    "classification_confidence",
    "sporocilo",
    "from_state",
    "to_state",
    "reported_state",
)


def _resolve_from_root(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _artifact_metadata(path: Path) -> dict[str, object]:
    return {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}


def _install_staged_artifacts(staging: Path, output_dir: Path) -> None:
    if not output_dir.exists():
        staging.rename(output_dir)
        return

    existing_manifest = output_dir / "manifest.json"
    staged_manifest = staging / "manifest.json"
    unchanged = (
        existing_manifest.exists()
        and file_sha256(existing_manifest) == file_sha256(staged_manifest)
        and all(
            (output_dir / name).exists() and file_sha256(output_dir / name) == file_sha256(staging / name)
            for name in V1_1_ARTIFACT_FILENAMES
        )
    )
    if unchanged:
        return
    raise FileExistsError(f"{output_dir} already contains different v1.1 artifacts; use a new version directory")


def build_v1_1_artifacts(
    input_path: Path,
    input_manifest_path: Path,
    output_dir: Path,
    *,
    project_root: Path,
) -> dict[str, object]:
    """Build immutable, checksum-recorded v1.1 outputs from a verified handoff."""
    resolved_input = _resolve_from_root(input_path, project_root)
    resolved_manifest = _resolve_from_root(input_manifest_path, project_root)
    resolved_output = _resolve_from_root(output_dir, project_root)
    handoff, handoff_manifest = load_verified_handoff(
        resolved_input,
        resolved_manifest,
        project_root=project_root,
    )
    result = run_state_machine_v1_1(handoff)

    missing_event_columns = sorted(set(_EVENT_EXPORT_COLUMNS).difference(result.events.columns))
    if missing_event_columns:
        raise ValueError(f"handoff cannot produce viewer event export; missing {missing_event_columns}")
    events = result.events.loc[:, _EVENT_EXPORT_COLUMNS].copy()

    context_details = handoff.loc[:, _CONTEXT_DETAIL_COLUMNS].copy()
    context_links = result.context_links.merge(
        context_details,
        on="event_index",
        how="left",
        validate="one_to_one",
        suffixes=("", "_detail"),
    )

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{V1_1_ARTIFACT_DIRECTORY}-", dir=resolved_output.parent) as temporary:
        staging = Path(temporary) / V1_1_ARTIFACT_DIRECTORY
        staging.mkdir()
        tables = {
            "events.parquet": events,
            "episodes.parquet": result.episodes,
            "membership.parquet": result.membership,
            "context-links.parquet": context_links,
        }
        for name, table in tables.items():
            table.to_parquet(staging / name, index=False, compression="zstd")

        state_machine_path = Path(__file__).with_name("state_machine_v1_1.py")
        frozen_dependency_path = Path(__file__).with_name("state_machine_v1.py")
        builder_path = Path(__file__)
        manifest: dict[str, object] = {
            "schema_version": V1_1_ARTIFACT_SCHEMA_VERSION,
            "method": "asset_state_machine_v1_1",
            "parameters": {
                "grammar_version": ASSET_EPISODE_GRAMMAR_VERSION,
                "retry_grace_seconds": RETRY_GRACE.total_seconds(),
                "maximum_episode_duration_seconds": MAX_EPISODE_DURATION.total_seconds(),
                "control_restoration_rule": CONTROL_RESTORATION_RULE,
                "control_restoration_membership": False,
                "control_restoration_requires_same_open_breaker_point": True,
            },
            "reference_evaluation": {
                "nonzero_interval_end_extension_seconds": 60,
                "zero_duration_reference_kind": "point",
                "changes_delivered_boundaries": False,
            },
            "input": {
                "path": _relative_or_absolute(resolved_input, project_root),
                "sha256": file_sha256(resolved_input),
                "manifest_path": _relative_or_absolute(resolved_manifest, project_root),
                "manifest_sha256": file_sha256(resolved_manifest),
                "row_count": len(handoff),
                "schema_version": handoff_manifest["schema_version"],
            },
            "producer": {
                "state_machine_path": _relative_or_absolute(state_machine_path, project_root),
                "state_machine_sha256": file_sha256(state_machine_path),
                "frozen_v1_dependency_path": _relative_or_absolute(frozen_dependency_path, project_root),
                "frozen_v1_dependency_sha256": file_sha256(frozen_dependency_path),
                "artifact_builder_path": _relative_or_absolute(builder_path, project_root),
                "artifact_builder_sha256": file_sha256(builder_path),
            },
            "counts": {
                "selected_events": len(result.events),
                "assigned_events": len(result.membership),
                "episodes": len(result.episodes),
                "complete_episodes": int(result.episodes["episode_complete"].sum()),
                "incomplete_episodes": int(result.episodes["episode_complete"].eq(False).sum()),
                "control_confirmed_restorations": int(
                    result.episodes["closure_reason"].eq("control_confirmed_restoration").sum()
                ),
                "context_links": len(result.context_links),
            },
            "artifacts": {name: _artifact_metadata(staging / name) for name in V1_1_ARTIFACT_FILENAMES},
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _install_staged_artifacts(staging, resolved_output)
    return manifest


def load_v1_1_artifacts(output_dir: Path) -> dict[str, pd.DataFrame]:
    """Verify and load a previously built v1.1 artifact directory."""
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != V1_1_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"unsupported v1.1 artifact schema: {manifest.get('schema_version')!r}")
    if manifest.get("method") != "asset_state_machine_v1_1":
        raise ValueError(f"unexpected artifact method: {manifest.get('method')!r}")

    tables: dict[str, pd.DataFrame] = {}
    metadata_by_name = manifest.get("artifacts", {})
    for name in V1_1_ARTIFACT_FILENAMES:
        path = output_dir / name
        metadata = metadata_by_name.get(name)
        if metadata is None or file_sha256(path) != metadata["sha256"]:
            raise ValueError(f"artifact checksum mismatch: {name}")
        if path.stat().st_size != metadata["size_bytes"]:
            raise ValueError(f"artifact size mismatch: {name}")
        tables[name.removesuffix(".parquet").replace("-", "_")] = pd.read_parquet(path)
    return tables


def _argument_parser(project_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Service 4 state-machine v1.1 artifacts.")
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "data/processed/service4-clustering-input-v1.parquet",
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=project_root / "data/processed/service4-clustering-input-v1.manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / f"data/processed/{V1_1_ARTIFACT_DIRECTORY}",
    )
    return parser


def main() -> None:
    """Command-line entry point for deterministic v1.1 artifact generation."""
    project_root = find_project_root()
    args = _argument_parser(project_root).parse_args()
    manifest = build_v1_1_artifacts(
        args.input,
        args.input_manifest,
        args.output,
        project_root=project_root,
    )
    counts = manifest["counts"]
    print(
        "Built state-machine v1.1 artifacts: "
        f"{counts['episodes']:,} episodes, {counts['assigned_events']:,} assigned events, "
        f"{counts['control_confirmed_restorations']:,} control-confirmed closures"
    )


if __name__ == "__main__":
    main()
