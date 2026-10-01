from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from service4.artifacts import (
    V1_ARTIFACT_FILENAMES,
    build_v1_artifacts,
    file_sha256,
    load_v1_artifacts,
)
from service4.artifacts_v1_1 import (
    V1_1_ARTIFACT_FILENAMES,
    build_v1_1_artifacts,
    load_v1_1_artifacts,
)
from service4.artifacts_v2 import (
    V2_ARTIFACT_FILENAMES,
    build_v2_artifacts,
    load_v2_artifacts,
)

START = pd.Timestamp("2025-01-01 00:00:00")


class V1ArtifactsTest(unittest.TestCase):
    def test_builds_reloads_and_rebuilds_identically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_dir = root / "data"
            data_dir.mkdir()
            producer = root / "producer.py"
            source = data_dir / "source.parquet"
            producer.write_text("# fixture producer\n", encoding="utf-8")
            source.write_text("fixture source\n", encoding="utf-8")

            handoff = self._handoff()
            input_path = data_dir / "handoff.parquet"
            handoff.to_parquet(input_path, index=False)
            input_manifest = data_dir / "handoff.manifest.json"
            input_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "artifact_sha256": file_sha256(input_path),
                        "producer": "producer.py",
                        "producer_sha256": file_sha256(producer),
                        "source": "data/source.parquet",
                        "source_sha256": file_sha256(source),
                        "row_count": len(handoff),
                    }
                ),
                encoding="utf-8",
            )
            output_dir = data_dir / "service4-state-machine-v1"

            first_manifest = build_v1_artifacts(
                input_path,
                input_manifest,
                output_dir,
                project_root=root,
            )
            second_manifest = build_v1_artifacts(
                input_path,
                input_manifest,
                output_dir,
                project_root=root,
            )
            tables = load_v1_artifacts(output_dir)

            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(first_manifest["counts"]["episodes"], 1)
            self.assertEqual(first_manifest["counts"]["assigned_events"], 1)
            self.assertEqual(set(tables), {"events", "episodes", "membership", "context_links"})
            self.assertEqual(tables["context_links"]["event_index"].tolist(), [1])
            for name in V1_ARTIFACT_FILENAMES:
                self.assertEqual(
                    file_sha256(output_dir / name),
                    first_manifest["artifacts"][name]["sha256"],
                )

    @staticmethod
    def _handoff() -> pd.DataFrame:
        rows = [
            {
                "event_index": 0,
                "date_time": START,
                "DevKey": "protection-a",
                "SifraSredstva": "asset-a",
                "signal_description": "PROTECTION",
                "event_family": "automatic_state",
                "point_family": "protection",
                "can_start_episode": True,
                "from_state": "ALARM",
                "to_state": "ALARM",
                "reported_state": "ALARM",
                "command_target_state": pd.NA,
            },
            {
                "event_index": 1,
                "date_time": START + pd.Timedelta(seconds=10),
                "DevKey": "command-a",
                "SifraSredstva": "asset-a",
                "signal_description": "CONTROL",
                "event_family": "control_command",
                "point_family": "control",
                "can_start_episode": False,
                "from_state": pd.NA,
                "to_state": pd.NA,
                "reported_state": pd.NA,
                "command_target_state": "VKLOP",
            },
            {
                "event_index": 2,
                "date_time": START + pd.Timedelta(seconds=46),
                "DevKey": "switch-a",
                "SifraSredstva": "asset-a",
                "signal_description": "SWITCH",
                "event_family": "automatic_state",
                "point_family": "switchgear",
                "can_start_episode": True,
                "from_state": "NORMAL",
                "to_state": "NORMAL",
                "reported_state": "NORMAL",
                "command_target_state": pd.NA,
            },
        ]
        frame = pd.DataFrame.from_records(rows)
        frame["aoj"] = "AREA"
        frame["lokacija"] = "LOCATION"
        frame["point_description"] = "POINT"
        frame["prioriteta"] = "A"
        frame["classification_rule"] = "fixture"
        frame["classification_confidence"] = "high"
        frame["sporocilo"] = "synthetic fixture"
        return frame


class V1_1ArtifactsTest(unittest.TestCase):
    def test_builds_reloads_and_rebuilds_identically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_dir = root / "data"
            data_dir.mkdir()
            producer = root / "producer.py"
            source = data_dir / "source.parquet"
            producer.write_text("# fixture producer\n", encoding="utf-8")
            source.write_text("fixture source\n", encoding="utf-8")

            handoff = V1ArtifactsTest._handoff()
            input_path = data_dir / "handoff.parquet"
            handoff.to_parquet(input_path, index=False)
            input_manifest = data_dir / "handoff.manifest.json"
            input_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "artifact_sha256": file_sha256(input_path),
                        "producer": "producer.py",
                        "producer_sha256": file_sha256(producer),
                        "source": "data/source.parquet",
                        "source_sha256": file_sha256(source),
                        "row_count": len(handoff),
                    }
                ),
                encoding="utf-8",
            )
            output_dir = data_dir / "service4-state-machine-v1-1"

            first_manifest = build_v1_1_artifacts(
                input_path,
                input_manifest,
                output_dir,
                project_root=root,
            )
            second_manifest = build_v1_1_artifacts(
                input_path,
                input_manifest,
                output_dir,
                project_root=root,
            )
            tables = load_v1_1_artifacts(output_dir)

            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(first_manifest["method"], "asset_state_machine_v1_1")
            self.assertEqual(first_manifest["counts"]["episodes"], 1)
            self.assertEqual(set(tables), {"events", "episodes", "membership", "context_links"})
            self.assertIn("point_count", tables["episodes"].columns)
            self.assertNotIn("device_count", tables["episodes"].columns)
            for name in V1_1_ARTIFACT_FILENAMES:
                self.assertEqual(file_sha256(output_dir / name), first_manifest["artifacts"][name]["sha256"])


class V2ArtifactsTest(unittest.TestCase):
    def test_builds_reloads_and_rebuilds_identically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_dir = root / "data"
            data_dir.mkdir()
            producer = root / "producer.py"
            source = data_dir / "source.parquet"
            producer.write_text("# fixture producer\n", encoding="utf-8")
            source.write_text("fixture source\n", encoding="utf-8")

            handoff = V1ArtifactsTest._handoff()
            input_path = data_dir / "handoff.parquet"
            handoff.to_parquet(input_path, index=False)
            input_manifest = data_dir / "handoff.manifest.json"
            input_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "artifact_sha256": file_sha256(input_path),
                        "producer": "producer.py",
                        "producer_sha256": file_sha256(producer),
                        "source": "data/source.parquet",
                        "source_sha256": file_sha256(source),
                        "row_count": len(handoff),
                    }
                ),
                encoding="utf-8",
            )
            output_dir = data_dir / "service4-state-machine-v2"

            first_manifest = build_v2_artifacts(
                input_path,
                input_manifest,
                output_dir,
                project_root=root,
            )
            second_manifest = build_v2_artifacts(
                input_path,
                input_manifest,
                output_dir,
                project_root=root,
            )
            tables = load_v2_artifacts(output_dir)

            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(first_manifest["method"], "asset_state_machine_v2")
            self.assertEqual(first_manifest["counts"]["episodes"], 1)
            self.assertEqual(set(tables), {"events", "episodes", "membership", "context_links"})
            for name in V2_ARTIFACT_FILENAMES:
                self.assertEqual(file_sha256(output_dir / name), first_manifest["artifacts"][name]["sha256"])


if __name__ == "__main__":
    unittest.main()
