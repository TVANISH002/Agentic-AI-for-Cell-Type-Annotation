from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


class AnnotationOrchestrator:
    def __init__(self, context_profile: str | Path, project_root: str | Path = ".") -> None:
        self.project_root = Path(project_root).resolve()
        self.profile_path = self._resolve_path(context_profile)
        self.profile = json.loads(self.profile_path.read_text(encoding="utf-8"))

        self.output_dir = self._resolve_path(self.profile["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, execute_stage_commands: bool = False, force: bool = False) -> dict[str, Any]:
        timings: list[dict[str, Any]] = []

        input_summary = self._validate_inputs()
        timings.append(input_summary["timing"])

        for stage in self.profile.get("stages", []):
            timings.append(
                self._run_stage(
                    stage=stage,
                    execute_stage_commands=execute_stage_commands,
                    force=force,
                )
            )

        summary = {
            "context_name": self.profile.get("context_name"),
            "species": self.profile.get("species"),
            "tissue": self.profile.get("tissue"),
            "disease_context": self.profile.get("disease_context"),
            "platform": self.profile.get("platform"),
            "marker_file": str(self._resolve_path(self.profile["marker_file"])),
            "cluster_column": self.profile.get("cluster_column"),
            "gene_column": self.profile.get("gene_column"),
            "cluster_count": input_summary["cluster_count"],
            "cluster_ids": input_summary["cluster_ids"],
            "total_marker_rows": input_summary["total_marker_rows"],
            "stage_timings": timings,
            "note": "Orchestrator only automates existing stage commands. Agent logic and scientific output schemas are unchanged.",
            "outputs": {
                "orchestrator_manifest": str(self.output_dir / "orchestrator_run_manifest.json"),
                "orchestrator_stage_timing": str(self.output_dir / "orchestrator_stage_timing.csv"),
            },
        }

        self._write_outputs(summary, timings)
        return summary

    def _validate_inputs(self) -> dict[str, Any]:
        started = time.perf_counter()

        marker_file = self._resolve_path(self.profile["marker_file"])
        cluster_column = self.profile["cluster_column"]
        gene_column = self.profile["gene_column"]

        if not marker_file.exists():
            raise FileNotFoundError(f"Marker file does not exist: {marker_file}")

        cluster_ids: set[str] = set()
        total_rows = 0

        with marker_file.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)

            if reader.fieldnames is None:
                raise ValueError(f"Marker file has no header: {marker_file}")

            missing_columns = [
                column
                for column in (cluster_column, gene_column)
                if column not in reader.fieldnames
            ]
            if missing_columns:
                raise ValueError(
                    f"Marker file is missing required columns {missing_columns}. "
                    f"Available columns: {reader.fieldnames}"
                )

            for row in reader:
                total_rows += 1
                cluster_ids.add(str(row[cluster_column]))

        elapsed = time.perf_counter() - started

        return {
            "cluster_count": len(cluster_ids),
            "cluster_ids": sorted(cluster_ids, key=self._natural_sort_key),
            "total_marker_rows": total_rows,
            "timing": {
                "stage": "input_validation",
                "status": "executed",
                "elapsed_seconds": elapsed,
                "command": "",
                "outputs": [],
                "note": "",
            },
        }

    def _run_stage(
        self,
        stage: dict[str, Any],
        execute_stage_commands: bool,
        force: bool,
    ) -> dict[str, Any]:
        name = stage["name"]
        command = stage.get("command", "")
        expected_outputs = [
            self._resolve_path(output_path)
            for output_path in stage.get("outputs", [])
        ]

        if expected_outputs and not force and all(path.exists() for path in expected_outputs):
            return {
                "stage": name,
                "status": "skipped_existing_outputs",
                "elapsed_seconds": 0.0,
                "command": command,
                "outputs": [str(path) for path in expected_outputs],
                "note": "All expected outputs already exist. Use --force to rerun.",
            }

        if not execute_stage_commands:
            return {
                "stage": name,
                "status": "planned",
                "elapsed_seconds": 0.0,
                "command": command,
                "outputs": [str(path) for path in expected_outputs],
                "note": "Stage command was not executed because --execute-stage-commands was not provided.",
            }

        if not command:
            raise ValueError(f"Stage {name} has no command.")

        for output_path in expected_outputs:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")

        started = time.perf_counter()

        completed = subprocess.run(
            command,
            cwd=self.project_root,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

        elapsed = time.perf_counter() - started

        if completed.returncode != 0:
            raise RuntimeError(
                f"Stage failed: {name}\n"
                f"Command: {command}\n"
                f"STDOUT:\n{completed.stdout}\n"
                f"STDERR:\n{completed.stderr}"
            )

        missing = [str(path) for path in expected_outputs if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Stage {name} finished but expected outputs are missing: {missing}"
            )

        return {
            "stage": name,
            "status": "executed",
            "elapsed_seconds": elapsed,
            "command": command,
            "outputs": [str(path) for path in expected_outputs],
            "note": "",
        }

    def _write_outputs(
        self,
        summary: dict[str, Any],
        timings: list[dict[str, Any]],
    ) -> None:
        manifest_path = self.output_dir / "orchestrator_run_manifest.json"
        timing_path = self.output_dir / "orchestrator_stage_timing.csv"

        manifest_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        fieldnames = ["stage", "status", "elapsed_seconds", "command", "outputs", "note"]

        with timing_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()

            for row in timings:
                csv_row = dict(row)
                csv_row["outputs"] = json.dumps(csv_row.get("outputs", []))
                writer.writerow(csv_row)

    def _resolve_path(self, path: str | Path) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path
        return self.project_root / path

    @staticmethod
    def _natural_sort_key(value: str) -> tuple[int, str]:
        if value.isdigit():
            return (0, f"{int(value):020d}")
        return (1, value)