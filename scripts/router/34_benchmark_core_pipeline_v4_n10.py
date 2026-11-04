#!/usr/bin/env python3
"""
Step 34: publication-oriented repeated runtime benchmark for the current
deterministic Agent 1 -> Agent 2 -> Router V4 workflows.

This script does NOT modify source code or configuration files.
It re-executes the existing orchestrator profiles with --force, records
per-run wall-clock time, reads the orchestrator's own per-stage timing CSV,
and writes analysis-only benchmark outputs.

Important:
- Agent 3 is intentionally excluded here because literature/network/LLM
  latency is a separate experiment.
- HPA validation is intentionally excluded because HPA is supplemental.
- Expert/reference labels are not used by this benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_BREAST_PROFILE = Path("configs/orchestrator/breast_context_marker_only_clean.json")
DEFAULT_LUNG_PROFILE = Path("configs/orchestrator/lung_context_validated_clean.json")
DEFAULT_OUTPUT_DIR = Path("data/analysis/runtime_benchmark_core_v4_n10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the current breast and lung deterministic orchestrator "
            "workflows repeatedly and summarize stage + end-to-end runtime."
        )
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--breast-profile", type=Path, default=DEFAULT_BREAST_PROFILE)
    parser.add_argument("--lung-profile", type=Path, default=DEFAULT_LUNG_PROFILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: List[dict], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        seen = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def resolve_profile_path(project_root: Path, profile: Path) -> Path:
    return profile if profile.is_absolute() else (project_root / profile)


def numeric(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        x = float(text)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def detect_duration_column(rows: List[dict]) -> str:
    if not rows:
        raise RuntimeError("Orchestrator timing CSV is empty.")

    preferred = [
        "duration_seconds",
        "elapsed_seconds",
        "runtime_seconds",
        "seconds",
        "duration",
        "elapsed",
        "runtime",
    ]
    keys = list(rows[0].keys())

    for candidate in preferred:
        if candidate in keys and any(numeric(r.get(candidate)) is not None for r in rows):
            return candidate

    # Conservative fallback: find exactly one numeric-looking column other than
    # obvious identifiers/indexes.
    ignored = {
        "index",
        "run",
        "run_id",
        "cluster_id",
        "stage_index",
        "return_code",
    }
    numeric_keys = []
    for key in keys:
        if key.lower() in ignored:
            continue
        values = [numeric(r.get(key)) for r in rows]
        if values and all(v is not None for v in values):
            numeric_keys.append(key)

    if len(numeric_keys) == 1:
        return numeric_keys[0]

    raise RuntimeError(
        "Could not determine duration column in orchestrator_stage_timing.csv. "
        f"Columns were: {keys}"
    )


def stage_name_column(rows: List[dict]) -> str:
    if not rows:
        raise RuntimeError("Orchestrator timing CSV is empty.")
    for candidate in ("stage", "stage_name", "name"):
        if candidate in rows[0]:
            return candidate
    raise RuntimeError(
        "Could not determine stage-name column in orchestrator timing CSV. "
        f"Columns were: {list(rows[0].keys())}"
    )


def validate_profile(profile_path: Path) -> dict:
    cfg = load_json(profile_path)
    required = {"context_name", "output_dir", "stages"}
    missing = required - set(cfg)
    if missing:
        raise RuntimeError(f"{profile_path}: missing required keys {sorted(missing)}")

    stages = cfg["stages"]
    names = [str(s.get("name", "")) for s in stages]

    router_matches = [n for n in names if n == "router_v4_scored_evidence_uncertainty"]
    if len(router_matches) != 1:
        raise RuntimeError(
            f"{profile_path}: expected exactly one Router V4 stage, found "
            f"{len(router_matches)}. Stages: {names}"
        )

    forbidden_fragments = (
        "review_router",
        "reference_annotation",
        "expert_annotation",
    )
    serialized = json.dumps(cfg).lower()
    found_forbidden = [x for x in forbidden_fragments if x in serialized]
    if found_forbidden:
        raise RuntimeError(
            f"{profile_path}: benchmark profile contains prohibited/legacy "
            f"routing text: {found_forbidden}"
        )

    return cfg


def expected_outputs_exist(project_root: Path, cfg: dict) -> Tuple[bool, List[str]]:
    missing: List[str] = []
    for stage in cfg["stages"]:
        for output in stage.get("outputs", []):
            p = Path(output)
            if not p.is_absolute():
                p = project_root / p
            if not p.exists():
                missing.append(str(p))
    return (len(missing) == 0, missing)


def classify_scientific_component(workflow: str, stage: str) -> str:
    s = stage.lower()

    if s == "input_validation":
        return "input_validation"
    if "prepare_input" in s:
        return "input_preparation"
    if "agent1" in s:
        return "agent1"
    if "agent2" in s:
        return "agent2"
    if "router_v4" in s or "scored_evidence_uncertainty" in s:
        return "router_v4"
    return "other"


def summarize(values: Iterable[float]) -> dict:
    vals = list(values)
    if not vals:
        return {
            "count": 0,
            "mean_seconds": "",
            "std_seconds": "",
            "median_seconds": "",
            "min_seconds": "",
            "max_seconds": "",
        }
    return {
        "count": len(vals),
        "mean_seconds": statistics.mean(vals),
        "std_seconds": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "median_seconds": statistics.median(vals),
        "min_seconds": min(vals),
        "max_seconds": max(vals),
    }


def main() -> int:
    args = parse_args()
    if args.runs < 2:
        raise SystemExit("--runs must be at least 2 for an SD estimate.")

    project_root = args.project_root.resolve()
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else project_root / args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    src_dir = project_root / "src"
    if not src_dir.exists():
        raise RuntimeError(f"src directory not found: {src_dir}")

    workflows = [
        ("breast", resolve_profile_path(project_root, args.breast_profile)),
        ("lung", resolve_profile_path(project_root, args.lung_profile)),
    ]

    profiles: Dict[str, dict] = {}
    for workflow, path in workflows:
        if not path.exists():
            raise RuntimeError(f"Profile not found: {path}")
        profiles[workflow] = validate_profile(path)

    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(src_dir)
        if not old_pythonpath
        else str(src_dir) + os.pathsep + old_pythonpath
    )

    raw_stage_rows: List[dict] = []
    run_rows: List[dict] = []
    command_log: List[dict] = []

    print("=" * 72)
    print("STEP 34 — CORE DETERMINISTIC RUNTIME BENCHMARK")
    print("=" * 72)
    print(f"Project root : {project_root}")
    print(f"Runs/dataset : {args.runs}")
    print("Agent 3      : EXCLUDED (separate network/LLM experiment)")
    print("HPA          : EXCLUDED (supplemental evidence experiment)")
    print("Router       : V4 heuristic baseline")
    print()

    for workflow, profile_path in workflows:
        cfg = profiles[workflow]
        context_name = str(cfg["context_name"])
        workflow_output = Path(cfg["output_dir"])
        if not workflow_output.is_absolute():
            workflow_output = project_root / workflow_output

        timing_path = workflow_output / "orchestrator_stage_timing.csv"
        manifest_path = workflow_output / "orchestrator_run_manifest.json"

        print(f"\n--- {workflow.upper()} : {context_name} ---")

        for run_idx in range(1, args.runs + 1):
            cmd = [
                sys.executable,
                "-m",
                "annotation_orchestrator.runners.run_orchestrator",
                "--context-profile",
                str(profile_path),
                "--project-root",
                str(project_root),
                "--execute-stage-commands",
                "--force",
            ]

            print(f"[{workflow}] run {run_idx:02d}/{args.runs} ...", flush=True)
            started = time.perf_counter()
            completed = subprocess.run(
                cmd,
                cwd=project_root,
                env=env,
                text=True,
                capture_output=True,
            )
            wall_seconds = time.perf_counter() - started

            command_log.append(
                {
                    "workflow": workflow,
                    "run": run_idx,
                    "return_code": completed.returncode,
                    "command": " ".join(cmd),
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            )

            if completed.returncode != 0:
                write_csv(output_dir / "benchmark_command_log_partial.csv", command_log)
                raise RuntimeError(
                    f"{workflow} run {run_idx} failed with return code "
                    f"{completed.returncode}.\nSTDOUT:\n{completed.stdout}\n"
                    f"STDERR:\n{completed.stderr}"
                )

            if not timing_path.exists():
                raise RuntimeError(
                    f"{workflow} run {run_idx}: orchestrator timing file not found: "
                    f"{timing_path}"
                )

            timing_rows = read_csv(timing_path)
            duration_col = detect_duration_column(timing_rows)
            stage_col = stage_name_column(timing_rows)

            stage_seen = set()
            stage_sum = 0.0
            component_totals: Dict[str, float] = {}

            for row in timing_rows:
                stage = str(row.get(stage_col, "")).strip()
                duration = numeric(row.get(duration_col))
                if not stage or duration is None:
                    continue

                stage_seen.add(stage)
                stage_sum += duration
                component = classify_scientific_component(workflow, stage)
                component_totals[component] = component_totals.get(component, 0.0) + duration

                raw_stage_rows.append(
                    {
                        "workflow": workflow,
                        "context_name": context_name,
                        "run": run_idx,
                        "stage": stage,
                        "scientific_component": component,
                        "seconds": duration,
                    }
                )

            configured_stage_names = {str(s["name"]) for s in cfg["stages"]}
            missing_configured_stages = sorted(configured_stage_names - stage_seen)
            if missing_configured_stages:
                raise RuntimeError(
                    f"{workflow} run {run_idx}: timing CSV omitted configured stages: "
                    f"{missing_configured_stages}. Seen: {sorted(stage_seen)}"
                )

            if "router_v4_scored_evidence_uncertainty" not in stage_seen:
                raise RuntimeError(
                    f"{workflow} run {run_idx}: Router V4 timing row is missing."
                )

            outputs_ok, missing_outputs = expected_outputs_exist(project_root, cfg)
            if not outputs_ok:
                raise RuntimeError(
                    f"{workflow} run {run_idx}: expected outputs missing: "
                    f"{missing_outputs}"
                )

            run_rows.append(
                {
                    "workflow": workflow,
                    "context_name": context_name,
                    "run": run_idx,
                    "orchestrator_wall_seconds": wall_seconds,
                    "timed_stage_sum_seconds": stage_sum,
                    "input_validation_seconds": component_totals.get("input_validation", 0.0),
                    "input_preparation_seconds": component_totals.get("input_preparation", 0.0),
                    "agent1_seconds": component_totals.get("agent1", 0.0),
                    "agent2_seconds": component_totals.get("agent2", 0.0),
                    "router_v4_seconds": component_totals.get("router_v4", 0.0),
                    "other_stage_seconds": component_totals.get("other", 0.0),
                    "all_expected_outputs_exist": True,
                    "manifest_exists": manifest_path.exists(),
                }
            )

            print(
                f"    wall={wall_seconds:.3f}s | "
                f"A1={component_totals.get('agent1', 0.0):.3f}s | "
                f"A2={component_totals.get('agent2', 0.0):.3f}s | "
                f"Router={component_totals.get('router_v4', 0.0):.3f}s"
            )

    # Per-stage summary.
    stage_groups: Dict[Tuple[str, str, str], List[float]] = {}
    for row in raw_stage_rows:
        key = (
            str(row["workflow"]),
            str(row["stage"]),
            str(row["scientific_component"]),
        )
        stage_groups.setdefault(key, []).append(float(row["seconds"]))

    stage_summary: List[dict] = []
    for (workflow, stage, component), values in sorted(stage_groups.items()):
        row = {
            "workflow": workflow,
            "stage": stage,
            "scientific_component": component,
        }
        row.update(summarize(values))
        stage_summary.append(row)

    # Scientific-component summary.
    component_fields = [
        ("input_validation", "input_validation_seconds"),
        ("input_preparation", "input_preparation_seconds"),
        ("agent1", "agent1_seconds"),
        ("agent2", "agent2_seconds"),
        ("router_v4", "router_v4_seconds"),
        ("orchestrator_wall_clock", "orchestrator_wall_seconds"),
        ("timed_stage_sum", "timed_stage_sum_seconds"),
    ]

    component_summary: List[dict] = []
    for workflow in ("breast", "lung"):
        subset = [r for r in run_rows if r["workflow"] == workflow]
        for component, field in component_fields:
            values = [float(r[field]) for r in subset]
            # Omit all-zero optional components (e.g., breast input_preparation).
            if component in {"input_preparation"} and all(v == 0.0 for v in values):
                continue
            row = {"workflow": workflow, "component": component}
            row.update(summarize(values))
            component_summary.append(row)

    # Compact paper-oriented summary.
    paper_rows: List[dict] = []
    for workflow in ("breast", "lung"):
        subset = [r for r in run_rows if r["workflow"] == workflow]
        for label, field in [
            ("Agent 1", "agent1_seconds"),
            ("Agent 2 total", "agent2_seconds"),
            ("Router V4", "router_v4_seconds"),
            ("Whole orchestrated workflow", "orchestrator_wall_seconds"),
        ]:
            stats = summarize(float(r[field]) for r in subset)
            paper_rows.append(
                {
                    "dataset": workflow,
                    "measurement": label,
                    **stats,
                }
            )

    write_csv(output_dir / "core_v4_runtime_per_run.csv", run_rows)
    write_csv(output_dir / "core_v4_runtime_stage_per_run.csv", raw_stage_rows)
    write_csv(output_dir / "core_v4_runtime_stage_summary.csv", stage_summary)
    write_csv(output_dir / "core_v4_runtime_component_summary.csv", component_summary)
    write_csv(output_dir / "core_v4_runtime_paper_table.csv", paper_rows)

    # Keep command output separately because it can be verbose.
    write_csv(output_dir / "benchmark_command_log.csv", command_log)

    protocol = {
        "created_at_local": datetime.now().astimezone().isoformat(),
        "runs_per_dataset": args.runs,
        "datasets": ["breast", "lung"],
        "breast_profile": str(workflows[0][1]),
        "lung_profile": str(workflows[1][1]),
        "execution": "existing orchestrator, --execute-stage-commands --force",
        "router_role": "Router V4 heuristic baseline",
        "agent3_included": False,
        "hpa_included": False,
        "expert_or_reference_labels_required_for_benchmark": False,
        "source_code_modified": False,
        "configuration_modified": False,
        "timing_sources": {
            "stage_timing": "orchestrator_stage_timing.csv produced by each run",
            "end_to_end": "external time.perf_counter around complete orchestrator process",
        },
        "important_interpretation": (
            "Whole orchestrated workflow runtime is measured directly per run. "
            "Do not construct total runtime by adding means from separate experiments."
        ),
    }
    (output_dir / "core_v4_runtime_protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("BENCHMARK COMPLETE")
    print("=" * 72)
    print(f"Output directory: {output_dir}")
    print("Router V4 present in every timed workflow: YES")
    print("Agent 3 included: NO")
    print("HPA included: NO")
    print("Existing source/config files modified: NO")
    print("\nPaper-oriented summary:")
    for row in paper_rows:
        print(
            f"{row['dataset']:>6} | {row['measurement']:<28} | "
            f"{float(row['mean_seconds']):.3f} ± "
            f"{float(row['std_seconds']):.3f} s | "
            f"median {float(row['median_seconds']):.3f} s | "
            f"min {float(row['min_seconds']):.3f} | "
            f"max {float(row['max_seconds']):.3f}"
        )

    print("\nSTEP 34 STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
