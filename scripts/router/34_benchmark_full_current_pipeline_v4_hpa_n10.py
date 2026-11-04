#!/usr/bin/env python3
"""
Step 34 V2: repeated runtime benchmark for the CURRENT project workflows,
including supplemental HPA validation.

Per dataset and per repetition, this script executes:

  CORE ANNOTATION/ROUTING WORKFLOW
    - existing marker/preparation stage(s)
    - Agent 1
    - Agent 2
    - Router V4

  SUPPLEMENTAL HPA WORKFLOW
    - existing HPA validation orchestrator profile

It records:
  - core workflow wall-clock time
  - Agent 1 time
  - Agent 2 time
  - Router V4 time
  - HPA workflow wall-clock time
  - directly measured combined suite wall-clock time
  - mean, SD, median, min, max across repeated runs

Important scientific interpretation:
  * HPA is supplemental evidence/validation, not a Router inference feature.
  * Therefore core runtime and HPA runtime are reported separately.
  * "Combined suite" is the time required to execute both current workflows
    sequentially for a dataset; it is NOT evidence that HPA feeds Router V4.
  * Agent 3 remains excluded because literature/network/LLM benchmarking is a
    separate experiment.
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


DEFAULT_OUTPUT_DIR = Path("data/analysis/runtime_benchmark_full_current_v4_hpa_n10")

WORKFLOWS = {
    "breast": {
        "core_profile": Path("configs/orchestrator/breast_context_marker_only_clean.json"),
        "hpa_profile": Path("configs/orchestrator/breast_hpa_validation_clean.json"),
    },
    "lung": {
        "core_profile": Path("configs/orchestrator/lung_context_validated_clean.json"),
        "hpa_profile": Path("configs/orchestrator/lung_hpa_validation_clean.json"),
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark core Agent1->Agent2->Router V4 plus supplemental HPA."
    )
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def num(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        x = float(str(value).strip())
    except (ValueError, TypeError):
        return None
    return x if math.isfinite(x) else None


def detect_duration_column(rows: List[dict]) -> str:
    if not rows:
        raise RuntimeError("Timing CSV is empty.")
    candidates = [
        "duration_seconds",
        "elapsed_seconds",
        "runtime_seconds",
        "seconds",
        "duration",
        "elapsed",
        "runtime",
    ]
    keys = list(rows[0])
    for key in candidates:
        if key in keys and any(num(r.get(key)) is not None for r in rows):
            return key

    ignored = {"index", "run", "run_id", "stage_index", "return_code"}
    numeric = []
    for key in keys:
        if key.lower() in ignored:
            continue
        vals = [num(r.get(key)) for r in rows]
        if vals and all(v is not None for v in vals):
            numeric.append(key)
    if len(numeric) == 1:
        return numeric[0]
    raise RuntimeError(f"Cannot identify timing column. Columns: {keys}")


def detect_stage_column(rows: List[dict]) -> str:
    if not rows:
        raise RuntimeError("Timing CSV is empty.")
    for key in ("stage", "stage_name", "name"):
        if key in rows[0]:
            return key
    raise RuntimeError(f"Cannot identify stage column. Columns: {list(rows[0])}")


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


def classify_core_stage(stage: str) -> str:
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


def expected_outputs_exist(root: Path, cfg: dict) -> Tuple[bool, List[str]]:
    missing = []
    for stage in cfg.get("stages", []):
        for item in stage.get("outputs", []):
            p = Path(item)
            if not p.is_absolute():
                p = root / p
            if not p.exists():
                missing.append(str(p))
    return not missing, missing


def run_profile(
    *,
    project_root: Path,
    profile_path: Path,
    env: dict,
    dataset: str,
    workflow_kind: str,
    run_idx: int,
) -> dict:
    cfg = read_json(profile_path)

    if workflow_kind == "core":
        names = [str(s.get("name", "")) for s in cfg.get("stages", [])]
        if "router_v4_scored_evidence_uncertainty" not in names:
            raise RuntimeError(
                f"{profile_path} does not contain Router V4. Stages: {names}"
            )

        # Protect benchmark from accidentally using the older label-comparison router.
        serialized = json.dumps(cfg).lower()
        if "review_router" in serialized:
            raise RuntimeError(
                f"{profile_path} contains legacy review_router; refusing benchmark."
            )

    output_dir = Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    timing_path = output_dir / "orchestrator_stage_timing.csv"

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

    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=project_root,
        env=env,
        text=True,
        capture_output=True,
    )
    wall = time.perf_counter() - started

    if proc.returncode != 0:
        raise RuntimeError(
            f"{dataset} {workflow_kind} run {run_idx} failed.\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    if not timing_path.exists():
        raise RuntimeError(f"Missing timing file: {timing_path}")

    rows = read_csv(timing_path)
    duration_col = detect_duration_column(rows)
    stage_col = detect_stage_column(rows)

    stages = []
    components: Dict[str, float] = {}
    stage_sum = 0.0

    for row in rows:
        stage = str(row.get(stage_col, "")).strip()
        seconds = num(row.get(duration_col))
        if not stage or seconds is None:
            continue

        stage_sum += seconds
        component = (
            classify_core_stage(stage)
            if workflow_kind == "core"
            else "hpa"
        )
        components[component] = components.get(component, 0.0) + seconds
        stages.append(
            {
                "dataset": dataset,
                "run": run_idx,
                "workflow_kind": workflow_kind,
                "context_name": cfg.get("context_name", ""),
                "stage": stage,
                "component": component,
                "seconds": seconds,
            }
        )

    if workflow_kind == "core":
        timed_names = {r["stage"] for r in stages}
        if "router_v4_scored_evidence_uncertainty" not in timed_names:
            raise RuntimeError(
                f"{dataset} core run {run_idx}: Router V4 stage not found in timing CSV."
            )

    outputs_ok, missing_outputs = expected_outputs_exist(project_root, cfg)
    if not outputs_ok:
        raise RuntimeError(
            f"{dataset} {workflow_kind} run {run_idx}: missing expected outputs: "
            f"{missing_outputs}"
        )

    return {
        "wall_seconds": wall,
        "stage_sum_seconds": stage_sum,
        "components": components,
        "stages": stages,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "command": " ".join(cmd),
        "context_name": cfg.get("context_name", ""),
    }


def main() -> int:
    args = parse_args()
    if args.runs < 2:
        raise SystemExit("--runs must be >= 2.")

    root = args.project_root.resolve()
    out = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    src = root / "src"
    if not src.exists():
        raise RuntimeError(f"src directory not found: {src}")

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src) if not existing else str(src) + os.pathsep + existing

    resolved = {}
    for dataset, cfg in WORKFLOWS.items():
        resolved[dataset] = {}
        for kind, rel in cfg.items():
            p = rel if rel.is_absolute() else root / rel
            if not p.exists():
                raise RuntimeError(f"Required profile not found: {p}")
            resolved[dataset][kind] = p

    per_run: List[dict] = []
    stage_rows: List[dict] = []
    logs: List[dict] = []

    print("=" * 78)
    print("STEP 34 V2 — CORE + HPA REPEATED RUNTIME BENCHMARK")
    print("=" * 78)
    print(f"Runs per dataset: {args.runs}")
    print("Core: Agent 1 -> Agent 2 -> Router V4")
    print("Supplemental: HPA validation")
    print("Agent 3: excluded; benchmarked separately later")
    print()

    for dataset in ("breast", "lung"):
        print(f"\n### {dataset.upper()} ###")

        core_profile = resolved[dataset]["core_profile"]
        hpa_profile = resolved[dataset]["hpa_profile"]

        for run_idx in range(1, args.runs + 1):
            print(f"[{dataset}] run {run_idx:02d}/{args.runs}: core ...", flush=True)

            suite_start = time.perf_counter()

            core = run_profile(
                project_root=root,
                profile_path=core_profile,
                env=env,
                dataset=dataset,
                workflow_kind="core",
                run_idx=run_idx,
            )

            print(f"[{dataset}] run {run_idx:02d}/{args.runs}: HPA ...", flush=True)

            hpa = run_profile(
                project_root=root,
                profile_path=hpa_profile,
                env=env,
                dataset=dataset,
                workflow_kind="hpa",
                run_idx=run_idx,
            )

            suite_wall = time.perf_counter() - suite_start

            stage_rows.extend(core["stages"])
            stage_rows.extend(hpa["stages"])

            c = core["components"]
            per_run.append(
                {
                    "dataset": dataset,
                    "run": run_idx,
                    "input_validation_seconds": c.get("input_validation", 0.0),
                    "input_preparation_seconds": c.get("input_preparation", 0.0),
                    "agent1_seconds": c.get("agent1", 0.0),
                    "agent2_seconds": c.get("agent2", 0.0),
                    "router_v4_seconds": c.get("router_v4", 0.0),
                    "core_orchestrator_wall_seconds": core["wall_seconds"],
                    "hpa_orchestrator_wall_seconds": hpa["wall_seconds"],
                    "combined_suite_wall_seconds": suite_wall,
                    "core_expected_outputs_ok": True,
                    "hpa_expected_outputs_ok": True,
                }
            )

            logs.extend(
                [
                    {
                        "dataset": dataset,
                        "run": run_idx,
                        "workflow_kind": "core",
                        "command": core["command"],
                        "stdout": core["stdout"],
                        "stderr": core["stderr"],
                    },
                    {
                        "dataset": dataset,
                        "run": run_idx,
                        "workflow_kind": "hpa",
                        "command": hpa["command"],
                        "stdout": hpa["stdout"],
                        "stderr": hpa["stderr"],
                    },
                ]
            )

            print(
                f"  A1={c.get('agent1', 0.0):.3f}s | "
                f"A2={c.get('agent2', 0.0):.3f}s | "
                f"Router={c.get('router_v4', 0.0):.3f}s | "
                f"Core={core['wall_seconds']:.3f}s | "
                f"HPA={hpa['wall_seconds']:.3f}s | "
                f"Combined={suite_wall:.3f}s"
            )

    summary_rows: List[dict] = []
    measures = [
        ("Agent 1", "agent1_seconds"),
        ("Agent 2 total", "agent2_seconds"),
        ("Router V4", "router_v4_seconds"),
        ("Core orchestrated workflow", "core_orchestrator_wall_seconds"),
        ("Supplemental HPA workflow", "hpa_orchestrator_wall_seconds"),
        ("Combined current suite", "combined_suite_wall_seconds"),
    ]

    for dataset in ("breast", "lung"):
        subset = [r for r in per_run if r["dataset"] == dataset]
        for label, field in measures:
            row = {
                "dataset": dataset,
                "measurement": label,
                **summarize(float(r[field]) for r in subset),
            }
            summary_rows.append(row)

    write_csv(out / "full_current_runtime_per_run.csv", per_run)
    write_csv(out / "full_current_runtime_stage_per_run.csv", stage_rows)
    write_csv(out / "full_current_runtime_summary.csv", summary_rows)
    write_csv(out / "full_current_runtime_command_log.csv", logs)

    protocol = {
        "created_at_local": datetime.now().astimezone().isoformat(),
        "runs_per_dataset": args.runs,
        "datasets": ["breast", "lung"],
        "core_profiles": {
            k: str(v["core_profile"]) for k, v in resolved.items()
        },
        "hpa_profiles": {
            k: str(v["hpa_profile"]) for k, v in resolved.items()
        },
        "core_definition": "current Agent1/Agent2 workflow plus Router V4",
        "hpa_definition": "supplemental HPA validation workflow",
        "combined_suite_definition": (
            "direct wall-clock measurement around sequential execution of the "
            "core workflow followed by supplemental HPA workflow"
        ),
        "hpa_is_router_inference_feature": False,
        "agent3_included": False,
        "expert_reference_labels_required_at_inference": False,
        "source_code_modified": False,
        "config_files_modified": False,
        "warning": (
            "HPA is network-sensitive and may show large runtime variance/outliers. "
            "Report core and HPA runtimes separately even when combined-suite time "
            "is also reported."
        ),
    }
    (out / "full_current_runtime_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 78)
    print("BENCHMARK COMPLETE")
    print("=" * 78)
    for row in summary_rows:
        print(
            f"{row['dataset']:>6} | {row['measurement']:<28} | "
            f"{float(row['mean_seconds']):.3f} ± "
            f"{float(row['std_seconds']):.3f} s | "
            f"median {float(row['median_seconds']):.3f} s | "
            f"min {float(row['min_seconds']):.3f} | "
            f"max {float(row['max_seconds']):.3f}"
        )

    print("\nInterpretation:")
    print("- Core runtime = Agent 1 + Agent 2 + Router V4 workflow.")
    print("- HPA runtime = supplemental HPA workflow.")
    print("- Combined suite = directly measured sequential core + HPA execution.")
    print("- HPA is NOT represented as a Router V4 inference feature.")
    print("- Agent 3 remains a separate benchmark.")
    print("\nSTEP 34 V2 STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
