from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(".").resolve()


STEP_NAMES = {
    1: "Literature request construction",
    2: "Fresh context retrieval",
    3: "Paper ranking",
    4: "Evidence corpus construction",
    5: "PubMedBERT semantic reranking",
    6: "PubMedBERT evidence audit",
    7: "Rank-aware evidence coverage",
    8: "Conditional marker-rescue decision",
    9: "Candidate-independent rescue retrieval",
    10: "Cell Ontology candidate discovery",
    11: "Target-context confirmation query generation",
    12: "Fresh confirmation retrieval",
    13: "NCBI gene terminology",
    14: "Terminology-aware grounding",
    15: "Confirmation PubMedBERT reranking",
    16: "Evidence provenance packet",
    17: "Final evidence/result packet",
    18: "Decision summary and final packaging",
}


LABEL_TO_STEP = {
    "02 FRESH CONTEXT RETRIEVAL": 2,
    "03 PAPER RANKING": 3,
    "04 EVIDENCE CORPUS": 4,
    "05 PUBMEDBERT RERANKING": 5,
    "06 PUBMEDBERT EVIDENCE AUDIT": 6,
    "07 RANK-AWARE EVIDENCE COVERAGE": 7,
    "08 MARKER-RESCUE DECISION": 8,
    "09 CANDIDATE-INDEPENDENT RESCUE RETRIEVAL": 9,
    "10 ONTOLOGY-GROUNDED CANDIDATE DISCOVERY": 10,
    "11 TARGET-CONTEXT CONFIRMATION QUERY PLAN": 11,
    "12 FRESH TARGET-CONTEXT CONFIRMATION RETRIEVAL": 12,
    "13 NCBI GENE TERMINOLOGY": 13,
    "14 TERMINOLOGY-AWARE GROUNDING": 14,
    "15 CONFIRMATION PUBMEDBERT": 15,
    "16 CANDIDATE PROVENANCE": 16,
    "17 RESCUE EVIDENCE PACKET": 17,
    "STANDARDIZED FREEZE AUDIT": 17,
    "Agent 3 decision summary added.": 18,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the standardized Agent 3 workflow and record stage-level "
            "wall-clock runtime for any dataset, tissue, and cluster."
        )
    )

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cluster-id", required=True)
    parser.add_argument("--disease-context", required=True)
    parser.add_argument("--agent2-json", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--candidate-count", type=int, default=3)
    parser.add_argument("--context-page-size", type=int, default=3)
    parser.add_argument("--rescue-page-size", type=int, default=10)
    parser.add_argument("--confirmation-page-size", type=int, default=10)
    parser.add_argument("--top-k-passages", type=int, default=10)
    parser.add_argument("--top-marker-count", type=int, default=25)
    parser.add_argument("--rescue-top-marker-limit", type=int, default=5)
    parser.add_argument(
        "--cell-ontology",
        type=Path,
        default=Path("data/resources/cell_ontology/v2026-06-08/cl-base.json"),
    )
    parser.add_argument(
        "--timing-note",
        default="warm-cache command-level wall-clock timing",
        help=(
            "Free-text note stored in the timing manifest. Use this to mark "
            "warm-cache, cold-cache, CPU-only, or hardware details."
        ),
    )

    return parser.parse_args()


def default_output_root(dataset: str, cluster_id: str) -> Path:
    return (
        ROOT
        / "data"
        / "analysis"
        / "agent3_validation"
        / "runtime_timing_stepwise"
        / dataset
        / f"cluster_{cluster_id}"
    )


def build_command(args: argparse.Namespace, output_root: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        "scripts/agent3_validation/run_agent3_standardized_full_with_decision.py",
        "--dataset",
        args.dataset,
        "--cluster-id",
        str(args.cluster_id),
        "--disease-context",
        args.disease_context,
        "--candidate-count",
        str(args.candidate_count),
        "--context-page-size",
        str(args.context_page_size),
        "--rescue-page-size",
        str(args.rescue_page_size),
        "--confirmation-page-size",
        str(args.confirmation_page_size),
        "--top-k-passages",
        str(args.top_k_passages),
        "--top-marker-count",
        str(args.top_marker_count),
        "--rescue-top-marker-limit",
        str(args.rescue_top_marker_limit),
        "--cell-ontology",
        str(args.cell_ontology),
        "--output-root",
        str(output_root),
    ]

    if args.agent2_json:
        cmd.extend(["--agent2-json", str(args.agent2_json)])

    return cmd


def find_step(line: str) -> int | None:
    stripped = line.strip()

    for label, step in LABEL_TO_STEP.items():
        if stripped == label:
            return step

    return None


def append_row(
    rows: list[dict],
    *,
    dataset: str,
    cluster_id: str,
    step_number: int,
    status: str,
    elapsed_seconds: float,
    notes: str = "",
) -> None:
    rows.append(
        {
            "dataset": dataset,
            "cluster_id": str(cluster_id),
            "step_number": step_number,
            "step_name": STEP_NAMES[step_number],
            "status": status,
            "elapsed_seconds": round(elapsed_seconds, 6),
            "elapsed_minutes": round(elapsed_seconds / 60.0, 6),
            "notes": notes,
        }
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "cluster_id",
        "step_number",
        "step_name",
        "status",
        "elapsed_seconds",
        "elapsed_minutes",
        "notes",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    output_root = args.output_root or default_output_root(
        args.dataset,
        str(args.cluster_id),
    )
    output_root.mkdir(parents=True, exist_ok=True)

    timing_csv = (
        output_root
        / f"cluster_{args.cluster_id}_agent3_stepwise_runtime.csv"
    )
    manifest_json = (
        output_root
        / f"cluster_{args.cluster_id}_agent3_timing_manifest.json"
    )
    log_path = (
        output_root
        / f"cluster_{args.cluster_id}_agent3_timing_stdout.log"
    )

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"


    src = str(ROOT / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        src
        if not existing_pythonpath
        else src + os.pathsep + existing_pythonpath
    )

    cmd = build_command(args, output_root)

    print()
    print("=" * 100)
    print("STANDARDIZED AGENT 3 TIMING AUDIT")
    print("=" * 100)
    print("Dataset:", args.dataset)
    print("Cluster:", args.cluster_id)
    print("Output root:", output_root)
    print("Timing CSV:", timing_csv)
    print("Manifest:", manifest_json)
    print("Note:", args.timing_note)
    print()
    print("RUNNING:")
    print(" ".join(cmd))
    print("=" * 100)

    rows: list[dict] = []
    observed_steps: set[int] = set()
    current_step = 1
    current_start = time.perf_counter()
    overall_start = current_start
    return_code = None

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        assert process.stdout is not None

        try:
            for line in process.stdout:
                print(line, end="")
                log.write(line)
                log.flush()

                next_step = find_step(line)
                if next_step is None or next_step == current_step:
                    continue

                elapsed = time.perf_counter() - current_start
                append_row(
                    rows,
                    dataset=args.dataset,
                    cluster_id=str(args.cluster_id),
                    step_number=current_step,
                    status="completed",
                    elapsed_seconds=elapsed,
                )
                observed_steps.add(current_step)
                current_step = next_step
                current_start = time.perf_counter()

            return_code = process.wait()

        finally:
            if return_code is None:
                return_code = process.wait()

    elapsed = time.perf_counter() - current_start
    final_status = "completed" if return_code == 0 else "failed"
    append_row(
        rows,
        dataset=args.dataset,
        cluster_id=str(args.cluster_id),
        step_number=current_step,
        status=final_status,
        elapsed_seconds=elapsed,
    )
    observed_steps.add(current_step)

    for step_number in range(1, 19):
        if step_number in observed_steps:
            continue

        append_row(
            rows,
            dataset=args.dataset,
            cluster_id=str(args.cluster_id),
            step_number=step_number,
            status="skipped",
            elapsed_seconds=0.0,
            notes=(
                "Stage label was not observed. This commonly means the "
                "rescue branch was not triggered, or the run stopped before "
                "this stage."
            ),
        )

    rows.sort(key=lambda row: int(row["step_number"]))

    total_completed_seconds = sum(
        float(row["elapsed_seconds"])
        for row in rows
        if row["status"] == "completed"
    )

    total_row = {
        "dataset": args.dataset,
        "cluster_id": str(args.cluster_id),
        "step_number": 99,
        "step_name": "TOTAL_AGENT3_STEPWISE_RUNTIME",
        "status": final_status,
        "elapsed_seconds": round(total_completed_seconds, 6),
        "elapsed_minutes": round(total_completed_seconds / 60.0, 6),
        "notes": "Sum of completed stage-level wall-clock measurements.",
    }
    rows.append(total_row)

    write_csv(timing_csv, rows)

    manifest = {
        "schema_name": "Agent3TimingAuditManifest",
        "schema_version": "1.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "cluster_id": str(args.cluster_id),
        "disease_context": args.disease_context,
        "agent2_json": str(args.agent2_json) if args.agent2_json else None,
        "cell_ontology": str(args.cell_ontology),
        "output_root": str(output_root),
        "timing_csv": str(timing_csv),
        "stdout_log": str(log_path),
        "return_code": return_code,
        "status": final_status,
        "timing_note": args.timing_note,
        "command": cmd,
        "total_completed_seconds": round(total_completed_seconds, 6),
        "total_completed_minutes": round(total_completed_seconds / 60.0, 6),
        "method_note": (
            "This audit records command-level wall-clock runtime by observing "
            "the standardized Agent 3 stage labels printed by the existing "
            "runner. It does not change Agent 3 evidence logic."
        ),
    }
    write_json(manifest_json, manifest)

    print()
    print("=" * 100)
    print("TIMING AUDIT COMPLETE" if return_code == 0 else "TIMING AUDIT FAILED")
    print("=" * 100)
    print("Timing CSV:", timing_csv)
    print("Manifest:", manifest_json)
    print("Stdout log:", log_path)
    print("Total seconds:", round(total_completed_seconds, 3))
    print("Total minutes:", round(total_completed_seconds / 60.0, 3))

    if return_code != 0:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
