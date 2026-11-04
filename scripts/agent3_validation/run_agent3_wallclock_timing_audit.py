from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


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

HEADER_TO_STEP = {
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
    "STANDARDIZED AGENT 3 COMPLETE": 18,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cluster-id", required=True)
    parser.add_argument("--output-root", required=True)
    args, _ = parser.parse_known_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    cluster_id = str(args.cluster_id)
    timing_csv = output_root / f"cluster_{cluster_id}_agent3_stepwise_runtime.csv"
    manifest_json = output_root / f"cluster_{cluster_id}_agent3_timing_manifest.json"
    stdout_log = output_root / f"cluster_{cluster_id}_agent3_timing_stdout.log"

    cmd = [
        sys.executable,
        "scripts/agent3_validation/run_agent3_standardized_full_with_decision.py",
        *sys.argv[1:],
    ]

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env["PYTHONPATH"] = str(Path.cwd() / "src")

    rows = []
    current_step = 1
    current_start = time.perf_counter()
    total_start = current_start
    log_lines = []

    print("RUNNING TIMED AGENT 3:")
    print(" ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    assert proc.stdout is not None

    for line in proc.stdout:
        print(line, end="")
        log_lines.append(line)

        clean = line.strip()
        if clean in HEADER_TO_STEP:
            now = time.perf_counter()
            elapsed = now - current_start

            if current_step not in [row["step_number"] for row in rows]:
                rows.append({
                    "step_number": current_step,
                    "step_name": STEP_NAMES[current_step],
                    "status": "completed",
                    "elapsed_seconds": elapsed,
                    "elapsed_minutes": elapsed / 60.0,
                })

            current_step = HEADER_TO_STEP[clean]
            current_start = now

    return_code = proc.wait()
    total_end = time.perf_counter()

    if current_step not in [row["step_number"] for row in rows]:
        elapsed = total_end - current_start
        rows.append({
            "step_number": current_step,
            "step_name": STEP_NAMES[current_step],
            "status": "completed",
            "elapsed_seconds": elapsed,
            "elapsed_minutes": elapsed / 60.0,
        })

    completed = {row["step_number"] for row in rows}
    for step in range(1, 19):
        if step not in completed:
            rows.append({
                "step_number": step,
                "step_name": STEP_NAMES[step],
                "status": "skipped",
                "elapsed_seconds": 0.0,
                "elapsed_minutes": 0.0,
            })

    rows = sorted(rows, key=lambda row: row["step_number"])

    total_seconds = total_end - total_start
    rows.append({
        "step_number": 99,
        "step_name": "TOTAL_AGENT3_STEPWISE_RUNTIME",
        "status": "completed" if return_code == 0 else "failed",
        "elapsed_seconds": total_seconds,
        "elapsed_minutes": total_seconds / 60.0,
    })

    with timing_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step_number",
                "step_name",
                "status",
                "elapsed_seconds",
                "elapsed_minutes",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "elapsed_seconds": round(float(row["elapsed_seconds"]), 6),
                "elapsed_minutes": round(float(row["elapsed_minutes"]), 6),
            })

    stdout_log.write_text("".join(log_lines), encoding="utf-8")

    manifest_json.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "cluster_id": cluster_id,
                "return_code": return_code,
                "timing_csv": str(timing_csv),
                "stdout_log": str(stdout_log),
                "total_seconds": total_seconds,
                "total_minutes": total_seconds / 60.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("TIMING AUDIT COMPLETE" if return_code == 0 else "TIMING AUDIT FAILED")
    print("=" * 100)
    print("Timing CSV:", timing_csv)
    print("Manifest:", manifest_json)
    print("Stdout log:", stdout_log)
    print(f"Total seconds: {total_seconds:.3f}")
    print(f"Total minutes: {total_seconds / 60.0:.3f}")

    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
