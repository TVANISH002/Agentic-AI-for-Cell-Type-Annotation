from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def resolve_project_path(
    path: Path,
) -> Path:
    if path.is_absolute():
        return path

    return (
        PROJECT_ROOT
        / path
    ).resolve()


def read_router_queue(
    path: Path,
) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Router queue not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        if (
            reader.fieldnames is None
            or "cluster_id"
            not in reader.fieldnames
        ):
            raise ValueError(
                "Router queue must contain "
                "a 'cluster_id' column."
            )

        cluster_ids: list[str] = []

        for row in reader:
            cluster_id = str(
                row.get(
                    "cluster_id",
                    "",
                )
            ).strip()

            if not cluster_id:
                continue

            if cluster_id not in cluster_ids:
                cluster_ids.append(
                    cluster_id
                )

    if not cluster_ids:
        raise ValueError(
            "Router queue contains no clusters."
        )

    return cluster_ids


def write_manifest(
    output_root: Path,
    payload: dict[str, Any],
) -> None:
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        output_root
        / "agent3_batch_manifest.json"
    )

    csv_path = (
        output_root
        / "agent3_batch_manifest.csv"
    )

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    rows = payload.get(
        "clusters",
        [],
    )

    fieldnames = [
        "cluster_id",
        "status",
        "elapsed_seconds",
        "agent2_json",
        "output_root",
        "decision_json",
        "command",
        "error",
    ]

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: row.get(
                        key,
                        "",
                    )
                    for key in fieldnames
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run standardized Agent 3 for "
            "clusters selected by the Router."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--queue-csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--agent2-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--disease-context",
        required=True,
    )

    parser.add_argument(
        "--candidate-count",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--context-page-size",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--rescue-page-size",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--confirmation-page-size",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--top-k-passages",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--top-marker-count",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--rescue-top-marker-limit",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--cell-ontology",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    queue_csv = resolve_project_path(
        args.queue_csv
    )

    agent2_root = resolve_project_path(
        args.agent2_root
    )

    output_root = resolve_project_path(
        args.output_root
    )

    cell_ontology = resolve_project_path(
        args.cell_ontology
    )

    runner = (
        PROJECT_ROOT
        / "scripts"
        / "agent3_validation"
        / "run_agent3_standardized_full_with_decision.py"
    )

    if not runner.exists():
        raise FileNotFoundError(
            f"Agent 3 runner not found: {runner}"
        )

    if not cell_ontology.exists():
        raise FileNotFoundError(
            f"Cell Ontology file not found: "
            f"{cell_ontology}"
        )

    cluster_ids = read_router_queue(
        queue_csv
    )

    print("=" * 90)
    print(
        "ROUTER-DRIVEN STANDARDIZED AGENT 3"
    )
    print("=" * 90)

    print(
        f"Dataset: {args.dataset}"
    )

    print(
        f"Disease context: "
        f"{args.disease_context}"
    )

    print(
        "Router-selected clusters: "
        + ", ".join(cluster_ids)
    )

    print(
        f"Cluster count: "
        f"{len(cluster_ids)}"
    )

    print(
        f"Output root: {output_root}"
    )

    print(
        f"Dry run: {args.dry_run}"
    )

    print("=" * 90)

    manifest: dict[str, Any] = {
        "dataset": args.dataset,
        "disease_context": (
            args.disease_context
        ),
        "queue_csv": str(
            queue_csv
        ),
        "agent2_root": str(
            agent2_root
        ),
        "output_root": str(
            output_root
        ),
        "cell_ontology": str(
            cell_ontology
        ),
        "candidate_count": (
            args.candidate_count
        ),
        "context_page_size": (
            args.context_page_size
        ),
        "rescue_page_size": (
            args.rescue_page_size
        ),
        "confirmation_page_size": (
            args.confirmation_page_size
        ),
        "top_k_passages": (
            args.top_k_passages
        ),
        "top_marker_count": (
            args.top_marker_count
        ),
        "rescue_top_marker_limit": (
            args.rescue_top_marker_limit
        ),
        "wait_seconds": (
            args.wait_seconds
        ),
        "dry_run": (
            args.dry_run
        ),
        "started_at_utc": (
            utc_now()
        ),
        "router_selected_clusters": (
            cluster_ids
        ),
        "cluster_count": (
            len(cluster_ids)
        ),
        "reference_labels_used": False,
        "expert_labels_used": False,
        "clusters": [],
    }

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index, cluster_id in enumerate(
        cluster_ids
    ):
        cluster_output = (
            output_root
            / f"cluster_{cluster_id}"
        )

        agent2_json = (
            agent2_root
            / "clusters"
            / f"cluster_{cluster_id}"
            / (
                f"cluster_{cluster_id}"
                "_marker_evidence.json"
            )
        )

        decision_json = (
            cluster_output
            / "result"
            / (
                f"cluster_{cluster_id}"
                "_agent3_decision_summary.json"
            )
        )

        if not agent2_json.exists():
            raise FileNotFoundError(
                "Agent 2 cluster evidence "
                f"not found: {agent2_json}"
            )

        command = [
            sys.executable,
            "-u",
            str(runner),
            "--dataset",
            args.dataset,
            "--cluster-id",
            cluster_id,
            "--disease-context",
            args.disease_context,
            "--agent2-json",
            str(agent2_json),
            "--output-root",
            str(cluster_output),
            "--candidate-count",
            str(
                args.candidate_count
            ),
            "--context-page-size",
            str(
                args.context_page_size
            ),
            "--rescue-page-size",
            str(
                args.rescue_page_size
            ),
            "--confirmation-page-size",
            str(
                args.confirmation_page_size
            ),
            "--top-k-passages",
            str(
                args.top_k_passages
            ),
            "--top-marker-count",
            str(
                args.top_marker_count
            ),
            "--rescue-top-marker-limit",
            str(
                args.rescue_top_marker_limit
            ),
            "--cell-ontology",
            str(cell_ontology),
        ]

        command_text = (
            subprocess.list2cmdline(
                command
            )
        )

        row: dict[str, Any] = {
            "cluster_id": cluster_id,
            "status": "",
            "elapsed_seconds": "",
            "agent2_json": str(
                agent2_json
            ),
            "output_root": str(
                cluster_output
            ),
            "decision_json": str(
                decision_json
            ),
            "command": command_text,
            "error": "",
        }

        print()
        print("=" * 90)
        print(
            f"CLUSTER {cluster_id}"
        )
        print("=" * 90)

        if args.dry_run:
            row["status"] = "PLANNED"

            print("PLANNED:")
            print(command_text)

            manifest["clusters"].append(
                row
            )

            write_manifest(
                output_root,
                manifest,
            )

            continue

        if decision_json.exists():
            row["status"] = (
                "SKIPPED_COMPLETE"
            )

            print(
                "Existing completed Agent 3 "
                "decision found; skipping."
            )

            manifest["clusters"].append(
                row
            )

            write_manifest(
                output_root,
                manifest,
            )

            continue

        if cluster_output.exists():
            existing_items = list(
                cluster_output.iterdir()
            )

            if existing_items:
                row["status"] = (
                    "BLOCKED_PARTIAL_OUTPUT"
                )

                row["error"] = (
                    "Cluster output directory "
                    "already contains files but "
                    "no final decision summary "
                    "was found."
                )

                manifest["clusters"].append(
                    row
                )

                manifest[
                    "completed_at_utc"
                ] = utc_now()

                write_manifest(
                    output_root,
                    manifest,
                )

                raise RuntimeError(
                    row["error"]
                    + "\nDirectory: "
                    + str(cluster_output)
                )

        print("RUNNING:")
        print(command_text)

        started = (
            time.perf_counter()
        )

        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        row["elapsed_seconds"] = round(
            elapsed,
            3,
        )

        if completed.returncode != 0:
            row["status"] = "FAILED"

            row["error"] = (
                "Agent 3 returned non-zero "
                f"exit code "
                f"{completed.returncode}."
            )

            manifest["clusters"].append(
                row
            )

            manifest[
                "completed_at_utc"
            ] = utc_now()

            write_manifest(
                output_root,
                manifest,
            )

            return (
                completed.returncode
            )

        if not decision_json.exists():
            row["status"] = "FAILED"

            row["error"] = (
                "Agent 3 process completed "
                "but final decision summary "
                "was not produced."
            )

            manifest["clusters"].append(
                row
            )

            manifest[
                "completed_at_utc"
            ] = utc_now()

            write_manifest(
                output_root,
                manifest,
            )

            return 1

        row["status"] = "COMPLETE"

        manifest["clusters"].append(
            row
        )

        write_manifest(
            output_root,
            manifest,
        )

        print(
            f"Completed cluster "
            f"{cluster_id}."
        )

        remaining = (
            index
            < len(cluster_ids) - 1
        )

        if (
            remaining
            and args.wait_seconds > 0
        ):
            print(
                f"Waiting "
                f"{args.wait_seconds} "
                "seconds before the "
                "next cluster..."
            )

            time.sleep(
                args.wait_seconds
            )

    manifest[
        "completed_at_utc"
    ] = utc_now()

    completed_clusters = [
        row["cluster_id"]
        for row in manifest["clusters"]
        if row["status"]
        == "COMPLETE"
    ]

    skipped_clusters = [
        row["cluster_id"]
        for row in manifest["clusters"]
        if row["status"]
        == "SKIPPED_COMPLETE"
    ]

    manifest[
        "completed_clusters"
    ] = completed_clusters

    manifest[
        "skipped_complete_clusters"
    ] = skipped_clusters

    write_manifest(
        output_root,
        manifest,
    )

    print()
    print("=" * 90)
    print(
        "ROUTER-DRIVEN AGENT 3 COMPLETE"
        if not args.dry_run
        else
        "ROUTER-DRIVEN AGENT 3 DRY RUN COMPLETE"
    )
    print("=" * 90)

    print(
        "Clusters: "
        + ", ".join(cluster_ids)
    )

    print(
        "Manifest: "
        + str(
            output_root
            / "agent3_batch_manifest.json"
        )
    )

    print("=" * 90)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )