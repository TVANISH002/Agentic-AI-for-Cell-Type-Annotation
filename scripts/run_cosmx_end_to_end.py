from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any



DATASET = "cosmx_brcr_validated_clean"

SOURCE_PROFILE = Path(
    "configs/orchestrator/"
    "cosmx_brcr_context_validated_clean.json"
)

SOURCE_ROUTER_CONFIG = Path(
    "configs/router/"
    "cosmx_brcr_router_v4_orchestration_validation.json"
)

FROZEN_BASELINE_ROOT = Path(
    "data/pipeline_runs/"
    "cosmx_brcr_orchestration_validation"
)

RUNS_PARENT = Path(
    "data/pipeline_runs/"
    "cosmx_brcr_end_to_end_runs"
)

CELL_ONTOLOGY = Path(
    "data/resources/cell_ontology/"
    "v2026-06-08/cl-base.json"
)

DISEASE_CONTEXT = "breast cancer"



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Non-destructive CosMX end-to-end runner: "
            "Agent 1 -> Agent 2 -> Router -> Agent 3."
        )
    )

    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Optional unique run ID. "
            "If omitted, UTC timestamp is used."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Create and validate isolated run configuration "
            "without executing Agent 1/2/Router/Agent 3."
        ),
    )

    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=120,
        help=(
            "Wait time between Agent 3 cluster executions."
        ),
    )

    return parser.parse_args()



def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def read_json(path: Path) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def write_json(
    path: Path,
    payload: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:
        while True:
            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(
                chunk
            )

    return h.hexdigest()



def replace_root_in_object(
    value: Any,
    old_path: Path,
    new_path: Path,
) -> Any:
    """
    Recursively replace an existing path inside strings
    contained in JSON objects.

    This operates only on an in-memory copy.
    Original config files are never modified.
    """

    if isinstance(
        value,
        dict,
    ):
        return {
            key: replace_root_in_object(
                item,
                old_path,
                new_path,
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        list,
    ):
        return [
            replace_root_in_object(
                item,
                old_path,
                new_path,
            )
            for item in value
        ]

    if isinstance(
        value,
        str,
    ):
        old_forward = str(
            old_path
        ).replace(
            "\\",
            "/",
        )

        new_forward = str(
            new_path
        ).replace(
            "\\",
            "/",
        )

        old_backward = str(
            old_path
        ).replace(
            "/",
            "\\",
        )

        new_backward = str(
            new_path
        ).replace(
            "/",
            "\\",
        )

        updated = value.replace(
            old_forward,
            new_forward,
        )

        updated = updated.replace(
            old_backward,
            new_backward,
        )

        return updated

    return value



def run_command(
    command: list[str],
    env: dict[str, str],
    label: str,
) -> None:

    print()
    print("=" * 90)
    print(label)
    print("=" * 90)

    print(
        subprocess.list2cmdline(
            command
        )
    )

    result = subprocess.run(
        command,
        env=env,
        check=False,
    )

    if result.returncode != 0:
        raise SystemExit(
            f"\n{label} failed with "
            f"exit code {result.returncode}."
        )



def extract_possible_labels(
    payload: dict[str, Any],
) -> list[str]:

    labels = payload.get(
        "possible_labels",
        [],
    )

    result: list[str] = []

    if not isinstance(
        labels,
        list,
    ):
        return result

    for item in labels:

        if isinstance(
            item,
            str,
        ):
            result.append(
                item
            )

            continue

        if isinstance(
            item,
            dict,
        ):
            for key in (
                "label",
                "cell_type",
                "candidate_cell_type",
            ):
                value = item.get(
                    key
                )

                if value:
                    result.append(
                        str(
                            value
                        )
                    )
                    break

    return result


def collect_agent3_results(
    agent3_root: Path,
) -> list[dict[str, Any]]:

    rows: list[
        dict[str, Any]
    ] = []

    if not agent3_root.exists():
        return rows

    cluster_dirs = [
        path
        for path in agent3_root.glob(
            "cluster_*"
        )
        if path.is_dir()
    ]

    def cluster_number(
        path: Path,
    ) -> int:

        try:
            return int(
                path.name.replace(
                    "cluster_",
                    "",
                )
            )

        except ValueError:
            return 999999

    for cluster_dir in sorted(
        cluster_dirs,
        key=cluster_number,
    ):

        cluster_id = (
            cluster_dir.name.replace(
                "cluster_",
                "",
            )
        )

        decision_file = (
            cluster_dir
            / "result"
            / (
                f"cluster_{cluster_id}"
                "_agent3_decision_summary.json"
            )
        )

        if not decision_file.exists():

            rows.append(
                {
                    "cluster_id": cluster_id,
                    "decision_state": "MISSING",
                    "ambiguity_subtype": "",
                    "possible_labels": "",
                    "possible_label_count": "",
                    "reference_label_used": "",
                    "expert_label_used": "",
                    "human_review_required": "",
                }
            )

            continue

        payload = read_json(
            decision_file
        )

        possible_labels = (
            extract_possible_labels(
                payload
            )
        )

        decision_state = str(
            payload.get(
                "decision_state",
                "",
            )
        )

        rows.append(
            {
                "cluster_id": cluster_id,
                "decision_state": decision_state,
                "ambiguity_subtype": (
                    payload.get(
                        "ambiguity_subtype"
                    )
                    or ""
                ),
                "possible_labels": (
                    "; ".join(
                        possible_labels
                    )
                ),
                "possible_label_count": (
                    payload.get(
                        "possible_label_count",
                        len(
                            possible_labels
                        ),
                    )
                ),
                "reference_label_used": (
                    payload.get(
                        "reference_label_used",
                        False,
                    )
                ),
                "expert_label_used": (
                    payload.get(
                        "expert_label_used",
                        False,
                    )
                ),
                "human_review_required": (
                    decision_state
                    == "UNRESOLVED"
                ),
            }
        )

    return rows


def write_summary_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "cluster_id",
        "decision_state",
        "ambiguity_subtype",
        "possible_labels",
        "possible_label_count",
        "reference_label_used",
        "expert_label_used",
        "human_review_required",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )



def main() -> None:

    args = parse_args()

    project_root = (
        Path.cwd().resolve()
    )



    if not SOURCE_PROFILE.exists():
        raise SystemExit(
            "Missing source orchestrator profile:\n"
            f"{SOURCE_PROFILE}"
        )

    if not SOURCE_ROUTER_CONFIG.exists():
        raise SystemExit(
            "Missing source Router config:\n"
            f"{SOURCE_ROUTER_CONFIG}"
        )

    if not CELL_ONTOLOGY.exists():
        raise SystemExit(
            "Missing Cell Ontology:\n"
            f"{CELL_ONTOLOGY}"
        )



    run_id = (
        args.run_id
        or datetime.now(
            timezone.utc
        ).strftime(
            "cosmx_%Y%m%d_%H%M%S"
        )
    )

    run_root = (
        RUNS_PARENT
        / run_id
    )

    # Never allow the wrapper to target the frozen baseline.
    if (
        run_root.resolve()
        == FROZEN_BASELINE_ROOT.resolve()
    ):
        raise SystemExit(
            "\nSAFETY STOP\n"
            "The frozen validation baseline cannot "
            "be used as a new run directory."
        )

    # Never overwrite a previous end-to-end run.
    if run_root.exists():
        raise SystemExit(
            "\nSAFETY STOP\n"
            "Run directory already exists:\n"
            f"{run_root}\n\n"
            "Nothing was overwritten."
        )

    run_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    manifests_dir = (
        run_root
        / "manifests"
    )

    manifests_dir.mkdir(
        parents=True,
        exist_ok=True,
    )



    original_router_config = (
        read_json(
            SOURCE_ROUTER_CONFIG
        )
    )

    run_router_config = (
        replace_root_in_object(
            original_router_config,
            FROZEN_BASELINE_ROOT,
            run_root,
        )
    )

    run_router_config_path = (
        run_root
        / "run_router_config.json"
    )

    write_json(
        run_router_config_path,
        run_router_config,
    )



    original_profile = read_json(
        SOURCE_PROFILE
    )

    # Replace frozen validation root with
    # this isolated end-to-end run root.
    run_profile = (
        replace_root_in_object(
            original_profile,
            FROZEN_BASELINE_ROOT,
            run_root,
        )
    )

    # Replace original Router config path with
    # this run's Router config copy.
    run_profile = (
        replace_root_in_object(
            run_profile,
            SOURCE_ROUTER_CONFIG,
            run_router_config_path,
        )
    )

    run_profile_path = (
        run_root
        / "run_context_profile.json"
    )

    write_json(
        run_profile_path,
        run_profile,
    )



    agent1_root = (
        run_root
        / "agent1_sources"
    )

    agent2_root = (
        run_root
        / "agent2_output"
    )

    router_root = (
        run_root
        / "router_output_v4"
    )

    queue_csv = (
        router_root
        / "agent3_queue_if_no_expert.csv"
    )

    agent3_root = (
        run_root
        / "agent3_primary"
    )

    final_root = (
        run_root
        / "final"
    )


    manifest_path = (
        manifests_dir
        / "cosmx_end_to_end_run_manifest.json"
    )

    manifest = {
        "schema_name": (
            "CosMXEndToEndRunManifest"
        ),
        "schema_version": "1.1",
        "dataset": DATASET,
        "created_at_utc": utc_now(),
        "run_id": run_id,
        "run_root": str(
            run_root
        ),
        "source_context_profile": str(
            SOURCE_PROFILE
        ),
        "source_context_profile_sha256": (
            sha256_file(
                SOURCE_PROFILE
            )
        ),
        "source_router_config": str(
            SOURCE_ROUTER_CONFIG
        ),
        "source_router_config_sha256": (
            sha256_file(
                SOURCE_ROUTER_CONFIG
            )
        ),
        "run_context_profile": str(
            run_profile_path
        ),
        "run_context_profile_sha256": (
            sha256_file(
                run_profile_path
            )
        ),
        "run_router_config": str(
            run_router_config_path
        ),
        "run_router_config_sha256": (
            sha256_file(
                run_router_config_path
            )
        ),
        "frozen_baseline_root": str(
            FROZEN_BASELINE_ROOT
        ),
        "frozen_baseline_modified": False,
        "reference_labels_allowed_in_inference": False,
        "expert_labels_allowed_in_inference": False,
        "agent3_routing_policy": (
            "ROUTER_SELECTED_NO_EXPERT_POLICY"
        ),
        "supplemental_audit_automatic": False,
        "supplemental_audit_policy": (
            "HUMAN_MANUAL_INVOCATION_ONLY"
        ),
        "dry_run": args.dry_run,
    }

    write_json(
        manifest_path,
        manifest,
    )


    env = os.environ.copy()

    src_path = str(
        (
            project_root
            / "src"
        ).resolve()
    )

    previous_pythonpath = (
        env.get(
            "PYTHONPATH",
            "",
        )
    )

    if previous_pythonpath:
        env["PYTHONPATH"] = (
            src_path
            + os.pathsep
            + previous_pythonpath
        )

    else:
        env["PYTHONPATH"] = (
            src_path
        )



    if args.dry_run:

        print()
        print(
            "=============================================="
        )
        print(
            "COSMX END-TO-END DRY RUN"
        )
        print(
            "=============================================="
        )

        print(
            f"Fresh run root: "
            f"{run_root}"
        )

        print(
            f"Copied profile: "
            f"{run_profile_path}"
        )

        print(
            f"Copied Router config: "
            f"{run_router_config_path}"
        )

        print(
            f"Expected Agent 1 root: "
            f"{agent1_root}"
        )

        print(
            f"Expected Agent 2 root: "
            f"{agent2_root}"
        )

        print(
            f"Expected Router root: "
            f"{router_root}"
        )

        print(
            f"Expected Router queue: "
            f"{queue_csv}"
        )

        print(
            f"Expected Agent 3 root: "
            f"{agent3_root}"
        )

        print()
        print(
            "Frozen validation baseline untouched:"
        )

        print(
            FROZEN_BASELINE_ROOT
        )

        print()
        print(
            "Supplemental audit automatic: False"
        )

        print(
            "STATUS: DRY_RUN_PASSED"
        )

        return



    orchestrator_command = [
        sys.executable,
        "-m",
        "annotation_orchestrator.runners.run_orchestrator",
        "--context-profile",
        str(
            run_profile_path
        ),
        "--project-root",
        ".",
        "--execute-stage-commands",
    ]

    # IMPORTANT:
    # NO --force
    # NO overwrite
    # NO expert/reference labels
    run_command(
        orchestrator_command,
        env,
        (
            "COSMX PIPELINE: "
            "AGENT 1 -> AGENT 2 -> ROUTER"
        ),
    )




    if not agent2_root.exists():
        raise SystemExit(
            "\nSAFETY STOP\n"
            "Agent 2 output directory was not created:\n"
            f"{agent2_root}"
        )



    expected_router_outputs = [
        router_root
        / "router_decisions.csv",

        router_root
        / "router_summary.json",

        router_root
        / "agent3_queue.csv",

        router_root
        / "monitor_queue.csv",

        router_root
        / "agent3_queue_if_no_expert.csv",
    ]

    missing_router_outputs = [
        str(path)
        for path in expected_router_outputs
        if not path.exists()
    ]

    if missing_router_outputs:

        print()
        print(
            "Router output verification failed."
        )

        print(
            "Missing:"
        )

        for missing in (
            missing_router_outputs
        ):
            print(
                f"  - {missing}"
            )

        raise SystemExit(
            "\nSAFETY STOP\n"
            "Router stage did not produce all "
            "expected isolated outputs.\n"
            "Agent 3 was NOT started."
        )



    agent3_command = [
        sys.executable,
        (
            "scripts/agent3_validation/"
            "run_agent3_from_router_queue.py"
        ),
        "--dataset",
        DATASET,
        "--queue-csv",
        str(
            queue_csv
        ),
        "--agent2-root",
        str(
            agent2_root
        ),
        "--output-root",
        str(
            agent3_root
        ),
        "--disease-context",
        DISEASE_CONTEXT,
        "--cell-ontology",
        str(
            CELL_ONTOLOGY
        ),
        "--wait-seconds",
        str(
            args.wait_seconds
        ),
    ]

    run_command(
        agent3_command,
        env,
        (
            "COSMX PIPELINE: "
            "ROUTER-SELECTED AGENT 3"
        ),
    )


    result_rows = (
        collect_agent3_results(
            agent3_root
        )
    )

    summary_path = (
        final_root
        / "cosmx_agent3_final_summary.csv"
    )

    write_summary_csv(
        summary_path,
        result_rows,
    )



    unresolved_rows = [
        row
        for row in result_rows
        if row[
            "decision_state"
        ]
        == "UNRESOLVED"
    ]

    unresolved_clusters = [
        row[
            "cluster_id"
        ]
        for row in unresolved_rows
    ]


    final_manifest = {
        **manifest,
        "completed_at_utc": (
            utc_now()
        ),
        "status": (
            "COMPLETE_AGENT1_TO_AGENT3"
        ),
        "agent1_completed": (
            agent1_root.exists()
        ),
        "agent2_completed": (
            agent2_root.exists()
        ),
        "router_completed": True,
        "agent3_completed": True,
        "agent3_cluster_count": len(
            result_rows
        ),
        "unresolved_cluster_count": len(
            unresolved_clusters
        ),
        "unresolved_clusters": (
            unresolved_clusters
        ),
        "human_review_required": (
            len(
                unresolved_clusters
            ) > 0
        ),
        "supplemental_audit_executed": False,
        "final_agent3_summary": str(
            summary_path
        ),
        "frozen_baseline_modified": False,
    }

    write_json(
        manifest_path,
        final_manifest,
    )



    print()
    print(
        "=============================================="
    )
    print(
        "COSMX END-TO-END PIPELINE COMPLETE"
    )
    print(
        "=============================================="
    )

    print(
        f"Dataset: {DATASET}"
    )

    print(
        f"Run ID: {run_id}"
    )

    print(
        f"Run root: {run_root}"
    )

    print()
    print(
        "Agent 1: COMPLETE"
    )

    print(
        "Agent 2: COMPLETE"
    )

    print(
        "Router V4: COMPLETE"
    )

    print(
        "Router-selected Agent 3: COMPLETE"
    )

    print()
    print(
        "Agent 3 cluster count:",
        len(
            result_rows
        ),
    )

    print(
        "UNRESOLVED cluster count:",
        len(
            unresolved_clusters
        ),
    )

    if unresolved_clusters:

        print()
        print(
            "HUMAN REVIEW REQUIRED:"
        )

        for cluster_id in (
            unresolved_clusters
        ):
            print(
                f"  - Cluster {cluster_id}"
            )

        print()
        print(
            "Supplemental unresolved audit "
            "was NOT automatically executed."
        )

    else:

        print()
        print(
            "No Agent 3 UNRESOLVED "
            "clusters were detected."
        )

    print()
    print(
        "Final Agent 3 summary:"
    )

    print(
        summary_path
    )

    print()
    print(
        "Run manifest:"
    )

    print(
        manifest_path
    )

    print()
    print(
        "Frozen validation baseline:"
    )

    print(
        FROZEN_BASELINE_ROOT
    )

    print(
        "UNCHANGED BY THIS WRAPPER: True"
    )

    print()
    print(
        "STATUS: "
        "COSMX_END_TO_END_COMPLETE"
    )


if __name__ == "__main__":
    main()