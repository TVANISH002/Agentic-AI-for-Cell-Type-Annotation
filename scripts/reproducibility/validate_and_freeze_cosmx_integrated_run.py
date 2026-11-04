from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ============================================================
# PATHS
# ============================================================

DATASET = "cosmx_brcr_validated_clean"

BASELINE_ROOT = Path(
    "data/pipeline_runs/"
    "cosmx_brcr_orchestration_validation"
)

BASELINE_HASH_MANIFEST = (
    BASELINE_ROOT
    / "final_freeze"
    / "cosmx_final_file_manifest_sha256.csv"
)

INTEGRATED_RUNS_ROOT = Path(
    "data/pipeline_runs/"
    "cosmx_brcr_end_to_end_runs"
)


# ============================================================
# BASIC UTILITIES
# ============================================================

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def read_json(
    path: Path,
) -> Any:
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


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def sha256_file(
    path: Path,
) -> str:
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


def normalize(
    value: Any,
) -> str:
    if value is None:
        return ""

    return " ".join(
        str(value)
        .strip()
        .split()
    )


def normalize_label_list(
    values: list[str],
) -> list[str]:
    return [
        normalize(value)
        for value in values
        if normalize(value)
    ]


# ============================================================
# FIND LATEST SUCCESSFUL INTEGRATED RUN
# ============================================================

def find_latest_completed_run() -> Path:

    if not INTEGRATED_RUNS_ROOT.exists():
        raise SystemExit(
            "Integrated-run directory does not exist:\n"
            f"{INTEGRATED_RUNS_ROOT}"
        )

    candidates: list[Path] = []

    for run_dir in (
        INTEGRATED_RUNS_ROOT.iterdir()
    ):
        if not run_dir.is_dir():
            continue

        manifest = (
            run_dir
            / "manifests"
            / "cosmx_end_to_end_run_manifest.json"
        )

        if not manifest.exists():
            continue

        try:
            payload = read_json(
                manifest
            )
        except Exception:
            continue

        if (
            payload.get("status")
            == "COMPLETE_AGENT1_TO_AGENT3"
        ):
            candidates.append(
                run_dir
            )

    if not candidates:
        raise SystemExit(
            "No completed CosMX integrated run was found."
        )

    return max(
        candidates,
        key=lambda p: p.stat().st_mtime,
    )


# ============================================================
# STEP 1
# VERIFY ORIGINAL 329-FILE FROZEN BASELINE
# ============================================================

def verify_baseline_integrity() -> dict[str, Any]:

    if not BASELINE_HASH_MANIFEST.exists():
        raise SystemExit(
            "Frozen baseline SHA256 manifest not found:\n"
            f"{BASELINE_HASH_MANIFEST}"
        )

    rows = read_csv(
        BASELINE_HASH_MANIFEST
    )

    failures: list[
        dict[str, str]
    ] = []

    checked = 0

    for row in rows:

        raw_path = row.get(
            "relative_path",
            "",
        )

        expected_hash = (
            row.get(
                "sha256",
                "",
            )
            .strip()
            .lower()
        )

        if not raw_path:
            failures.append(
                {
                    "status": "INVALID_MANIFEST_ROW",
                    "path": "",
                }
            )
            continue

        path = Path(
            raw_path
        )

        if not path.exists():

            failures.append(
                {
                    "status": "MISSING",
                    "path": str(path),
                }
            )

            continue

        actual_hash = (
            sha256_file(
                path
            )
            .lower()
        )

        checked += 1

        if actual_hash != expected_hash:

            failures.append(
                {
                    "status": "HASH_CHANGED",
                    "path": str(path),
                }
            )

    return {
        "manifest_row_count": len(
            rows
        ),
        "files_checked": checked,
        "failure_count": len(
            failures
        ),
        "failures": failures,
        "passed": (
            len(failures)
            == 0
            and checked
            == len(rows)
        ),
    }


# ============================================================
# AGENT 2 TOP CALLS
# ============================================================

CANDIDATE_COLUMNS = (
    "candidate_cell_type",
    "cell_type",
    "candidate",
    "label",
    "candidate_label",
)


def detect_column(
    row: dict[str, Any],
    candidates: tuple[str, ...],
) -> str | None:

    normalized_columns = {
        key.strip().lower(): key
        for key in row.keys()
        if key is not None
    }

    for candidate in candidates:

        if candidate.lower() in (
            normalized_columns
        ):
            return normalized_columns[
                candidate.lower()
            ]

    return None


def load_agent2_top_calls(
    run_root: Path,
) -> dict[str, str]:

    clusters_root = (
        run_root
        / "agent2_output"
        / "clusters"
    )

    if not clusters_root.exists():
        raise FileNotFoundError(
            f"Agent 2 clusters directory missing: "
            f"{clusters_root}"
        )

    result: dict[
        str,
        str,
    ] = {}

    cluster_dirs = sorted(
        [
            path
            for path in (
                clusters_root.glob(
                    "cluster_*"
                )
            )
            if path.is_dir()
        ],
        key=lambda p: int(
            p.name.replace(
                "cluster_",
                "",
            )
        ),
    )

    for cluster_dir in cluster_dirs:

        cluster_id = (
            cluster_dir.name.replace(
                "cluster_",
                "",
            )
        )

        ranked_file = (
            cluster_dir
            / (
                f"cluster_{cluster_id}"
                "_ranked_candidate_cell_types.csv"
            )
        )

        if not ranked_file.exists():
            raise FileNotFoundError(
                "Missing Agent 2 ranked candidates:\n"
                f"{ranked_file}"
            )

        rows = read_csv(
            ranked_file
        )

        if not rows:
            raise RuntimeError(
                "Agent 2 ranked candidate file is empty:\n"
                f"{ranked_file}"
            )

        candidate_column = (
            detect_column(
                rows[0],
                CANDIDATE_COLUMNS,
            )
        )

        if candidate_column is None:
            raise RuntimeError(
                "Could not identify candidate column in:\n"
                f"{ranked_file}\n"
                f"Columns: {list(rows[0].keys())}"
            )

        top_call = normalize(
            rows[0].get(
                candidate_column,
                "",
            )
        )

        result[
            cluster_id
        ] = top_call

    return result


# ============================================================
# ROUTER DECISIONS
# ============================================================

CLUSTER_COLUMNS = (
    "cluster_id",
    "cluster",
)

ROUTER_DECISION_COLUMNS = (
    "router_decision",
    "decision",
    "routing_decision",
    "route_decision",
)


def load_router_decisions(
    run_root: Path,
) -> dict[str, str]:

    path = (
        run_root
        / "router_output_v4"
        / "router_decisions.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Router decision file missing: {path}"
        )

    rows = read_csv(
        path
    )

    if not rows:
        raise RuntimeError(
            f"Router decision file is empty: {path}"
        )

    cluster_column = detect_column(
        rows[0],
        CLUSTER_COLUMNS,
    )

    decision_column = detect_column(
        rows[0],
        ROUTER_DECISION_COLUMNS,
    )

    if cluster_column is None:
        raise RuntimeError(
            "Could not determine Router cluster column.\n"
            f"Columns: {list(rows[0].keys())}"
        )

    if decision_column is None:
        raise RuntimeError(
            "Could not determine Router decision column.\n"
            f"Columns: {list(rows[0].keys())}"
        )

    return {
        normalize(
            row.get(
                cluster_column,
                "",
            )
        ):
        normalize(
            row.get(
                decision_column,
                "",
            )
        )
        for row in rows
    }


# ============================================================
# ROUTER AGENT 3 QUEUE
# ============================================================

def load_agent3_queue(
    run_root: Path,
) -> list[dict[str, str]]:

    path = (
        run_root
        / "router_output_v4"
        / "agent3_queue_if_no_expert.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Agent 3 queue missing: {path}"
        )

    rows = read_csv(
        path
    )

    selected_fields = (
        "cluster_id",
        "leading_candidate",
        "top_supporting_gene_count",
        "second_supporting_gene_count",
        "support_gap",
        "support_ratio",
    )

    cleaned: list[
        dict[str, str]
    ] = []

    for row in rows:

        item: dict[
            str,
            str,
        ] = {}

        for field in selected_fields:

            if field in row:
                item[
                    field
                ] = normalize(
                    row.get(
                        field,
                        "",
                    )
                )

        cleaned.append(
            item
        )

    return cleaned


# ============================================================
# AGENT 3 DECISIONS
# ============================================================

def extract_possible_labels(
    payload: dict[str, Any],
) -> list[str]:

    raw = payload.get(
        "possible_labels",
        [],
    )

    if not isinstance(
        raw,
        list,
    ):
        return []

    labels: list[
        str
    ] = []

    for item in raw:

        if isinstance(
            item,
            str,
        ):
            labels.append(
                normalize(
                    item
                )
            )

        elif isinstance(
            item,
            dict,
        ):

            for key in (
                "label",
                "cell_type",
                "candidate_cell_type",
            ):

                if item.get(key):

                    labels.append(
                        normalize(
                            item[key]
                        )
                    )

                    break

    return normalize_label_list(
        labels
    )


def load_agent3_decisions(
    run_root: Path,
) -> dict[str, dict[str, Any]]:

    root = (
        run_root
        / "agent3_primary"
    )

    if not root.exists():
        raise FileNotFoundError(
            f"Agent 3 directory missing: {root}"
        )

    result: dict[
        str,
        dict[str, Any],
    ] = {}

    cluster_dirs = sorted(
        [
            path
            for path in root.glob(
                "cluster_*"
            )
            if path.is_dir()
        ],
        key=lambda p: int(
            p.name.replace(
                "cluster_",
                "",
            )
        ),
    )

    for cluster_dir in cluster_dirs:

        cluster_id = (
            cluster_dir.name.replace(
                "cluster_",
                "",
            )
        )

        decision_path = (
            cluster_dir
            / "result"
            / (
                f"cluster_{cluster_id}"
                "_agent3_decision_summary.json"
            )
        )

        if not decision_path.exists():
            raise FileNotFoundError(
                "Agent 3 decision missing:\n"
                f"{decision_path}"
            )

        payload = read_json(
            decision_path
        )

        result[
            cluster_id
        ] = {
            "decision_state": normalize(
                payload.get(
                    "decision_state",
                    "",
                )
            ),
            "ambiguity_subtype": normalize(
                payload.get(
                    "ambiguity_subtype",
                    "",
                )
            ),
            "possible_labels": (
                extract_possible_labels(
                    payload
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
        }

    return result


# ============================================================
# COMPARISON HELPERS
# ============================================================

def compare_mapping(
    category: str,
    baseline: dict[str, Any],
    integrated: dict[str, Any],
) -> list[dict[str, Any]]:

    rows: list[
        dict[str, Any]
    ] = []

    all_keys = sorted(
        set(
            baseline
        )
        | set(
            integrated
        ),
        key=lambda x: (
            int(x)
            if str(x).isdigit()
            else str(x)
        ),
    )

    for key in all_keys:

        baseline_value = (
            baseline.get(
                key,
                "<MISSING>",
            )
        )

        integrated_value = (
            integrated.get(
                key,
                "<MISSING>",
            )
        )

        rows.append(
            {
                "category": category,
                "item": (
                    f"cluster_{key}"
                ),
                "baseline_value": (
                    json.dumps(
                        baseline_value,
                        sort_keys=True,
                    )
                    if isinstance(
                        baseline_value,
                        (
                            dict,
                            list,
                        ),
                    )
                    else str(
                        baseline_value
                    )
                ),
                "integrated_value": (
                    json.dumps(
                        integrated_value,
                        sort_keys=True,
                    )
                    if isinstance(
                        integrated_value,
                        (
                            dict,
                            list,
                        ),
                    )
                    else str(
                        integrated_value
                    )
                ),
                "match": (
                    baseline_value
                    == integrated_value
                ),
            }
        )

    return rows


# ============================================================
# FREEZE NEW INTEGRATED RUN
# ============================================================

def build_integrated_hash_manifest(
    run_root: Path,
    freeze_root: Path,
    validation_root: Path,
) -> list[dict[str, Any]]:

    rows: list[
        dict[str, Any]
    ] = []

    files = sorted(
        [
            path
            for path in (
                run_root.rglob("*")
            )
            if path.is_file()
        ],
        key=lambda p: str(
            p
        ).lower(),
    )

    for path in files:

        # Do not hash the freeze output into itself.
        if (
            freeze_root
            in path.parents
        ):
            continue

        # Do not include generated comparison
        # reports in the scientific-run hash set.
        if (
            validation_root
            in path.parents
        ):
            continue

        rows.append(
            {
                "relative_path": str(
                    path
                ),
                "size_bytes": (
                    path.stat().st_size
                ),
                "sha256": (
                    sha256_file(
                        path
                    )
                ),
            }
        )

    return rows


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print(
        "=" * 76
    )
    print(
        "COSMX INTEGRATED RUN REGRESSION VALIDATION"
    )
    print(
        "=" * 76
    )

    # --------------------------------------------------------
    # FIND SUCCESSFUL NEW RUN
    # --------------------------------------------------------

    integrated_root = (
        find_latest_completed_run()
    )

    print(
        f"Baseline:   {BASELINE_ROOT}"
    )

    print(
        f"Integrated: {integrated_root}"
    )

    # --------------------------------------------------------
    # VERIFY FROZEN BASELINE FIRST
    # --------------------------------------------------------

    print()
    print(
        "1. Verifying frozen baseline integrity..."
    )

    baseline_integrity = (
        verify_baseline_integrity()
    )

    print(
        "   Frozen files expected:",
        baseline_integrity[
            "manifest_row_count"
        ],
    )

    print(
        "   Frozen files checked:",
        baseline_integrity[
            "files_checked"
        ],
    )

    print(
        "   Differences:",
        baseline_integrity[
            "failure_count"
        ],
    )

    if not baseline_integrity[
        "passed"
    ]:

        print()
        print(
            "STATUS: FAIL_BASELINE_INTEGRITY"
        )

        print(
            "The frozen validation run has changed."
        )

        print(
            "No integrated-run freeze was created."
        )

        for failure in (
            baseline_integrity[
                "failures"
            ]
        ):

            print(
                failure[
                    "status"
                ],
                failure[
                    "path"
                ],
            )

        raise SystemExit(1)

    print(
        "   PASS: frozen baseline unchanged."
    )

    # --------------------------------------------------------
    # LOAD BOTH RUNS
    # --------------------------------------------------------

    print()
    print(
        "2. Comparing Agent 2 top calls..."
    )

    baseline_agent2 = (
        load_agent2_top_calls(
            BASELINE_ROOT
        )
    )

    integrated_agent2 = (
        load_agent2_top_calls(
            integrated_root
        )
    )

    print(
        "3. Comparing Router decisions..."
    )

    baseline_router = (
        load_router_decisions(
            BASELINE_ROOT
        )
    )

    integrated_router = (
        load_router_decisions(
            integrated_root
        )
    )

    print(
        "4. Comparing no-expert Agent 3 queue..."
    )

    baseline_queue = (
        load_agent3_queue(
            BASELINE_ROOT
        )
    )

    integrated_queue = (
        load_agent3_queue(
            integrated_root
        )
    )

    print(
        "5. Comparing Agent 3 decisions..."
    )

    baseline_agent3 = (
        load_agent3_decisions(
            BASELINE_ROOT
        )
    )

    integrated_agent3 = (
        load_agent3_decisions(
            integrated_root
        )
    )

    # --------------------------------------------------------
    # BUILD DETAILED COMPARISON
    # --------------------------------------------------------

    comparison_rows: list[
        dict[str, Any]
    ] = []

    comparison_rows.extend(
        compare_mapping(
            "AGENT2_TOP_CALL",
            baseline_agent2,
            integrated_agent2,
        )
    )

    comparison_rows.extend(
        compare_mapping(
            "ROUTER_DECISION",
            baseline_router,
            integrated_router,
        )
    )

    comparison_rows.append(
        {
            "category": (
                "AGENT3_QUEUE"
            ),
            "item": (
                "ordered_no_expert_queue"
            ),
            "baseline_value": (
                json.dumps(
                    baseline_queue,
                    sort_keys=True,
                )
            ),
            "integrated_value": (
                json.dumps(
                    integrated_queue,
                    sort_keys=True,
                )
            ),
            "match": (
                baseline_queue
                == integrated_queue
            ),
        }
    )

    # --------------------------------------------------------
    # SPLIT AGENT 3 INTO THE EXACT MANUSCRIPT-SAFE CHECKS
    # --------------------------------------------------------

    baseline_states = {
        cluster: value[
            "decision_state"
        ]
        for cluster, value
        in baseline_agent3.items()
    }

    integrated_states = {
        cluster: value[
            "decision_state"
        ]
        for cluster, value
        in integrated_agent3.items()
    }

    comparison_rows.extend(
        compare_mapping(
            "AGENT3_DECISION_STATE",
            baseline_states,
            integrated_states,
        )
    )

    baseline_subtypes = {
        cluster: value[
            "ambiguity_subtype"
        ]
        for cluster, value
        in baseline_agent3.items()
    }

    integrated_subtypes = {
        cluster: value[
            "ambiguity_subtype"
        ]
        for cluster, value
        in integrated_agent3.items()
    }

    comparison_rows.extend(
        compare_mapping(
            "AGENT3_AMBIGUITY_SUBTYPE",
            baseline_subtypes,
            integrated_subtypes,
        )
    )

    baseline_labels = {
        cluster: value[
            "possible_labels"
        ]
        for cluster, value
        in baseline_agent3.items()
    }

    integrated_labels = {
        cluster: value[
            "possible_labels"
        ]
        for cluster, value
        in integrated_agent3.items()
    }

    comparison_rows.extend(
        compare_mapping(
            "AGENT3_POSSIBLE_LABELS",
            baseline_labels,
            integrated_labels,
        )
    )

    # --------------------------------------------------------
    # REFERENCE / EXPERT SAFETY CHECK
    # --------------------------------------------------------

    integrated_reference_safe = all(
        value[
            "reference_label_used"
        ]
        is False
        for value
        in integrated_agent3.values()
    )

    integrated_expert_safe = all(
        value[
            "expert_label_used"
        ]
        is False
        for value
        in integrated_agent3.values()
    )

    comparison_rows.append(
        {
            "category": (
                "INFERENCE_SAFETY"
            ),
            "item": (
                "reference_labels_not_used"
            ),
            "baseline_value": "True",
            "integrated_value": str(
                integrated_reference_safe
            ),
            "match": (
                integrated_reference_safe
            ),
        }
    )

    comparison_rows.append(
        {
            "category": (
                "INFERENCE_SAFETY"
            ),
            "item": (
                "expert_labels_not_used"
            ),
            "baseline_value": "True",
            "integrated_value": str(
                integrated_expert_safe
            ),
            "match": (
                integrated_expert_safe
            ),
        }
    )

    # --------------------------------------------------------
    # SUMMARY BY CATEGORY
    # --------------------------------------------------------

    category_summary: dict[
        str,
        dict[str, int],
    ] = {}

    for row in comparison_rows:

        category = str(
            row[
                "category"
            ]
        )

        category_summary.setdefault(
            category,
            {
                "total": 0,
                "passed": 0,
            },
        )

        category_summary[
            category
        ][
            "total"
        ] += 1

        if row[
            "match"
        ]:
            category_summary[
                category
            ][
                "passed"
            ] += 1

    overall_pass = all(
        bool(
            row[
                "match"
            ]
        )
        for row
        in comparison_rows
    )

    # --------------------------------------------------------
    # SAVE COMPARISON
    # --------------------------------------------------------

    validation_root = (
        integrated_root
        / "validation_against_frozen"
    )

    validation_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_csv = (
        validation_root
        / "cosmx_integrated_vs_frozen_comparison.csv"
    )

    write_csv(
        comparison_csv,
        comparison_rows,
        [
            "category",
            "item",
            "baseline_value",
            "integrated_value",
            "match",
        ],
    )

    comparison_json = (
        validation_root
        / "cosmx_integrated_vs_frozen_report.json"
    )

    report = {
        "schema_name": (
            "CosMXIntegratedRegressionValidation"
        ),
        "schema_version": "1.0",
        "created_at_utc": (
            utc_now()
        ),
        "dataset": DATASET,
        "baseline_root": str(
            BASELINE_ROOT
        ),
        "integrated_run_root": str(
            integrated_root
        ),
        "baseline_integrity": (
            baseline_integrity
        ),
        "category_summary": (
            category_summary
        ),
        "reference_labels_used_in_integrated_agent3": (
            not integrated_reference_safe
        ),
        "expert_labels_used_in_integrated_agent3": (
            not integrated_expert_safe
        ),
        "overall_regression_pass": (
            overall_pass
        ),
        "comparisons": (
            comparison_rows
        ),
    }

    write_json(
        comparison_json,
        report,
    )

    # --------------------------------------------------------
    # PRINT COMPACT RESULT
    # --------------------------------------------------------

    print()
    print(
        "=" * 76
    )
    print(
        "REGRESSION RESULTS"
    )
    print(
        "=" * 76
    )

    print(
        "Frozen baseline integrity: PASS"
    )

    for category, values in (
        category_summary.items()
    ):

        passed = values[
            "passed"
        ]

        total = values[
            "total"
        ]

        status = (
            "PASS"
            if passed == total
            else "FAIL"
        )

        print(
            f"{category}: "
            f"{passed}/{total} {status}"
        )

    # --------------------------------------------------------
    # FAIL SAFELY IF ANY DIFFERENCE
    # --------------------------------------------------------

    if not overall_pass:

        print()
        print(
            "Differences detected:"
        )

        for row in comparison_rows:

            if row[
                "match"
            ]:
                continue

            print()
            print(
                f"[{row['category']}] "
                f"{row['item']}"
            )

            print(
                "  Baseline:",
                row[
                    "baseline_value"
                ],
            )

            print(
                "  Integrated:",
                row[
                    "integrated_value"
                ],
            )

        print()
        print(
            "STATUS: REGRESSION_COMPARISON_FAILED"
        )

        print(
            "The integrated run was NOT frozen."
        )

        raise SystemExit(1)

    # ========================================================
    # ONLY AFTER FULL PASS:
    # FREEZE THE NEW INTEGRATED RUN
    # ========================================================

    print()
    print(
        "All decision-level comparisons passed."
    )

    print(
        "Creating integrated-run freeze..."
    )

    freeze_root = (
        integrated_root
        / "integrated_freeze"
    )

    if freeze_root.exists():
        raise SystemExit(
            "\nSAFETY STOP:\n"
            "Integrated freeze directory already exists:\n"
            f"{freeze_root}\n"
            "Nothing was overwritten."
        )

    freeze_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    hash_rows = (
        build_integrated_hash_manifest(
            integrated_root,
            freeze_root,
            validation_root,
        )
    )

    hash_manifest_csv = (
        freeze_root
        / "cosmx_integrated_file_manifest_sha256.csv"
    )

    write_csv(
        hash_manifest_csv,
        hash_rows,
        [
            "relative_path",
            "size_bytes",
            "sha256",
        ],
    )

    freeze_manifest = {
        "schema_name": (
            "CosMXIntegratedRunFreezeManifest"
        ),
        "schema_version": "1.0",
        "created_at_utc": (
            utc_now()
        ),
        "dataset": DATASET,
        "integrated_run_root": str(
            integrated_root
        ),
        "baseline_root": str(
            BASELINE_ROOT
        ),
        "baseline_integrity_verified": True,
        "baseline_manifest_file_count": (
            baseline_integrity[
                "manifest_row_count"
            ]
        ),
        "decision_level_regression_pass": True,
        "agent2_top_calls_match": True,
        "router_decisions_match": True,
        "agent3_queue_match": True,
        "agent3_decision_states_match": True,
        "agent3_ambiguity_subtypes_match": True,
        "agent3_possible_labels_match": True,
        "reference_labels_used_for_inference": False,
        "expert_labels_used_for_inference": False,
        "supplemental_audit_included_in_integrated_run": False,
        "hashed_integrated_file_count": len(
            hash_rows
        ),
        "sha256_manifest": str(
            hash_manifest_csv
        ),
        "regression_report": str(
            comparison_json
        ),
        "regression_comparison_csv": str(
            comparison_csv
        ),
        "scientific_interpretation": (
            "The one-command integrated CosMX run reproduced "
            "the previously frozen decision-level outputs for "
            "Agent 2 top calls, Router decisions, the no-expert "
            "Agent 3 queue, Agent 3 decision states, ambiguity "
            "subtypes, and possible labels. This validation does "
            "not claim byte-identical reproduction of dynamic "
            "literature retrieval artifacts."
        ),
        "status": (
            "COSMX_INTEGRATED_RUN_VALIDATED_AND_FROZEN"
        ),
    }

    freeze_manifest_json = (
        freeze_root
        / "cosmx_integrated_freeze_manifest.json"
    )

    write_json(
        freeze_manifest_json,
        freeze_manifest,
    )

    print()
    print(
        "=" * 76
    )
    print(
        "COSMX INTEGRATED VALIDATION COMPLETE"
    )
    print(
        "=" * 76
    )

    print(
        "Baseline files verified:",
        baseline_integrity[
            "files_checked"
        ],
    )

    print(
        "Baseline differences: 0"
    )

    print(
        "Agent 2 top calls: MATCH"
    )

    print(
        "Router decisions: MATCH"
    )

    print(
        "Agent 3 queue: MATCH"
    )

    print(
        "Agent 3 decision states: MATCH"
    )

    print(
        "Agent 3 ambiguity subtypes: MATCH"
    )

    print(
        "Agent 3 possible labels: MATCH"
    )

    print(
        "Reference labels used: False"
    )

    print(
        "Expert labels used: False"
    )

    print()
    print(
        "Integrated files frozen:",
        len(
            hash_rows
        ),
    )

    print()
    print(
        "Regression report:"
    )

    print(
        comparison_json
    )

    print()
    print(
        "Integrated freeze manifest:"
    )

    print(
        freeze_manifest_json
    )

    print()
    print(
        "STATUS: "
        "COSMX_INTEGRATED_RUN_VALIDATED_AND_FROZEN"
    )


if __name__ == "__main__":
    main()