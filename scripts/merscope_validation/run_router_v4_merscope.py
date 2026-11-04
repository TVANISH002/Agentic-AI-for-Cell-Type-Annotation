
"""MERSCOPE Router V4 — reuse the frozen CosMX routing policy.

Uses the existing Router V4 implementation, without editing it.
Creates a MERSCOPE-specific configuration and output directory.
Does not rerun Agent 1/2 or use expert/reference annotations.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from annotation_router.runners.run_scored_router import run_router


DATASET = "merscope_brcr_validated_clean"

BASE = ROOT / "data" / "pipeline_runs" / DATASET
AGENT2 = BASE / "agent2_output"

SUMMARY_CSV = AGENT2 / "all_clusters_summary.csv"
CANDIDATES_CSV = (
    AGENT2 / "all_clusters_threshold_candidates.csv"
)

COSMX_CONFIG = (
    ROOT / "configs/router/cosmx_brcr_router_v4.json"
)

NEW_CONFIG = (
    ROOT / "configs/router/merscope_brcr_router_v4.json"
)

ROUTER_SOURCE = (
    ROOT
    / "src/annotation_router/runners/run_scored_router.py"
)

OUTPUT = BASE / "router_output_v4"


def digest(path: Path) -> str:
    sha = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            sha.update(block)

    return sha.hexdigest()


def load_rows(path: Path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        rows = list(reader)

    if not fields or not rows:
        raise ValueError(
            f"Missing CSV header or rows: {path}"
        )

    return fields, rows


def validate_inputs() -> None:

    if Path.cwd().resolve() != ROOT.resolve():
        raise RuntimeError(
            "Run from the VS Code project root:\n"
            f"{ROOT}"
        )

    required_files = (
        SUMMARY_CSV,
        CANDIDATES_CSV,
        COSMX_CONFIG,
        ROUTER_SOURCE,
    )

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    if NEW_CONFIG.exists():
        raise FileExistsError(
            "MERSCOPE Router configuration already exists:\n"
            f"{NEW_CONFIG}"
        )

    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(
            "Router output folder already contains files:\n"
            f"{OUTPUT}"
        )

    fields, rows = load_rows(SUMMARY_CSV)

    required_columns = {
        "cluster_id",
        "leading_candidate",
        "human_review_required",
    }

    if not required_columns.issubset(fields):
        raise ValueError(
            "Missing Agent 2 summary columns: "
            f"{required_columns - set(fields)}"
        )

    clusters = [
        str(row["cluster_id"]).strip()
        for row in rows
    ]

    if (
        len(clusters) != 8
        or sorted(clusters) != [str(i) for i in range(8)]
    ):
        raise ValueError(
            f"Expected clusters 0-7; found {clusters}"
        )

    if any(
        not row["leading_candidate"].strip()
        for row in rows
    ):
        raise ValueError(
            "Agent 2 has an empty leading candidate"
        )

    candidate_fields, candidate_rows = load_rows(
        CANDIDATES_CSV
    )

    required_candidates = {
        "cluster_id",
        "candidate_rank",
        "unique_supporting_gene_count",
    }

    if not required_candidates.issubset(
        candidate_fields
    ):
        raise ValueError(
            "Missing Agent 2 candidate columns: "
            f"{required_candidates - set(candidate_fields)}"
        )

    candidate_clusters = {
        str(row["cluster_id"]).strip()
        for row in candidate_rows
    }

    if candidate_clusters != set(clusters):
        raise ValueError(
            "Threshold candidate evidence must cover "
            "all 8 MERSCOPE clusters.\n"
            f"Found: {sorted(candidate_clusters)}"
        )


def main() -> None:

    validate_inputs()

    # ======================================================
    # 1. LOAD FROZEN COSMX ROUTER V4 CONFIGURATION
    # ======================================================

    source_config = json.loads(
        COSMX_CONFIG.read_text(
            encoding="utf-8-sig"
        )
    )

    required_keys = (
        "dataset_name",
        "agent2_summary_csv",
        "agent2_candidates_csv",
        "output_dir",
    )

    for field in required_keys:
        if field not in source_config:
            raise ValueError(
                f"Frozen Router configuration lacks {field}"
            )

    # ======================================================
    # 2. CHANGE ONLY DATASET-SPECIFIC FIELDS
    # ======================================================

    config = dict(source_config)

    config.update({
        "dataset_name": DATASET,
        "agent2_summary_csv": (
            SUMMARY_CSV.relative_to(ROOT).as_posix()
        ),
        "agent2_candidates_csv": (
            CANDIDATES_CSV.relative_to(ROOT).as_posix()
        ),
        "output_dir": (
            OUTPUT.relative_to(ROOT).as_posix()
        ),
    })

    changed_fields = {
        key
        for key in config
        if config[key] != source_config.get(key)
    }

    allowed_changes = set(required_keys)

    if not changed_fields.issubset(allowed_changes):
        raise AssertionError(
            "Unexpected Router policy modification: "
            f"{changed_fields - allowed_changes}"
        )

    leftovers = {
        key: value
        for key, value in config.items()
        if (
            key not in allowed_changes
            and isinstance(value, str)
            and "cosmx" in value.casefold()
        )
    }

    if leftovers:
        raise ValueError(
            "Unadapted CosMX configuration fields: "
            f"{leftovers}"
        )

    # ======================================================
    # 3. HASH ALL REUSED INPUTS AND FROZEN CODE
    # ======================================================

    protected_files = (
        SUMMARY_CSV,
        CANDIDATES_CSV,
        COSMX_CONFIG,
        ROUTER_SOURCE,
    )

    hashes_before = {
        str(path.relative_to(ROOT)): digest(path)
        for path in protected_files
    }

    wrapper_start = time.perf_counter()

    NEW_CONFIG.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Exclusive creation: existing configuration is protected.
    with NEW_CONFIG.open(
        "x",
        encoding="utf-8",
    ) as stream:
        json.dump(
            config,
            stream,
            indent=2,
            ensure_ascii=False,
        )
        stream.write("\n")

    # ======================================================
    # 4. RUN EXISTING FROZEN ROUTER V4
    # ======================================================

    router_start = time.perf_counter()

    summary = run_router(NEW_CONFIG)

    router_seconds = (
        time.perf_counter() - router_start
    )

    # ======================================================
    # 5. VERIFY PROTECTED INPUT FILES ARE UNCHANGED
    # ======================================================

    for path in protected_files:
        key = str(path.relative_to(ROOT))

        if digest(path) != hashes_before[key]:
            raise RuntimeError(
                f"Existing input/code changed: {key}"
            )

    expected_files = (
        "router_decisions.csv",
        "router_decisions.json",
        "router_summary.json",
        "agent3_queue.csv",
        "monitor_queue.csv",
        "agent3_queue_if_no_expert.csv",
    )

    for filename in expected_files:
        if not (OUTPUT / filename).is_file():
            raise RuntimeError(
                f"Missing Router output: {filename}"
            )

    _, decisions = load_rows(
        OUTPUT / "router_decisions.csv"
    )

    valid_decisions = {
        "accept_agent2_annotation",
        "monitor_uncertainty",
        "route_to_agent3",
    }

    if (
        len(decisions) != 8
        or {row["cluster_id"] for row in decisions}
        != {str(i) for i in range(8)}
    ):
        raise RuntimeError(
            "Router did not return all 8 clusters"
        )

    if any(
        row["router_decision"] not in valid_decisions
        for row in decisions
    ):
        raise RuntimeError(
            "Unexpected Router decision state"
        )

    counts = Counter(
        row["router_decision"]
        for row in decisions
    )

    if (
        counts["accept_agent2_annotation"]
        != summary["accepted_clusters"]
        or counts["monitor_uncertainty"]
        != summary["monitored_clusters"]
        or counts["route_to_agent3"]
        != summary["routed_to_agent3_clusters"]
    ):
        raise RuntimeError(
            "Router summary disagrees with decision CSV"
        )

    wrapper_seconds = (
        time.perf_counter() - wrapper_start
    )

    # ======================================================
    # 6. SAVE REPRODUCIBILITY AND TIMING MANIFEST
    # ======================================================

    manifest = {
        "dataset": DATASET,
        "status": "ROUTER_V4_COMPLETED_VALIDATED",
        "code_reused": str(
            ROUTER_SOURCE.relative_to(ROOT)
        ),
        "policy_cloned_from": str(
            COSMX_CONFIG.relative_to(ROOT)
        ),
        "policy_changed_fields": sorted(
            changed_fields
        ),
        "policy_parameters_unchanged": True,
        "routing_uses_reference_or_expert_labels": False,
        "source_input_hashes_sha256": hashes_before,
        "merscope_router_config_sha256": digest(
            NEW_CONFIG
        ),
        "output_hashes_sha256": {
            name: digest(OUTPUT / name)
            for name in expected_files
        },
        "router_runtime_seconds": round(
            router_seconds, 6
        ),
        "wrapper_total_runtime_seconds": round(
            wrapper_seconds, 6
        ),
        "decision_counts": dict(counts),
        "router_summary": summary,
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    manifest_path = (
        OUTPUT / "merscope_router_v4_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    # ======================================================
    # 7. DISPLAY CLUSTER RESULTS
    # ======================================================

    print("\n" + "=" * 64)
    print("MERSCOPE ROUTER V4 COMPLETED")
    print("=" * 64)

    for row in sorted(
        decisions,
        key=lambda row: int(row["cluster_id"]),
    ):
        print(
            f"C{row['cluster_id']}: "
            f"{row['leading_candidate']} | "
            f"{row['router_decision']} | "
            f"score={row['router_confidence_score']}"
        )

    print()
    print(
        "ACCEPT  :",
        counts["accept_agent2_annotation"],
    )
    print(
        "MONITOR :",
        counts["monitor_uncertainty"],
    )
    print(
        "ROUTE   :",
        counts["route_to_agent3"],
    )
    print(
        "Agent 3 if no expert:",
        summary["agent3_if_no_expert_clusters"],
    )

    print()
    print(
        f"Router runtime : "
        f"{router_seconds:.6f} seconds"
    )
    print(
        f"Wrapper total  : "
        f"{wrapper_seconds:.6f} seconds"
    )
    print(f"Output folder  : {OUTPUT}")
    print(f"Configuration  : {NEW_CONFIG}")
    print(f"Manifest       : {manifest_path}")
    print("CosMX and other datasets: NOT MODIFIED")
    print("=" * 64)


if __name__ == "__main__":
    main()
