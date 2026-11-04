from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_MODULES = [
    "runtime_context.py",
    "rescue_query_builder.py",
    "rescue_candidate_discovery.py",
    "rescue_confirmation_query_builder.py",
    "rescue_confirmation_retrieval.py",
    "rescue_gene_terminology.py",
    "rescue_confirmation_grounding.py",
    "rescue_confirmation_pubmedbert.py",
    "rescue_candidate_provenance.py",
    "rescue_evidence_packet.py",
]


SUMMARY_FIELDS = [
    "candidate",
    "ontology_id",
    "discovery_rank",
    "is_upstream_candidate",
    "is_novel_to_agent2",
    "discovery_marker_linked_papers",
    "discovery_linked_markers",
    "provenance_total_papers",
    "provenance_target_context_papers",
    "provenance_title_mention_papers",
    "provenance_pairwise_marker_papers",
    "provenance_same_sentence_marker_papers",
    "provenance_same_sentence_markers",
    "confirmation_candidate_mention_papers",
    "confirmation_candidate_marker_papers",
    "target_context_candidate_papers",
    "target_context_candidate_marker_papers",
    "pubmedbert_eligible_passages",
    "pubmedbert_eligible_unique_papers",
    "pubmedbert_selected_top_k",
    "pubmedbert_mean_similarity",
    "pubmedbert_max_similarity",
    "pubmedbert_median_similarity",
    "semantic_evidence_available",
    "evidence_state",
    "reporting_state",
    "automatic_acceptance",
    "final_annotation",
]


DEFAULT_FOCUS_CANDIDATES = [
    "B cell",
    "memory B cell",
    "mature B cell",
    "follicular B cell",
    "plasma cell",
    "T cell",
    "dendritic cell",
    "myeloid dendritic cell",
    "macrophage",
    "mast cell",
    "epithelial cell",
    "fibroblast",
    "endothelial cell",
]


def clean(value: Any) -> str:
    return str(value or "").strip()


def safe_name(value: str) -> str:
    value = clean(value).lower()
    keep = []

    for ch in value:
        if ch.isalnum():
            keep.append(ch)
        elif ch in {" ", "-", "_"}:
            keep.append("_")

    out = "".join(keep).strip("_")

    while "__" in out:
        out = out.replace("__", "_")

    return out or "unknown"


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8-sig")
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "modified_time": datetime.fromtimestamp(
            path.stat().st_mtime
        ).isoformat(timespec="seconds"),
    }


def should_skip_artifact(path: Path) -> bool:
    lowered_parts = {
        part.casefold()
        for part in path.parts
    }

    skip_markers = {
        "agent3_lung_c6_frozen_manifest",
        "agent3_standardized_frozen_manifest",
        "standardized_frozen_manifest",
    }

    return bool(
        lowered_parts.intersection(skip_markers)
    )


def discover_artifacts(cluster_root: Path) -> dict[str, dict[str, Any]]:
    records = {}

    for path in sorted(cluster_root.rglob("*")):
        if not path.is_file():
            continue

        if should_skip_artifact(path):
            continue

        if path.suffix.casefold() not in {
            ".json",
            ".csv",
            ".txt",
        }:
            continue

        rel = path.relative_to(cluster_root)
        key = str(rel).replace("\\", "/")

        records[key] = file_record(path)

    return records


def find_single_file(
    root: Path,
    pattern: str,
    *,
    required: bool = True,
) -> Path | None:
    hits = [
        path
        for path in root.rglob(pattern)
        if path.is_file()
        and not should_skip_artifact(path)
    ]

    if not hits:
        if required:
            raise FileNotFoundError(
                f"No file found for pattern: {pattern}"
            )
        return None

    if len(hits) > 1:
        raise RuntimeError(
            "Multiple files found for pattern "
            f"{pattern}:\n"
            + "\n".join(str(p) for p in hits)
        )

    return hits[0]


def discover_source_files(
    source_root: Path,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    records = {}
    missing = []

    for filename in DEFAULT_SOURCE_MODULES:
        path = source_root / filename

        if path.exists():
            records[filename] = file_record(path)
        else:
            missing.append(str(path))

    return records, missing


def scan_label_flags(
    obj: Any,
    *,
    path: str = "",
) -> list[str]:
    problems = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            current_path = f"{path}.{key}" if path else str(key)

            if key in {
                "reference_label_used",
                "expert_label_used",
            } and value is True:
                problems.append(
                    f"{current_path}=True"
                )

            problems.extend(
                scan_label_flags(
                    value,
                    path=current_path,
                )
            )

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            problems.extend(
                scan_label_flags(
                    item,
                    path=f"{path}[{i}]",
                )
            )

    return problems


def summarize_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        field: row.get(field)
        for field in SUMMARY_FIELDS
    }


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
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
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Standardize and freeze-audit a completed Agent 3 "
            "cluster evidence packet without rerunning inference."
        )
    )

    parser.add_argument(
        "--cluster-root",
        required=True,
        type=Path,
        help="Completed Agent 3 cluster folder.",
    )

    parser.add_argument(
        "--dataset",
        default="unknown",
        help="Dataset name, e.g., lung or breast.",
    )

    parser.add_argument(
        "--cluster-id",
        default=None,
        help="Expected cluster id. Optional but recommended.",
    )

    parser.add_argument(
        "--source-root",
        default=Path(
            "src/literature_retrieval_agent"
        ),
        type=Path,
        help="Agent 3 source module directory.",
    )

    parser.add_argument(
        "--output-directory",
        default=None,
        type=Path,
        help="Optional output directory for manifest and summaries.",
    )

    parser.add_argument(
        "--focus-candidates",
        default=",".join(DEFAULT_FOCUS_CANDIDATES),
        help="Comma-separated candidates to highlight.",
    )

    args = parser.parse_args()

    cluster_root = args.cluster_root

    if not cluster_root.exists():
        raise FileNotFoundError(
            f"Cluster root does not exist: {cluster_root}"
        )

    final_packet_path = find_single_file(
        cluster_root,
        "cluster_*_rescue_evidence_packet.json",
        required=True,
    )

    final_packet = load_json(final_packet_path)

    candidates = final_packet.get(
        "candidates",
        [],
    )

    if not isinstance(candidates, list):
        raise ValueError(
            "Final evidence packet field 'candidates' is not a list."
        )

    packet_cluster_id = clean(
        final_packet.get("cluster_id")
    )

    expected_cluster_id = (
        clean(args.cluster_id)
        if args.cluster_id is not None
        else packet_cluster_id
    )

    dataset_name = safe_name(args.dataset)
    cluster_name = safe_name(expected_cluster_id)

    outdir = (
        args.output_directory
        if args.output_directory is not None
        else cluster_root / "agent3_standardized_frozen_manifest"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact_records = discover_artifacts(
        cluster_root
    )

    source_records, missing_source_files = discover_source_files(
        args.source_root
    )

    label_flag_problems = scan_label_flags(
        final_packet
    )

    # Also scan all JSON artifacts under the cluster folder.
    json_scan_errors = []

    for path in sorted(cluster_root.rglob("*.json")):
        if should_skip_artifact(path):
            continue

        try:
            obj = load_json(path)
        except Exception as exc:
            json_scan_errors.append(
                {
                    "path": str(path),
                    "error": repr(exc),
                }
            )
            continue

        problems = scan_label_flags(obj)

        if problems:
            for problem in problems:
                label_flag_problems.append(
                    f"{path}: {problem}"
                )

    rescue_plan_path = find_single_file(
        cluster_root,
        "cluster_*_marker_rescue_query_plan.json",
        required=False,
    )

    rescue_plan_candidate_name_ok = True
    rescue_plan_candidate_names_used = None

    if rescue_plan_path is not None:
        rescue_plan = load_json(rescue_plan_path)
        rescue_plan_candidate_names_used = rescue_plan.get(
            "candidate_names_used"
        )
        rescue_plan_candidate_name_ok = (
            rescue_plan_candidate_names_used is False
            or rescue_plan_candidate_names_used is None
        )

    candidate_by_name = {
        clean(row.get("candidate")).casefold(): row
        for row in candidates
    }

    focus_candidates = [
        item.strip()
        for item in args.focus_candidates.split(",")
        if item.strip()
    ]

    focus_rows = []

    for name in focus_candidates:
        row = candidate_by_name.get(
            name.casefold()
        )

        if row is not None:
            focus_rows.append(
                summarize_candidate(row)
            )

    all_summary_rows = [
        summarize_candidate(row)
        for row in candidates
    ]

    invariants = {
        "final_packet_schema_present": bool(
            final_packet.get("schema_name")
        ),
        "cluster_id_present": bool(
            packet_cluster_id
        ),
        "cluster_id_matches_expected": (
            packet_cluster_id == expected_cluster_id
        ),
        "candidate_count_matches_rows": (
            final_packet.get("candidate_count") == len(candidates)
        ),
        "candidate_count_nonzero": len(candidates) > 0,
        "reference_label_not_used": (
            final_packet.get("reference_label_used") is False
        ),
        "expert_label_not_used": (
            final_packet.get("expert_label_used") is False
        ),
        "automatic_acceptance_not_calibrated": (
            final_packet.get("automatic_acceptance")
            == "NOT_CALIBRATED"
        ),
        "no_candidate_has_final_annotation": all(
            row.get("final_annotation") is None
            for row in candidates
        ),
        "all_candidates_not_calibrated": all(
            row.get("automatic_acceptance")
            == "NOT_CALIBRATED"
            for row in candidates
        ),
        "no_true_reference_or_expert_flags_found": (
            len(label_flag_problems) == 0
        ),
        "json_artifacts_readable": (
            len(json_scan_errors) == 0
        ),
        "source_modules_present": (
            len(missing_source_files) == 0
        ),
        "candidate_independent_rescue_plan_has_no_candidate_names": (
            rescue_plan_candidate_name_ok
        ),
        "pubmedbert_note_not_confidence": (
            "not probability"
            in clean(
                final_packet.get("important_note")
            ).casefold()
            or "semantic retrieval relevance"
            in clean(
                final_packet.get("important_note")
            ).casefold()
        ),
    }

    overall_pass = all(
        invariants.values()
    )

    manifest = {
        "schema_name": "Agent3StandardizedClusterFreezeManifest",
        "schema_version": "1.0",
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "dataset": args.dataset,
        "cluster_root": str(cluster_root),
        "final_evidence_packet": str(final_packet_path),
        "cluster_id": packet_cluster_id,
        "expected_cluster_id": expected_cluster_id,
        "freeze_scope": (
            "Standardized final audit/freeze of an already completed "
            "Agent 3 cluster evidence packet. This script does not rerun "
            "literature retrieval, PubMedBERT, grounding, or candidate discovery."
        ),
        "important_scientific_note": (
            "automatic_acceptance remains NOT_CALIBRATED and final_annotation "
            "remains None. PubMedBERT values are semantic retrieval relevance only, "
            "not annotation confidence or probability."
        ),
        "overall_pass": overall_pass,
        "invariants": invariants,
        "label_flag_problems": label_flag_problems,
        "json_scan_errors": json_scan_errors,
        "missing_source_files": missing_source_files,
        "rescue_plan_path": (
            str(rescue_plan_path)
            if rescue_plan_path is not None
            else None
        ),
        "rescue_plan_candidate_names_used": rescue_plan_candidate_names_used,
        "artifact_count": len(artifact_records),
        "artifacts": artifact_records,
        "source_files": source_records,
        "candidate_count": len(candidates),
        "candidate_ordering": final_packet.get(
            "candidate_ordering"
        ),
        "state_rule": final_packet.get(
            "state_rule"
        ),
        "top_15_candidates": all_summary_rows[:15],
        "focus_candidate_summary": focus_rows,
    }

    manifest_path = (
        outdir
        / (
            f"agent3_{dataset_name}_cluster_{cluster_name}"
            "_standardized_freeze_manifest.json"
        )
    )

    candidate_csv_path = (
        outdir
        / (
            f"agent3_{dataset_name}_cluster_{cluster_name}"
            "_all_candidate_summary.csv"
        )
    )

    focus_csv_path = (
        outdir
        / (
            f"agent3_{dataset_name}_cluster_{cluster_name}"
            "_focus_candidate_summary.csv"
        )
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    write_csv(
        candidate_csv_path,
        all_summary_rows,
        SUMMARY_FIELDS,
    )

    write_csv(
        focus_csv_path,
        focus_rows,
        SUMMARY_FIELDS,
    )

    print("=" * 90)
    print("STANDARDIZED AGENT 3 CLUSTER FREEZE AUDIT")
    print("=" * 90)
    print("Dataset:", args.dataset)
    print("Cluster root:", cluster_root)
    print("Final packet:", final_packet_path)
    print("Cluster:", packet_cluster_id)
    print("Candidates:", len(candidates))
    print("Overall:", "PASS" if overall_pass else "FAIL")
    print("Manifest:", manifest_path)
    print("All candidates CSV:", candidate_csv_path)
    print("Focus candidates CSV:", focus_csv_path)

    print()
    print("Invariant failures:")
    failed = [
        name
        for name, ok in invariants.items()
        if not ok
    ]

    if failed:
        for name in failed:
            print(" -", name)
    else:
        print(" none")

    if label_flag_problems:
        print()
        print("Label flag problems:")
        for problem in label_flag_problems:
            print(" -", problem)

    if missing_source_files:
        print()
        print("Missing source files:")
        for path in missing_source_files:
            print(" -", path)

    print()
    print("=" * 90)
    print("TOP 15 CANDIDATES")
    print("=" * 90)

    for row in all_summary_rows[:15]:
        print(
            f"{row.get('discovery_rank')}. "
            f"{row.get('candidate')} "
            f"[{row.get('ontology_id')}] "
            f"| upstream={row.get('is_upstream_candidate')} "
            f"| state={row.get('evidence_state')} "
            f"| auto={row.get('automatic_acceptance')} "
            f"| final={row.get('final_annotation')}"
        )

    print()
    print("=" * 90)
    print("FOCUS CANDIDATES")
    print("=" * 90)

    if not focus_rows:
        print("No focus candidates found.")
    else:
        for row in focus_rows:
            print(
                f"{row.get('candidate')} "
                f"[{row.get('ontology_id')}] "
                f"| rank={row.get('discovery_rank')} "
                f"| upstream={row.get('is_upstream_candidate')} "
                f"| state={row.get('evidence_state')} "
                f"| auto={row.get('automatic_acceptance')} "
                f"| final={row.get('final_annotation')}"
            )

    print()
    print("=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    main()
