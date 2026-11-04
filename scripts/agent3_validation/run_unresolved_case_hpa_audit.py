from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


T_CELL_TERMS = [
    "t cell",
    "t-cell",
    "t cells",
    "t-cells",
    "cytotoxic t cell",
    "cytotoxic t-cell",
    "cd8+ t cell",
    "cd8-positive",
    "alpha-beta t cell",
]

HPA_TERMS = [
    "hpa",
    "human protein atlas",
    "rna single cell",
    "rna_single_cell",
    "single cell type",
    "expression cell type",
]

STRUCTURED_SOURCE_TERMS = [
    "cellmarker",
    "panglao",
    "panglaodb",
    "direct_marker_supported",
    "marker_supported",
    "structured",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_text(value: Any) -> str:
    return str(value).replace("_", " ").replace("-", " ").lower()


def flatten_text(obj: Any) -> str:
    parts: list[str] = []

    def visit(x: Any) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                parts.append(str(k))
                visit(v)
        elif isinstance(x, list):
            for item in x:
                visit(item)
        elif x is not None:
            parts.append(str(x))

    visit(obj)
    return normalize_text(" ".join(parts))


def marker_regex(marker: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(marker)}(?![A-Za-z0-9])", re.IGNORECASE)


def subtree_contains_marker(obj: Any, marker: str) -> bool:
    return bool(marker_regex(marker).search(flatten_text(obj)))


def find_marker_subtrees(obj: Any, marker: str) -> list[Any]:
    hits: list[Any] = []

    def visit(x: Any) -> None:
        if isinstance(x, dict):
            if subtree_contains_marker(x, marker):
                hits.append(x)
            for v in x.values():
                visit(v)
        elif isinstance(x, list):
            for item in x:
                visit(item)

    visit(obj)
    return hits


def contains_any(text: str, terms: list[str]) -> bool:
    text_norm = normalize_text(text)
    return any(normalize_text(term) in text_norm for term in terms)


def detect_support(agent2_json: Any, marker: str) -> dict[str, Any]:
    marker_found = subtree_contains_marker(agent2_json, marker)
    marker_subtrees = find_marker_subtrees(agent2_json, marker)

    hpa_hits = []
    structured_hits = []

    for subtree in marker_subtrees:
        text = flatten_text(subtree)
        has_t_cell = contains_any(text, T_CELL_TERMS)
        has_hpa = contains_any(text, HPA_TERMS)
        has_structured = contains_any(text, STRUCTURED_SOURCE_TERMS)

        if has_t_cell and has_hpa:
            hpa_hits.append(text[:1000])

        if has_t_cell and has_structured:
            structured_hits.append(text[:1000])

    return {
        "marker_found_in_agent2": marker_found,
        "hpa_t_cell_support_detected": bool(hpa_hits),
        "structured_t_cell_support_detected": bool(structured_hits),
        "hpa_hit_count": len(hpa_hits),
        "structured_hit_count": len(structured_hits),
        "hpa_hit_examples": hpa_hits[:3],
        "structured_hit_examples": structured_hits[:3],
    }


def extract_candidate_names_from_agent2(agent2_json: Any) -> list[str]:
    names: list[str] = []

    def visit(x: Any) -> None:
        if isinstance(x, dict):
            for key, value in x.items():
                key_l = str(key).lower()
                if key_l in {"name", "label", "cell_type", "celltype", "candidate", "candidate_name"}:
                    if isinstance(value, str):
                        v = value.strip()
                        if v and len(v) <= 100:
                            names.append(v)
                visit(value)
        elif isinstance(x, list):
            for item in x:
                visit(item)

    visit(agent2_json)

    seen = set()
    ordered = []
    for name in names:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(name)

    return ordered


def choose_projected_labels(candidate_names: list[str]) -> str:
    lowered = {name.lower(): name for name in candidate_names}

    selected: list[str] = []

    for preferred in ["T cell", "Cytotoxic T cell"]:
        for original in candidate_names:
            if original.lower() == preferred.lower():
                selected.append(original)
                break

    if not selected:
        for original in candidate_names:
            if "t cell" in original.lower() or "t-cell" in original.lower():
                selected.append(original)

    if not selected:
        selected = ["T cell", "Cytotoxic T cell"]

    seen = set()
    clean = []
    for item in selected:
        k = item.lower()
        if k not in seen:
            seen.add(k)
            clean.append(item)

    return "; ".join(clean)


def get_agent2_marker_evidence_path(root: Path, dataset: str, cluster: str) -> Path:
    cluster_id = cluster.replace("C", "").replace("c", "")

    dataset_roots = {
        "breast": "breast_xenium_validated_clean",
        "lung": "lung_validated_clean",
    }

    dataset_root = dataset_roots.get(dataset.lower(), dataset)
    return (
        root
        / "data"
        / "pipeline_runs"
        / dataset_root
        / "agent2_output"
        / "clusters"
        / f"cluster_{cluster_id}"
        / f"cluster_{cluster_id}_marker_evidence.json"
    )


def get_rank_coverage_path(root: Path, dataset: str, cluster: str) -> Path:
    cluster_id = cluster.replace("C", "").replace("c", "")
    return (
        root
        / "data"
        / "analysis"
        / "agent3_locked_standard_results"
        / dataset.lower()
        / f"cluster_{cluster_id}"
        / "rank_coverage"
        / f"cluster_{cluster_id}_pubmedbert_rank_coverage.json"
    )


def inspect_rank_coverage(root: Path, dataset: str, cluster: str, marker: str) -> str:
    path = get_rank_coverage_path(root, dataset, cluster)

    if not path.exists():
        return "RANK_COVERAGE_FILE_NOT_FOUND"

    text = flatten_text(read_json(path))

    if not marker_regex(marker).search(text):
        return "MARKER_NOT_FOUND_IN_RANK_COVERAGE"

    marker_window = text

    if "unexplained" in marker_window:
        return "MARKER_PRESENT_IN_RANK_COVERAGE_UNEXPLAINED_CONTEXT"

    if "explained" in marker_window:
        return "MARKER_PRESENT_IN_RANK_COVERAGE_EXPLAINED_CONTEXT"

    return "MARKER_PRESENT_IN_RANK_COVERAGE_STATUS_UNCLEAR"


def read_locked_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--locked-summary",
        default="data/analysis/agent3_locked_standard_results/summary_tables/agent3_locked_standard_cluster_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="data/analysis/agent3_unresolved_case_audit/summary_tables",
    )
    parser.add_argument(
        "--markers",
        nargs="+",
        default=["TRAC"],
        help="Markers to audit in unresolved cases. Default: TRAC",
    )
    args = parser.parse_args()

    root = Path.cwd()
    locked_summary_path = root / args.locked_summary
    output_dir = root / args.output_dir

    if not locked_summary_path.exists():
        raise FileNotFoundError(locked_summary_path)

    locked_rows = read_locked_summary(locked_summary_path)
    unresolved_rows = [
        row for row in locked_rows
        if row.get("Decision", "").strip().upper() == "UNRESOLVED"
    ]

    audit_rows: list[dict[str, Any]] = []
    detail_records: list[dict[str, Any]] = []

    for row in unresolved_rows:
        dataset = row.get("Dataset", "").strip()
        cluster = row.get("Cluster", "").strip()
        original_decision = row.get("Decision", "").strip()
        original_labels = row.get("Labels", "").strip()

        agent2_path = get_agent2_marker_evidence_path(root, dataset, cluster)

        if not agent2_path.exists():
            audit_row = {
                "Dataset": dataset,
                "Cluster": cluster,
                "OriginalDecision": original_decision,
                "OriginalLabels": original_labels,
                "Marker": ";".join(args.markers),
                "Agent2MarkerEvidenceFound": False,
                "MarkerFoundInAgent2": False,
                "HPATCellSupportDetected": False,
                "StructuredTCellSupportDetected": False,
                "RankCoverageStatus": "NOT_CHECKED_AGENT2_FILE_MISSING",
                "ProjectedDecisionUnderAudit": "NO_CHANGE",
                "ProjectedLabelsUnderAudit": original_labels,
                "Interpretation": "Agent 2 marker evidence file not found; no audit reinterpretation applied.",
                "CoreAgent3OutputChanged": False,
                "Agent2MarkerEvidencePath": str(agent2_path),
            }
            audit_rows.append(audit_row)
            detail_records.append(audit_row)
            continue

        agent2_json = read_json(agent2_path)
        candidate_names = extract_candidate_names_from_agent2(agent2_json)

        for marker in args.markers:
            support = detect_support(agent2_json, marker)
            rank_status = inspect_rank_coverage(root, dataset, cluster, marker)

            should_project = (
                support["marker_found_in_agent2"]
                and (
                    support["hpa_t_cell_support_detected"]
                    or support["structured_t_cell_support_detected"]
                )
                and marker.upper() == "TRAC"
            )

            if should_project:
                projected_decision = "AMBIGUOUS"
                projected_labels = choose_projected_labels(candidate_names)
                interpretation = (
                    "T-cell lineage supported by upstream supplemental/structured evidence; "
                    "subtype remains unresolved. Core Agent 3 output is unchanged."
                )
            else:
                projected_decision = "NO_CHANGE"
                projected_labels = original_labels
                interpretation = (
                    "No qualifying upstream supplemental evidence was detected for this marker; "
                    "core Agent 3 unresolved decision remains unchanged."
                )

            audit_row = {
                "Dataset": dataset,
                "Cluster": cluster,
                "OriginalDecision": original_decision,
                "OriginalLabels": original_labels,
                "Marker": marker,
                "Agent2MarkerEvidenceFound": True,
                "MarkerFoundInAgent2": support["marker_found_in_agent2"],
                "HPATCellSupportDetected": support["hpa_t_cell_support_detected"],
                "StructuredTCellSupportDetected": support["structured_t_cell_support_detected"],
                "RankCoverageStatus": rank_status,
                "ProjectedDecisionUnderAudit": projected_decision,
                "ProjectedLabelsUnderAudit": projected_labels,
                "Interpretation": interpretation,
                "CoreAgent3OutputChanged": False,
                "Agent2MarkerEvidencePath": str(agent2_path),
            }

            audit_rows.append(audit_row)
            detail_records.append({
                **audit_row,
                "CandidateNamesDetected": candidate_names,
                "SupportDetails": support,
            })

    summary_csv = output_dir / "agent3_unresolved_case_hpa_audit_summary.csv"
    detail_json = output_dir / "agent3_unresolved_case_hpa_audit_details.json"

    write_csv(summary_csv, audit_rows)
    write_json(detail_json, detail_records)

    print("")
    print("=" * 80)
    print("UNRESOLVED-CASE HPA/TRAC AUDIT COMPLETE")
    print("=" * 80)
    print(f"Unresolved clusters audited: {len(unresolved_rows)}")
    print(f"Rows written: {len(audit_rows)}")
    print("")
    print("Saved:")
    print(summary_csv)
    print(detail_json)
    print("=" * 80)


if __name__ == "__main__":
    main()
