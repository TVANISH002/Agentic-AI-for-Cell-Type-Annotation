from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def as_int(value, default=0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def as_float(value, default=0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def as_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def split_cell_types(value: str) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in str(value).replace("|", ";").split(";") if x.strip()]


def load_candidates(path_text: str) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    if not path_text:
        return grouped

    path = Path(path_text)
    if not path.exists():
        return grouped

    for row in read_csv(path):
        cluster_id = str(row.get("cluster_id", "")).strip()
        if cluster_id:
            grouped[cluster_id].append(row)

    for cluster_id in grouped:
        grouped[cluster_id].sort(key=lambda r: as_int(r.get("candidate_rank"), 999999))

    return grouped


def is_broad_label(label: str, broad_terms: list[str]) -> bool:
    return str(label).strip().lower() in {x.lower() for x in broad_terms}


def route_one_cluster(row: dict, candidates: list[dict], config: dict) -> dict:
    cluster_id = str(row.get("cluster_id", "")).strip()
    leading_candidate = str(row.get("leading_candidate", "")).strip()

    selected = split_cell_types(row.get("selected_candidate_cell_types", ""))
    candidate_count = max(
        as_int(row.get("selected_candidate_count"), len(selected)),
        as_int(row.get("threshold_candidate_count"), len(candidates)),
        len(candidates),
        1,
    )

    if candidates:
        top_support = as_int(candidates[0].get("unique_supporting_gene_count"), 0)
        second_support = as_int(candidates[1].get("unique_supporting_gene_count"), 0) if len(candidates) > 1 else 0
    else:
        top_support = as_int(
            row.get("supporting_gene_count"),
            as_int(row.get("direct_marker_supported_genes"), 0),
        )
        second_support = 0

    both_db_support = as_int(
        row.get("both_database_supporting_gene_count"),
        as_int(row.get("both_source_supporting_gene_count"), 0),
    )

    human_review_required = as_bool(row.get("human_review_required"))
    flag_codes = row.get("review_flag_codes") or row.get("warning_flag_codes") or ""

    min_top_support = as_int(config.get("min_top_support_to_accept"), 8)
    close_ratio_cutoff = as_float(config.get("close_candidate_ratio"), 1.25)
    broad_candidate_threshold = as_int(config.get("broad_label_candidate_threshold"), 3)
    high_candidate_count = as_int(config.get("high_candidate_count"), 10)
    high_candidate_min_support = as_int(config.get("high_candidate_min_support"), 8)
    broad_terms = config.get("broad_label_terms", [])

    support_gap = top_support - second_support
    support_ratio = round(top_support / second_support, 3) if second_support > 0 else ""

    core_reasons = []
    notes = []

    if top_support < min_top_support:
        core_reasons.append("weak_top_marker_support")

    if second_support > 0 and (top_support / second_support) <= close_ratio_cutoff:
        core_reasons.append("close_competing_candidates")

    if is_broad_label(leading_candidate, broad_terms) and candidate_count >= broad_candidate_threshold:
        core_reasons.append("broad_label_with_competing_candidates")

    if candidate_count >= high_candidate_count and top_support < high_candidate_min_support:
        core_reasons.append("large_candidate_set_with_limited_top_support")

    if human_review_required and core_reasons:
        notes.append("agent2_review_flag_supports_routing")
    elif human_review_required:
        notes.append("agent2_review_flag_only_not_routed")

    if flag_codes:
        notes.append(f"agent2_flags:{flag_codes}")

    route_to_agent3 = bool(core_reasons)

    if top_support >= min_top_support + 2 and not core_reasons:
        confidence_level = "high"
    elif route_to_agent3:
        confidence_level = "uncertain"
    else:
        confidence_level = "moderate"

    router_score = 1.0
    if top_support < min_top_support:
        router_score -= 0.25
    if second_support > 0 and (top_support / second_support) <= close_ratio_cutoff:
        router_score -= 0.25
    if candidate_count >= high_candidate_count:
        router_score -= 0.15
    if is_broad_label(leading_candidate, broad_terms):
        router_score -= 0.10
    if human_review_required and core_reasons:
        router_score -= 0.10
    router_score = round(max(router_score, 0.0), 3)

    return {
        "cluster_id": cluster_id,
        "leading_candidate": leading_candidate,
        "top_supporting_gene_count": top_support,
        "second_supporting_gene_count": second_support,
        "support_gap": support_gap,
        "support_ratio": support_ratio,
        "candidate_count": candidate_count,
        "human_review_required": str(human_review_required),
        "agent2_flag_codes": flag_codes,
        "router_score": router_score,
        "router_confidence_level": confidence_level,
        "router_decision": "route_to_agent3" if route_to_agent3 else "accept_agent2_annotation",
        "router_reasons": " | ".join(core_reasons) if core_reasons else "clear_or_sufficient_agent2_evidence",
        "router_notes": " | ".join(notes),
        "selected_candidate_cell_types": row.get("selected_candidate_cell_types", ""),
        "supporting_markers": row.get("supporting_markers", ""),
    }


def run_router(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))

    summary_rows = read_csv(Path(config["agent2_summary_csv"]))
    candidates_by_cluster = load_candidates(config.get("agent2_candidates_csv", ""))

    output_dir = Path(config["output_dir"])
    decisions = []

    for row in summary_rows:
        cluster_id = str(row.get("cluster_id", "")).strip()
        decisions.append(route_one_cluster(row, candidates_by_cluster.get(cluster_id, []), config))

    accepted = [r for r in decisions if r["router_decision"] == "accept_agent2_annotation"]
    routed = [r for r in decisions if r["router_decision"] == "route_to_agent3"]

    write_csv(output_dir / "router_decisions.csv", decisions)
    write_json(output_dir / "router_decisions.json", decisions)
    write_csv(output_dir / "agent3_queue.csv", routed)

    summary = {
        "dataset_name": config.get("dataset_name", ""),
        "router_basis": "Agent 2 evidence and uncertainty only. Reference annotations are not used for routing.",
        "total_clusters": len(decisions),
        "accepted_clusters": len(accepted),
        "routed_to_agent3_clusters": len(routed),
        "accepted_cluster_ids": [r["cluster_id"] for r in accepted],
        "routed_cluster_ids": [r["cluster_id"] for r in routed],
    }

    write_json(output_dir / "router_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-config", required=True)
    args = parser.parse_args()

    print(json.dumps(run_router(Path(args.router_config)), indent=2))


if __name__ == "__main__":
    main()
