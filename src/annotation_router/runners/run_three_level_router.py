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
        rows = [{"note": "no rows"}]
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


def classify_cluster(row: dict, candidates: list[dict], config: dict) -> dict:
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

    support_gap = top_support - second_support
    support_ratio = round(top_support / second_support, 3) if second_support > 0 else ""

    human_review_required = as_bool(row.get("human_review_required"))
    flag_codes = row.get("review_flag_codes") or row.get("warning_flag_codes") or ""

    min_accept_support = as_int(config.get("min_accept_support"), 8)
    min_route_support = as_int(config.get("min_route_support"), 6)
    route_close_ratio = as_float(config.get("route_close_ratio"), 1.15)
    monitor_close_ratio = as_float(config.get("monitor_close_ratio"), 1.50)
    route_candidate_count = as_int(config.get("route_candidate_count"), 20)
    monitor_candidate_count = as_int(config.get("monitor_candidate_count"), 8)
    broad_terms = config.get("broad_label_terms", [])

    broad_label = is_broad_label(leading_candidate, broad_terms)
    close_route = second_support > 0 and (top_support / second_support) <= route_close_ratio
    close_monitor = second_support > 0 and (top_support / second_support) <= monitor_close_ratio

    route_reasons = []
    monitor_reasons = []

    if top_support < min_route_support:
        route_reasons.append("very_weak_top_marker_support")
    elif top_support < min_accept_support:
        monitor_reasons.append("moderate_top_marker_support")

    if close_route:
        route_reasons.append("near_tie_between_top_candidates")
    elif close_monitor:
        monitor_reasons.append("moderate_candidate_competition")

    if broad_label and close_monitor:
        route_reasons.append("broad_label_with_competing_candidate")
    elif broad_label:
        monitor_reasons.append("broad_label")

    if candidate_count >= route_candidate_count:
        route_reasons.append("large_candidate_set")
    elif candidate_count >= monitor_candidate_count:
        monitor_reasons.append("moderate_candidate_set")

    if human_review_required and route_reasons:
        route_reasons.append("agent2_review_flag_supports_route")
    elif human_review_required:
        monitor_reasons.append("agent2_review_flag")

    if route_reasons:
        decision = "route_to_agent3"
        confidence = "low"
        reasons = route_reasons
    elif monitor_reasons:
        decision = "monitor_uncertainty"
        confidence = "moderate"
        reasons = monitor_reasons
    else:
        decision = "accept_agent2_annotation"
        confidence = "high"
        reasons = ["clear_agent2_evidence"]

    score = 1.0
    if top_support < min_route_support:
        score -= 0.30
    elif top_support < min_accept_support:
        score -= 0.15
    if close_route:
        score -= 0.25
    elif close_monitor:
        score -= 0.10
    if broad_label:
        score -= 0.10
    if candidate_count >= route_candidate_count:
        score -= 0.20
    elif candidate_count >= monitor_candidate_count:
        score -= 0.10
    if human_review_required and decision == "route_to_agent3":
        score -= 0.10
    elif human_review_required:
        score -= 0.05

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
        "router_score": round(max(score, 0.0), 3),
        "router_confidence_level": confidence,
        "router_decision": decision,
        "router_reasons": " | ".join(reasons),
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
        decisions.append(classify_cluster(row, candidates_by_cluster.get(cluster_id, []), config))

    accepted = [r for r in decisions if r["router_decision"] == "accept_agent2_annotation"]
    monitored = [r for r in decisions if r["router_decision"] == "monitor_uncertainty"]
    routed = [r for r in decisions if r["router_decision"] == "route_to_agent3"]

    write_csv(output_dir / "router_decisions.csv", decisions)
    write_json(output_dir / "router_decisions.json", decisions)
    write_csv(output_dir / "monitor_queue.csv", monitored)
    write_csv(output_dir / "agent3_queue.csv", routed)

    summary = {
        "dataset_name": config.get("dataset_name", ""),
        "router_version": "v3_three_level_evidence_uncertainty_router",
        "router_basis": "Agent 2 evidence and uncertainty only. Reference annotations are not used for routing.",
        "total_clusters": len(decisions),
        "accepted_clusters": len(accepted),
        "monitored_clusters": len(monitored),
        "routed_to_agent3_clusters": len(routed),
        "accepted_cluster_ids": [r["cluster_id"] for r in accepted],
        "monitored_cluster_ids": [r["cluster_id"] for r in monitored],
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
