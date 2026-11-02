from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


PENALTIES = {
    "weak_top_marker_support": 0.25,
    "close_competing_candidate_support": 0.25,
    "moderate_candidate_competition": 0.10,
    "large_candidate_set": 0.15,
    "broad_or_nonspecific_leading_label": 0.15,
    "agent2_review_flag": 0.05,
}

ROUTER_DECISION_FIELDNAMES = [
    "cluster_id", "leading_candidate",
    "top_supporting_gene_count", "second_supporting_gene_count",
    "support_gap", "support_ratio", "candidate_count",
    "human_review_required", "agent2_flag_codes",
    "signal_weak_top_marker_support",
    "signal_close_competing_candidate_support",
    "signal_moderate_candidate_competition",
    "signal_large_candidate_set",
    "signal_broad_or_nonspecific_leading_label",
    "signal_agent2_review_flag",
    "penalty_weak_top_marker_support",
    "penalty_close_competing_candidate_support",
    "penalty_moderate_candidate_competition",
    "penalty_large_candidate_set",
    "penalty_broad_or_nonspecific_leading_label",
    "penalty_agent2_review_flag",
    "total_penalty", "score_calculation",
    "router_confidence_score", "router_confidence_level",
    "router_decision", "human_in_loop_default_action",
    "router_reasons", "critical_reasons",
    "selected_candidate_cell_types", "supporting_markers",
]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"Cannot write {path} without rows or explicit fieldnames.")
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
        top_support = as_int(row.get("supporting_gene_count"), as_int(row.get("direct_marker_supported_genes"), 0))
        second_support = 0

    human_review_required = as_bool(row.get("human_review_required"))
    flag_codes = row.get("review_flag_codes") or row.get("warning_flag_codes") or ""

    min_top_support = as_int(config.get("min_top_support_to_accept"), 6)
    close_ratio = as_float(config.get("close_candidate_ratio"), 1.25)
    clear_ratio = as_float(config.get("clear_candidate_ratio"), 1.50)
    high_candidate_count = as_int(config.get("high_candidate_count"), 10)
    high_candidate_min_support = as_int(config.get("high_candidate_min_support"), min_top_support)
    broad_threshold = as_int(config.get("broad_label_candidate_threshold"), 3)
    broad_terms = config.get("broad_label_terms", [])

    accept_score = as_float(config.get("accept_score_threshold"), 0.75)
    monitor_score = as_float(config.get("monitor_score_threshold"), 0.55)
    critical_count_to_route = as_int(config.get("critical_reason_count_to_route"), 2)

    support_gap = top_support - second_support
    support_ratio = round(top_support / second_support, 3) if second_support > 0 else ""

    score = 1.0
    reasons = []
    critical_reasons = []
    signal_values = {key: False for key in PENALTIES}
    applied_penalties = {key: 0.0 for key in PENALTIES}

    if top_support < min_top_support:
        signal_values["weak_top_marker_support"] = True
        applied_penalties["weak_top_marker_support"] = PENALTIES["weak_top_marker_support"]
        score -= applied_penalties["weak_top_marker_support"]
        reasons.append("weak_top_marker_support")
        critical_reasons.append("weak_top_marker_support")

    if second_support > 0:
        ratio = top_support / second_support
        if ratio <= close_ratio:
            signal_values["close_competing_candidate_support"] = True
            applied_penalties["close_competing_candidate_support"] = PENALTIES["close_competing_candidate_support"]
            score -= applied_penalties["close_competing_candidate_support"]
            reasons.append("close_competing_candidate_support")
            critical_reasons.append("close_competing_candidate_support")
        elif ratio < clear_ratio:
            signal_values["moderate_candidate_competition"] = True
            applied_penalties["moderate_candidate_competition"] = PENALTIES["moderate_candidate_competition"]
            score -= applied_penalties["moderate_candidate_competition"]
            reasons.append("moderate_candidate_competition")

    if candidate_count >= high_candidate_count:
        signal_values["large_candidate_set"] = True
        applied_penalties["large_candidate_set"] = PENALTIES["large_candidate_set"]
        score -= applied_penalties["large_candidate_set"]
        reasons.append("large_candidate_set")
        if top_support < high_candidate_min_support:
            critical_reasons.append("large_candidate_set_with_limited_top_support")

    if is_broad_label(leading_candidate, broad_terms) and candidate_count >= broad_threshold:
        signal_values["broad_or_nonspecific_leading_label"] = True
        applied_penalties["broad_or_nonspecific_leading_label"] = PENALTIES["broad_or_nonspecific_leading_label"]
        score -= applied_penalties["broad_or_nonspecific_leading_label"]
        reasons.append("broad_or_nonspecific_leading_label")

    if human_review_required:
        signal_values["agent2_review_flag"] = True
        applied_penalties["agent2_review_flag"] = PENALTIES["agent2_review_flag"]
        score -= applied_penalties["agent2_review_flag"]
        reasons.append("agent2_review_flag")

    total_penalty = round(sum(applied_penalties.values()), 3)
    score = round(max(score, 0.0), 3)

    if len(critical_reasons) >= critical_count_to_route:
        decision = "route_to_agent3"
    elif score >= accept_score and not critical_reasons:
        decision = "accept_agent2_annotation"
    elif score >= monitor_score:
        decision = "monitor_uncertainty"
    else:
        decision = "route_to_agent3"

    confidence_level = "high" if score >= accept_score else "moderate" if score >= monitor_score else "low"

    if decision == "accept_agent2_annotation":
        human_in_loop_default_action = "accept"
    elif decision == "monitor_uncertainty":
        human_in_loop_default_action = "expert_review_or_route_to_agent3_if_no_expert"
    else:
        human_in_loop_default_action = "route_to_agent3"

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
        "signal_weak_top_marker_support": str(signal_values["weak_top_marker_support"]),
        "signal_close_competing_candidate_support": str(signal_values["close_competing_candidate_support"]),
        "signal_moderate_candidate_competition": str(signal_values["moderate_candidate_competition"]),
        "signal_large_candidate_set": str(signal_values["large_candidate_set"]),
        "signal_broad_or_nonspecific_leading_label": str(signal_values["broad_or_nonspecific_leading_label"]),
        "signal_agent2_review_flag": str(signal_values["agent2_review_flag"]),
        "penalty_weak_top_marker_support": applied_penalties["weak_top_marker_support"],
        "penalty_close_competing_candidate_support": applied_penalties["close_competing_candidate_support"],
        "penalty_moderate_candidate_competition": applied_penalties["moderate_candidate_competition"],
        "penalty_large_candidate_set": applied_penalties["large_candidate_set"],
        "penalty_broad_or_nonspecific_leading_label": applied_penalties["broad_or_nonspecific_leading_label"],
        "penalty_agent2_review_flag": applied_penalties["agent2_review_flag"],
        "total_penalty": total_penalty,
        "score_calculation": f"max(0, 1.0 - {total_penalty:.3f})",
        "router_confidence_score": score,
        "router_confidence_level": confidence_level,
        "router_decision": decision,
        "human_in_loop_default_action": human_in_loop_default_action,
        "router_reasons": " | ".join(reasons) if reasons else "clear_agent2_evidence",
        "critical_reasons": " | ".join(critical_reasons),
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
    monitored = [r for r in decisions if r["router_decision"] == "monitor_uncertainty"]
    routed = [r for r in decisions if r["router_decision"] == "route_to_agent3"]
    no_expert_agent3 = routed + monitored

    write_csv(output_dir / "router_decisions.csv", decisions, ROUTER_DECISION_FIELDNAMES)
    write_json(output_dir / "router_decisions.json", decisions)
    write_csv(output_dir / "agent3_queue.csv", routed, ROUTER_DECISION_FIELDNAMES)
    write_csv(output_dir / "monitor_queue.csv", monitored, ROUTER_DECISION_FIELDNAMES)
    write_csv(output_dir / "agent3_queue_if_no_expert.csv", no_expert_agent3, ROUTER_DECISION_FIELDNAMES)

    summary = {
        "dataset_name": config.get("dataset_name", ""),
        "router_version": "V4 scoring-based evidence and uncertainty router",
        "router_basis": "Agent 2 evidence and uncertainty only. Reference annotations are not used for routing.",
        "accept_score_threshold": as_float(config.get("accept_score_threshold"), 0.75),
        "monitor_score_threshold": as_float(config.get("monitor_score_threshold"), 0.55),
        "critical_reason_count_to_route": as_int(config.get("critical_reason_count_to_route"), 2),
        "penalty_weights": PENALTIES,
        "critical_indicators": [
            "weak_top_marker_support",
            "close_competing_candidate_support",
            "large_candidate_set_with_limited_top_support",
        ],
        "monitor_policy": (
            "Monitored clusters are reserved for human-in-the-loop review. "
            "If no expert review is available, monitored clusters are "
            "conservatively included in agent3_queue_if_no_expert.csv."
        ),
        "total_clusters": len(decisions),
        "accepted_clusters": len(accepted),
        "monitored_clusters": len(monitored),
        "routed_to_agent3_clusters": len(routed),
        "agent3_if_no_expert_clusters": len(no_expert_agent3),
        "accepted_cluster_ids": [r["cluster_id"] for r in accepted],
        "monitored_cluster_ids": [r["cluster_id"] for r in monitored],
        "routed_cluster_ids": [r["cluster_id"] for r in routed],
        "agent3_if_no_expert_cluster_ids": [r["cluster_id"] for r in no_expert_agent3],
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
