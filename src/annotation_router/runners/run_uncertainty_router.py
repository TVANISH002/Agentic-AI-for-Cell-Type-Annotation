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
    parts = []
    for chunk in str(value).replace("|", ";").split(";"):
        cleaned = chunk.strip()
        if cleaned:
            parts.append(cleaned)
    return parts


def load_candidate_support(candidate_path: Path | None) -> dict[str, list[dict]]:
    grouped = defaultdict(list)

    if not candidate_path or not candidate_path.exists():
        return grouped

    for row in read_csv(candidate_path):
        cluster_id = str(row.get("cluster_id", "")).strip()
        if not cluster_id:
            continue
        grouped[cluster_id].append(row)

    for cluster_id in grouped:
        grouped[cluster_id].sort(key=lambda r: as_int(r.get("candidate_rank"), 999999))

    return grouped


def has_close_competing_candidate(candidates: list[dict], ratio: float) -> bool:
    if len(candidates) < 2:
        return False

    top_support = as_float(candidates[0].get("unique_supporting_gene_count"), 0.0)
    second_support = as_float(candidates[1].get("unique_supporting_gene_count"), 0.0)

    if top_support <= 0 or second_support <= 0:
        return False

    return (top_support / second_support) <= ratio


def is_broad_label(label: str, broad_terms: list[str]) -> bool:
    normalized = str(label).strip().lower()
    if not normalized:
        return False

    return normalized in {term.lower() for term in broad_terms}


def compute_router_score(
    top_support: int,
    both_db_support: int,
    candidate_count: int,
    human_review_required: bool,
    close_competition: bool,
    min_high_support: int,
    max_acceptable_candidates: int,
) -> float:
    support_score = min(top_support / max(min_high_support, 1), 1.0)

    if top_support > 0:
        both_db_score = min(both_db_support / top_support, 1.0)
    else:
        both_db_score = 0.0

    ambiguity_score = 1.0
    if close_competition:
        ambiguity_score -= 0.45
    if candidate_count > max_acceptable_candidates:
        ambiguity_score -= 0.35
    ambiguity_score = max(ambiguity_score, 0.0)

    review_score = 0.0 if human_review_required else 1.0

    score = (
        0.40 * support_score
        + 0.20 * both_db_score
        + 0.25 * ambiguity_score
        + 0.15 * review_score
    )

    return round(score, 3)


def route_cluster(row: dict, candidates: list[dict], config: dict) -> dict:
    cluster_id = str(row.get("cluster_id", "")).strip()
    leading_candidate = str(row.get("leading_candidate", "")).strip()

    selected_candidates = split_cell_types(row.get("selected_candidate_cell_types", ""))
    selected_candidate_count = as_int(row.get("selected_candidate_count"), len(selected_candidates))
    threshold_candidate_count = as_int(row.get("threshold_candidate_count"), selected_candidate_count)

    candidate_count = max(selected_candidate_count, threshold_candidate_count, len(candidates), 1)

    top_support = as_int(
        row.get("supporting_gene_count"),
        as_int(row.get("direct_marker_supported_genes"), 0),
    )

    if candidates:
        top_support = max(top_support, as_int(candidates[0].get("unique_supporting_gene_count"), top_support))

    both_db_support = as_int(
        row.get("both_database_supporting_gene_count"),
        as_int(row.get("both_source_supporting_gene_count"), 0),
    )

    human_review_required = as_bool(row.get("human_review_required"))

    flag_codes = (
        row.get("review_flag_codes")
        or row.get("warning_flag_codes")
        or ""
    ).strip()

    min_supporting_genes = as_int(config.get("min_supporting_genes"), 2)
    min_high_support = as_int(config.get("min_high_supporting_genes"), 8)
    max_acceptable_candidates = as_int(config.get("max_acceptable_candidates"), 4)
    close_candidate_ratio = as_float(config.get("close_candidate_ratio"), 1.15)
    broad_terms = config.get("broad_label_terms", [])

    close_competition = has_close_competing_candidate(candidates, close_candidate_ratio)
    broad_label = is_broad_label(leading_candidate, broad_terms)

    reasons = []

    if human_review_required:
        reasons.append("agent2_human_review_required")

    if flag_codes:
        reasons.append(f"agent2_flag_codes:{flag_codes}")

    if top_support < min_supporting_genes:
        reasons.append("low_marker_support")

    if close_competition:
        reasons.append("close_competing_candidate_support")

    if candidate_count > max_acceptable_candidates:
        reasons.append("multiple_candidate_cell_types")

    if broad_label and candidate_count > 1:
        reasons.append("broad_cell_type_label")

    router_score = compute_router_score(
        top_support=top_support,
        both_db_support=both_db_support,
        candidate_count=candidate_count,
        human_review_required=human_review_required,
        close_competition=close_competition,
        min_high_support=min_high_support,
        max_acceptable_candidates=max_acceptable_candidates,
    )

    route_to_agent3 = bool(reasons)

    return {
        "cluster_id": cluster_id,
        "leading_candidate": leading_candidate,
        "selected_candidate_cell_types": row.get("selected_candidate_cell_types", ""),
        "top_supporting_gene_count": top_support,
        "both_database_supporting_gene_count": both_db_support,
        "candidate_count": candidate_count,
        "human_review_required": str(human_review_required),
        "agent2_flag_codes": flag_codes,
        "router_score": router_score,
        "router_decision": "route_to_agent3" if route_to_agent3 else "accept_agent2_annotation",
        "router_reasons": " | ".join(reasons) if reasons else "clear_agent2_evidence",
        "supporting_markers": row.get("supporting_markers", ""),
    }


def run_router(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))

    summary_path = Path(config["agent2_summary_csv"])
    candidate_path = Path(config["agent2_candidates_csv"]) if config.get("agent2_candidates_csv") else None
    output_dir = Path(config["output_dir"])

    summary_rows = read_csv(summary_path)
    candidates_by_cluster = load_candidate_support(candidate_path)

    decision_rows = []
    for row in summary_rows:
        cluster_id = str(row.get("cluster_id", "")).strip()
        decision_rows.append(route_cluster(row, candidates_by_cluster.get(cluster_id, []), config))

    accepted = [r for r in decision_rows if r["router_decision"] == "accept_agent2_annotation"]
    routed = [r for r in decision_rows if r["router_decision"] == "route_to_agent3"]

    decision_csv = output_dir / "router_decisions.csv"
    decision_json = output_dir / "router_decisions.json"
    queue_csv = output_dir / "agent3_queue.csv"
    summary_json = output_dir / "router_summary.json"

    write_csv(decision_csv, decision_rows)
    write_json(decision_json, decision_rows)
    write_csv(queue_csv, routed)

    summary = {
        "dataset_name": config.get("dataset_name", ""),
        "router_basis": "Agent 2 evidence and uncertainty only; reference annotations are not used for routing.",
        "total_clusters": len(decision_rows),
        "accepted_clusters": len(accepted),
        "routed_to_agent3_clusters": len(routed),
        "accepted_cluster_ids": [r["cluster_id"] for r in accepted],
        "routed_cluster_ids": [r["cluster_id"] for r in routed],
        "outputs": {
            "router_decisions_csv": str(decision_csv),
            "router_decisions_json": str(decision_json),
            "agent3_queue_csv": str(queue_csv),
            "router_summary_json": str(summary_json),
        },
    }

    write_json(summary_json, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-config", required=True)
    args = parser.parse_args()

    summary = run_router(Path(args.router_config))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

