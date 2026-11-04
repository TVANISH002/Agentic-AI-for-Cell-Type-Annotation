from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd




# This script reads the Agent 2 cluster summary created in Step 3 and produces:
#
#   1. router_evidence_table.csv
#      A clean table containing only observable Agent 2 evidence.
#
#   2. router_evidence_observations.json
#      A reproducible summary of simple descriptive observations.
#
# IMPORTANT:
# - No expert labels are used.
# - No ChatGPT/LLM reference labels are used.
# - No "confidence score" is created.
# - No accept/escalate decision is made.
# - No arbitrary threshold is introduced.


def count_sources(value: object) -> int:
    text = str(value or "").strip()

    if not text or text.lower() == "nan":
        return 0

    return len(
        {
            part.strip()
            for part in text.split("|")
            if part.strip()
        }
    )


def safe_fraction(numerator: float, denominator: float) -> float | None:
    if denominator is None or denominator <= 0:
        return None

    return round(float(numerator) / float(denominator), 4)


def build_router_evidence_table(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset",
        "cluster_id",
        "candidate_count",
        "top1_candidate",
        "top1_support_count",
        "top1_evidence_sources",
        "top1_cross_database_support",
        "top2_candidate",
        "top2_support_count",
        "top3_candidate",
        "top3_support_count",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Input file is missing required columns: "
            + ", ".join(sorted(missing))
        )

    output = pd.DataFrame()

    
    # RAW EVIDENCE
    # These values already come from Agent 2.
   
    output["dataset"] = df["dataset"]
    output["cluster_id"] = df["cluster_id"]
    output["candidate_count"] = df["candidate_count"]

    output["top1_candidate"] = df["top1_candidate"]
    output["top1_support_count"] = df["top1_support_count"]

    output["top2_candidate"] = df["top2_candidate"]
    output["top2_support_count"] = df["top2_support_count"]

    output["top3_candidate"] = df["top3_candidate"]
    output["top3_support_count"] = df["top3_support_count"]

    output["top1_evidence_sources"] = df["top1_evidence_sources"]
    output["top1_cross_database_support_count"] = (
        df["top1_cross_database_support"]
    )

    
    # TRANSPARENT DESCRIPTIVE QUANTITIES
    # These are NOT probabilities and NOT confidence scores.
    # They are included only because their biological meaning is explicit.
    
    # How many more supporting genes does the leading candidate have than the runner-up?
    output["top1_top2_support_gap"] = (
        df["top1_support_count"] - df["top2_support_count"]
    )

    # Is there an exact tie in direct marker support?
    output["top1_top2_exact_tie"] = (
        df["top1_support_count"] == df["top2_support_count"]
    )

    # How many independent knowledge sources support the leading candidate?
    output["top1_evidence_source_count"] = (
        df["top1_evidence_sources"].apply(count_sources)
    )

    # How many of the leading candidate's supporting genes were supported
    # by both CellMarker and PanglaoDB?
    output["top1_cross_database_fraction"] = [
        safe_fraction(cross_db, support)
        for cross_db, support in zip(
            df["top1_cross_database_support"],
            df["top1_support_count"],
        )
    ]

    # Simple binary descriptor. This is NOT a confidence label.
    output["top1_has_cross_database_corroboration"] = (
        df["top1_cross_database_support"] > 0
    )

    return output


def build_observations(table: pd.DataFrame) -> dict:
    observations: dict[str, object] = {
        "important_note": (
            "These are descriptive observations only. "
            "They do not define Router thresholds or confidence."
        ),
        "total_clusters": int(len(table)),
        "datasets": {},
    }

    for dataset, group in table.groupby("dataset", sort=True):
        group = group.copy()

        exact_ties = group.loc[
            group["top1_top2_exact_tie"],
            "cluster_id",
        ].tolist()

        corroborated = group.loc[
            group["top1_has_cross_database_corroboration"],
            "cluster_id",
        ].tolist()

        smallest_gap = int(
            group["top1_top2_support_gap"].min()
        )

        smallest_gap_clusters = group.loc[
            group["top1_top2_support_gap"] == smallest_gap,
            "cluster_id",
        ].tolist()

        largest_candidate_count = int(
            group["candidate_count"].max()
        )

        largest_candidate_clusters = group.loc[
            group["candidate_count"] == largest_candidate_count,
            "cluster_id",
        ].tolist()

        observations["datasets"][str(dataset)] = {
            "cluster_count": int(len(group)),
            "exact_top1_top2_tie_clusters": exact_ties,
            "cross_database_corroborated_top1_clusters": corroborated,
            "smallest_top1_top2_support_gap": smallest_gap,
            "clusters_with_smallest_gap": smallest_gap_clusters,
            "largest_candidate_count": largest_candidate_count,
            "clusters_with_largest_candidate_count": largest_candidate_clusters,
        }

    return observations


def write_markdown_report(
    table: pd.DataFrame,
    observations: dict,
    output_path: Path,
) -> None:
    lines: list[str] = []

    lines.append("# Router Evidence Observations")
    lines.append("")
    lines.append(
        "These observations are descriptive only. "
        "No routing threshold, confidence score, or expert label was used."
    )
    lines.append("")
    lines.append(
        f"Total clusters summarized: **{observations['total_clusters']}**."
    )
    lines.append("")

    for dataset, info in observations["datasets"].items():
        lines.append(f"## {dataset.capitalize()}")
        lines.append("")
        lines.append(
            f"- Clusters summarized: {info['cluster_count']}"
        )
        lines.append(
            "- Exact top-1/top-2 support ties: "
            + (
                ", ".join(map(str, info["exact_top1_top2_tie_clusters"]))
                if info["exact_top1_top2_tie_clusters"]
                else "none"
            )
        )
        lines.append(
            "- Clusters whose leading candidate has at least one "
            "supporting gene corroborated by both CellMarker and PanglaoDB: "
            + (
                ", ".join(
                    map(
                        str,
                        info["cross_database_corroborated_top1_clusters"],
                    )
                )
                if info["cross_database_corroborated_top1_clusters"]
                else "none"
            )
        )
        lines.append(
            "- Smallest top-1/top-2 support gap: "
            f"{info['smallest_top1_top2_support_gap']} "
            "supporting gene(s), observed in cluster(s): "
            + ", ".join(
                map(str, info["clusters_with_smallest_gap"])
            )
        )
        lines.append(
            "- Largest number of threshold-passing candidates: "
            f"{info['largest_candidate_count']}, observed in cluster(s): "
            + ", ".join(
                map(str, info["clusters_with_largest_candidate_count"])
            )
        )
        lines.append("")

    # Add a few directly observable breast cases for discussion.
    breast = table[table["dataset"].astype(str).str.lower() == "breast"]

    if not breast.empty:
        lines.append("## Breast examples worth discussing")
        lines.append("")

        for cluster_id in [3, 4, 5, 6]:
            row = breast[
                breast["cluster_id"].astype(int) == cluster_id
            ]

            if row.empty:
                continue

            r = row.iloc[0]

            lines.append(
                f"- **Cluster {cluster_id}:** "
                f"{r['top1_candidate']} ({int(r['top1_support_count'])}) "
                f"vs {r['top2_candidate']} "
                f"({int(r['top2_support_count'])}); "
                f"support gap = {int(r['top1_top2_support_gap'])}; "
                f"top-1 cross-database corroborated genes = "
                f"{int(r['top1_cross_database_support_count'])}."
            )

        lines.append("")
        lines.append(
            "These examples show why support counts are useful descriptors "
            "but should not be interpreted as calibrated confidence."
        )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Router evidence table from existing Agent 2 summaries "
            "without modifying any existing agent."
        )
    )

    parser.add_argument(
        "--agent2-features",
        type=Path,
        required=True,
        help="Path to agent2_evidence_features.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("router_evidence_outputs"),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        args.agent2_features
    )

    table = build_router_evidence_table(
        df
    )

    observations = build_observations(
        table
    )

    table_path = (
        args.output_dir
        / "router_evidence_table.csv"
    )

    json_path = (
        args.output_dir
        / "router_evidence_observations.json"
    )

    md_path = (
        args.output_dir
        / "router_evidence_observations.md"
    )

    table.to_csv(
        table_path,
        index=False,
    )

    json_path.write_text(
        json.dumps(
            observations,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_markdown_report(
        table,
        observations,
        md_path,
    )

    print(
        f"Router evidence table: {table_path}"
    )
    print(
        f"Observation JSON: {json_path}"
    )
    print(
        f"Observation report: {md_path}"
    )
    print(
        "No Router decision was made. "
        "No expert/LLM reference label was used."
    )


if __name__ == "__main__":
    main()
