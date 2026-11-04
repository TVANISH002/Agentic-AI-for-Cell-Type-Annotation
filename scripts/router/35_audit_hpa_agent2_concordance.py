from pathlib import Path
import pandas as pd

OUT_DIR = Path("data/analysis/hpa_agent2_concordance_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "breast": {
        "agent2": Path(
            "data/pipeline_runs/breast_xenium_marker_only_clean/"
            "agent2_output/all_clusters_summary.csv"
        ),
        "hpa": Path(
            "data/pipeline_runs/breast_hpa_validation_clean/"
            "hpa_output/breast_hpa_selected_candidate_celltypes.csv"
        ),
    },
    "lung": {
        "agent2": Path(
            "data/pipeline_runs/lung_validated_clean/"
            "agent2_output/all_clusters_summary.csv"
        ),
        "hpa": Path(
            "data/pipeline_runs/lung_hpa_validation_clean/"
            "hpa_output/lung_hpa_selected_candidate_celltypes.csv"
        ),
    },
}

# Deterministic lineage harmonization used only for this supplemental
# Agent2-HPA concordance analysis. These mappings are evaluation metadata
# and are NOT inputs to Agent 2, Router V4, or the Selective Router.
AGENT2_LINEAGE = {
    "luminal epithelial cell": "epithelial cell",
    "epithelial cell": "epithelial cell",
    "macrophage": "macrophage",
    "fibroblast": "fibroblast",
    "cytotoxic t cell": "t cell",
    "t cell": "t cell",
    "endothelial cell": "endothelial cell",
    "b cell": "b cell",
    "plasma cell": "plasma cell",
    "mast cell": "mast cell",
}

# HPA broad labels differ slightly between breast and lung.
# Harmonize only biologically obvious broad-lineage synonyms.
HPA_LINEAGE = {
    "epithelial cell": "epithelial cell",
    "immune cell": "immune cell",
    "stromal cell": "stromal cell",
    "t cell": "t cell",
    "endothelial cell": "endothelial cell",
    "macrophage": "macrophage",
    "fibroblast": "fibroblast",
    "b cell": "b cell",
    "plasma cell": "plasma cell",
    "mast cell": "mast cell",
}

# Explicit compatible parent/child lineage relations.
COMPATIBLE = {
    ("macrophage", "immune cell"),
    ("fibroblast", "stromal cell"),
}


def norm(value):
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())


def classify(agent2_label, hpa_broad):
    a2 = AGENT2_LINEAGE.get(norm(agent2_label))
    hpa = HPA_LINEAGE.get(norm(hpa_broad))

    if a2 is None:
        return "not_evaluable", f"Unmapped Agent 2 label: {agent2_label}"

    if hpa is None:
        return "not_evaluable", f"Unmapped HPA broad lineage: {hpa_broad}"

    if a2 == hpa:
        return "lineage_concordant", f"Shared lineage: {a2}"

    if (a2, hpa) in COMPATIBLE or (hpa, a2) in COMPATIBLE:
        return (
            "lineage_concordant",
            f"Compatible hierarchical lineages: {a2} vs {hpa}",
        )

    return "discordant", f"Different lineages: {a2} vs {hpa}"


rows = []

for dataset, paths in DATASETS.items():
    agent2 = pd.read_csv(paths["agent2"])
    hpa = pd.read_csv(paths["hpa"])

    agent2["cluster_id"] = agent2["cluster_id"].astype(str)
    hpa["cluster_id"] = hpa["cluster_id"].astype(str)

    top_hpa = (
        hpa[hpa["candidate_rank"].astype(int) == 1]
        .sort_values("cluster_id")
        .drop_duplicates("cluster_id", keep="first")
    )

    for _, a2_row in agent2.iterrows():
        cluster_id = str(a2_row["cluster_id"])

        h = top_hpa[top_hpa["cluster_id"] == cluster_id]

        if h.empty:
            rows.append(
                {
                    "dataset": dataset,
                    "cluster_id": cluster_id,
                    "agent2_leading_candidate": a2_row["leading_candidate"],
                    "hpa_top_candidate": "",
                    "hpa_broad_lineage": "",
                    "hpa_supporting_gene_count": "",
                    "concordance_class": "not_evaluable",
                    "concordance_reason": "No rank-1 selected HPA candidate",
                }
            )
            continue

        h = h.iloc[0]

        concordance_class, reason = classify(
            a2_row["leading_candidate"],
            h["hpa_broad_lineage"],
        )

        rows.append(
            {
                "dataset": dataset,
                "cluster_id": cluster_id,
                "agent2_leading_candidate": a2_row["leading_candidate"],
                "hpa_top_candidate": h["hpa_cell_type"],
                "hpa_broad_lineage": h["hpa_broad_lineage"],
                "hpa_supporting_gene_count": h["unique_supporting_gene_count"],
                "concordance_class": concordance_class,
                "concordance_reason": reason,
            }
        )

audit = pd.DataFrame(rows)

summary_rows = []

for dataset, group in audit.groupby("dataset", sort=False):
    evaluable = group[group["concordance_class"] != "not_evaluable"]
    concordant = (
        evaluable["concordance_class"] == "lineage_concordant"
    ).sum()
    discordant = (
        evaluable["concordance_class"] == "discordant"
    ).sum()

    summary_rows.append(
        {
            "dataset": dataset,
            "total_clusters": len(group),
            "evaluable_clusters": len(evaluable),
            "lineage_concordant_clusters": int(concordant),
            "discordant_clusters": int(discordant),
            "not_evaluable_clusters": int(
                (group["concordance_class"] == "not_evaluable").sum()
            ),
            "lineage_concordance_percent": (
                100.0 * concordant / len(evaluable)
                if len(evaluable)
                else float("nan")
            ),
        }
    )

summary = pd.DataFrame(summary_rows)

audit_path = OUT_DIR / "hpa_agent2_cluster_concordance_v1.csv"
summary_path = OUT_DIR / "hpa_agent2_concordance_summary_v1.csv"

audit.to_csv(audit_path, index=False)
summary.to_csv(summary_path, index=False)

print("\n================ HPA-AGENT 2 CONCORDANCE AUDIT ================\n")
print(audit.to_string(index=False))

print("\n================ SUMMARY ================\n")
print(summary.to_string(index=False))

print("\nSaved:")
print(audit_path)
print(summary_path)
