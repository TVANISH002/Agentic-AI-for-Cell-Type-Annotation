from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hpa_evidence_agent.config import DEFAULT_CONFIG
from hpa_evidence_agent.runners.run_hpa_panel import run


def broad_lineage(cell_type):
    text = str(cell_type or "").lower()

    if any(x in text for x in ["secretory", "lactating", "hormone-responsive", "myoepithelial", "epithelial"]):
        return "Epithelial cell"
    if "t-cell" in text or "t cell" in text:
        return "T cell"
    if "b-cell" in text or "b cell" in text:
        return "B cell"
    if "plasma" in text:
        return "Plasma cell"
    if any(x in text for x in ["macrophage", "monocyte", "cdc", "pdc", "immune", "lymphoid"]):
        return "Immune cell"
    if "endothelial" in text:
        return "Endothelial cell"
    if any(x in text for x in ["fibroblast", "pericyte", "smooth muscle", "stromal"]):
        return "Stromal cell"
    if text in ["", "nan", "none"]:
        return ""
    return "Other"


def existing_broad(label):
    text = str(label or "").lower()

    if "epithelial" in text or "cancer" in text or "tumor" in text:
        return "Epithelial cell"
    if "t cell" in text:
        return "T cell"
    if "b cell" in text:
        return "B cell"
    if "plasma" in text:
        return "Plasma cell"
    if any(x in text for x in ["macrophage", "monocyte", "myeloid", "dendritic"]):
        return "Immune cell"
    if "endothelial" in text:
        return "Endothelial cell"
    if any(x in text for x in ["fibroblast", "stromal", "pericyte"]):
        return "Stromal cell"
    return ""


def standardize_outputs():
    hpa_root = ROOT / "data/pipeline_runs/breast_hpa_validation_clean"
    hpa_output = hpa_root / "hpa_output"
    final_output = hpa_root / "final"
    final_output.mkdir(parents=True, exist_ok=True)

    gene_path = hpa_output / "all_clusters_hpa_gene_summary.csv"
    metadata_path = hpa_output / "panel_hpa_source_validation.json"
    agent2_path = ROOT / "data/pipeline_runs/breast_xenium_validated_clean/agent2_output/all_clusters_summary.csv"

    gene_df = pd.read_csv(gene_path)
    agent2_df = pd.read_csv(agent2_path)

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    gene_out = pd.DataFrame({
        "cluster_id": gene_df["cluster_id"],
        "marker_rank": gene_df["rank_within_cluster"],
        "gene": gene_df["gene"],
        "log_fold_change": gene_df["log_fold_change"],
        "adjusted_p_value": gene_df["adjusted_p_value"],
        "hpa_status": gene_df["source_status"],
        "hpa_top_cell_type": gene_df["top_breast_cell_type"],
        "hpa_top_broad_lineage": gene_df["top_breast_cell_type"].map(broad_lineage),
        "hpa_top_max_ncpm": gene_df["highest_ncpm"],
        "hpa_second_cell_type": gene_df["second_breast_cell_type"],
        "hpa_second_max_ncpm": gene_df["second_highest_ncpm"],
        "hpa_top_to_second_ratio": gene_df["top_to_second_ncpm_ratio"],
        "hpa_detected_cell_type_count": gene_df["detected_cell_type_count"],
        "hpa_specificity_fraction": gene_df["detected_cell_type_count"].apply(
            lambda x: 1 / x if pd.notna(x) and x > 0 else 0
        ),
        "evidence_scope": gene_df["evidence_scope"],
    })

    gene_out_path = hpa_output / "breast_hpa_gene_evidence.csv"
    gene_out.to_csv(gene_out_path, index=False)

    candidate_rows = []
    valid_gene_out = gene_out.dropna(subset=["hpa_top_cell_type"])

    for (cluster_id, cell_type), group in valid_gene_out.groupby(["cluster_id", "hpa_top_cell_type"]):
        supporting_genes = sorted(set(group["gene"].astype(str)))

        candidate_rows.append({
            "cluster_id": cluster_id,
            "hpa_cell_type": cell_type,
            "hpa_broad_lineage": broad_lineage(cell_type),
            "selected_for_downstream": len(supporting_genes) >= 2,
            "minimum_supporting_genes": 2,
            "unique_supporting_gene_count": len(supporting_genes),
            "supporting_genes": " | ".join(supporting_genes),
            "specificity_weighted_support": round(group["hpa_specificity_fraction"].sum(), 6),
            "mean_supporting_gene_rank": round(group["marker_rank"].mean(), 3),
            "mean_supporting_log_fold_change": round(group["log_fold_change"].mean(), 3),
            "mean_hpa_top_ncpm": round(group["hpa_top_max_ncpm"].mean(), 3),
            "max_hpa_top_ncpm": round(group["hpa_top_max_ncpm"].max(), 3),
            "evidence_source": metadata.get("source", "Human Protein Atlas"),
            "source_release": metadata.get("source_release", "25.1"),
            "evidence_scope": "healthy_breast_reference",
            "warnings": "HPA evidence is healthy-breast expression evidence and is not a ground-truth annotation.",
        })

    cand = pd.DataFrame(candidate_rows)

    cand = cand.sort_values(
        ["cluster_id", "unique_supporting_gene_count", "specificity_weighted_support", "mean_supporting_gene_rank"],
        ascending=[True, False, False, True],
    )

    cand["candidate_rank"] = cand.groupby("cluster_id").cumcount() + 1

    cand = cand[[
        "cluster_id",
        "candidate_rank",
        "hpa_cell_type",
        "hpa_broad_lineage",
        "selected_for_downstream",
        "minimum_supporting_genes",
        "unique_supporting_gene_count",
        "supporting_genes",
        "specificity_weighted_support",
        "mean_supporting_gene_rank",
        "mean_supporting_log_fold_change",
        "mean_hpa_top_ncpm",
        "max_hpa_top_ncpm",
        "evidence_source",
        "source_release",
        "evidence_scope",
        "warnings",
    ]]

    ranked_path = hpa_output / "breast_hpa_ranked_candidate_celltypes.csv"
    selected_path = hpa_output / "breast_hpa_selected_candidate_celltypes.csv"

    cand.to_csv(ranked_path, index=False)
    cand[cand["selected_for_downstream"]].to_csv(selected_path, index=False)

    top = cand[cand["candidate_rank"] == 1].copy()

    selected = (
        cand[cand["selected_for_downstream"]]
        .groupby("cluster_id")["hpa_cell_type"]
        .apply(lambda x: " | ".join(x))
        .reset_index()
        .rename(columns={"hpa_cell_type": "hpa_selected_candidate_cell_types"})
    )

    summary = agent2_df.merge(top, on="cluster_id", how="left")
    summary = summary.merge(selected, on="cluster_id", how="left")

    positive_counts = (
        gene_out[gene_out["hpa_status"] == "breast_expression_found"]
        .groupby("cluster_id")
        .size()
    )
    queried_counts = gene_out.groupby("cluster_id").size()

    summary["hpa_queried_marker_gene_count"] = summary["cluster_id"].map(queried_counts).fillna(0).astype(int)
    summary["hpa_positive_breast_expression_gene_count"] = summary["cluster_id"].map(positive_counts).fillna(0).astype(int)

    summary["hpa_agrees_with_existing_broad_lineage"] = summary.apply(
        lambda r: existing_broad(r.get("leading_candidate")) == r.get("hpa_broad_lineage"),
        axis=1,
    )

    summary["hpa_evidence_scope"] = "healthy_breast_reference"
    summary["hpa_review_note"] = (
        "HPA provides healthy-breast expression evidence and does not replace "
        "CellMarker/PanglaoDB marker interpretation."
    )

    annotations_out = summary.rename(columns={
        "hpa_cell_type": "hpa_top_cell_type",
        "hpa_broad_lineage": "hpa_top_broad_lineage",
        "unique_supporting_gene_count": "hpa_top_supporting_gene_count",
        "supporting_genes": "hpa_top_supporting_genes",
    })

    annotations_out = annotations_out[[
        *agent2_df.columns,
        "hpa_top_cell_type",
        "hpa_top_broad_lineage",
        "hpa_top_supporting_gene_count",
        "hpa_top_supporting_genes",
        "hpa_selected_candidate_cell_types",
        "hpa_queried_marker_gene_count",
        "hpa_positive_breast_expression_gene_count",
        "hpa_agrees_with_existing_broad_lineage",
        "hpa_evidence_scope",
        "hpa_review_note",
    ]]

    annotations_path = final_output / "breast_cluster_annotations_with_hpa.csv"
    annotations_out.to_csv(annotations_path, index=False)

    validation_summary = annotations_out[[
        "cluster_id",
        "leading_candidate",
        "hpa_top_cell_type",
        "hpa_top_broad_lineage",
        "hpa_top_supporting_gene_count",
        "hpa_top_supporting_genes",
        "hpa_agrees_with_existing_broad_lineage",
    ]].rename(columns={"leading_candidate": "existing_leading_candidate"})

    validation_summary["reference_cell_type"] = ""
    validation_summary["reference_subtype_or_state"] = ""
    validation_summary["hpa_reference_candidate_rank"] = ""
    validation_summary["hpa_reference_candidate_supporting_gene_count"] = ""
    validation_summary["hpa_reference_candidate_supporting_genes"] = ""

    validation_summary["interpretation"] = validation_summary["hpa_agrees_with_existing_broad_lineage"].apply(
        lambda agrees: (
            "HPA agrees with the existing prediction at the broad-lineage level. "
            "HPA provides healthy-breast expression evidence and does not replace "
            "the CellMarker/PanglaoDB annotation."
            if agrees else
            "HPA does not agree with the existing prediction at the broad-lineage level. "
            "This should be reviewed as supplemental expression evidence, not ground truth."
        )
    )

    summary_path = final_output / "breast_hpa_validation_summary.csv"
    validation_summary.to_csv(summary_path, index=False)

    run_metadata = {
        "dataset_name": "breast_hpa_validation_clean",
        "hpa_tissue": "breast",
        "source_release": metadata.get("source_release"),
        "source_sha256": metadata.get("source_sha256"),
        "scanned_hpa_rows": metadata.get("scanned_row_count"),
        "requested_panel_genes": metadata.get("requested_gene_count"),
        "genes_present_in_hpa": metadata.get("requested_gene_count", 0) - metadata.get("gene_not_found_in_hpa_count", 0),
        "genes_with_breast_expression": metadata.get("breast_expression_found_count"),
        "expression_record_count": metadata.get("expression_record_count"),
        "output_contract": "lung_style_hpa_validation_outputs",
        "outputs": {
            "breast_hpa_gene_evidence_csv": str(gene_out_path),
            "breast_hpa_ranked_candidate_celltypes_csv": str(ranked_path),
            "breast_hpa_selected_candidate_celltypes_csv": str(selected_path),
            "breast_cluster_annotations_with_hpa_csv": str(annotations_path),
            "breast_hpa_validation_summary_csv": str(summary_path),
        },
        "note": "Standardized export only. HPA remains supplemental tissue-expression evidence, not ground truth.",
    }

    metadata_out = hpa_output / "breast_hpa_run_metadata.json"
    metadata_out.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    print("Created standardized breast HPA outputs:")
    for path in [gene_out_path, ranked_path, selected_path, metadata_out, annotations_path, summary_path]:
        print(path)


def main():
    config = replace(
        DEFAULT_CONFIG,
        panel_gene_list_path=Path("data/agent2_v2/panel/panel_gene_list.csv"),
        cluster_marker_path=Path("data/processed/brcr_top50_primary_marker_genes_by_cluster.csv"),
        hpa_source_path=Path("data/reference/hpa/rna_single_cell_cluster_hpa_v25_1.tsv.zip"),
        output_dir=Path("data/pipeline_runs/breast_hpa_validation_clean/hpa_output"),
        query_tissue="breast",
        evidence_scope="healthy_breast_reference",
    )

    run(config)
    standardize_outputs()


if __name__ == "__main__":
    main()
