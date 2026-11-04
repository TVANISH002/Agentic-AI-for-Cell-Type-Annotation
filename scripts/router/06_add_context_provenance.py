from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd




# This script only attaches provenance already present in Agent 2's original
# threshold-candidate outputs to the Router evidence table.
# Added fields for the TOP-RANKED candidate:
#   - tissue_contexts
#   - disease_contexts
#   - cell_ontology_ids
#   - specificity_weighted_support
#   - mean_supporting_gene_rank
#   - mean_supporting_log_fold_change
#
# IMPORTANT:
# - No expert labels are used.
# - No LLM reference labels are used.
# - No new confidence score is created.
# - No routing threshold is introduced.


REQUIRED_CANDIDATE_COLUMNS = {
    "cluster_id",
    "candidate_rank",
    "cell_type",
    "tissue_contexts",
    "disease_contexts",
    "cell_ontology_ids",
    "specificity_weighted_support",
    "mean_supporting_gene_rank",
    "mean_supporting_log_fold_change",
}


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_top1_provenance(path: Path, dataset: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = REQUIRED_CANDIDATE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{dataset} candidate file is missing required columns: "
            + ", ".join(sorted(missing))
        )

    top1 = df.loc[df["candidate_rank"] == 1].copy()

    if top1.empty:
        raise ValueError(
            f"No candidate_rank == 1 rows found for {dataset}."
        )

    duplicate_clusters = top1.loc[
        top1.duplicated(subset=["cluster_id"], keep=False),
        "cluster_id",
    ].tolist()

    if duplicate_clusters:
        raise ValueError(
            f"Multiple rank-1 candidates found for {dataset} cluster(s): "
            + ", ".join(map(str, sorted(set(duplicate_clusters))))
        )

    top1 = top1[
        [
            "cluster_id",
            "cell_type",
            "tissue_contexts",
            "disease_contexts",
            "cell_ontology_ids",
            "specificity_weighted_support",
            "mean_supporting_gene_rank",
            "mean_supporting_log_fold_change",
        ]
    ].copy()

    top1["dataset"] = dataset

    top1 = top1.rename(
        columns={
            "cell_type": "source_top1_candidate",
            "tissue_contexts": "top1_tissue_contexts",
            "disease_contexts": "top1_disease_contexts",
            "cell_ontology_ids": "top1_cell_ontology_ids",
            "specificity_weighted_support": "top1_specificity_weighted_support",
            "mean_supporting_gene_rank": "top1_mean_supporting_gene_rank",
            "mean_supporting_log_fold_change": "top1_mean_supporting_log_fold_change",
        }
    )

    return top1


def validate_candidate_identity(merged: pd.DataFrame) -> None:
    lhs = merged["top1_candidate"].map(clean_text).str.casefold()
    rhs = merged["source_top1_candidate"].map(clean_text).str.casefold()

    mismatch_mask = lhs != rhs

    if mismatch_mask.any():
        bad = merged.loc[
            mismatch_mask,
            [
                "dataset",
                "cluster_id",
                "top1_candidate",
                "source_top1_candidate",
            ],
        ]

        raise ValueError(
            "Top-candidate mismatch detected. "
            "The analysis table and Agent 2 source output are not aligned:\n"
            + bad.to_string(index=False)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Attach original Agent 2 top-candidate context/provenance "
            "to the Router evidence table."
        )
    )

    parser.add_argument(
        "--router-table",
        type=Path,
        required=True,
        help=(
            "Path to router_evidence_table_with_coverage.csv"
        ),
    )

    parser.add_argument(
        "--breast-candidates",
        type=Path,
        required=True,
        help=(
            "Breast all_clusters_threshold_candidates.csv"
        ),
    )

    parser.add_argument(
        "--lung-candidates",
        type=Path,
        required=True,
        help=(
            "Lung all_clusters_threshold_candidates.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "router_evidence_outputs_v3"
        ),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    router = pd.read_csv(
        args.router_table
    )

    breast = load_top1_provenance(
        args.breast_candidates,
        "breast",
    )

    lung = load_top1_provenance(
        args.lung_candidates,
        "lung",
    )

    provenance = pd.concat(
        [breast, lung],
        ignore_index=True,
    )

    merged = router.merge(
        provenance,
        on=["dataset", "cluster_id"],
        how="left",
        validate="one_to_one",
    )

    required_after_merge = [
        "source_top1_candidate",
        "top1_tissue_contexts",
        "top1_disease_contexts",
        "top1_cell_ontology_ids",
        "top1_specificity_weighted_support",
        "top1_mean_supporting_gene_rank",
        "top1_mean_supporting_log_fold_change",
    ]

    if merged[required_after_merge].isna().all(axis=1).any():
        bad = merged.loc[
            merged[required_after_merge].isna().all(axis=1),
            ["dataset", "cluster_id"],
        ]

        raise ValueError(
            "Some clusters could not be matched to Agent 2 provenance:\n"
            + bad.to_string(index=False)
        )

    validate_candidate_identity(
        merged
    )

    # The duplicate validation column is useful only for the audit check,
    # so remove it from the final analysis table.
    merged = merged.drop(
        columns=["source_top1_candidate"]
    )

    output_table = (
        args.output_dir
        / "router_evidence_table_with_context.csv"
    )

    merged.to_csv(
        output_table,
        index=False,
    )

    audit = {
        "purpose": (
            "Attach original Agent 2 top-candidate context and provenance "
            "without changing any candidate, annotation, or Router decision."
        ),
        "input_router_rows": int(len(router)),
        "output_rows": int(len(merged)),
        "breast_top1_rows": int(len(breast)),
        "lung_top1_rows": int(len(lung)),
        "top_candidate_identity_check": "PASS",
        "expert_labels_used": False,
        "llm_reference_labels_used": False,
        "router_decision_created": False,
        "confidence_score_created": False,
        "added_fields": [
            "top1_tissue_contexts",
            "top1_disease_contexts",
            "top1_cell_ontology_ids",
            "top1_specificity_weighted_support",
            "top1_mean_supporting_gene_rank",
            "top1_mean_supporting_log_fold_change",
        ],
    }

    audit_file = (
        args.output_dir
        / "context_provenance_audit.json"
    )

    audit_file.write_text(
        json.dumps(
            audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Updated Router evidence table: {output_table}"
    )
    print(
        f"Context/provenance audit: {audit_file}"
    )
    print(
        "Top-candidate identity check: PASS"
    )
    print(
        "No annotation, candidate ranking, or Router decision was changed."
    )


if __name__ == "__main__":
    main()
