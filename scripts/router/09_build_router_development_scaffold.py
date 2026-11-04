from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


# This script therefore:
#   1. reads the FROZEN tissue-agnostic feature matrix;
#   2. creates only transparent structural evidence descriptors;
#   3. keeps metadata separate from predictive features;
#   4. writes the future training/evaluation protocol;
#   5. creates NO confidence score;
#   6. creates NO Router decision;
#   7. uses NO expert/LLM labels.



REQUIRED_FEATURES = {
    "row_id",
    "candidate_count",
    "top1_support_count",
    "top2_present",
    "top3_present",
    "top2_support_count",
    "top3_support_count",
    "top1_cross_database_support_count",
    "top1_cross_database_fraction",
    "direct_evidence_coverage",
    "top_marker_direct_supported",
    "top1_support_fraction_of_input",
    "top2_support_fraction_of_input",
    "top1_top2_gap_fraction_of_input",
    "leading_non_direct_fraction",
    "first_direct_supported_rank_fraction",
    "top1_mean_supporting_gene_rank_fraction",
}

REQUIRED_METADATA = {
    "row_id",
    "dataset",
    "cluster_id",
    "top1_candidate",
    "top1_evidence_sources",
    "top1_tissue_contexts",
    "top1_disease_contexts",
    "top1_cell_ontology_ids",
    "total_input_genes",
    "direct_marker_supported_genes",
    "follow_up_gene_count",
}


def require_columns(
    df: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} is missing required columns: "
            + ", ".join(sorted(missing))
        )


def build_structural_state(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    # Join ONLY for analysis/audit output.
    # Dataset, cluster and candidate names remain metadata and are NOT added to
    # the frozen predictive feature matrix.
    merged = metadata.merge(
        features,
        on="row_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(features):
        raise ValueError(
            "Feature/metadata merge lost rows."
        )

    out = merged[
        [
            "row_id",
            "dataset",
            "cluster_id",
            "top1_candidate",
        ]
    ].copy()

    top1 = pd.to_numeric(
        merged["top1_support_count"],
        errors="raise",
    )

    top2 = pd.to_numeric(
        merged["top2_support_count"],
        errors="coerce",
    )

    top2_present = pd.to_numeric(
        merged["top2_present"],
        errors="raise",
    ).astype(int)

    top_marker_supported = pd.to_numeric(
        merged["top_marker_direct_supported"],
        errors="raise",
    ).astype(int)

    cross_db_count = pd.to_numeric(
        merged["top1_cross_database_support_count"],
        errors="raise",
    )

    leading_non_direct_fraction = pd.to_numeric(
        merged["leading_non_direct_fraction"],
        errors="raise",
    )

    # -----------------------------------------------------------------
    # STRUCTURAL DESCRIPTORS ONLY
    # -----------------------------------------------------------------
    # These are not "uncertainty scores" and do not automatically trigger
    # routing. They simply describe observable properties of the evidence.
    # -----------------------------------------------------------------

    out["competitor_present"] = top2_present

    out["top1_top2_gap_count"] = (
        top1 - top2
    ).where(
        top2_present == 1
    )

    out["exact_top_support_tie"] = (
        (top2_present == 1)
        & (top1 == top2)
    ).astype(int)

    out["top_marker_lacks_direct_support"] = (
        1 - top_marker_supported
    ).astype(int)

    out["leading_ranked_markers_lack_direct_support"] = (
        leading_non_direct_fraction > 0
    ).astype(int)

    out["zero_cross_database_corroboration"] = (
        cross_db_count == 0
    ).astype(int)

    out["direct_evidence_coverage"] = pd.to_numeric(
        merged["direct_evidence_coverage"],
        errors="raise",
    ).round(4)

    out["top1_cross_database_fraction"] = pd.to_numeric(
        merged["top1_cross_database_fraction"],
        errors="coerce",
    ).round(4)

    out["leading_non_direct_fraction"] = (
        leading_non_direct_fraction.round(4)
    )

    out["first_direct_supported_rank_fraction"] = pd.to_numeric(
        merged["first_direct_supported_rank_fraction"],
        errors="coerce",
    ).round(4)

    # Explicitly state that this table is descriptive.
    out["router_decision"] = "NOT_ASSIGNED"
    out["confidence_score"] = "NOT_CREATED"

    return out


def build_protocol() -> dict[str, object]:
    return {
        "protocol_version": "selective_router_development_v1",
        "status": "DEVELOPMENT_PROTOCOL_NOT_FINAL_MODEL",
        "scientific_goal": (
            "Estimate whether Agent 2 evidence is sufficient to stop "
            "or whether the cluster should be escalated to Agent 3."
        ),
        "core_principle": (
            "The Router should learn evidence sufficiency from development "
            "data rather than use manually assigned uncertainty penalties."
        ),
        "current_pilot_data_role": (
            "Breast and lung clusters are for engineering, feature inspection, "
            "and case-study analysis; they are not sufficient for final Router "
            "calibration."
        ),
        "future_supervised_target": {
            "name": "agent2_top1_correct",
            "definition": (
                "1 when Agent 2's top-ranked candidate agrees with an "
                "independent expert-curated or published reference annotation "
                "after pre-specified label/ontology harmonization; 0 otherwise."
            ),
            "inference_input": False,
            "important_note": (
                "Reference annotations are revealed only for Router "
                "development/evaluation and are never supplied to Agent 1, "
                "Agent 2, or Router inference."
            ),
        },
        "recommended_primary_router_model": {
            "model": "regularized_logistic_regression",
            "reason": (
                "Interpretable coefficients, limited complexity, probabilistic "
                "output suitable for selective prediction, and easier auditing "
                "than high-capacity black-box models."
            ),
            "note": (
                "The model must not be fit on the current 20 clusters as final "
                "evidence of performance."
            ),
        },
        "data_partitioning": [
            (
                "Split at dataset level, not random cluster level, so clusters "
                "from the same biological dataset do not leak between training "
                "and held-out evaluation."
            ),
            (
                "When enough datasets are available, additionally report "
                "leave-one-dataset-out or leave-one-tissue-out generalization."
            ),
            (
                "All preprocessing, feature definitions and ontology "
                "harmonization rules must be frozen before held-out testing."
            ),
        ],
        "threshold_selection": {
            "rule": (
                "Do not choose a threshold from the held-out test set."
            ),
            "preferred_analysis": (
                "Use validation data to select operating points, while "
                "reporting the full risk-coverage curve and AURC so conclusions "
                "do not depend on one arbitrary threshold."
            ),
        },
        "router_evaluation_metrics": [
            "risk_coverage_curve",
            "AURC",
            "coverage_at_pre_specified_risk_levels",
            "selective_accuracy",
            "error_enrichment_among_escalated_clusters",
            "fraction_of_clusters_escalated",
        ],
        "full_pipeline_metrics": [
            "Agent2_only_annotation_accuracy_when_reference_available",
            "selective_pipeline_annotation_accuracy",
            "Agent3_correction_rate",
            "Agent3_unchanged_rate",
            "Agent3_worsening_rate",
            "Agent3_abstention_rate",
            "literature_retrieval_calls_avoided_vs_Agent3_for_all",
            "latency",
            "computational_or_API_cost",
        ],
        "agent3_claim_level_metrics": [
            "exact_quote_grounding_rate",
            "semantic_relation_validation_precision",
            "semantic_relation_validation_recall",
            "semantic_relation_validation_macro_f1",
        ],
        "required_baselines": [
            "Agent2_only",
            "existing_Router_V4_heuristic",
            "new_selective_router",
            "Agent3_for_every_cluster",
            "direct_LLM_marker_list_annotation",
            "compatible_established_cell_annotation_baseline",
        ],
        "forbidden_claims_at_current_stage": [
            "The new Router is validated.",
            "The current 20 clusters establish generalization.",
            "Evidence coverage is confidence.",
            "Cross-database corroboration guarantees correctness.",
            "A large candidate support gap guarantees correctness.",
            "Lung LLM labels are ground truth.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the selective Router development scaffold without "
            "training or routing decisions."
        )
    )

    parser.add_argument(
        "--feature-matrix",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/analysis/router_development_v1"
        ),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    features = pd.read_csv(
        args.feature_matrix
    )

    metadata = pd.read_csv(
        args.metadata
    )

    require_columns(
        features,
        REQUIRED_FEATURES,
        "Feature matrix",
    )

    require_columns(
        metadata,
        REQUIRED_METADATA,
        "Metadata",
    )

    if features["row_id"].duplicated().any():
        raise ValueError(
            "Duplicate row_id values in feature matrix."
        )

    if metadata["row_id"].duplicated().any():
        raise ValueError(
            "Duplicate row_id values in metadata."
        )

    if set(features["row_id"]) != set(metadata["row_id"]):
        raise ValueError(
            "Feature and metadata row_id sets do not match."
        )

    state = build_structural_state(
        features,
        metadata,
    )

    protocol = build_protocol()

    state_file = (
        args.output_dir
        / "router_evidence_state_v1.csv"
    )

    protocol_file = (
        args.output_dir
        / "router_development_protocol_v1.json"
    )

    audit_file = (
        args.output_dir
        / "router_development_audit_v1.json"
    )

    state.to_csv(
        state_file,
        index=False,
    )

    protocol_file.write_text(
        json.dumps(
            protocol,
            indent=2,
        ),
        encoding="utf-8",
    )

    audit = {
        "rows_processed": int(len(state)),
        "expert_labels_used": False,
        "llm_reference_labels_used": False,
        "router_model_trained": False,
        "router_threshold_selected": False,
        "router_decisions_created": False,
        "confidence_score_created": False,
        "existing_agent_outputs_modified": False,
        "exact_tie_clusters": state.loc[
            state["exact_top_support_tie"] == 1,
            ["dataset", "cluster_id"],
        ].to_dict(orient="records"),
        "clusters_with_top_marker_lacking_direct_support": state.loc[
            state["top_marker_lacks_direct_support"] == 1,
            ["dataset", "cluster_id"],
        ].to_dict(orient="records"),
    }

    audit_file.write_text(
        json.dumps(
            audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Router evidence state: {state_file}"
    )
    print(
        f"Router development protocol: {protocol_file}"
    )
    print(
        f"Router development audit: {audit_file}"
    )
    print(
        f"Rows processed: {len(state)}"
    )
    print(
        "No Router model was trained."
    )
    print(
        "No threshold, confidence score, or routing decision was created."
    )
    print(
        "No existing Agent output or annotation was changed."
    )


if __name__ == "__main__":
    main()
