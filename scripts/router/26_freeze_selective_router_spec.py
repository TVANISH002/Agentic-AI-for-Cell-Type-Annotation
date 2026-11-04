from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


# Freeze the publication-facing design of the proposed Router BEFORE model
# fitting or threshold selection.



ROUTER_NAME = "Selective Evidence-Sufficiency Router"
ROUTER_SPEC_VERSION = "v1"

PREDICTIVE_FEATURES = [
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
]

PROHIBITED_PREDICTIVE_FIELDS = [
    "dataset",
    "dataset_name",
    "cluster",
    "cluster_id",
    "tissue",
    "disease",
    "disease_context",
    "expert_label",
    "expert_annotation",
    "reference_label",
    "reference_annotation",
    "ground_truth",
    "ground_truth_label",
    "expected_label",
    "expected_cell_type",
    "chatgpt_label",
    "llm_label",
    "llm_annotation",
    "hpa_label",
    "hpa_result",
    "router_confidence_score",
    "router_decision",
]

PRIMARY_TARGET_POLICY = {
    "target_name": "agent2_evidence_sufficient",
    "positive_class": 1,
    "negative_class": 0,
    "positive_relations": [
        "exact",
        "predicted_child",
    ],
    "negative_relations": [
        "predicted_parent",
        "sibling",
        "no_match",
    ],
    "excluded_relations": [
        "biologically_compatible_nonhierarchical",
        "unmapped",
    ],
    "inference_dependency_on_reference_labels": False,
    "note": (
        "Reference/expert annotations may generate development/evaluation "
        "targets only. They are prohibited from Router inference features."
    ),
}

MODEL_POLICY = {
    "primary_model_family": "regularized_logistic_regression",
    "final_regularization_choice": "pending_development_data",
    "class_weight_policy": "evaluate_if_class_imbalance_requires_it",
    "probability_interpretation": (
        "model output is an evidence-sufficiency score; calibration must be "
        "evaluated before probabilistic language is used"
    ),
    "final_model_fitted": False,
    "final_coefficients_available": False,
}

SPLIT_POLICY = {
    "unit_of_separation": "dataset",
    "random_cluster_split_allowed_for_final_claims": False,
    "preferred_strategy_when_enough_datasets": [
        "development_datasets",
        "validation_dataset",
        "held_out_test_dataset",
    ],
    "fallback_strategy_when_dataset_count_is_limited": (
        "leave_one_dataset_out_evaluation"
    ),
    "test_labels_may_influence_model_or_threshold": False,
}

FEATURE_REDUCTION_POLICY = {
    "status": "pending",
    "performed_before_final_model_fit": True,
    "required_analyses": [
        "pairwise_feature_correlation",
        "redundancy_review",
        "feature_ablation",
    ],
    "selection_principle": (
        "remove or consolidate redundant evidence variables only using "
        "development data and scientific interpretability; never tune feature "
        "membership on the held-out test set"
    ),
}

THRESHOLD_POLICY = {
    "final_threshold_selected": False,
    "manual_fixed_threshold_allowed": False,
    "selection_data": "validation_data_only",
    "selection_method": "risk_coverage_analysis",
    "primary_operating_goal": (
        "select the highest practical coverage while controlling error risk "
        "among clusters accepted without Agent 3"
    ),
    "threshold_frozen_before_held_out_test": True,
}

INFERENCE_CONTRACT = {
    "input": "tissue_agnostic_agent2_evidence_feature_vector",
    "reference_label_required": False,
    "expert_required": False,
    "hpa_required": False,
    "actions": {
        "sufficient": "accept_agent2_annotation",
        "insufficient": "route_to_agent3",
    },
    "monitor_class_in_final_primary_router": False,
    "note": (
        "The primary proposed Router is a selective stop/escalate decision. "
        "A separate human-review policy may be layered on top operationally."
    ),
}

EVALUATION_POLICY = {
    "primary_metrics": [
        "risk_coverage_curve",
        "area_under_risk_coverage_curve",
        "coverage",
        "selective_risk",
        "selective_accuracy",
        "routed_error_enrichment",
        "fraction_escalated",
    ],
    "full_pipeline_metrics": [
        "agent2_only_accuracy",
        "agent3_correction_rate",
        "agent3_unchanged_rate",
        "agent3_worsening_rate",
        "agent3_abstention_rate",
        "full_pipeline_accuracy",
        "literature_retrieval_calls_avoided",
        "latency",
        "api_or_compute_cost",
    ],
}

BASELINE_POLICY = {
    "router_v4_role": "heuristic_baseline_only",
    "router_v4_final_proposed_method": False,
    "router_v4_score_interpretation": (
        "heuristic evidence/uncertainty routing score, not calibrated confidence"
    ),
    "minimum_comparisons": [
        "agent2_only",
        "router_v4_heuristic_baseline",
        "proposed_selective_router",
        "agent3_for_all_clusters",
        "proposed_selective_router_plus_agent3",
    ],
}

CURRENT_DATA_STATUS = {
    "current_cluster_rows": 20,
    "current_independent_eligible_targets": 7,
    "current_positive_targets": 4,
    "current_negative_targets": 3,
    "lung_llm_labels_are_ground_truth": False,
    "sufficient_for_final_router_calibration": False,
    "reason": (
        "Seven independent expert-derived targets are insufficient for "
        "defensible final coefficient estimation, threshold calibration, "
        "and held-out evaluation."
    ),
}

REPLACEMENT_POLICY = {
    "router_v4_replaced_now": False,
    "replacement_conditions": [
        "additional independent expert_or_published labeled datasets available",
        "feature redundancy and ablation protocol completed",
        "model trained using development datasets only",
        "operating threshold selected using validation data only",
        "model and threshold frozen before held-out test",
        "held-out selective performance reported",
        "comparison against Router V4 baseline completed",
    ],
    "intended_final_pipeline_position": (
        "Agent2 -> Selective Evidence-Sufficiency Router -> "
        "accept Agent2 or escalate to Agent3"
    ),
}


def normalized_name(value: Any) -> str:
    return str(value).strip().casefold()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feature-matrix",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/analysis/router_selective_spec_v1"
        ),
    )

    args = parser.parse_args()

    if not args.feature_matrix.exists():
        raise FileNotFoundError(
            f"Feature matrix not found: {args.feature_matrix}"
        )

    feature_df = pd.read_csv(
        args.feature_matrix
    )

    columns = list(
        feature_df.columns
    )

    normalized_columns = {
        normalized_name(column): column
        for column in columns
    }

    missing_features = [
        feature
        for feature in PREDICTIVE_FEATURES
        if normalized_name(feature)
        not in normalized_columns
    ]

    present_features = [
        normalized_columns[
            normalized_name(feature)
        ]
        for feature in PREDICTIVE_FEATURES
        if normalized_name(feature)
        in normalized_columns
    ]

    prohibited_present_in_table = [
        column
        for column in columns
        if normalized_name(column)
        in {
            normalized_name(field)
            for field in PROHIBITED_PREDICTIVE_FIELDS
        }
    ]

    # Prohibited metadata fields are allowed to exist in the table.
    # They are simply forbidden from the predictive feature contract.
    prohibited_in_predictive_contract = [
        feature
        for feature in PREDICTIVE_FEATURES
        if normalized_name(feature)
        in {
            normalized_name(field)
            for field in PROHIBITED_PREDICTIVE_FIELDS
        }
    ]

    duplicate_feature_names = sorted(
        {
            feature
            for feature in PREDICTIVE_FEATURES
            if PREDICTIVE_FEATURES.count(feature) > 1
        }
    )

    feature_contract_pass = (
        not missing_features
        and not prohibited_in_predictive_contract
        and not duplicate_feature_names
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    spec = {
        "router_name": ROUTER_NAME,
        "spec_version": ROUTER_SPEC_VERSION,
        "scientific_role": (
            "Predict whether Agent 2 evidence is sufficient to stop or "
            "whether the cluster should be selectively escalated to Agent 3."
        ),
        "status": (
            "SPECIFICATION_FROZEN_MODEL_NOT_YET_FITTED"
        ),
        "feature_matrix_source": str(
            args.feature_matrix
        ),
        "feature_matrix_row_count": int(
            len(
                feature_df
            )
        ),
        "predictive_feature_count": len(
            PREDICTIVE_FEATURES
        ),
        "predictive_features": PREDICTIVE_FEATURES,
        "feature_contract_validation": {
            "missing_required_features": missing_features,
            "duplicate_predictive_features": duplicate_feature_names,
            "prohibited_fields_in_predictive_contract": (
                prohibited_in_predictive_contract
            ),
            "metadata_or_evaluation_fields_present_in_table": (
                prohibited_present_in_table
            ),
            "note": (
                "Metadata/evaluation columns may remain in the feature table "
                "for joins, auditing, and evaluation, but are excluded from "
                "the predictive feature contract."
            ),
            "status": (
                "PASS"
                if feature_contract_pass
                else "FAIL"
            ),
        },
        "target_policy": PRIMARY_TARGET_POLICY,
        "model_policy": MODEL_POLICY,
        "split_policy": SPLIT_POLICY,
        "feature_reduction_policy": FEATURE_REDUCTION_POLICY,
        "threshold_policy": THRESHOLD_POLICY,
        "inference_contract": INFERENCE_CONTRACT,
        "evaluation_policy": EVALUATION_POLICY,
        "baseline_policy": BASELINE_POLICY,
        "current_data_status": CURRENT_DATA_STATUS,
        "replacement_policy": REPLACEMENT_POLICY,
        "label_leakage_policy": {
            "expert_reference_labels_predictive": False,
            "llm_labels_predictive": False,
            "tissue_predictive": False,
            "disease_predictive": False,
            "cluster_id_predictive": False,
            "router_v4_score_predictive": False,
            "router_v4_decision_predictive": False,
            "hpa_predictive": False,
        },
        "existing_outputs_modified": False,
    }

    spec_path = (
        args.output_dir
        / "selective_router_spec_v1.json"
    )

    audit_path = (
        args.output_dir
        / "selective_router_spec_audit_v1.json"
    )

    spec_path.write_text(
        json.dumps(
            spec,
            indent=2,
        ),
        encoding="utf-8",
    )

    audit = {
        "feature_matrix": str(
            args.feature_matrix
        ),
        "row_count": int(
            len(
                feature_df
            )
        ),
        "required_predictive_features": len(
            PREDICTIVE_FEATURES
        ),
        "present_required_predictive_features": len(
            present_features
        ),
        "missing_required_predictive_features": missing_features,
        "prohibited_fields_in_predictive_contract": (
            prohibited_in_predictive_contract
        ),
        "duplicate_predictive_features": duplicate_feature_names,
        "metadata_or_evaluation_fields_present_but_not_predictive": (
            prohibited_present_in_table
        ),
        "final_model_fitted": False,
        "final_threshold_selected": False,
        "router_v4_replaced": False,
        "status": (
            "PASS"
            if feature_contract_pass
            else "FAIL"
        ),
    }

    audit_path.write_text(
        json.dumps(
            audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Selective Router specification: {spec_path}"
    )
    print(
        f"Specification audit: {audit_path}"
    )
    print()
    print(
        f"Feature-matrix rows: {len(feature_df)}"
    )
    print(
        f"Required predictive features: {len(PREDICTIVE_FEATURES)}"
    )
    print(
        f"Present required predictive features: {len(present_features)}"
    )
    print(
        f"Missing required predictive features: {len(missing_features)}"
    )
    print(
        "Prohibited fields in predictive contract: "
        f"{len(prohibited_in_predictive_contract)}"
    )
    print(
        f"Duplicate predictive features: {len(duplicate_feature_names)}"
    )
    print()
    print(
        "SPECIFICATION CONTRACT STATUS: "
        + (
            "PASS"
            if feature_contract_pass
            else "FAIL"
        )
    )
    print()
    print(
        "Final Router model fitted: NO"
    )
    print(
        "Final Router threshold selected: NO"
    )
    print(
        "Router V4 replaced: NO"
    )
    print(
        "Expert/reference labels required at inference: NO"
    )
    print(
        "No existing Agent/Router/orchestrator outputs were modified."
    )


if __name__ == "__main__":
    main()
