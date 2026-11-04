from __future__ import annotations

import argparse
import json
from pathlib import Path


PROVISIONAL_MODEL_FEATURES = [
    "candidate_count",
    "top1_support_count",
    "top2_present",
    "top3_present",
    "top2_support_count",
    "top3_support_count",
    "top1_cross_database_fraction",
    "direct_evidence_coverage",
    "top1_support_fraction_of_input",
    "top2_support_fraction_of_input",
    "top1_top2_gap_fraction_of_input",
    "leading_non_direct_fraction",
    "first_direct_supported_rank_fraction",
    "top1_mean_supporting_gene_rank_fraction",
]


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/analysis/"
            "router_selective_training_protocol_v1"
        ),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    protocol = {
        "protocol_name": (
            "Selective Evidence-Sufficiency Router "
            "Training and Evaluation Protocol"
        ),
        "version": "v1",
        "status": (
            "FROZEN_BEFORE_FINAL_MODEL_FITTING"
        ),
        "scientific_objective": (
            "Predict whether Agent 2 evidence is sufficient to stop "
            "annotation or whether the cluster should be selectively "
            "escalated to Agent 3."
        ),
        "prediction_target": {
            "name": "agent2_evidence_sufficient",
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
            "reference_labels_used_at_inference": False,
        },
        "predictive_features": {
            "count": len(
                PROVISIONAL_MODEL_FEATURES
            ),
            "features": (
                PROVISIONAL_MODEL_FEATURES
            ),
            "status": (
                "provisional_pending_development_data_ablation"
            ),
            "forbidden_predictors": [
                "dataset",
                "cluster_id",
                "tissue",
                "disease",
                "expert/reference annotation",
                "LLM annotation",
                "HPA result",
                "Router V4 score",
                "Router V4 decision",
            ],
        },
        "model": {
            "primary_family": (
                "L2_regularized_logistic_regression"
            ),
            "reason": (
                "Interpretable, low-complexity classifier suitable for "
                "structured evidence features and coefficient inspection."
            ),
            "final_regularization_strength": (
                "selected_using_development_data_only"
            ),
            "class_weight": (
                "evaluated_on_development_data_only_if_imbalance_requires"
            ),
            "nonlinear_models": (
                "optional_secondary_comparators_only_if_sample_size_supports"
            ),
        },
        "preprocessing": {
            "zero_variance_filter": (
                "fit_on_development_training_data_only"
            ),
            "continuous_scaling": (
                "fit_on_development_training_data_only"
            ),
            "structural_missingness": {
                "top3_support_count": (
                    "encode_as_zero_only_when_top3_present_equals_zero; "
                    "retain top3_present as the structural indicator"
                )
            },
            "unexpected_missingness": (
                "data_quality_error_unless_predeclared"
            ),
        },
        "data_partitioning": {
            "unit": "dataset",
            "random_cluster_split_for_final_claims": False,
            "preferred_roles": [
                "development",
                "validation",
                "held_out_test",
            ],
            "preferred_design": (
                "one_or_more_development_datasets + "
                "independent_validation_dataset + "
                "independent_held_out_test_dataset"
            ),
            "limited_dataset_fallback": (
                "leave_one_dataset_out_evaluation; "
                "do_not describe as a single fixed held-out test design"
            ),
            "test_data_may_influence_training": False,
            "test_data_may_influence_feature_selection": False,
            "test_data_may_influence_threshold": False,
        },
        "feature_ablation": {
            "performed_on": (
                "development_data_only"
            ),
            "predeclared_groups": [
                [
                    "top2_support_count",
                    "top3_support_count",
                ],
                [
                    "top1_support_count",
                    "top1_support_fraction_of_input",
                ],
                [
                    "leading_non_direct_fraction",
                    "first_direct_supported_rank_fraction",
                ],
                [
                    "top1_support_fraction_of_input",
                    "top2_support_fraction_of_input",
                    "top1_top2_gap_fraction_of_input",
                ],
            ],
            "selection_basis": [
                "predictive stability",
                "selective performance",
                "coefficient stability",
                "scientific interpretability",
            ],
            "held_out_test_used_for_ablation": False,
        },
        "threshold_selection": {
            "manual_fixed_threshold": False,
            "selection_partition": "validation_only",
            "method": "risk_coverage_analysis",
            "primary_principle": (
                "maximize useful Agent 2 coverage subject to an acceptable "
                "error risk among automatically accepted clusters"
            ),
            "threshold_frozen_before_test": True,
        },
        "router_actions": {
            "sufficient": (
                "accept_agent2_annotation"
            ),
            "insufficient": (
                "route_to_agent3"
            ),
            "primary_model_has_monitor_class": False,
        },
        "primary_selective_metrics": [
            "coverage",
            "selective_risk",
            "selective_accuracy",
            "risk_coverage_curve",
            "area_under_risk_coverage_curve",
            "fraction_escalated",
            "routed_error_enrichment",
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
        "required_baselines": [
            "Agent2_only",
            "Router_V4_heuristic",
            "Selective_Evidence_Sufficiency_Router",
            "Agent3_for_all",
            "Selective_Router_plus_Agent3",
        ],
        "router_v4": {
            "role": "heuristic_baseline",
            "final_proposed_method": False,
            "score_language": (
                "heuristic routing score, not calibrated confidence"
            ),
        },
        "replacement_gate": {
            "replace_v4_only_after": [
                "additional independently labeled datasets are available",
                "development-only feature ablation is completed",
                "development-only model fitting is completed",
                "validation-only operating threshold is selected",
                "model and threshold are frozen",
                "held-out selective performance is evaluated",
                "Router V4 baseline comparison is completed",
                "end-to-end Agent3 escalation performance is evaluated",
            ],
            "v4_replaced_now": False,
        },
        "current_data_status": {
            "independent_labeled_clusters": 7,
            "sufficient": 4,
            "insufficient": 3,
            "independently_labeled_datasets": 1,
            "dataset_names": [
                "breast"
            ],
            "provisional_feature_count": 14,
            "ready_for_final_fitting": False,
        },
        "current_allowed_uses": [
            "engineering validation",
            "case studies",
            "feature-definition audits",
            "software testing",
        ],
        "current_disallowed_claims": [
            "final calibrated Router",
            "held-out Router accuracy",
            "cross-tissue predictive generalization",
            "calibrated confidence",
            "validated final operating threshold",
        ],
        "expert_required_at_inference": False,
        "model_trained_by_this_step": False,
        "threshold_selected_by_this_step": False,
        "existing_pipeline_outputs_modified": False,
    }

    protocol_path = (
        args.output_dir
        / "selective_router_training_protocol_v1.json"
    )

    protocol_path.write_text(
        json.dumps(
            protocol,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Training/evaluation protocol: {protocol_path}"
    )
    print()
    print(
        "Protocol status: FROZEN BEFORE FINAL MODEL FITTING"
    )
    print(
        f"Provisional predictive features: "
        f"{len(PROVISIONAL_MODEL_FEATURES)}"
    )
    print(
        "Primary model family: L2-regularized logistic regression"
    )
    print(
        "Final split unit: DATASET"
    )
    print(
        "Threshold selection: VALIDATION risk-coverage analysis"
    )
    print(
        "Held-out test may influence threshold: NO"
    )
    print(
        "Expert/reference labels required at inference: NO"
    )
    print(
        "Router V4 role: HEURISTIC BASELINE"
    )
    print(
        "Router V4 replaced now: NO"
    )
    print(
        "Model trained: NO"
    )
    print(
        "Threshold selected: NO"
    )
    print(
        "No existing Agent/Router/orchestrator outputs were modified."
    )


if __name__ == "__main__":
    main()
