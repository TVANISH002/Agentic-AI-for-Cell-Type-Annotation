from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd



# Freeze how ontology relations will later be converted into evaluation
# targets BEFORE any Router model is trained.



REQUIRED_COLUMNS = {
    "row_id",
    "dataset",
    "cluster_id",
    "agent2_top1_candidate",
    "agent2_top1_cell_ontology_ids",
    "reference_label_raw",
    "reference_type",
    "eligible_for_supervised_router",
    "harmonization_status",
}


ALLOWED_RELATIONS = {
    "exact",
    "predicted_child_of_reference",
    "predicted_parent_of_reference",
    "sibling",
    "biologically_compatible_nonhierarchical",
    "no_match",
    "unmapped",
}


PRIMARY_REFERENCE_TYPES = {
    "biology_expert",
    "published_reference",
    "expert_curated",
}


def require_columns(
    df: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} missing required columns: "
            + ", ".join(sorted(missing))
        )


def build_review_template(
    worklist: pd.DataFrame,
) -> pd.DataFrame:

    eligible = pd.to_numeric(
        worklist["eligible_for_supervised_router"],
        errors="raise",
    ).astype(int) == 1

    review = worklist.loc[
        eligible,
        [
            "row_id",
            "dataset",
            "cluster_id",
            "agent2_top1_candidate",
            "agent2_top1_cell_ontology_ids",
            "reference_label_raw",
            "reference_type",
        ],
    ].copy()

    
    # Fields that must be resolved BEFORE target creation.
    
    review["reference_label_harmonized"] = ""
    review["reference_cell_ontology_id"] = ""
    review["reference_mapping_status"] = "PENDING"

    review["ontology_relation"] = ""
    review["ontology_distance"] = pd.NA

    review["relation_evidence"] = ""
    review["relation_source"] = ""
    review["reviewer"] = ""
    review["review_status"] = "PENDING"

    # Targets remain unassigned until relation review is complete.
    review["strict_exact_target"] = pd.NA
    review["primary_router_sufficiency_target"] = pd.NA
    review["permissive_hierarchy_target"] = pd.NA

    review["target_status"] = "NOT_CREATED"

    return review


def build_policy() -> dict[str, object]:
    return {
        "policy_version": "router_correctness_policy_v1",
        "status": "FROZEN_BEFORE_TARGET_CREATION",
        "scientific_question": (
            "Is Agent 2's top-ranked annotation sufficiently compatible "
            "with an independent reference to stop, or does it remain "
            "biologically under-resolved / discordant and therefore warrant "
            "selective literature escalation?"
        ),
        "ontology_backbone": {
            "name": "Cell Ontology",
            "recommended_release": "2026-06-08",
            "release_policy": (
                "Use a version-pinned Cell Ontology release for reproducible "
                "mapping and relation traversal."
            ),
        },
        "allowed_relations": [
            "exact",
            "predicted_child_of_reference",
            "predicted_parent_of_reference",
            "sibling",
            "biologically_compatible_nonhierarchical",
            "no_match",
            "unmapped",
        ],
        "targets": {
            "strict_exact": {
                "correct_relations": [
                    "exact",
                ],
                "incorrect_relations": [
                    "predicted_child_of_reference",
                    "predicted_parent_of_reference",
                    "sibling",
                    "biologically_compatible_nonhierarchical",
                    "no_match",
                ],
                "excluded_relations": [
                    "unmapped",
                ],
                "role": (
                    "Secondary stringent reporting metric."
                ),
            },
            "primary_router_sufficiency": {
                "correct_relations": [
                    "exact",
                    "predicted_child_of_reference",
                ],
                "incorrect_relations": [
                    "predicted_parent_of_reference",
                    "sibling",
                    "no_match",
                ],
                "excluded_relations": [
                    "biologically_compatible_nonhierarchical",
                    "unmapped",
                ],
                "role": (
                    "Primary future Router development/evaluation target."
                ),
                "rationale": (
                    "A prediction that is more specific than a broader "
                    "reference can still be sufficient when it lies within "
                    "the same ontology branch. A prediction that is broader "
                    "than the reference is treated as under-resolved for the "
                    "primary Router objective."
                ),
            },
            "permissive_hierarchy": {
                "correct_relations": [
                    "exact",
                    "predicted_child_of_reference",
                    "predicted_parent_of_reference",
                ],
                "incorrect_relations": [
                    "sibling",
                    "no_match",
                ],
                "excluded_relations": [
                    "biologically_compatible_nonhierarchical",
                    "unmapped",
                ],
                "role": (
                    "Sensitivity analysis only; not the primary target."
                ),
            },
        },
        "nonhierarchical_policy": {
            "biologically_compatible_nonhierarchical": (
                "Report separately and exclude from primary binary target "
                "unless a future pre-specified external standard resolves "
                "the relation."
            ),
            "unmapped": (
                "Exclude from binary target creation and report mapping "
                "failure rate."
            ),
        },
        "ontology_distance_policy": (
            "Preserve ontology graph distance when available, but do not "
            "introduce an arbitrary maximum-distance cutoff in Version 1."
        ),
        "reference_policy": (
            "Only independent biology-expert, published-reference, or "
            "expert-curated labels may define supervised correctness."
        ),
        "llm_reference_policy": (
            "LLM-only labels remain descriptive comparison metadata and "
            "cannot define Router targets."
        ),
        "leakage_policy": (
            "Reference labels, harmonized ontology relations, and correctness "
            "targets are development/evaluation metadata only and must never "
            "enter Router inference features."
        ),
        "held_out_policy": (
            "Once targets are finalized, train/validation/test separation "
            "must occur at dataset level; held-out test labels cannot be used "
            "for feature selection, coefficient tuning, or threshold choice."
        ),
        "reporting_requirements": [
            "Report exact-match performance separately.",
            "Report ontology relation counts: exact, child, parent, sibling, no_match, unmapped/nonhierarchical.",
            "Report primary hierarchy-aware Router performance.",
            "Report permissive sensitivity analysis.",
            "Report excluded/unmapped fraction.",
        ],
    }


def validate_reference_types(
    worklist: pd.DataFrame,
) -> None:
    eligible = pd.to_numeric(
        worklist["eligible_for_supervised_router"],
        errors="raise",
    ).astype(int) == 1

    bad = worklist.loc[
        eligible
        & ~worklist["reference_type"]
        .fillna("")
        .astype(str)
        .isin(PRIMARY_REFERENCE_TYPES),
        [
            "dataset",
            "cluster_id",
            "reference_type",
        ],
    ]

    if not bad.empty:
        raise ValueError(
            "Eligible rows contain unsupported primary reference types:\n"
            + bad.to_string(index=False)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze ontology-aware Router correctness policy and create "
            "the harmonization review template."
        )
    )

    parser.add_argument(
        "--worklist",
        type=Path,
        required=True,
        help=(
            "router_reference_harmonization_worklist_v1.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/analysis/router_correctness_policy_v1"
        ),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    worklist = pd.read_csv(
        args.worklist
    )

    require_columns(
        worklist,
        REQUIRED_COLUMNS,
        "Reference harmonization worklist",
    )

    if worklist["row_id"].duplicated().any():
        raise ValueError(
            "Duplicate row_id values in harmonization worklist."
        )

    validate_reference_types(
        worklist
    )

    review = build_review_template(
        worklist
    )

    policy = build_policy()

    review_file = (
        args.output_dir
        / "router_harmonization_review_template_v1.csv"
    )

    policy_file = (
        args.output_dir
        / "router_correctness_policy_v1.json"
    )

    audit_file = (
        args.output_dir
        / "router_correctness_policy_audit_v1.json"
    )

    review.to_csv(
        review_file,
        index=False,
    )

    policy_file.write_text(
        json.dumps(
            policy,
            indent=2,
        ),
        encoding="utf-8",
    )

    audit = {
        "rows_in_input_worklist": int(
            len(worklist)
        ),
        "eligible_rows_for_review": int(
            len(review)
        ),
        "datasets_in_review": sorted(
            review["dataset"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        "correctness_policy_frozen": True,
        "ontology_relations_auto_assigned": False,
        "correctness_targets_created": False,
        "router_model_trained": False,
        "router_threshold_selected": False,
        "routing_decisions_created": False,
        "llm_labels_used_as_ground_truth": False,
        "existing_agent_outputs_modified": False,
    }

    audit_file.write_text(
        json.dumps(
            audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Router correctness policy: {policy_file}"
    )
    print(
        f"Harmonization review template: {review_file}"
    )
    print(
        f"Correctness policy audit: {audit_file}"
    )
    print(
        f"Eligible rows awaiting ontology review: {len(review)}"
    )
    print(
        "Correctness policy frozen: YES"
    )
    print(
        "Ontology relations were NOT auto-assigned."
    )
    print(
        "Correctness targets were NOT created."
    )
    print(
        "No Router model, threshold, or routing decision was created."
    )
    print(
        "No existing Agent output or annotation was changed."
    )


if __name__ == "__main__":
    main()
