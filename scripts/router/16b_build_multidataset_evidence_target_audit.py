from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# Generalize the pilot evidence–target audit across ANY dataset/tissue.



FEATURE_COLUMNS = [
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


REQUIRED_METADATA = {
    "row_id",
    "dataset",
    "cluster_id",
    "top1_candidate",
}


REQUIRED_RELATIONS = {
    "row_id",
    "ontology_relation",
    "primary_router_sufficiency_target",
    "target_eligible",
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


def feature_value_equal(
    a: object,
    b: object,
    tolerance: float = 1e-9,
) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True

    if pd.isna(a) or pd.isna(b):
        return False

    try:
        return abs(float(a) - float(b)) <= tolerance
    except (TypeError, ValueError):
        return str(a) == str(b)


def build_counterexamples(
    eligible: pd.DataFrame,
) -> list[dict[str, object]]:

    positive = eligible.loc[
        eligible["primary_router_sufficiency_target"] == 1
    ]

    negative = eligible.loc[
        eligible["primary_router_sufficiency_target"] == 0
    ]

    counterexamples: list[dict[str, object]] = []

    if positive.empty or negative.empty:
        return counterexamples

    for feature in FEATURE_COLUMNS:
        examples = []

        for _, pos in positive.iterrows():
            for _, neg in negative.iterrows():
                if feature_value_equal(
                    pos[feature],
                    neg[feature],
                ):
                    examples.append(
                        {
                            "feature_value": (
                                None
                                if pd.isna(pos[feature])
                                else pos[feature]
                            ),
                            "sufficient_dataset": str(
                                pos["dataset"]
                            ),
                            "sufficient_cluster": int(
                                pos["cluster_id"]
                            ),
                            "sufficient_candidate": str(
                                pos["top1_candidate"]
                            ),
                            "insufficient_dataset": str(
                                neg["dataset"]
                            ),
                            "insufficient_cluster": int(
                                neg["cluster_id"]
                            ),
                            "insufficient_candidate": str(
                                neg["top1_candidate"]
                            ),
                        }
                    )

        if examples:
            counterexamples.append(
                {
                    "feature": feature,
                    "interpretation": (
                        "The same observed feature value occurs in at least "
                        "one sufficient and one insufficient eligible case; "
                        "therefore this feature alone cannot deterministically "
                        "define Router sufficiency in the current development data."
                    ),
                    "examples": examples,
                }
            )

    return counterexamples


def build_dataset_summary(
    combined: pd.DataFrame,
) -> list[dict[str, object]]:

    summaries = []

    for dataset, group in combined.groupby(
        "dataset",
        sort=True,
    ):
        eligible = group.loc[
            pd.to_numeric(
                group["target_eligible"],
                errors="coerce",
            ) == 1
        ].copy()

        if not eligible.empty:
            targets = pd.to_numeric(
                eligible[
                    "primary_router_sufficiency_target"
                ],
                errors="coerce",
            )

            sufficient = int(
                (targets == 1).sum()
            )

            insufficient = int(
                (targets == 0).sum()
            )

        else:
            sufficient = 0
            insufficient = 0

        summaries.append(
            {
                "dataset": str(dataset),
                "cluster_rows": int(len(group)),
                "eligible_target_rows": int(len(eligible)),
                "sufficient_target_rows": sufficient,
                "insufficient_target_rows": insufficient,
                "supervised_router_role": (
                    "ELIGIBLE_FOR_DESCRIPTIVE_TARGET_AUDIT"
                    if len(eligible) > 0
                    else "NO_INDEPENDENT_PRIMARY_TARGETS"
                ),
            }
        )

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a tissue-agnostic multi-dataset evidence-target audit "
            "without training a Router."
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
        "--relations",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--path-audit",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/analysis/router_multidataset_evidence_target_audit_v1"
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

    relations = pd.read_csv(
        args.relations
    )

    path_audit = pd.read_csv(
        args.path_audit
    )

    require_columns(
        metadata,
        REQUIRED_METADATA,
        "Metadata",
    )

    require_columns(
        relations,
        REQUIRED_RELATIONS,
        "Relations",
    )

    missing_features = (
        set(FEATURE_COLUMNS)
        - set(features.columns)
    )

    if missing_features:
        raise ValueError(
            "Feature matrix missing columns: "
            + ", ".join(
                sorted(missing_features)
            )
        )

    for name, frame in {
        "features": features,
        "metadata": metadata,
        "relations": relations,
        "path_audit": path_audit,
    }.items():
        if "row_id" not in frame.columns:
            raise ValueError(
                f"{name} missing row_id."
            )

        if frame["row_id"].duplicated().any():
            raise ValueError(
                f"Duplicate row_id in {name}."
            )

    # -----------------------------------------------------------------
    # Relations/path audit currently exist only for rows with independent
    # primary references. Use LEFT joins so datasets without such references
    # (e.g., current lung dataset) remain visible rather than disappearing.
    # -----------------------------------------------------------------
    combined = (
        metadata[
            [
                "row_id",
                "dataset",
                "cluster_id",
                "top1_candidate",
            ]
        ]
        .merge(
            features[
                [
                    "row_id",
                    *FEATURE_COLUMNS,
                ]
            ],
            on="row_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            relations[
                [
                    "row_id",
                    "ontology_relation",
                    "primary_router_sufficiency_target",
                    "target_eligible",
                ]
            ],
            on="row_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            path_audit[
                [
                    "row_id",
                    "nearest_shared_ancestor_id",
                    "nearest_shared_ancestor_label",
                    "prediction_to_shared_distance",
                    "reference_to_shared_distance",
                ]
            ],
            on="row_id",
            how="left",
            validate="one_to_one",
        )
    )

    combined[
        "has_independent_primary_target"
    ] = (
        pd.to_numeric(
            combined["target_eligible"],
            errors="coerce",
        )
        == 1
    ).astype(int)

    eligible = combined.loc[
        combined[
            "has_independent_primary_target"
        ] == 1
    ].copy()

    if not eligible.empty:
        eligible[
            "primary_router_sufficiency_target"
        ] = pd.to_numeric(
            eligible[
                "primary_router_sufficiency_target"
            ],
            errors="raise",
        ).astype(int)

        eligible[
            "pilot_target_label"
        ] = eligible[
            "primary_router_sufficiency_target"
        ].map(
            {
                1: "SUFFICIENT",
                0: "INSUFFICIENT",
            }
        )

    dataset_summary = build_dataset_summary(
        combined
    )

    counterexamples = build_counterexamples(
        eligible
    )

    combined_file = (
        args.output_dir
        / "multidataset_evidence_target_audit_v1.csv"
    )

    summary_file = (
        args.output_dir
        / "multidataset_evidence_target_summary_v1.json"
    )

    manifest_file = (
        args.output_dir
        / "multidataset_evidence_target_manifest_v1.json"
    )

    combined.to_csv(
        combined_file,
        index=False,
    )

    summary = {
        "role": (
            "DESCRIPTIVE_MULTI_DATASET_DEVELOPMENT_AUDIT_ONLY"
        ),
        "dataset_summaries": dataset_summary,
        "total_cluster_rows": int(
            len(combined)
        ),
        "total_eligible_target_rows": int(
            len(eligible)
        ),
        "eligible_datasets": sorted(
            eligible[
                "dataset"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        if not eligible.empty
        else [],
        "datasets_without_primary_targets": sorted(
            [
                item["dataset"]
                for item in dataset_summary
                if item[
                    "eligible_target_rows"
                ] == 0
            ]
        ),
        "single_feature_counterexamples": counterexamples,
        "scientific_interpretation": (
            "The audit is tissue-agnostic. Datasets may contribute Router "
            "targets only when independent expert-curated or published "
            "references are available. Datasets without such references "
            "remain valid for inference, evidence analysis, Agent 3 testing, "
            "runtime analysis, and descriptive agreement, but not supervised "
            "Router correctness."
        ),
        "forbidden_interpretations": [
            "Do not treat LLM-only labels as ground truth.",
            "Do not claim tissue generalization until independent held-out datasets are evaluated.",
            "Do not fit the final Router on the current small pilot alone.",
            "Do not interpret individual evidence features as calibrated confidence.",
        ],
    }

    summary_file.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = {
        "dataset_count": int(
            combined["dataset"].nunique()
        ),
        "cluster_rows": int(
            len(combined)
        ),
        "eligible_target_rows": int(
            len(eligible)
        ),
        "hard_coded_tissue_names": False,
        "llm_only_labels_used_as_ground_truth": False,
        "router_model_trained": False,
        "feature_selection_performed": False,
        "statistical_hypothesis_test_performed": False,
        "router_threshold_selected": False,
        "routing_decisions_created": False,
        "existing_agent_outputs_modified": False,
        "next_required_stage": (
            "Add independent expert-curated/public datasets, run the same "
            "frozen Agent 1/Agent 2 evidence extraction, then perform "
            "dataset-level development/validation/held-out Router evaluation."
        ),
    }

    manifest_file.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Multi-dataset evidence-target audit: {combined_file}"
    )

    print(
        f"Multi-dataset summary: {summary_file}"
    )

    print(
        f"Multi-dataset manifest: {manifest_file}"
    )

    print(
        f"Datasets processed: {manifest['dataset_count']}"
    )

    print(
        f"Cluster rows processed: {manifest['cluster_rows']}"
    )

    print(
        f"Eligible independent target rows: {manifest['eligible_target_rows']}"
    )

    print(
        "Hard-coded tissue names: NO"
    )

    print(
        "LLM-only labels were NOT used as ground truth."
    )

    print(
        "No Router model, feature selection, threshold, "
        "or routing decision was created."
    )

    print(
        "No existing Agent output or annotation was changed."
    )


if __name__ == "__main__":
    main()
