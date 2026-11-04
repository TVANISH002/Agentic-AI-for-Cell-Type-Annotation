from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd



# Join the frozen Router evidence features with EVALUATION-ONLY ontology-aware
# targets for the current breast pilot and identify counterexamples showing why
# no single evidence heuristic should be treated as "confidence".



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
        eligible[
            "primary_router_sufficiency_target"
        ] == 1
    ].copy()

    negative = eligible.loc[
        eligible[
            "primary_router_sufficiency_target"
        ] == 0
    ].copy()

    counterexamples: list[
        dict[str, object]
    ] = []

    for feature in FEATURE_COLUMNS:
        matches = []

        for _, pos in positive.iterrows():
            for _, neg in negative.iterrows():
                if feature_value_equal(
                    pos[feature],
                    neg[feature],
                ):
                    matches.append(
                        {
                            "feature_value": (
                                None
                                if pd.isna(
                                    pos[feature]
                                )
                                else pos[feature]
                            ),
                            "sufficient_cluster": int(
                                pos["cluster_id"]
                            ),
                            "sufficient_candidate": str(
                                pos["top1_candidate"]
                            ),
                            "insufficient_cluster": int(
                                neg["cluster_id"]
                            ),
                            "insufficient_candidate": str(
                                neg["top1_candidate"]
                            ),
                        }
                    )

        if matches:
            counterexamples.append(
                {
                    "feature": feature,
                    "interpretation": (
                        "The same observed feature value occurs in at least "
                        "one sufficient and one insufficient pilot case; "
                        "therefore this feature alone cannot deterministically "
                        "define Router sufficiency in the current pilot."
                    ),
                    "examples": matches,
                }
            )

    return counterexamples


def build_near_counterexamples(
    eligible: pd.DataFrame,
) -> list[dict[str, object]]:
    """
    Identify a small set of semantically useful near-counterexamples for
    continuous features using exact values already present in the pilot.

    NO threshold is used to label cases; these are only descriptive pairs.
    """

    pairs = []

    sufficient = eligible.loc[
        eligible[
            "primary_router_sufficiency_target"
        ] == 1
    ]

    insufficient = eligible.loc[
        eligible[
            "primary_router_sufficiency_target"
        ] == 0
    ]

    continuous = [
        "direct_evidence_coverage",
        "top1_top2_gap_fraction_of_input",
        "top1_cross_database_fraction",
        "leading_non_direct_fraction",
    ]

    for feature in continuous:
        best = None

        for _, pos in sufficient.iterrows():
            for _, neg in insufficient.iterrows():
                if (
                    pd.isna(pos[feature])
                    or pd.isna(neg[feature])
                ):
                    continue

                difference = abs(
                    float(pos[feature])
                    - float(neg[feature])
                )

                candidate = {
                    "feature": feature,
                    "absolute_difference": round(
                        difference,
                        6,
                    ),
                    "sufficient_cluster": int(
                        pos["cluster_id"]
                    ),
                    "sufficient_value": float(
                        pos[feature]
                    ),
                    "insufficient_cluster": int(
                        neg["cluster_id"]
                    ),
                    "insufficient_value": float(
                        neg[feature]
                    ),
                }

                if (
                    best is None
                    or candidate[
                        "absolute_difference"
                    ]
                    < best[
                        "absolute_difference"
                    ]
                ):
                    best = candidate

        if best is not None:
            pairs.append(best)

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build descriptive pilot evidence-target audit without "
            "training a Router."
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
            "data/analysis/router_pilot_evidence_target_audit_v1"
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

    missing_features = set(
        FEATURE_COLUMNS
    ) - set(
        features.columns
    )

    if missing_features:
        raise ValueError(
            "Feature matrix missing columns: "
            + ", ".join(
                sorted(
                    missing_features
                )
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

    breast = combined.loc[
        combined["dataset"] == "breast"
    ].copy()

    eligible = breast.loc[
        pd.to_numeric(
            breast["target_eligible"],
            errors="coerce",
        ) == 1
    ].copy()

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

    counterexamples = build_counterexamples(
        eligible
    )

    near_counterexamples = (
        build_near_counterexamples(
            eligible
        )
    )

    output_table = (
        args.output_dir
        / "pilot_evidence_target_audit_v1.csv"
    )

    observation_file = (
        args.output_dir
        / "pilot_evidence_target_observations_v1.json"
    )

    audit_file = (
        args.output_dir
        / "pilot_evidence_target_audit_manifest_v1.json"
    )

    breast.to_csv(
        output_table,
        index=False,
    )

    observations = {
        "role": (
            "DESCRIPTIVE_PILOT_CASE_STUDY_ONLY"
        ),
        "eligible_cluster_count": int(
            len(eligible)
        ),
        "sufficient_count": int(
            (
                eligible[
                    "primary_router_sufficiency_target"
                ]
                == 1
            ).sum()
        ),
        "insufficient_count": int(
            (
                eligible[
                    "primary_router_sufficiency_target"
                ]
                == 0
            ).sum()
        ),
        "single_feature_exact_value_counterexamples": (
            counterexamples
        ),
        "nearest_cross_target_pairs_for_selected_continuous_features": (
            near_counterexamples
        ),
        "scientific_interpretation": (
            "The pilot contains sufficient and insufficient cases sharing "
            "several individual evidence characteristics. This motivates "
            "multivariate selective-routing evaluation but does not establish "
            "that any model will generalize."
        ),
        "forbidden_interpretations": [
            "Do not report statistical significance from these seven eligible cases.",
            "Do not fit or validate the final Router on this pilot alone.",
            "Do not interpret a single feature as calibrated confidence.",
            "Do not claim generalization from breast pilot cases.",
        ],
    }

    observation_file.write_text(
        json.dumps(
            observations,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = {
        "breast_rows": int(
            len(breast)
        ),
        "eligible_target_rows": int(
            len(eligible)
        ),
        "target_positive_rows": int(
            (
                eligible[
                    "primary_router_sufficiency_target"
                ]
                == 1
            ).sum()
        ),
        "target_negative_rows": int(
            (
                eligible[
                    "primary_router_sufficiency_target"
                ]
                == 0
            ).sum()
        ),
        "router_model_trained": False,
        "feature_selection_performed": False,
        "statistical_hypothesis_test_performed": False,
        "router_threshold_selected": False,
        "routing_decisions_created": False,
        "existing_agent_outputs_modified": False,
        "next_required_stage": (
            "Expand to multiple independent public datasets with "
            "expert-curated/published references before Router training."
        ),
    }

    audit_file.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Pilot evidence-target audit: {output_table}"
    )

    print(
        f"Pilot observations: {observation_file}"
    )

    print(
        f"Pilot audit manifest: {audit_file}"
    )

    print(
        f"Eligible pilot targets: {len(eligible)} "
        f"({manifest['target_positive_rows']} sufficient / "
        f"{manifest['target_negative_rows']} insufficient)"
    )

    print(
        "No Router model was trained."
    )

    print(
        "No feature selection, statistical hypothesis test, threshold, "
        "or routing decision was performed."
    )

    print(
        "No existing Agent output or annotation was changed."
    )

    print(
        "NEXT: expand to independent expert-curated/public datasets "
        "before Router training."
    )


if __name__ == "__main__":
    main()
