from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


# Freeze Version 1 of the generic Router evidence feature representation.


# Missing top-2/top-3 values are preserved as missing when those candidates
# did not pass Agent 2's minimum-support threshold. Presence indicators make
# this explicit instead of inventing artificial support values.



REQUIRED_COLUMNS = {
    "dataset",
    "cluster_id",
    "candidate_count",
    "top1_support_count",
    "top2_support_count",
    "top3_support_count",
    "top1_cross_database_support_count",
    "top1_cross_database_fraction",
    "total_input_genes",
    "direct_marker_supported_genes",
    "follow_up_gene_count",
    "direct_evidence_coverage",
    "top_marker_direct_supported",
    "first_direct_supported_rank",
    "leading_non_direct_marker_count",
    "top1_mean_supporting_gene_rank",
    "top1_candidate",
    "top1_evidence_sources",
    "top1_tissue_contexts",
    "top1_disease_contexts",
    "top1_cell_ontology_ids",
}


FORBIDDEN_PREDICTIVE_TOKENS = (
    "expert",
    "ground_truth",
    "reference_label",
    "llm_label",
    "chatgpt",
    "router_decision",
    "router_score",
    "confidence",
    "hpa_",
    "tissue_context",
    "disease_context",
    "cell_ontology",
    "top1_candidate",
    "dataset",
    "cluster_id",
)


def safe_fraction(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    numerator = pd.to_numeric(
        numerator,
        errors="coerce",
    )

    denominator = pd.to_numeric(
        denominator,
        errors="coerce",
    )

    return (
        numerator
        / denominator.where(denominator > 0)
    ).round(4)


def parse_bool_series(series: pd.Series) -> pd.Series:
    parsed = (
        series.astype(str)
        .str.strip()
        .str.casefold()
        .map(
            {
                "true": 1,
                "false": 0,
                "1": 1,
                "0": 0,
            }
        )
    )

    if parsed.isna().any():
        raise ValueError(
            "Could not parse one or more boolean evidence values."
        )

    return parsed.astype(int)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()

    # Candidate competition / support

    out["candidate_count"] = pd.to_numeric(
        df["candidate_count"],
        errors="raise",
    ).astype(int)

    out["top1_support_count"] = pd.to_numeric(
        df["top1_support_count"],
        errors="raise",
    ).astype(int)

    # A missing top-2/top-3 means that candidate rank did not exist among
    # threshold-passing candidates. Preserve NA; do not invent support = 0.
    top2 = pd.to_numeric(
        df["top2_support_count"],
        errors="coerce",
    )

    top3 = pd.to_numeric(
        df["top3_support_count"],
        errors="coerce",
    )

    out["top2_present"] = top2.notna().astype(int)
    out["top3_present"] = top3.notna().astype(int)

    out["top2_support_count"] = top2
    out["top3_support_count"] = top3

 
    # Cross-database corroboration

    out["top1_cross_database_support_count"] = pd.to_numeric(
        df["top1_cross_database_support_count"],
        errors="raise",
    ).astype(int)

    out["top1_cross_database_fraction"] = pd.to_numeric(
        df["top1_cross_database_fraction"],
        errors="coerce",
    ).round(4)

  
    # Evidence completeness

    out["direct_evidence_coverage"] = pd.to_numeric(
        df["direct_evidence_coverage"],
        errors="raise",
    ).round(4)


    # Rank-aware evidence
 
    out["top_marker_direct_supported"] = parse_bool_series(
        df["top_marker_direct_supported"]
    )

    
    # Marker-list-size-normalized descriptive variables
    # These are fractions with explicit denominators, NOT probabilities.

    out["top1_support_fraction_of_input"] = safe_fraction(
        df["top1_support_count"],
        df["total_input_genes"],
    )

    out["top2_support_fraction_of_input"] = safe_fraction(
        df["top2_support_count"],
        df["total_input_genes"],
    )

    out["top1_top2_gap_fraction_of_input"] = safe_fraction(
        (
            pd.to_numeric(
                df["top1_support_count"],
                errors="raise",
            )
            - pd.to_numeric(
                df["top2_support_count"],
                errors="coerce",
            )
        ),
        df["total_input_genes"],
    )

    out["leading_non_direct_fraction"] = safe_fraction(
        df["leading_non_direct_marker_count"],
        df["total_input_genes"],
    )

    out["first_direct_supported_rank_fraction"] = safe_fraction(
        df["first_direct_supported_rank"],
        df["total_input_genes"],
    )

    out["top1_mean_supporting_gene_rank_fraction"] = safe_fraction(
        df["top1_mean_supporting_gene_rank"],
        df["total_input_genes"],
    )

    return out


def build_metadata(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        [
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
        ]
    ].copy()


def validate_no_forbidden_features(
    feature_matrix: pd.DataFrame,
) -> None:
    bad: list[str] = []

    for column in feature_matrix.columns:
        lowered = column.casefold()

        if any(
            token in lowered
            for token in FORBIDDEN_PREDICTIVE_TOKENS
        ):
            bad.append(column)

    if bad:
        raise ValueError(
            "Forbidden predictive columns detected: "
            + ", ".join(bad)
        )


def build_schema(
    feature_matrix: pd.DataFrame,
) -> dict[str, object]:
    return {
        "schema_version": "router_features_v1",
        "status": "FROZEN_FOR_INITIAL_DEVELOPMENT",
        "purpose": (
            "Tissue-agnostic and cluster-agnostic evidence representation "
            "for future selective Router development."
        ),
        "important_scientific_constraints": [
            "No expert or LLM reference annotation is a predictive input.",
            "Dataset identity and cluster identity are metadata only.",
            "Tissue and disease strings are provenance only, not predictive features.",
            "HPA-specific fields are excluded from the core Router feature matrix.",
            "No field is interpreted as calibrated confidence.",
            "No Router threshold or decision is defined by this schema.",
            "Missing top-2/top-3 support means that rank was absent among threshold-passing candidates; it is not replaced with an artificial zero.",
            "Feature utility must be evaluated on additional/held-out benchmark data."
        ],
        "predictive_feature_columns": list(
            feature_matrix.columns
        ),
        "feature_definitions": {
            "candidate_count": (
                "Number of candidate cell types passing Agent 2's minimum-support criterion."
            ),
            "top1_support_count": (
                "Unique input markers directly supporting the highest-ranked candidate."
            ),
            "top2_present": (
                "Whether a second candidate passed Agent 2's minimum-support criterion."
            ),
            "top3_present": (
                "Whether a third candidate passed Agent 2's minimum-support criterion."
            ),
            "top2_support_count": (
                "Unique input markers directly supporting the second-ranked candidate; missing if no second threshold-passing candidate exists."
            ),
            "top3_support_count": (
                "Unique input markers directly supporting the third-ranked candidate; missing if no third threshold-passing candidate exists."
            ),
            "top1_cross_database_support_count": (
                "Leading-candidate supporting genes corroborated by both CellMarker and PanglaoDB."
            ),
            "top1_cross_database_fraction": (
                "top1_cross_database_support_count / top1_support_count."
            ),
            "direct_evidence_coverage": (
                "direct_marker_supported_genes / total_input_genes."
            ),
            "top_marker_direct_supported": (
                "Whether the highest-ranked input marker has direct context-matched marker evidence."
            ),
            "top1_support_fraction_of_input": (
                "top1_support_count / total_input_genes."
            ),
            "top2_support_fraction_of_input": (
                "top2_support_count / total_input_genes; missing if no second threshold-passing candidate exists."
            ),
            "top1_top2_gap_fraction_of_input": (
                "(top1_support_count - top2_support_count) / total_input_genes; missing if no second threshold-passing candidate exists."
            ),
            "leading_non_direct_fraction": (
                "leading_non_direct_marker_count / total_input_genes."
            ),
            "first_direct_supported_rank_fraction": (
                "first_direct_supported_rank / total_input_genes."
            ),
            "top1_mean_supporting_gene_rank_fraction": (
                "top1_mean_supporting_gene_rank / total_input_genes."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze tissue-agnostic Router feature matrix Version 1."
        )
    )

    parser.add_argument(
        "--rank-aware-table",
        type=Path,
        required=True,
        help="Path to router_evidence_table_rank_aware.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("router_feature_matrix_v1"),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        args.rank_aware_table
    )

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            "Rank-aware evidence table is missing columns: "
            + ", ".join(sorted(missing))
        )

    feature_matrix = build_feature_matrix(
        df
    )

    metadata = build_metadata(
        df
    )

    validate_no_forbidden_features(
        feature_matrix
    )

    if len(feature_matrix) != len(metadata):
        raise ValueError(
            "Feature matrix and metadata row counts differ."
        )

    feature_matrix.insert(
        0,
        "row_id",
        range(1, len(feature_matrix) + 1),
    )

    metadata.insert(
        0,
        "row_id",
        range(1, len(metadata) + 1),
    )

    matrix_file = (
        args.output_dir
        / "router_feature_matrix_v1.csv"
    )

    metadata_file = (
        args.output_dir
        / "router_feature_metadata_v1.csv"
    )

    schema_file = (
        args.output_dir
        / "router_feature_schema_v1.json"
    )

    feature_matrix.to_csv(
        matrix_file,
        index=False,
    )

    metadata.to_csv(
        metadata_file,
        index=False,
    )

    schema = build_schema(
        feature_matrix.drop(
            columns=["row_id"]
        )
    )

    schema_file.write_text(
        json.dumps(
            schema,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Frozen Router feature matrix: {matrix_file}"
    )

    print(
        f"Router feature metadata: {metadata_file}"
    )

    print(
        f"Router feature schema: {schema_file}"
    )

    print(
        f"Rows frozen: {len(feature_matrix)}"
    )

    print(
        "Generic-schema check: PASS"
    )

    print(
        "Tissue names, cluster IDs, expert labels, LLM labels, HPA fields, "
        "confidence scores, and Router decisions are NOT predictive features."
    )

    print(
        "No existing Agent or annotation result was changed."
    )


if __name__ == "__main__":
    main()
