from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import pandas as pd



# READ-ONLY / UNSUPERVISED.


TOL = 1e-9


def numeric(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    series = frame[
        column
    ]

    if series.dtype == object:
        lowered = (
            series.astype(str)
            .str.strip()
            .str.casefold()
        )

        boolean_map = {
            "true": 1.0,
            "false": 0.0,
            "yes": 1.0,
            "no": 0.0,
        }

        mapped = lowered.map(
            boolean_map
        )

        parsed = pd.to_numeric(
            series,
            errors="coerce",
        )

        mask = (
            parsed.isna()
            & mapped.notna()
        )

        parsed.loc[
            mask
        ] = mapped.loc[
            mask
        ]

        return parsed

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def close(
    left: Any,
    right: Any,
    tol: float = TOL,
) -> bool:
    if pd.isna(
        left
    ) and pd.isna(
        right
    ):
        return True

    if pd.isna(
        left
    ) or pd.isna(
        right
    ):
        return False

    return math.isclose(
        float(
            left
        ),
        float(
            right
        ),
        rel_tol=tol,
        abs_tol=tol,
    )


def check_formula(
    frame: pd.DataFrame,
    *,
    name: str,
    required_columns: list[str],
    expected_fn: Callable[
        [pd.Series],
        float | None,
    ],
    observed_column: str,
) -> dict[str, Any]:

    missing = [
        column
        for column in required_columns
        + [
            observed_column
        ]
        if column not in frame.columns
    ]

    if missing:
        return {
            "relationship": name,
            "status": "NOT_TESTED",
            "reason": (
                "missing columns: "
                + ", ".join(
                    missing
                )
            ),
            "rows_tested": 0,
            "rows_matching": 0,
            "rows_mismatching": 0,
            "mismatch_examples": [],
        }

    mismatch_examples: list[
        dict[str, Any]
    ] = []

    tested = 0
    matching = 0

    for index, row in frame.iterrows():
        expected = expected_fn(
            row
        )

        observed = row[
            observed_column
        ]

        if expected is None:
            continue

        tested += 1

        if close(
            observed,
            expected,
            tol=1e-6,
        ):
            matching += 1
        else:
            if len(
                mismatch_examples
            ) < 10:
                mismatch_examples.append(
                    {
                        "row_index": int(
                            index
                        ),
                        "observed": (
                            None
                            if pd.isna(
                                observed
                            )
                            else float(
                                observed
                            )
                        ),
                        "expected": (
                            None
                            if expected is None
                            or pd.isna(
                                expected
                            )
                            else float(
                                expected
                            )
                        ),
                    }
                )

    mismatching = (
        tested
        - matching
    )

    return {
        "relationship": name,
        "status": (
            "EXACT_IN_CURRENT_TABLE"
            if tested > 0
            and mismatching == 0
            else (
                "NOT_EXACT"
                if tested > 0
                else "NOT_TESTED"
            )
        ),
        "rows_tested": int(
            tested
        ),
        "rows_matching": int(
            matching
        ),
        "rows_mismatching": int(
            mismatching
        ),
        "mismatch_examples": (
            mismatch_examples
        ),
    }


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
            "data/analysis/"
            "router_selective_structural_dependency_v1"
        ),
    )

    args = parser.parse_args()

    if not args.feature_matrix.exists():
        raise FileNotFoundError(
            f"Feature matrix not found: "
            f"{args.feature_matrix}"
        )

    raw = pd.read_csv(
        args.feature_matrix
    )

    frame = raw.copy()

    # Convert relevant columns to numeric where present.
    candidate_columns = [
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
        "total_input_genes",
    ]

    for column in candidate_columns:
        if column in frame.columns:
            frame[
                column
            ] = numeric(
                frame,
                column,
            )

    relationships: list[
        dict[str, Any]
    ] = []

    # -------------------------------------------------------------------------
    # 1. Cross-database fraction should generally be:
    #    cross-database supporting genes / top1 supporting genes
    # -------------------------------------------------------------------------
    relationships.append(
        check_formula(
            frame,
            name=(
                "top1_cross_database_fraction = "
                "top1_cross_database_support_count / top1_support_count"
            ),
            required_columns=[
                "top1_cross_database_support_count",
                "top1_support_count",
            ],
            observed_column=(
                "top1_cross_database_fraction"
            ),
            expected_fn=lambda row: (
                float(
                    row[
                        "top1_cross_database_support_count"
                    ]
                )
                / float(
                    row[
                        "top1_support_count"
                    ]
                )
                if (
                    not pd.isna(
                        row[
                            "top1_cross_database_support_count"
                        ]
                    )
                    and not pd.isna(
                        row[
                            "top1_support_count"
                        ]
                    )
                    and float(
                        row[
                            "top1_support_count"
                        ]
                    )
                    > 0
                )
                else None
            ),
        )
    )

    # -------------------------------------------------------------------------
    # 2. top marker direct support vs leading non-direct fraction.
    #    If no leading unsupported markers exist, the top marker should be
    #    directly supported. This tests the logical equivalence in the table.
    # -------------------------------------------------------------------------
    if {
        "top_marker_direct_supported",
        "leading_non_direct_fraction",
    }.issubset(
        frame.columns
    ):
        mismatches = []

        for index, row in frame.iterrows():
            direct = row[
                "top_marker_direct_supported"
            ]

            leading_fraction = row[
                "leading_non_direct_fraction"
            ]

            if pd.isna(
                direct
            ) or pd.isna(
                leading_fraction
            ):
                continue

            expected_direct = (
                1.0
                if math.isclose(
                    float(
                        leading_fraction
                    ),
                    0.0,
                    abs_tol=1e-9,
                )
                else 0.0
            )

            if not close(
                direct,
                expected_direct,
            ):
                mismatches.append(
                    {
                        "row_index": int(
                            index
                        ),
                        "top_marker_direct_supported": float(
                            direct
                        ),
                        "leading_non_direct_fraction": float(
                            leading_fraction
                        ),
                        "expected_direct": float(
                            expected_direct
                        ),
                    }
                )

        tested = int(
            frame[
                [
                    "top_marker_direct_supported",
                    "leading_non_direct_fraction",
                ]
            ]
            .dropna()
            .shape[
                0
            ]
        )

        relationships.append(
            {
                "relationship": (
                    "top_marker_direct_supported = 1 "
                    "iff leading_non_direct_fraction = 0"
                ),
                "status": (
                    "EXACT_IN_CURRENT_TABLE"
                    if tested > 0
                    and not mismatches
                    else (
                        "NOT_EXACT"
                        if tested > 0
                        else "NOT_TESTED"
                    )
                ),
                "rows_tested": tested,
                "rows_matching": (
                    tested
                    - len(
                        mismatches
                    )
                ),
                "rows_mismatching": len(
                    mismatches
                ),
                "mismatch_examples": (
                    mismatches[
                        :10
                    ]
                ),
            }
        )
    else:
        relationships.append(
            {
                "relationship": (
                    "top_marker_direct_supported = 1 "
                    "iff leading_non_direct_fraction = 0"
                ),
                "status": "NOT_TESTED",
                "reason": "required columns missing",
                "rows_tested": 0,
                "rows_matching": 0,
                "rows_mismatching": 0,
                "mismatch_examples": [],
            }
        )

    # -------------------------------------------------------------------------
    # 3. top3 structural missingness check.
    # -------------------------------------------------------------------------
    if {
        "top3_present",
        "top3_support_count",
    }.issubset(
        frame.columns
    ):
        mismatch_examples = []

        for index, row in frame.iterrows():
            present = row[
                "top3_present"
            ]

            support = row[
                "top3_support_count"
            ]

            if pd.isna(
                present
            ):
                continue

            expected_missing = (
                float(
                    present
                )
                == 0.0
            )

            actual_missing = pd.isna(
                support
            )

            if (
                expected_missing
                != actual_missing
            ):
                mismatch_examples.append(
                    {
                        "row_index": int(
                            index
                        ),
                        "top3_present": float(
                            present
                        ),
                        "top3_support_count": (
                            None
                            if pd.isna(
                                support
                            )
                            else float(
                                support
                            )
                        ),
                    }
                )

        tested = int(
            frame[
                "top3_present"
            ]
            .notna()
            .sum()
        )

        relationships.append(
            {
                "relationship": (
                    "top3_support_count is missing "
                    "iff top3_present = 0"
                ),
                "status": (
                    "EXACT_IN_CURRENT_TABLE"
                    if tested > 0
                    and not mismatch_examples
                    else (
                        "NOT_EXACT"
                        if tested > 0
                        else "NOT_TESTED"
                    )
                ),
                "rows_tested": tested,
                "rows_matching": (
                    tested
                    - len(
                        mismatch_examples
                    )
                ),
                "rows_mismatching": len(
                    mismatch_examples
                ),
                "mismatch_examples": (
                    mismatch_examples[
                        :10
                    ]
                ),
            }
        )
    else:
        relationships.append(
            {
                "relationship": (
                    "top3_support_count is missing "
                    "iff top3_present = 0"
                ),
                "status": "NOT_TESTED",
                "reason": "required columns missing",
                "rows_tested": 0,
                "rows_matching": 0,
                "rows_mismatching": 0,
                "mismatch_examples": [],
            }
        )

    # -------------------------------------------------------------------------
    # 4. If total_input_genes exists in the matrix, validate normalized support
    #    fractions. If not, record NOT_TESTED instead of inferring.
    # -------------------------------------------------------------------------
    relationships.append(
        check_formula(
            frame,
            name=(
                "top1_support_fraction_of_input = "
                "top1_support_count / total_input_genes"
            ),
            required_columns=[
                "top1_support_count",
                "total_input_genes",
            ],
            observed_column=(
                "top1_support_fraction_of_input"
            ),
            expected_fn=lambda row: (
                float(
                    row[
                        "top1_support_count"
                    ]
                )
                / float(
                    row[
                        "total_input_genes"
                    ]
                )
                if (
                    not pd.isna(
                        row[
                            "top1_support_count"
                        ]
                    )
                    and not pd.isna(
                        row[
                            "total_input_genes"
                        ]
                    )
                    and float(
                        row[
                            "total_input_genes"
                        ]
                    )
                    > 0
                )
                else None
            ),
        )
    )

    relationships.append(
        check_formula(
            frame,
            name=(
                "top2_support_fraction_of_input = "
                "top2_support_count / total_input_genes"
            ),
            required_columns=[
                "top2_support_count",
                "total_input_genes",
            ],
            observed_column=(
                "top2_support_fraction_of_input"
            ),
            expected_fn=lambda row: (
                float(
                    row[
                        "top2_support_count"
                    ]
                )
                / float(
                    row[
                        "total_input_genes"
                    ]
                )
                if (
                    not pd.isna(
                        row[
                            "top2_support_count"
                        ]
                    )
                    and not pd.isna(
                        row[
                            "total_input_genes"
                        ]
                    )
                    and float(
                        row[
                            "total_input_genes"
                        ]
                    )
                    > 0
                )
                else None
            ),
        )
    )

    relationships.append(
        check_formula(
            frame,
            name=(
                "top1_top2_gap_fraction_of_input = "
                "(top1_support_count - top2_support_count) / "
                "total_input_genes"
            ),
            required_columns=[
                "top1_support_count",
                "top2_support_count",
                "total_input_genes",
            ],
            observed_column=(
                "top1_top2_gap_fraction_of_input"
            ),
            expected_fn=lambda row: (
                (
                    float(
                        row[
                            "top1_support_count"
                        ]
                    )
                    - float(
                        row[
                            "top2_support_count"
                        ]
                    )
                )
                / float(
                    row[
                        "total_input_genes"
                    ]
                )
                if (
                    not pd.isna(
                        row[
                            "top1_support_count"
                        ]
                    )
                    and not pd.isna(
                        row[
                            "top2_support_count"
                        ]
                    )
                    and not pd.isna(
                        row[
                            "total_input_genes"
                        ]
                    )
                    and float(
                        row[
                            "total_input_genes"
                        ]
                    )
                    > 0
                )
                else None
            ),
        )
    )

    # -------------------------------------------------------------------------
    # 5. Current-sample variance observations.
    # -------------------------------------------------------------------------
    variance_notes = []

    for feature in [
        "top2_present",
        "top3_present",
    ]:
        if feature not in frame.columns:
            continue

        series = frame[
            feature
        ].dropna()

        variance_notes.append(
            {
                "feature": feature,
                "unique_values": int(
                    series.nunique()
                ),
                "dominant_fraction": (
                    float(
                        series.value_counts(
                            normalize=True
                        ).iloc[
                            0
                        ]
                    )
                    if not series.empty
                    else None
                ),
                "interpretation": (
                    "current_sample_only; do not remove from "
                    "tissue-agnostic raw schema solely for this reason"
                ),
            }
        )

    # -------------------------------------------------------------------------
    # Provisional interpretation categories.
    # These are NOT automatic feature-removal decisions.
    # -------------------------------------------------------------------------
    exact_relationships = [
        item[
            "relationship"
        ]
        for item in relationships
        if item.get(
            "status"
        )
        == "EXACT_IN_CURRENT_TABLE"
    ]

    report = {
        "audit_role": (
            "UNSUPERVISED_STRUCTURAL_FEATURE_DEPENDENCY_AUDIT"
        ),
        "feature_matrix": str(
            args.feature_matrix
        ),
        "row_count": int(
            len(
                frame
            )
        ),
        "expert_or_reference_targets_used": False,
        "model_trained": False,
        "routing_threshold_selected": False,
        "features_removed": False,
        "relationships": relationships,
        "current_sample_variance_notes": (
            variance_notes
        ),
        "exact_relationships_in_current_table": (
            exact_relationships
        ),
        "interpretation_rules": {
            "structural_dependency": (
                "May justify excluding one representation from the "
                "eventual fitted model if the relationship is confirmed "
                "by feature definitions and future development data."
            ),
            "constant_or_near_constant": (
                "Does not justify deletion from the raw generic schema; "
                "may justify exclusion from a fitted development model "
                "when the feature has no usable variance."
            ),
            "empirical_high_correlation": (
                "Does not by itself justify feature deletion. "
                "Retain until development-data ablation."
            ),
        },
        "existing_outputs_modified": False,
    }

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        args.output_dir
        / "router_structural_feature_dependency_audit_v1.json"
    )

    table_path = (
        args.output_dir
        / "router_structural_feature_dependency_checks_v1.csv"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {
                "relationship": item.get(
                    "relationship"
                ),
                "status": item.get(
                    "status"
                ),
                "rows_tested": item.get(
                    "rows_tested",
                    0,
                ),
                "rows_matching": item.get(
                    "rows_matching",
                    0,
                ),
                "rows_mismatching": item.get(
                    "rows_mismatching",
                    0,
                ),
                "reason": item.get(
                    "reason",
                    "",
                ),
            }
            for item in relationships
        ]
    ).to_csv(
        table_path,
        index=False,
    )

    print(
        f"Structural dependency audit: "
        f"{report_path}"
    )

    print(
        f"Relationship checks: "
        f"{table_path}"
    )

    print()
    print(
        f"Rows: {len(frame)}"
    )

    print()
    print(
        "STRUCTURAL RELATIONSHIP RESULTS:"
    )

    for item in relationships:
        print(
            f"  {item['status']}: "
            f"{item['relationship']}"
        )

        if (
            item.get(
                "status"
            )
            != "NOT_TESTED"
        ):
            print(
                "    "
                f"matches "
                f"{item.get('rows_matching', 0)}/"
                f"{item.get('rows_tested', 0)}"
            )
        elif item.get(
            "reason"
        ):
            print(
                "    "
                f"{item['reason']}"
            )

    print()
    print(
        "CURRENT-SAMPLE VARIANCE:"
    )

    for item in variance_notes:
        print(
            "  "
            f"{item['feature']}: "
            f"unique={item['unique_values']}, "
            f"dominant_fraction="
            f"{item['dominant_fraction']:.3f}"
        )

    print()
    print(
        "No target labels were used."
    )
    print(
        "No features were removed."
    )
    print(
        "No model was trained."
    )
    print(
        "No Router threshold was selected."
    )
    print(
        "No existing Agent/Router/orchestrator outputs were modified."
    )


if __name__ == "__main__":
    main()
