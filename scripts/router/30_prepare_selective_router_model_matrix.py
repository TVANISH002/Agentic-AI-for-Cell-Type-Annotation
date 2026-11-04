from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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

IDENTIFIER_COLUMNS = [
    "dataset",
    "cluster_id",
]


def as_numeric(
    series: pd.Series,
) -> pd.Series:
    if series.dtype == object:
        lowered = (
            series.astype(str)
            .str.strip()
            .str.casefold()
        )

        bool_map = {
            "true": 1.0,
            "false": 0.0,
            "yes": 1.0,
            "no": 0.0,
        }

        mapped = lowered.map(
            bool_map
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

        return parsed.astype(
            float
        )

    return pd.to_numeric(
        series,
        errors="coerce",
    ).astype(
        float
    )


def finite_or_nan(
    value: Any,
) -> bool:
    if pd.isna(
        value
    ):
        return True

    return math.isfinite(
        float(
            value
        )
    )


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
            "router_selective_model_matrix_v1"
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

    missing_features = [
        feature
        for feature in PROVISIONAL_MODEL_FEATURES
        if feature not in raw.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing provisional model features: "
            + ", ".join(
                missing_features
            )
        )

    output = pd.DataFrame(
        index=raw.index
    )

    for column in IDENTIFIER_COLUMNS:
        if column in raw.columns:
            output[
                column
            ] = raw[
                column
            ]

    numeric = pd.DataFrame(
        {
            feature: as_numeric(
                raw[
                    feature
                ]
            )
            for feature in PROVISIONAL_MODEL_FEATURES
        }
    )

    # -------------------------------------------------------------------------
    # Structural missingness handling for top3 support.
    # -------------------------------------------------------------------------
    structural_missing_mask = (
        numeric[
            "top3_present"
        ]
        == 0
    ) & (
        numeric[
            "top3_support_count"
        ]
        .isna()
    )

    invalid_missing_top3_mask = (
        numeric[
            "top3_present"
        ]
        == 1
    ) & (
        numeric[
            "top3_support_count"
        ]
        .isna()
    )

    contradictory_top3_mask = (
        numeric[
            "top3_present"
        ]
        == 0
    ) & (
        numeric[
            "top3_support_count"
        ]
        .notna()
    )

    # Only structural absence receives zero encoding.
    numeric.loc[
        structural_missing_mask,
        "top3_support_count",
    ] = 0.0

    # -------------------------------------------------------------------------
    # Basic binary-field validation.
    # -------------------------------------------------------------------------
    binary_columns = [
        "top2_present",
        "top3_present",
    ]

    invalid_binary_rows: list[
        dict[str, Any]
    ] = []

    for column in binary_columns:
        for index, value in numeric[
            column
        ].items():
            if pd.isna(
                value
            ) or value not in {
                0.0,
                1.0,
            }:
                invalid_binary_rows.append(
                    {
                        "row_index": int(
                            index
                        ),
                        "feature": column,
                        "value": (
                            None
                            if pd.isna(
                                value
                            )
                            else float(
                                value
                            )
                        ),
                    }
                )

    # -------------------------------------------------------------------------
    # Missing/non-finite audit after structural handling.
    # -------------------------------------------------------------------------
    remaining_missing = []

    nonfinite_values = []

    for feature in PROVISIONAL_MODEL_FEATURES:
        for index, value in numeric[
            feature
        ].items():
            if pd.isna(
                value
            ):
                remaining_missing.append(
                    {
                        "row_index": int(
                            index
                        ),
                        "feature": feature,
                    }
                )
            elif not finite_or_nan(
                value
            ):
                nonfinite_values.append(
                    {
                        "row_index": int(
                            index
                        ),
                        "feature": feature,
                        "value": str(
                            value
                        ),
                    }
                )

    # -------------------------------------------------------------------------
    # Current-sample variance profile.
    # This is descriptive only; no feature is automatically removed.
    # -------------------------------------------------------------------------
    variance_rows = []

    zero_variance_features = []

    for feature in PROVISIONAL_MODEL_FEATURES:
        series = numeric[
            feature
        ]

        unique_values = int(
            series.nunique(
                dropna=True
            )
        )

        variance = (
            float(
                series.var(
                    ddof=1
                )
            )
            if series.notna().sum() > 1
            else np.nan
        )

        zero_variance = (
            unique_values <= 1
        )

        if zero_variance:
            zero_variance_features.append(
                feature
            )

        variance_rows.append(
            {
                "feature": feature,
                "unique_nonmissing_values": (
                    unique_values
                ),
                "variance": variance,
                "zero_variance_current_sample": (
                    zero_variance
                ),
            }
        )

    # -------------------------------------------------------------------------
    # Build model-ready table.
    # -------------------------------------------------------------------------
    for feature in PROVISIONAL_MODEL_FEATURES:
        output[
            feature
        ] = numeric[
            feature
        ]

    structural_missing_rows = int(
        structural_missing_mask.sum()
    )

    invalid_missing_top3_rows = int(
        invalid_missing_top3_mask.sum()
    )

    contradictory_top3_rows = int(
        contradictory_top3_mask.sum()
    )

    status_pass = (
        not invalid_binary_rows
        and not remaining_missing
        and not nonfinite_values
        and invalid_missing_top3_rows == 0
        and contradictory_top3_rows == 0
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_matrix_path = (
        args.output_dir
        / "selective_router_model_ready_features_v1.csv"
    )

    variance_path = (
        args.output_dir
        / "selective_router_model_ready_variance_v1.csv"
    )

    audit_path = (
        args.output_dir
        / "selective_router_model_ready_audit_v1.json"
    )

    output.to_csv(
        model_matrix_path,
        index=False,
    )

    pd.DataFrame(
        variance_rows
    ).to_csv(
        variance_path,
        index=False,
    )

    audit = {
        "audit_role": (
            "SELECTIVE_ROUTER_MODEL_MATRIX_PREPARATION"
        ),
        "source_feature_matrix": str(
            args.feature_matrix
        ),
        "row_count": int(
            len(
                raw
            )
        ),
        "provisional_model_feature_count": len(
            PROVISIONAL_MODEL_FEATURES
        ),
        "provisional_model_features": (
            PROVISIONAL_MODEL_FEATURES
        ),
        "identifier_columns_present": [
            column
            for column in IDENTIFIER_COLUMNS
            if column in raw.columns
        ],
        "structural_missingness_policy": {
            "feature": "top3_support_count",
            "gate": "top3_present",
            "structural_missing_rows_encoded_as_zero": (
                structural_missing_rows
            ),
            "missing_when_top3_present_equals_1": (
                invalid_missing_top3_rows
            ),
            "nonmissing_when_top3_present_equals_0": (
                contradictory_top3_rows
            ),
        },
        "invalid_binary_value_count": len(
            invalid_binary_rows
        ),
        "remaining_missing_value_count": len(
            remaining_missing
        ),
        "nonfinite_value_count": len(
            nonfinite_values
        ),
        "zero_variance_features_current_sample": (
            zero_variance_features
        ),
        "zero_variance_note": (
            "Current-sample zero variance does not delete the feature from "
            "the generic schema. During future model fitting, variance "
            "filtering must be learned from development/training data only."
        ),
        "scaling_fitted": False,
        "model_trained": False,
        "target_labels_used": False,
        "threshold_selected": False,
        "router_v4_replaced": False,
        "existing_outputs_modified": False,
        "status": (
            "PASS"
            if status_pass
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
        f"Model-ready matrix: "
        f"{model_matrix_path}"
    )
    print(
        f"Variance profile: "
        f"{variance_path}"
    )
    print(
        f"Audit: {audit_path}"
    )

    print()
    print(
        f"Rows: {len(raw)}"
    )
    print(
        "Provisional model features: "
        f"{len(PROVISIONAL_MODEL_FEATURES)}"
    )
    print(
        "Structural top3 missing rows encoded as 0: "
        f"{structural_missing_rows}"
    )
    print(
        "Invalid missing top3 support rows: "
        f"{invalid_missing_top3_rows}"
    )
    print(
        "Contradictory top3 rows: "
        f"{contradictory_top3_rows}"
    )
    print(
        "Remaining missing model values: "
        f"{len(remaining_missing)}"
    )
    print(
        "Non-finite model values: "
        f"{len(nonfinite_values)}"
    )

    print()
    print(
        "CURRENT-SAMPLE ZERO-VARIANCE FEATURES:"
    )

    if zero_variance_features:
        for feature in (
            zero_variance_features
        ):
            print(
                f"  {feature}"
            )
    else:
        print(
            "  none"
        )

    print()
    print(
        "MODEL-READY MATRIX STATUS: "
        + (
            "PASS"
            if status_pass
            else "FAIL"
        )
    )

    print()
    print(
        "Scaling fitted: NO"
    )
    print(
        "Target labels used: NO"
    )
    print(
        "Model trained: NO"
    )
    print(
        "Router threshold selected: NO"
    )
    print(
        "Router V4 replaced: NO"
    )
    print(
        "No existing Agent/Router/orchestrator outputs were modified."
    )


if __name__ == "__main__":
    main()
