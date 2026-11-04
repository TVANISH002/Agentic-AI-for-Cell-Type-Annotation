from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


#  SELECTIVE ROUTER FEATURE REDUNDANCY AUDIT


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

HIGH_CORRELATION_THRESHOLD = 0.85
VERY_HIGH_CORRELATION_THRESHOLD = 0.95
NEAR_CONSTANT_DOMINANT_FRACTION = 0.95


def clean_numeric(series: pd.Series) -> pd.Series:
    mapping = {
        "true": 1.0,
        "false": 0.0,
        "yes": 1.0,
        "no": 0.0,
    }

    if series.dtype == object:
        lowered = (
            series.astype(str)
            .str.strip()
            .str.casefold()
        )

        converted = lowered.map(
            mapping
        )

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        use_converted = (
            converted.notna()
            & numeric.isna()
        )

        numeric.loc[
            use_converted
        ] = converted.loc[
            use_converted
        ]

        return numeric

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def dominant_fraction(
    series: pd.Series,
) -> float:
    nonmissing = series.dropna()

    if nonmissing.empty:
        return math.nan

    counts = nonmissing.value_counts(
        dropna=False
    )

    return float(
        counts.iloc[0]
        / len(
            nonmissing
        )
    )


def build_high_corr_pairs(
    matrix: pd.DataFrame,
    *,
    method: str,
) -> list[dict[str, Any]]:

    rows: list[
        dict[str, Any]
    ] = []

    columns = list(
        matrix.columns
    )

    for i, first in enumerate(
        columns
    ):
        for second in columns[
            i + 1 :
        ]:
            value = matrix.loc[
                first,
                second,
            ]

            if pd.isna(
                value
            ):
                continue

            abs_value = abs(
                float(
                    value
                )
            )

            if (
                abs_value
                < HIGH_CORRELATION_THRESHOLD
            ):
                continue

            rows.append(
                {
                    "method": method,
                    "feature_1": first,
                    "feature_2": second,
                    "correlation": float(
                        value
                    ),
                    "absolute_correlation": abs_value,
                    "correlation_band": (
                        "very_high"
                        if abs_value
                        >= VERY_HIGH_CORRELATION_THRESHOLD
                        else "high"
                    ),
                }
            )

    return sorted(
        rows,
        key=lambda row: (
            -row[
                "absolute_correlation"
            ],
            row[
                "feature_1"
            ],
            row[
                "feature_2"
            ],
        ),
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
            "router_selective_feature_redundancy_v1"
        ),
    )

    args = parser.parse_args()

    if not args.feature_matrix.exists():
        raise FileNotFoundError(
            f"Feature matrix not found: "
            f"{args.feature_matrix}"
        )

    frame = pd.read_csv(
        args.feature_matrix
    )

    missing_features = [
        feature
        for feature in PREDICTIVE_FEATURES
        if feature not in frame.columns
    ]

    if missing_features:
        raise ValueError(
            "Feature matrix is missing required "
            "predictive features: "
            + ", ".join(
                missing_features
            )
        )

    numeric = pd.DataFrame(
        {
            feature: clean_numeric(
                frame[
                    feature
                ]
            )
            for feature in PREDICTIVE_FEATURES
        }
    )

    profile_rows: list[
        dict[str, Any]
    ] = []

    constant_features: list[
        str
    ] = []

    near_constant_features: list[
        str
    ] = []

    features_with_missingness: list[
        str
    ] = []

    for feature in PREDICTIVE_FEATURES:
        series = numeric[
            feature
        ]

        nonmissing = series.dropna()

        unique_nonmissing = int(
            nonmissing.nunique(
                dropna=True
            )
        )

        missing_count = int(
            series.isna().sum()
        )

        missing_fraction = float(
            missing_count
            / len(
                series
            )
        ) if len(
            series
        ) else math.nan

        dominant = dominant_fraction(
            series
        )

        is_constant = (
            unique_nonmissing
            <= 1
        )

        is_near_constant = (
            not is_constant
            and not math.isnan(
                dominant
            )
            and dominant
            >= NEAR_CONSTANT_DOMINANT_FRACTION
        )

        if is_constant:
            constant_features.append(
                feature
            )

        if is_near_constant:
            near_constant_features.append(
                feature
            )

        if missing_count > 0:
            features_with_missingness.append(
                feature
            )

        profile_rows.append(
            {
                "feature": feature,
                "row_count": int(
                    len(
                        series
                    )
                ),
                "nonmissing_count": int(
                    nonmissing.shape[
                        0
                    ]
                ),
                "missing_count": missing_count,
                "missing_fraction": round(
                    missing_fraction,
                    6,
                ),
                "unique_nonmissing_values": unique_nonmissing,
                "dominant_value_fraction": (
                    round(
                        dominant,
                        6,
                    )
                    if not math.isnan(
                        dominant
                    )
                    else np.nan
                ),
                "minimum": (
                    float(
                        nonmissing.min()
                    )
                    if not nonmissing.empty
                    else np.nan
                ),
                "maximum": (
                    float(
                        nonmissing.max()
                    )
                    if not nonmissing.empty
                    else np.nan
                ),
                "mean": (
                    float(
                        nonmissing.mean()
                    )
                    if not nonmissing.empty
                    else np.nan
                ),
                "standard_deviation": (
                    float(
                        nonmissing.std(
                            ddof=1
                        )
                    )
                    if len(
                        nonmissing
                    )
                    > 1
                    else np.nan
                ),
                "constant_feature": bool(
                    is_constant
                ),
                "near_constant_feature": bool(
                    is_near_constant
                ),
            }
        )

    pearson = numeric.corr(
        method="pearson",
        min_periods=3,
    )

    spearman = numeric.corr(
        method="spearman",
        min_periods=3,
    )

    high_pairs = (
        build_high_corr_pairs(
            pearson,
            method="pearson",
        )
        + build_high_corr_pairs(
            spearman,
            method="spearman",
        )
    )

    pair_frame = pd.DataFrame(
        high_pairs,
        columns=[
            "method",
            "feature_1",
            "feature_2",
            "correlation",
            "absolute_correlation",
            "correlation_band",
        ],
    )

    # A pair is stronger evidence of redundancy if both Pearson and Spearman
    # independently cross the high-correlation threshold.
    pair_methods: dict[
        tuple[str, str],
        set[str],
    ] = {}

    for row in high_pairs:
        key = tuple(
            sorted(
                (
                    row[
                        "feature_1"
                    ],
                    row[
                        "feature_2"
                    ],
                )
            )
        )

        pair_methods.setdefault(
            key,
            set(),
        ).add(
            row[
                "method"
            ]
        )

    dual_method_pairs = [
        {
            "feature_1": key[
                0
            ],
            "feature_2": key[
                1
            ],
            "pearson_and_spearman_high": True,
        }
        for key, methods in sorted(
            pair_methods.items()
        )
        if {
            "pearson",
            "spearman",
        }.issubset(
            methods
        )
    ]

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile_path = (
        args.output_dir
        / "selective_router_feature_profile_v1.csv"
    )

    pearson_path = (
        args.output_dir
        / "selective_router_pearson_correlation_v1.csv"
    )

    spearman_path = (
        args.output_dir
        / "selective_router_spearman_correlation_v1.csv"
    )

    high_pairs_path = (
        args.output_dir
        / "selective_router_high_correlation_pairs_v1.csv"
    )

    dual_pairs_path = (
        args.output_dir
        / "selective_router_dual_method_redundancy_candidates_v1.csv"
    )

    audit_path = (
        args.output_dir
        / "selective_router_feature_redundancy_audit_v1.json"
    )

    pd.DataFrame(
        profile_rows
    ).to_csv(
        profile_path,
        index=False,
    )

    pearson.to_csv(
        pearson_path
    )

    spearman.to_csv(
        spearman_path
    )

    pair_frame.to_csv(
        high_pairs_path,
        index=False,
    )

    pd.DataFrame(
        dual_method_pairs,
        columns=[
            "feature_1",
            "feature_2",
            "pearson_and_spearman_high",
        ],
    ).to_csv(
        dual_pairs_path,
        index=False,
    )

    audit = {
        "audit_role": (
            "UNSUPERVISED_ROUTER_FEATURE_REDUNDANCY_AUDIT"
        ),
        "feature_matrix": str(
            args.feature_matrix
        ),
        "row_count": int(
            len(
                frame
            )
        ),
        "predictive_feature_count": len(
            PREDICTIVE_FEATURES
        ),
        "expert_or_reference_targets_used": False,
        "model_trained": False,
        "routing_threshold_selected": False,
        "features_automatically_removed": False,
        "high_correlation_threshold_absolute": (
            HIGH_CORRELATION_THRESHOLD
        ),
        "very_high_correlation_threshold_absolute": (
            VERY_HIGH_CORRELATION_THRESHOLD
        ),
        "near_constant_dominant_fraction_threshold": (
            NEAR_CONSTANT_DOMINANT_FRACTION
        ),
        "constant_features": constant_features,
        "near_constant_features": near_constant_features,
        "features_with_missingness": (
            features_with_missingness
        ),
        "pearson_high_pair_count": int(
            sum(
                row[
                    "method"
                ]
                == "pearson"
                for row in high_pairs
            )
        ),
        "spearman_high_pair_count": int(
            sum(
                row[
                    "method"
                ]
                == "spearman"
                for row in high_pairs
            )
        ),
        "dual_method_high_pair_count": int(
            len(
                dual_method_pairs
            )
        ),
        "interpretation": (
            "This audit identifies candidate redundancy only. "
            "With the current 20-cluster engineering sample, "
            "correlation estimates are descriptive and must not "
            "be used alone to finalize the publication feature set."
        ),
        "next_methodological_action": (
            "Review highly correlated and structurally related features, "
            "define a provisional reduced feature set if scientifically "
            "justified, then confirm feature ablations using future "
            "development datasets only."
        ),
        "existing_outputs_modified": False,
    }

    audit_path.write_text(
        json.dumps(
            audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Feature profile: {profile_path}"
    )
    print(
        f"Pearson correlation matrix: {pearson_path}"
    )
    print(
        f"Spearman correlation matrix: {spearman_path}"
    )
    print(
        f"High-correlation pairs: {high_pairs_path}"
    )
    print(
        f"Dual-method redundancy candidates: {dual_pairs_path}"
    )
    print(
        f"Audit summary: {audit_path}"
    )

    print()

    print(
        f"Rows: {len(frame)}"
    )
    print(
        f"Predictive features: "
        f"{len(PREDICTIVE_FEATURES)}"
    )
    print(
        f"Constant features: "
        f"{len(constant_features)}"
    )
    print(
        f"Near-constant features: "
        f"{len(near_constant_features)}"
    )
    print(
        "Features with missing values: "
        f"{len(features_with_missingness)}"
    )

    print(
        "Pearson |r| >= "
        f"{HIGH_CORRELATION_THRESHOLD}: "
        f"{audit['pearson_high_pair_count']}"
    )

    print(
        "Spearman |rho| >= "
        f"{HIGH_CORRELATION_THRESHOLD}: "
        f"{audit['spearman_high_pair_count']}"
    )

    print(
        "Pairs high by BOTH methods: "
        f"{audit['dual_method_high_pair_count']}"
    )

    print()

    if dual_method_pairs:
        print(
            "DUAL-METHOD REDUNDANCY CANDIDATES:"
        )

        for item in dual_method_pairs:
            print(
                "  "
                f"{item['feature_1']} <-> "
                f"{item['feature_2']}"
            )
    else:
        print(
            "DUAL-METHOD REDUNDANCY CANDIDATES: none"
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
