from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


# ============================================================
# INPUT / OUTPUT
# ============================================================

COSMX_ROOT = Path(
    "data/pipeline_runs/"
    "cosmx_brcr_orchestration_validation/"
    "agent2_output"
)

SUMMARY_FILE = (
    COSMX_ROOT
    / "all_clusters_summary.csv"
)

CLUSTERS_ROOT = (
    COSMX_ROOT
    / "clusters"
)

OUTPUT_DIR = Path(
    "data/analysis/"
    "router_feature_matrix_cosmx_v1"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "cosmx_original_six_router_features.csv"
)


FEATURE_COLUMNS = [
    "direct_evidence_coverage",
    "top1_support_fraction",
    "top1_top2_support_gap",
    "cross_database_support_fraction",
    "first_direct_supported_marker_rank_norm",
    "rank_weighted_unexplained_marker_score",
]


# ============================================================
# HELPERS
# ============================================================

def read_csv(
    path: Path,
) -> list[dict[str, str]]:

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        return list(
            csv.DictReader(f)
        )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "dataset",
        "platform",
        "cluster_id",
        "top1_candidate",
        *FEATURE_COLUMNS,
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def to_int(
    value: Any,
) -> int:

    return int(
        float(
            str(value).strip()
        )
    )


def to_bool(
    value: Any,
) -> bool:

    normalized = (
        str(value)
        .strip()
        .casefold()
    )

    if normalized in {
        "true",
        "1",
        "yes",
    }:
        return True

    if normalized in {
        "false",
        "0",
        "no",
        "",
    }:
        return False

    raise ValueError(
        f"Cannot interpret boolean value: {value!r}"
    )


def round_feature(
    value: float,
) -> float:

    # Same presentation precision as the existing
    # Breast/Lung original feature table.
    return round(
        value,
        4,
    )


# ============================================================
# FEATURE CALCULATION
# ============================================================

def calculate_cluster_features(
    summary_row: dict[str, str],
) -> dict[str, Any]:

    cluster_id = str(
        summary_row[
            "cluster_id"
        ]
    ).strip()

    total_genes = to_int(
        summary_row[
            "total_input_genes"
        ]
    )

    direct_supported_count = to_int(
        summary_row[
            "direct_marker_supported_genes"
        ]
    )

    cluster_dir = (
        CLUSTERS_ROOT
        / f"cluster_{cluster_id}"
    )

    candidate_file = (
        cluster_dir
        / (
            f"cluster_{cluster_id}"
            "_ranked_candidate_cell_types.csv"
        )
    )

    master_file = (
        cluster_dir
        / (
            f"cluster_{cluster_id}"
            "_master_gene_evidence.csv"
        )
    )

    if not candidate_file.exists():
        raise FileNotFoundError(
            f"Missing candidate file: {candidate_file}"
        )

    if not master_file.exists():
        raise FileNotFoundError(
            f"Missing master evidence file: {master_file}"
        )

    candidates = read_csv(
        candidate_file
    )

    master_rows = read_csv(
        master_file
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if len(candidates) < 2:
        raise RuntimeError(
            f"Cluster {cluster_id} has fewer than "
            f"2 ranked candidates."
        )

    if len(master_rows) != total_genes:
        raise RuntimeError(
            f"Cluster {cluster_id}: "
            f"total_input_genes={total_genes}, "
            f"but master evidence contains "
            f"{len(master_rows)} rows."
        )

    candidates = sorted(
        candidates,
        key=lambda row: to_int(
            row[
                "candidate_rank"
            ]
        ),
    )

    top1 = candidates[0]
    top2 = candidates[1]

    if to_int(
        top1[
            "candidate_rank"
        ]
    ) != 1:
        raise RuntimeError(
            f"Cluster {cluster_id}: "
            "Top-1 candidate rank is not 1."
        )

    if to_int(
        top2[
            "candidate_rank"
        ]
    ) != 2:
        raise RuntimeError(
            f"Cluster {cluster_id}: "
            "Top-2 candidate rank is not 2."
        )

    # --------------------------------------------------------
    # COUNTS
    # --------------------------------------------------------

    top1_support = to_int(
        top1[
            "unique_supporting_gene_count"
        ]
    )

    top2_support = to_int(
        top2[
            "unique_supporting_gene_count"
        ]
    )

    top1_cross_database_support = to_int(
        top1[
            "both_source_supporting_gene_count"
        ]
    )

    # --------------------------------------------------------
    # FEATURE 1:
    # DIRECT EVIDENCE COVERAGE
    #
    # direct_marker_supported_genes / total_input_genes
    # --------------------------------------------------------

    direct_evidence_coverage = (
        direct_supported_count
        / total_genes
    )

    # --------------------------------------------------------
    # FEATURE 2:
    # TOP-1 SUPPORT FRACTION
    #
    # top1_support_count / total_input_genes
    # --------------------------------------------------------

    top1_support_fraction = (
        top1_support
        / total_genes
    )

    # --------------------------------------------------------
    # FEATURE 3:
    # TOP-1 / TOP-2 SUPPORT GAP
    #
    # (top1_support - top2_support) / total_input_genes
    # --------------------------------------------------------

    top1_top2_support_gap = (
        (
            top1_support
            - top2_support
        )
        / total_genes
    )

    # --------------------------------------------------------
    # FEATURE 4:
    # CROSS-DATABASE SUPPORT FRACTION
    #
    # top1 genes supported by both databases
    # ---------------------------------------
    # top1 supporting genes
    # --------------------------------------------------------

    if top1_support > 0:

        cross_database_support_fraction = (
            top1_cross_database_support
            / top1_support
        )

    else:

        cross_database_support_fraction = 0.0

    # --------------------------------------------------------
    # SORT MASTER MARKERS BY RANK
    # --------------------------------------------------------

    master_rows = sorted(
        master_rows,
        key=lambda row: to_int(
            row[
                "rank_within_cluster"
            ]
        ),
    )

    expected_ranks = list(
        range(
            1,
            total_genes + 1,
        )
    )

    observed_ranks = [
        to_int(
            row[
                "rank_within_cluster"
            ]
        )
        for row in master_rows
    ]

    if observed_ranks != expected_ranks:
        raise RuntimeError(
            f"Cluster {cluster_id}: "
            "marker ranks are not exactly 1..N."
        )

    # --------------------------------------------------------
    # VERIFY DIRECT-SUPPORT COUNT
    # --------------------------------------------------------

    direct_rows = [
        row
        for row in master_rows
        if to_bool(
            row[
                "direct_marker_supported"
            ]
        )
    ]

    if (
        len(direct_rows)
        != direct_supported_count
    ):
        raise RuntimeError(
            f"Cluster {cluster_id}: "
            "summary direct-marker count "
            "does not match master evidence."
        )

    # --------------------------------------------------------
    # FEATURE 5:
    # FIRST DIRECT-SUPPORTED MARKER RANK / N
    # --------------------------------------------------------

    if not direct_rows:
        raise RuntimeError(
            f"Cluster {cluster_id}: "
            "no directly supported marker exists. "
            "Stopping rather than inventing a value."
        )

    first_direct_rank = min(
        to_int(
            row[
                "rank_within_cluster"
            ]
        )
        for row in direct_rows
    )

    first_direct_supported_marker_rank_norm = (
        first_direct_rank
        / total_genes
    )

    # --------------------------------------------------------
    # FEATURE 6:
    # RANK-WEIGHTED UNEXPLAINED MARKER SCORE
    #
    # weight(rank) = N - rank + 1
    #
    # numerator:
    # weights of markers WITHOUT direct evidence
    #
    # denominator:
    # weights of ALL markers
    # --------------------------------------------------------

    total_rank_weight = sum(
        (
            total_genes
            - rank
            + 1
        )
        for rank in expected_ranks
    )

    unexplained_rank_weight = 0

    for row in master_rows:

        is_direct = to_bool(
            row[
                "direct_marker_supported"
            ]
        )

        if is_direct:
            continue

        rank = to_int(
            row[
                "rank_within_cluster"
            ]
        )

        weight = (
            total_genes
            - rank
            + 1
        )

        unexplained_rank_weight += (
            weight
        )

    rank_weighted_unexplained_marker_score = (
        unexplained_rank_weight
        / total_rank_weight
    )

    # --------------------------------------------------------
    # RETURN ORIGINAL SIX FEATURES
    # --------------------------------------------------------

    return {
        "dataset": "breast_cosmx",
        "platform": "CosMX",
        "cluster_id": cluster_id,
        "top1_candidate": str(
            top1[
                "cell_type"
            ]
        ).strip(),

        "direct_evidence_coverage":
            round_feature(
                direct_evidence_coverage
            ),

        "top1_support_fraction":
            round_feature(
                top1_support_fraction
            ),

        "top1_top2_support_gap":
            round_feature(
                top1_top2_support_gap
            ),

        "cross_database_support_fraction":
            round_feature(
                cross_database_support_fraction
            ),

        "first_direct_supported_marker_rank_norm":
            round_feature(
                first_direct_supported_marker_rank_norm
            ),

        "rank_weighted_unexplained_marker_score":
            round_feature(
                rank_weighted_unexplained_marker_score
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print(
        "=" * 72
    )
    print(
        "COSMX ORIGINAL SIX ROUTER FEATURES"
    )
    print(
        "=" * 72
    )

    if not SUMMARY_FILE.exists():
        raise SystemExit(
            "Missing Agent 2 summary:\n"
            f"{SUMMARY_FILE}"
        )

    summary_rows = read_csv(
        SUMMARY_FILE
    )

    summary_rows = sorted(
        summary_rows,
        key=lambda row: to_int(
            row[
                "cluster_id"
            ]
        ),
    )

    if len(summary_rows) != 9:
        raise SystemExit(
            "Expected 9 CosMX clusters, "
            f"but found {len(summary_rows)}."
        )

    results: list[
        dict[str, Any]
    ] = []

    for summary_row in summary_rows:

        result = (
            calculate_cluster_features(
                summary_row
            )
        )

        results.append(
            result
        )

    write_csv(
        OUTPUT_FILE,
        results,
    )

    # ========================================================
    # KNOWN CLUSTER-0 SANITY CHECK
    # ========================================================

    cluster0 = next(
        row
        for row in results
        if row[
            "cluster_id"
        ] == "0"
    )

    expected_cluster0 = {
        "direct_evidence_coverage":
            0.32,

        "top1_support_fraction":
            0.18,

        "top1_top2_support_gap":
            0.10,

        "cross_database_support_fraction":
            0.1111,

        "first_direct_supported_marker_rank_norm":
            0.02,
    }

    for key, expected in (
        expected_cluster0.items()
    ):

        actual = cluster0[
            key
        ]

        if actual != expected:

            raise RuntimeError(
                "Cluster 0 sanity check failed: "
                f"{key}: "
                f"expected={expected}, "
                f"actual={actual}"
            )

    # ========================================================
    # PRINT TABLE
    # ========================================================

    print()
    print(
        "Cluster | Top-1 | "
        "DirectCov | Top1Frac | Gap | "
        "CrossDB | FirstRank | Unexplained"
    )

    print(
        "-" * 120
    )

    for row in results:

        print(
            f"{row['cluster_id']:>7} | "
            f"{row['top1_candidate']:<24} | "
            f"{row['direct_evidence_coverage']:<9} | "
            f"{row['top1_support_fraction']:<8} | "
            f"{row['top1_top2_support_gap']:<6} | "
            f"{row['cross_database_support_fraction']:<7} | "
            f"{row['first_direct_supported_marker_rank_norm']:<9} | "
            f"{row['rank_weighted_unexplained_marker_score']}"
        )

    print()
    print(
        "Rows:",
        len(
            results
        ),
    )

    print(
        "Feature columns:",
        len(
            FEATURE_COLUMNS
        ),
    )

    print()
    print(
        "Output:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "SOURCE COSMX VALIDATED RUN MODIFIED: False"
    )

    print(
        "STATUS: "
        "COSMX_ORIGINAL_SIX_FEATURES_COMPLETE"
    )


if __name__ == "__main__":
    main()