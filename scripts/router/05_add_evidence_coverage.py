from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd



# This script 
# combines:
#   - the Router evidence table already created
#   - the existing Agent 2 breast cluster summary
#   - the existing Agent 2 lung cluster summary
#
# It adds transparent evidence-coverage fields:
#   total_input_genes
#   direct_marker_supported_genes
#   follow_up_gene_count
#   direct_evidence_coverage
#
# direct_evidence_coverage =
#     direct_marker_supported_genes / total_input_genes

REQUIRED_SUMMARY_COLUMNS = {
    "cluster_id",
    "total_input_genes",
    "direct_marker_supported_genes",
    "follow_up_gene_count",
}


def load_summary(path: Path, dataset: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = REQUIRED_SUMMARY_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{dataset} summary is missing required columns: "
            + ", ".join(sorted(missing))
        )

    out = df[
        [
            "cluster_id",
            "total_input_genes",
            "direct_marker_supported_genes",
            "follow_up_gene_count",
        ]
    ].copy()

    out["dataset"] = dataset

    out["direct_evidence_coverage"] = (
        out["direct_marker_supported_genes"]
        / out["total_input_genes"]
    ).round(4)

    return out


def validate_bookkeeping(df: pd.DataFrame) -> None:
    expected = (
        df["direct_marker_supported_genes"]
        + df["follow_up_gene_count"]
    )

    bad = df.loc[
        expected != df["total_input_genes"],
        [
            "dataset",
            "cluster_id",
            "total_input_genes",
            "direct_marker_supported_genes",
            "follow_up_gene_count",
        ],
    ]

    if not bad.empty:
        raise ValueError(
            "Evidence bookkeeping mismatch found:\n"
            + bad.to_string(index=False)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add direct-evidence coverage to the existing Router evidence table."
        )
    )

    parser.add_argument(
        "--router-table",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--breast-summary",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--lung-summary",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("router_evidence_outputs_v2"),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    router = pd.read_csv(args.router_table)

    breast = load_summary(
        args.breast_summary,
        "breast",
    )

    lung = load_summary(
        args.lung_summary,
        "lung",
    )

    coverage = pd.concat(
        [breast, lung],
        ignore_index=True,
    )

    validate_bookkeeping(coverage)

    merged = router.merge(
        coverage,
        on=["dataset", "cluster_id"],
        how="left",
        validate="one_to_one",
    )

    if merged[
        [
            "total_input_genes",
            "direct_marker_supported_genes",
            "follow_up_gene_count",
            "direct_evidence_coverage",
        ]
    ].isna().any().any():
        raise ValueError(
            "One or more Router rows could not be matched to Agent 2 summary rows."
        )

    output_table = (
        args.output_dir
        / "router_evidence_table_with_coverage.csv"
    )

    merged.to_csv(
        output_table,
        index=False,
    )

    # Small reproducible summary for discussion.
    summary = (
        merged.groupby("dataset")
        .agg(
            cluster_count=("cluster_id", "count"),
            mean_direct_evidence_coverage=(
                "direct_evidence_coverage",
                "mean",
            ),
            min_direct_evidence_coverage=(
                "direct_evidence_coverage",
                "min",
            ),
            max_direct_evidence_coverage=(
                "direct_evidence_coverage",
                "max",
            ),
        )
        .round(4)
        .reset_index()
    )

    summary_file = (
        args.output_dir
        / "evidence_coverage_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    print(
        f"Updated Router evidence table: {output_table}"
    )
    print(
        f"Evidence coverage summary: {summary_file}"
    )
    print(
        "No annotation or Router decision was changed."
    )
    print(
        "direct_evidence_coverage is descriptive only, not confidence."
    )


if __name__ == "__main__":
    main()
