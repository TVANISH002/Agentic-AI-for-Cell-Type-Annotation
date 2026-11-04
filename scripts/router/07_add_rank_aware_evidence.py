from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd



#This script reads each cluster's existing master_gene_evidence.csv and measures
# whether the HIGHEST-RANKED input markers had direct biological evidence.
#
# Added descriptive fields:
#   top_marker_direct_supported
#   first_direct_supported_rank
#   leading_non_direct_marker_count
#   highest_ranked_non_direct_gene
#   first_direct_supported_gene
# These are descriptive evidence-quality variables, NOT confidence scores.


REQUIRED_MASTER_COLUMNS = {
    "cluster_id",
    "gene",
    "rank_within_cluster",
    "direct_marker_supported",
    "evidence_category",
}


def parse_bool(value: object) -> bool:
    text = str(value).strip().casefold()

    if text in {"true", "1", "yes", "y"}:
        return True

    if text in {"false", "0", "no", "n"}:
        return False

    raise ValueError(
        f"Cannot interpret direct_marker_supported value: {value!r}"
    )


def cluster_id_from_path(path: Path) -> int:
    match = re.search(
        r"cluster_(\d+)_master_gene_evidence\.csv$",
        path.name,
    )

    if not match:
        raise ValueError(
            f"Unexpected master evidence filename: {path.name}"
        )

    return int(match.group(1))


def summarize_cluster_file(
    path: Path,
    dataset: str,
) -> dict[str, object]:

    df = pd.read_csv(path)

    missing = REQUIRED_MASTER_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} is missing required columns: "
            + ", ".join(sorted(missing))
        )

    expected_cluster = cluster_id_from_path(path)

    actual_clusters = set(
        pd.to_numeric(
            df["cluster_id"],
            errors="raise",
        ).astype(int)
    )

    if actual_clusters != {expected_cluster}:
        raise ValueError(
            f"Cluster-id mismatch in {path}. "
            f"Expected {expected_cluster}; found {sorted(actual_clusters)}."
        )

    df = df.copy()

    df["rank_within_cluster"] = pd.to_numeric(
        df["rank_within_cluster"],
        errors="raise",
    ).astype(int)

    df["direct_marker_supported_bool"] = (
        df["direct_marker_supported"].map(parse_bool)
    )

    df = df.sort_values(
        "rank_within_cluster",
        kind="stable",
    ).reset_index(drop=True)

    if df["rank_within_cluster"].duplicated().any():
        duplicates = (
            df.loc[
                df["rank_within_cluster"].duplicated(keep=False),
                "rank_within_cluster",
            ]
            .astype(int)
            .tolist()
        )

        raise ValueError(
            f"Duplicate marker ranks in {path}: {duplicates}"
        )

    first_row = df.iloc[0]

    direct_rows = df.loc[
        df["direct_marker_supported_bool"]
    ].copy()

    non_direct_rows = df.loc[
        ~df["direct_marker_supported_bool"]
    ].copy()

    if direct_rows.empty:
        first_direct_rank = None
        first_direct_gene = ""
        leading_non_direct_count = int(len(df))
    else:
        first_direct = direct_rows.iloc[0]

        first_direct_rank = int(
            first_direct["rank_within_cluster"]
        )

        first_direct_gene = str(
            first_direct["gene"]
        ).strip()

        # Because ranks are sorted, this measures how many consecutive
        # highest-ranked markers lacked direct support before the first
        # direct-supported marker appeared.
        leading_non_direct_count = int(
            (
                df["rank_within_cluster"]
                < first_direct_rank
            ).sum()
        )

    highest_non_direct_gene = ""

    if not non_direct_rows.empty:
        highest_non_direct_gene = str(
            non_direct_rows.iloc[0]["gene"]
        ).strip()

    return {
        "dataset": dataset,
        "cluster_id": expected_cluster,
        "master_gene_count": int(len(df)),
        "master_direct_supported_count": int(
            df["direct_marker_supported_bool"].sum()
        ),
        "master_non_direct_count": int(
            (~df["direct_marker_supported_bool"]).sum()
        ),
        "top_marker_gene": str(
            first_row["gene"]
        ).strip(),
        "top_marker_direct_supported": bool(
            first_row["direct_marker_supported_bool"]
        ),
        "first_direct_supported_rank": first_direct_rank,
        "first_direct_supported_gene": first_direct_gene,
        "leading_non_direct_marker_count": (
            leading_non_direct_count
        ),
        "highest_ranked_non_direct_gene": (
            highest_non_direct_gene
        ),
    }


def collect_dataset(
    clusters_dir: Path,
    dataset: str,
) -> pd.DataFrame:

    files = sorted(
        clusters_dir.glob(
            "cluster_*/cluster_*_master_gene_evidence.csv"
        ),
        key=cluster_id_from_path,
    )

    if not files:
        raise FileNotFoundError(
            "No cluster_*_master_gene_evidence.csv files found under "
            f"{clusters_dir}"
        )

    rows = [
        summarize_cluster_file(
            path,
            dataset,
        )
        for path in files
    ]

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add rank-aware direct-evidence descriptors to the existing "
            "Router evidence table."
        )
    )

    parser.add_argument(
        "--router-table",
        type=Path,
        required=True,
        help=(
            "Path to router_evidence_table_with_context.csv"
        ),
    )

    parser.add_argument(
        "--breast-clusters-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--lung-clusters-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "router_evidence_outputs_v4"
        ),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    router = pd.read_csv(
        args.router_table
    )

    breast = collect_dataset(
        args.breast_clusters_dir,
        "breast",
    )

    lung = collect_dataset(
        args.lung_clusters_dir,
        "lung",
    )

    rank_evidence = pd.concat(
        [breast, lung],
        ignore_index=True,
    )

    merged = router.merge(
        rank_evidence,
        on=["dataset", "cluster_id"],
        how="left",
        validate="one_to_one",
    )

    if merged["master_gene_count"].isna().any():
        bad = merged.loc[
            merged["master_gene_count"].isna(),
            ["dataset", "cluster_id"],
        ]

        raise ValueError(
            "Some Router rows could not be matched to master gene evidence:\n"
            + bad.to_string(index=False)
        )

    
    # REPRODUCIBILITY CHECKS
    # Verify that gene counts reconstructed from cluster files match the
    # already existing Agent 2 cluster-level summaries.

    if "total_input_genes" in merged.columns:
        mismatch = (
            merged["master_gene_count"].astype(int)
            != merged["total_input_genes"].astype(int)
        )

        if mismatch.any():
            bad = merged.loc[
                mismatch,
                [
                    "dataset",
                    "cluster_id",
                    "master_gene_count",
                    "total_input_genes",
                ],
            ]

            raise ValueError(
                "Master-gene count disagrees with Agent 2 summary:\n"
                + bad.to_string(index=False)
            )

    if "direct_marker_supported_genes" in merged.columns:
        mismatch = (
            merged["master_direct_supported_count"].astype(int)
            != merged["direct_marker_supported_genes"].astype(int)
        )

        if mismatch.any():
            bad = merged.loc[
                mismatch,
                [
                    "dataset",
                    "cluster_id",
                    "master_direct_supported_count",
                    "direct_marker_supported_genes",
                ],
            ]

            raise ValueError(
                "Direct-supported count disagrees with Agent 2 summary:\n"
                + bad.to_string(index=False)
            )

    output_table = (
        args.output_dir
        / "router_evidence_table_rank_aware.csv"
    )

    merged.to_csv(
        output_table,
        index=False,
    )

    # Simple descriptive report.
    report_lines = [
        "# Rank-Aware Evidence Observations",
        "",
        (
            "These observations describe where direct marker-database evidence "
            "first appears in each ranked cluster marker list. They are not "
            "confidence scores and do not make Router decisions."
        ),
        "",
    ]

    for dataset, group in merged.groupby(
        "dataset",
        sort=True,
    ):
        report_lines.append(
            f"## {str(dataset).capitalize()}"
        )
        report_lines.append("")

        no_top_support = group.loc[
            ~group["top_marker_direct_supported"].astype(bool),
            "cluster_id",
        ].astype(int).tolist()

        max_leading = int(
            group["leading_non_direct_marker_count"].max()
        )

        max_leading_clusters = group.loc[
            group["leading_non_direct_marker_count"] == max_leading,
            "cluster_id",
        ].astype(int).tolist()

        report_lines.append(
            "- Clusters whose highest-ranked marker lacked direct "
            "context-matched marker evidence: "
            + (
                ", ".join(map(str, no_top_support))
                if no_top_support
                else "none"
            )
        )

        report_lines.append(
            "- Largest consecutive run of highest-ranked markers without "
            f"direct support: {max_leading}; cluster(s): "
            + ", ".join(
                map(str, max_leading_clusters)
            )
        )

        report_lines.append("")

    report_file = (
        args.output_dir
        / "rank_aware_evidence_observations.md"
    )

    report_file.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print(
        f"Rank-aware Router evidence table: {output_table}"
    )

    print(
        f"Rank-aware observation report: {report_file}"
    )

    print(
        "Reproducibility checks: PASS"
    )

    print(
        "No annotation, candidate ranking, confidence score, "
        "or Router decision was changed."
    )


if __name__ == "__main__":
    main()
