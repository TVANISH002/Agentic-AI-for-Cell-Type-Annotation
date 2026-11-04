from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--processed",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/analysis/pipeline_validation/breast_input_reconstruction"
        ),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw = pd.read_csv(
        args.raw
    )

    processed = pd.read_csv(
        args.processed
    )

    raw_required = {
        "spatial_cluster",
        "names",
        "pvals_adj",
    }

    processed_required = {
        "spatial_cluster",
        "rank_within_cluster",
        "names",
    }

    missing_raw = raw_required - set(
        raw.columns
    )

    missing_processed = processed_required - set(
        processed.columns
    )

    if missing_raw:
        raise ValueError(
            "Raw file missing columns: "
            + ", ".join(
                sorted(missing_raw)
            )
        )

    if missing_processed:
        raise ValueError(
            "Processed file missing columns: "
            + ", ".join(
                sorted(missing_processed)
            )
        )

    raw = raw.copy()
    processed = processed.copy()

    raw["spatial_cluster"] = pd.to_numeric(
        raw["spatial_cluster"],
        errors="raise",
    ).astype(int)

    processed["spatial_cluster"] = pd.to_numeric(
        processed["spatial_cluster"],
        errors="raise",
    ).astype(int)

    raw["pvals_adj"] = pd.to_numeric(
        raw["pvals_adj"],
        errors="coerce",
    )

    processed["rank_within_cluster"] = pd.to_numeric(
        processed["rank_within_cluster"],
        errors="raise",
    ).astype(int)

    # -------------------------------------------------------------
    # Candidate reconstruction
    # -------------------------------------------------------------
    candidate = raw.loc[
        raw["pvals_adj"] <= 0.05
    ].copy()

    candidate = (
        candidate
        .groupby(
            "spatial_cluster",
            sort=True,
            group_keys=False,
        )
        .head(50)
        .copy()
    )

    candidate = candidate.loc[
        ~candidate["names"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.startswith("BLANK")
    ].copy()

    candidate[
        "candidate_rank_after_filter"
    ] = (
        candidate
        .groupby(
            "spatial_cluster",
            sort=False,
        )
        .cumcount()
        + 1
    )

    # -------------------------------------------------------------
    # Compare gene identity and order cluster-by-cluster
    # -------------------------------------------------------------
    rows = []

    all_clusters = sorted(
        set(
            candidate["spatial_cluster"]
        )
        | set(
            processed["spatial_cluster"]
        )
    )

    exact_all = True

    for cluster_id in all_clusters:
        cand = candidate.loc[
            candidate["spatial_cluster"] == cluster_id
        ].copy()

        proc = processed.loc[
            processed["spatial_cluster"] == cluster_id
        ].copy()

        proc = proc.sort_values(
            "rank_within_cluster",
            kind="stable",
        )

        candidate_genes = (
            cand["names"]
            .astype(str)
            .tolist()
        )

        processed_genes = (
            proc["names"]
            .astype(str)
            .tolist()
        )

        same_count = (
            len(candidate_genes)
            == len(processed_genes)
        )

        same_order = (
            candidate_genes
            == processed_genes
        )

        same_set = (
            set(candidate_genes)
            == set(processed_genes)
        )

        cluster_exact = (
            same_count
            and same_order
        )

        exact_all = (
            exact_all
            and cluster_exact
        )

        only_candidate = sorted(
            set(candidate_genes)
            - set(processed_genes)
        )

        only_processed = sorted(
            set(processed_genes)
            - set(candidate_genes)
        )

        rows.append(
            {
                "cluster_id": int(
                    cluster_id
                ),
                "candidate_count": int(
                    len(candidate_genes)
                ),
                "processed_count": int(
                    len(processed_genes)
                ),
                "same_count": bool(
                    same_count
                ),
                "same_gene_set": bool(
                    same_set
                ),
                "same_gene_order": bool(
                    same_order
                ),
                "exact_cluster_match": bool(
                    cluster_exact
                ),
                "only_in_candidate_rule": " | ".join(
                    only_candidate
                ),
                "only_in_processed_input": " | ".join(
                    only_processed
                ),
            }
        )

    comparison = pd.DataFrame(
        rows
    )

    # -------------------------------------------------------------
    # Identify BLANK probes removed from the p_adj-only top50 stage
    # -------------------------------------------------------------
    pre_blank = (
        raw.loc[
            raw["pvals_adj"] <= 0.05
        ]
        .groupby(
            "spatial_cluster",
            sort=True,
            group_keys=False,
        )
        .head(50)
        .copy()
    )

    blank_rows = pre_blank.loc[
        pre_blank["names"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.startswith("BLANK")
    ].copy()

    blank_records = []

    for _, row in blank_rows.iterrows():
        blank_records.append(
            {
                "cluster_id": int(
                    row["spatial_cluster"]
                ),
                "gene": str(
                    row["names"]
                ),
                "pvals_adj": (
                    None
                    if pd.isna(
                        row["pvals_adj"]
                    )
                    else float(
                        row["pvals_adj"]
                    )
                ),
            }
        )

    report = {
        "audit_role": (
            "READ_ONLY_BREAST_INPUT_RECONSTRUCTION"
        ),
        "raw_rows": int(
            len(raw)
        ),
        "processed_rows": int(
            len(processed)
        ),
        "candidate_rule_rows": int(
            len(candidate)
        ),
        "cluster_count": int(
            len(all_clusters)
        ),
        "blank_rows_removed": blank_records,
        "all_clusters_exact_gene_and_order_match": bool(
            exact_all
        ),
        "exact_matching_cluster_count": int(
            comparison[
                "exact_cluster_match"
            ].sum()
        ),
        "total_cluster_count": int(
            len(comparison)
        ),
        "historical_rule_proven": False,
        "interpretation_if_pass": (
            "The candidate reconstruction exactly reproduces the actual "
            "processed Agent 1 input by gene identity and order. This is "
            "strong evidence of equivalence, but by itself does not prove "
            "which historical script originally created the file."
        ),
        "existing_outputs_modified": False,
    }

    comparison_path = (
        args.output_dir
        / "breast_input_reconstruction_comparison_v1.csv"
    )

    report_path = (
        args.output_dir
        / "breast_input_reconstruction_audit_v1.json"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Comparison table: {comparison_path}"
    )

    print(
        f"Audit report: {report_path}"
    )

    print(
        f"Raw rows: {len(raw)}"
    )

    print(
        f"Actual processed Agent 1 rows: {len(processed)}"
    )

    print(
        f"Candidate reconstruction rows: {len(candidate)}"
    )

    print(
        "BLANK rows removed:"
    )

    if blank_records:
        for item in blank_records:
            print(
                f"  cluster {item['cluster_id']}: {item['gene']}"
            )
    else:
        print(
            "  none"
        )

    print(
        "Exact gene identity + order across all clusters: "
        f"{exact_all}"
    )

    print(
        "Exact matching clusters: "
        f"{int(comparison['exact_cluster_match'].sum())}"
        f"/{len(comparison)}"
    )

    print(
        "Historical preprocessing rule was NOT asserted or rewritten."
    )

    print(
        "No existing Agent output, Router output, or annotation was changed."
    )


if __name__ == "__main__":
    main()
