from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd



CONTROL_PREFIXES = (
    "BLANK-",
    "NEGCONTROL",
    "UNASSIGNEDCODEWORD",
    "SYSTEMCONTROL",
)


def is_control(series: pd.Series) -> pd.Series:
    values = (
        series
        .fillna("")
        .astype(str)
        .str.upper()
    )

    mask = pd.Series(
        False,
        index=series.index,
    )

    for prefix in CONTROL_PREFIXES:
        mask = mask | values.str.startswith(prefix)

    return mask


def apply_filter(
    df: pd.DataFrame,
    filter_name: str,
) -> pd.DataFrame:

    out = df.copy()

    if filter_name == "none":
        return out

    if filter_name == "padj":
        return out.loc[
            out["pvals_adj"] <= 0.05
        ].copy()

    if filter_name == "logfc":
        return out.loc[
            out["logfoldchanges"] >= 0.5
        ].copy()

    if filter_name == "padj_and_logfc":
        return out.loc[
            (out["pvals_adj"] <= 0.05)
            & (out["logfoldchanges"] >= 0.5)
        ].copy()

    raise ValueError(
        f"Unknown filter: {filter_name}"
    )


def sort_group(
    group: pd.DataFrame,
    ranking_name: str,
) -> pd.DataFrame:

    if ranking_name == "original_order":
        return group.sort_values(
            "_raw_row_index",
            kind="stable",
        )

    if ranking_name == "scores_desc":
        return group.sort_values(
            ["scores", "_raw_row_index"],
            ascending=[False, True],
            kind="stable",
        )

    if ranking_name == "logfc_desc":
        return group.sort_values(
            ["logfoldchanges", "_raw_row_index"],
            ascending=[False, True],
            kind="stable",
        )

    if ranking_name == "padj_asc":
        return group.sort_values(
            ["pvals_adj", "_raw_row_index"],
            ascending=[True, True],
            kind="stable",
        )

    if ranking_name == "pval_asc":
        return group.sort_values(
            ["pvals", "_raw_row_index"],
            ascending=[True, True],
            kind="stable",
        )

    raise ValueError(
        f"Unknown ranking: {ranking_name}"
    )


def reconstruct(
    raw: pd.DataFrame,
    *,
    filter_name: str,
    ranking_name: str,
    control_timing: str,
) -> pd.DataFrame:

    df = apply_filter(
        raw,
        filter_name,
    )

    if control_timing == "before_top50":
        df = df.loc[
            ~is_control(df["names"])
        ].copy()

    selected_parts = []

    for cluster_id, group in df.groupby(
        "spatial_cluster",
        sort=True,
    ):
        ranked = sort_group(
            group,
            ranking_name,
        ).head(50)

        selected_parts.append(
            ranked
        )

    if selected_parts:
        selected = pd.concat(
            selected_parts,
            ignore_index=False,
        )
    else:
        selected = df.iloc[
            0:0
        ].copy()

    if control_timing == "after_top50":
        selected = selected.loc[
            ~is_control(
                selected["names"]
            )
        ].copy()

    selected_parts = []

    for cluster_id, group in selected.groupby(
        "spatial_cluster",
        sort=True,
    ):
        ranked = sort_group(
            group,
            ranking_name,
        ).copy()

        ranked[
            "_candidate_rank"
        ] = range(
            1,
            len(ranked) + 1,
        )

        selected_parts.append(
            ranked
        )

    if selected_parts:
        selected = pd.concat(
            selected_parts,
            ignore_index=True,
        )

    return selected


def compare_candidate(
    candidate: pd.DataFrame,
    processed: pd.DataFrame,
) -> dict[str, object]:

    exact_set_clusters = 0
    exact_order_clusters = 0
    total_overlap = 0
    total_processed = 0
    total_candidate = 0

    per_cluster = []

    clusters = sorted(
        set(
            candidate[
                "spatial_cluster"
            ].unique()
        )
        | set(
            processed[
                "spatial_cluster"
            ].unique()
        )
    )

    for cluster_id in clusters:
        cand = candidate.loc[
            candidate[
                "spatial_cluster"
            ] == cluster_id
        ].copy()

        proc = processed.loc[
            processed[
                "spatial_cluster"
            ] == cluster_id
        ].copy()

        cand = cand.sort_values(
            "_candidate_rank",
            kind="stable",
        )

        proc = proc.sort_values(
            "rank_within_cluster",
            kind="stable",
        )

        cand_genes = (
            cand["names"]
            .astype(str)
            .tolist()
        )

        proc_genes = (
            proc["names"]
            .astype(str)
            .tolist()
        )

        cand_set = set(
            cand_genes
        )

        proc_set = set(
            proc_genes
        )

        overlap = len(
            cand_set & proc_set
        )

        same_set = (
            cand_set == proc_set
            and len(cand_genes)
            == len(proc_genes)
        )

        same_order = (
            cand_genes == proc_genes
        )

        exact_set_clusters += int(
            same_set
        )

        exact_order_clusters += int(
            same_order
        )

        total_overlap += overlap
        total_processed += len(
            proc_genes
        )
        total_candidate += len(
            cand_genes
        )

        per_cluster.append(
            {
                "cluster_id": int(
                    cluster_id
                ),
                "candidate_count": int(
                    len(cand_genes)
                ),
                "processed_count": int(
                    len(proc_genes)
                ),
                "overlap_count": int(
                    overlap
                ),
                "same_gene_set": bool(
                    same_set
                ),
                "same_gene_order": bool(
                    same_order
                ),
            }
        )

    precision_like = (
        total_overlap / total_candidate
        if total_candidate
        else 0.0
    )

    recall_like = (
        total_overlap / total_processed
        if total_processed
        else 0.0
    )

    exact_full = (
        exact_order_clusters
        == len(clusters)
        and total_candidate
        == total_processed
    )

    return {
        "candidate_rows": int(
            total_candidate
        ),
        "processed_rows": int(
            total_processed
        ),
        "total_overlap": int(
            total_overlap
        ),
        "overlap_fraction_of_candidate": round(
            precision_like,
            6,
        ),
        "overlap_fraction_of_processed": round(
            recall_like,
            6,
        ),
        "exact_gene_set_clusters": int(
            exact_set_clusters
        ),
        "exact_gene_order_clusters": int(
            exact_order_clusters
        ),
        "cluster_count": int(
            len(clusters)
        ),
        "exact_full_reconstruction": bool(
            exact_full
        ),
        "per_cluster": per_cluster,
    }


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
            "data/analysis/pipeline_validation/"
            "breast_rule_identification"
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
        "scores",
        "logfoldchanges",
        "pvals",
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

    missing_processed = (
        processed_required
        - set(processed.columns)
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

    raw[
        "_raw_row_index"
    ] = range(
        len(raw)
    )

    raw[
        "spatial_cluster"
    ] = pd.to_numeric(
        raw["spatial_cluster"],
        errors="raise",
    ).astype(int)

    processed[
        "spatial_cluster"
    ] = pd.to_numeric(
        processed["spatial_cluster"],
        errors="raise",
    ).astype(int)

    processed[
        "rank_within_cluster"
    ] = pd.to_numeric(
        processed["rank_within_cluster"],
        errors="raise",
    ).astype(int)

    for column in [
        "scores",
        "logfoldchanges",
        "pvals",
        "pvals_adj",
    ]:
        raw[column] = pd.to_numeric(
            raw[column],
            errors="coerce",
        )

    # -------------------------------------------------------------
    # Sanity check: are processed genes present in raw same cluster?
    # -------------------------------------------------------------
    raw_pairs = set(
        zip(
            raw[
                "spatial_cluster"
            ].astype(int),
            raw[
                "names"
            ].astype(str),
        )
    )

    processed_pairs = list(
        zip(
            processed[
                "spatial_cluster"
            ].astype(int),
            processed[
                "names"
            ].astype(str),
        )
    )

    missing_processed_pairs = [
        {
            "cluster_id": int(
                cluster_id
            ),
            "gene": gene,
        }
        for (
            cluster_id,
            gene,
        ) in processed_pairs
        if (
            cluster_id,
            gene,
        )
        not in raw_pairs
    ]

    filters = [
        "none",
        "padj",
        "logfc",
        "padj_and_logfc",
    ]

    rankings = [
        "original_order",
        "scores_desc",
        "logfc_desc",
        "padj_asc",
        "pval_asc",
    ]

    control_timings = [
        "none",
        "before_top50",
        "after_top50",
    ]

    results = []

    detailed = []

    for (
        filter_name,
        ranking_name,
        control_timing,
    ) in itertools.product(
        filters,
        rankings,
        control_timings,
    ):
        candidate = reconstruct(
            raw,
            filter_name=filter_name,
            ranking_name=ranking_name,
            control_timing=control_timing,
        )

        comparison = compare_candidate(
            candidate,
            processed,
        )

        summary = {
            "filter": filter_name,
            "ranking": ranking_name,
            "control_timing": control_timing,
            "candidate_rows": comparison[
                "candidate_rows"
            ],
            "processed_rows": comparison[
                "processed_rows"
            ],
            "total_overlap": comparison[
                "total_overlap"
            ],
            "overlap_fraction_of_processed": (
                comparison[
                    "overlap_fraction_of_processed"
                ]
            ),
            "exact_gene_set_clusters": comparison[
                "exact_gene_set_clusters"
            ],
            "exact_gene_order_clusters": comparison[
                "exact_gene_order_clusters"
            ],
            "cluster_count": comparison[
                "cluster_count"
            ],
            "exact_full_reconstruction": comparison[
                "exact_full_reconstruction"
            ],
        }

        results.append(
            summary
        )

        detailed.append(
            {
                **summary,
                "per_cluster": comparison[
                    "per_cluster"
                ],
            }
        )

    ranking_df = pd.DataFrame(
        results
    )

    ranking_df = ranking_df.sort_values(
        by=[
            "exact_full_reconstruction",
            "exact_gene_order_clusters",
            "exact_gene_set_clusters",
            "total_overlap",
            "overlap_fraction_of_processed",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    ranking_df.insert(
        0,
        "rule_rank",
        range(
            1,
            len(ranking_df) + 1,
        ),
    )

    exact_matches = ranking_df.loc[
        ranking_df[
            "exact_full_reconstruction"
        ]
    ].copy()

    best = (
        ranking_df.iloc[0]
        .to_dict()
        if not ranking_df.empty
        else {}
    )

    ranking_path = (
        args.output_dir
        / "candidate_rule_ranking_v1.csv"
    )

    report_path = (
        args.output_dir
        / "candidate_rule_audit_v1.json"
    )

    ranking_df.to_csv(
        ranking_path,
        index=False,
    )

    report = {
        "audit_role": (
            "READ_ONLY_HISTORICAL_RULE_IDENTIFICATION"
        ),
        "raw_rows": int(
            len(raw)
        ),
        "processed_rows": int(
            len(processed)
        ),
        "processed_genes_missing_from_raw_same_cluster": (
            missing_processed_pairs
        ),
        "candidate_rules_tested": int(
            len(ranking_df)
        ),
        "exact_reconstruction_count": int(
            len(exact_matches)
        ),
        "best_candidate_rule": best,
        "historical_rule_proven": False,
        "interpretation": (
            "Candidate rules are ranked by agreement with the actual "
            "processed Agent-1 input. Even an exact reconstruction "
            "demonstrates equivalence but does not prove historical "
            "provenance unless supported by the original preprocessing "
            "script/configuration."
        ),
        "existing_outputs_modified": False,
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        f"Candidate-rule ranking: {ranking_path}"
    )

    print(
        f"Audit report: {report_path}"
    )

    print(
        f"Candidate rules tested: {len(ranking_df)}"
    )

    print(
        "Processed genes missing from raw same cluster: "
        f"{len(missing_processed_pairs)}"
    )

    print(
        "Exact full reconstructions found: "
        f"{len(exact_matches)}"
    )

    if not ranking_df.empty:
        print()
        print(
            "BEST CANDIDATE RULE:"
        )

        print(
            f"  filter: {best['filter']}"
        )

        print(
            f"  ranking: {best['ranking']}"
        )

        print(
            f"  control timing: {best['control_timing']}"
        )

        print(
            f"  candidate rows: {best['candidate_rows']}"
        )

        print(
            f"  total gene overlap: "
            f"{best['total_overlap']}/{best['processed_rows']}"
        )

        print(
            f"  exact gene-set clusters: "
            f"{best['exact_gene_set_clusters']}/"
            f"{best['cluster_count']}"
        )

        print(
            f"  exact gene-order clusters: "
            f"{best['exact_gene_order_clusters']}/"
            f"{best['cluster_count']}"
        )

        print(
            "  exact full reconstruction: "
            f"{best['exact_full_reconstruction']}"
        )

    print()
    print(
        "Historical preprocessing rule was NOT asserted or rewritten."
    )

    print(
        "No existing Agent output, Router output, or annotation was changed."
    )


if __name__ == "__main__":
    main()
