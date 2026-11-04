from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


# Checks:
#   - supporting gene count == number of unique supporting genes
#   - both-source count == number of unique both-source genes
#   - both-source genes are a subset of supporting genes
#   - candidate ranks are contiguous within each cluster
#   - unique-support counts are non-increasing by candidate rank
#   - threshold candidates meet their stored minimum support
#   - summary leading candidate agrees with threshold rank 1
#   - breast summary selected/threshold counts and cell-type lists agree
#     with the corresponding persisted CSVs when available
#   - lung summary supporting genes/counts agree with threshold rank 1
#   - direct-supported + follow-up == total input where those fields exist



def split_pipe(value: Any) -> list[str]:
    if value is None:
        return []

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return []

    return [
        part.strip()
        for part in text.split("|")
        if part.strip()
    ]


def norm_set(value: Any) -> set[str]:
    return {
        item.casefold()
        for item in split_pipe(value)
    }


def to_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return default

    return int(float(text))


def add_issue(
    issues: list[dict[str, Any]],
    dataset: str,
    cluster_id: Any,
    category: str,
    detail: str,
) -> None:
    issues.append(
        {
            "dataset": dataset,
            "cluster_id": cluster_id,
            "issue": category,
            "detail": detail,
        }
    )


def validate_threshold_table(
    dataset: str,
    df: pd.DataFrame,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:

    required = {
        "cluster_id",
        "candidate_rank",
        "cell_type",
        "unique_supporting_gene_count",
        "supporting_genes",
        "both_source_supporting_gene_count",
        "both_source_supporting_genes",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{dataset}: threshold table missing columns: "
            + ", ".join(sorted(missing))
        )

    supporting_count_failures = 0
    both_count_failures = 0
    both_subset_failures = 0
    rank_contiguity_failures = 0
    support_monotonicity_failures = 0
    minimum_support_failures = 0

    for _, row in df.iterrows():
        cluster = row["cluster_id"]

        supporting = split_pipe(
            row["supporting_genes"]
        )
        supporting_unique = {
            gene.casefold()
            for gene in supporting
        }

        stored_support_count = to_int(
            row["unique_supporting_gene_count"],
            default=-1,
        )

        if stored_support_count != len(supporting_unique):
            supporting_count_failures += 1
            add_issue(
                issues,
                dataset,
                cluster,
                "SUPPORTING_GENE_COUNT_MISMATCH",
                (
                    f"{row['cell_type']}: stored={stored_support_count}; "
                    f"unique_list={len(supporting_unique)}"
                ),
            )

        both = split_pipe(
            row["both_source_supporting_genes"]
        )
        both_unique = {
            gene.casefold()
            for gene in both
        }

        stored_both_count = to_int(
            row["both_source_supporting_gene_count"],
            default=-1,
        )

        if stored_both_count != len(both_unique):
            both_count_failures += 1
            add_issue(
                issues,
                dataset,
                cluster,
                "BOTH_SOURCE_COUNT_MISMATCH",
                (
                    f"{row['cell_type']}: stored={stored_both_count}; "
                    f"unique_list={len(both_unique)}"
                ),
            )

        if not both_unique.issubset(
            supporting_unique
        ):
            both_subset_failures += 1
            add_issue(
                issues,
                dataset,
                cluster,
                "BOTH_SOURCE_NOT_SUBSET_OF_SUPPORTING",
                (
                    f"{row['cell_type']}: extra="
                    f"{sorted(both_unique - supporting_unique)}"
                ),
            )

        if "minimum_supporting_genes" in df.columns:
            minimum = to_int(
                row["minimum_supporting_genes"]
            )

            if (
                minimum is not None
                and stored_support_count is not None
                and stored_support_count < minimum
            ):
                minimum_support_failures += 1
                add_issue(
                    issues,
                    dataset,
                    cluster,
                    "THRESHOLD_MINIMUM_SUPPORT_FAILURE",
                    (
                        f"{row['cell_type']}: support={stored_support_count}; "
                        f"minimum={minimum}"
                    ),
                )

    for cluster, group in df.groupby(
        "cluster_id",
        sort=True,
    ):
        group = group.copy()

        ranks = [
            int(float(value))
            for value in group[
                "candidate_rank"
            ]
        ]

        expected_ranks = list(
            range(
                1,
                len(ranks) + 1,
            )
        )

        if sorted(ranks) != expected_ranks:
            rank_contiguity_failures += 1
            add_issue(
                issues,
                dataset,
                cluster,
                "NONCONTIGUOUS_CANDIDATE_RANKS",
                (
                    f"observed={sorted(ranks)}; "
                    f"expected={expected_ranks}"
                ),
            )

        ordered = group.assign(
            _rank=pd.to_numeric(
                group["candidate_rank"],
                errors="raise",
            ),
            _support=pd.to_numeric(
                group[
                    "unique_supporting_gene_count"
                ],
                errors="raise",
            ),
        ).sort_values(
            "_rank",
            kind="stable",
        )

        supports = ordered[
            "_support"
        ].tolist()

        if any(
            supports[i] < supports[i + 1]
            for i in range(
                len(supports) - 1
            )
        ):
            support_monotonicity_failures += 1
            add_issue(
                issues,
                dataset,
                cluster,
                "SUPPORT_COUNT_INCREASES_WITH_WORSE_RANK",
                f"supports_by_rank={supports}",
            )

    return {
        "row_count": int(len(df)),
        "cluster_count": int(
            df["cluster_id"].nunique()
        ),
        "supporting_gene_count_failures": int(
            supporting_count_failures
        ),
        "both_source_count_failures": int(
            both_count_failures
        ),
        "both_source_subset_failures": int(
            both_subset_failures
        ),
        "rank_contiguity_failures": int(
            rank_contiguity_failures
        ),
        "support_monotonicity_failures": int(
            support_monotonicity_failures
        ),
        "minimum_support_failures": int(
            minimum_support_failures
        ),
    }


def validate_breast(
    threshold: pd.DataFrame,
    selected: pd.DataFrame,
    summary: pd.DataFrame,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:

    count_mismatches = 0
    type_list_mismatches = 0
    leading_mismatches = 0
    accounting_mismatches = 0

    for _, row in summary.iterrows():
        cluster = row["cluster_id"]

        threshold_group = threshold.loc[
            threshold[
                "cluster_id"
            ].astype(str)
            == str(cluster)
        ].copy()

        selected_group = selected.loc[
            selected[
                "cluster_id"
            ].astype(str)
            == str(cluster)
        ].copy()

        stored_threshold_count = to_int(
            row.get(
                "threshold_candidate_count"
            ),
            default=-1,
        )

        stored_selected_count = to_int(
            row.get(
                "selected_candidate_count"
            ),
            default=-1,
        )

        if stored_threshold_count != len(
            threshold_group
        ):
            count_mismatches += 1
            add_issue(
                issues,
                "breast",
                cluster,
                "SUMMARY_THRESHOLD_COUNT_MISMATCH",
                (
                    f"summary={stored_threshold_count}; "
                    f"csv={len(threshold_group)}"
                ),
            )

        if stored_selected_count != len(
            selected_group
        ):
            count_mismatches += 1
            add_issue(
                issues,
                "breast",
                cluster,
                "SUMMARY_SELECTED_COUNT_MISMATCH",
                (
                    f"summary={stored_selected_count}; "
                    f"csv={len(selected_group)}"
                ),
            )

        threshold_types_csv = {
            str(value).strip().casefold()
            for value in threshold_group[
                "cell_type"
            ].tolist()
            if str(value).strip()
        }

        threshold_types_summary = norm_set(
            row.get(
                "threshold_candidate_cell_types"
            )
        )

        if (
            threshold_types_csv
            != threshold_types_summary
        ):
            type_list_mismatches += 1
            add_issue(
                issues,
                "breast",
                cluster,
                "SUMMARY_THRESHOLD_CELL_TYPES_MISMATCH",
                (
                    f"summary={sorted(threshold_types_summary)}; "
                    f"csv={sorted(threshold_types_csv)}"
                ),
            )

        selected_types_csv = {
            str(value).strip().casefold()
            for value in selected_group[
                "cell_type"
            ].tolist()
            if str(value).strip()
        }

        selected_types_summary = norm_set(
            row.get(
                "selected_candidate_cell_types"
            )
        )

        if (
            selected_types_csv
            != selected_types_summary
        ):
            type_list_mismatches += 1
            add_issue(
                issues,
                "breast",
                cluster,
                "SUMMARY_SELECTED_CELL_TYPES_MISMATCH",
                (
                    f"summary={sorted(selected_types_summary)}; "
                    f"csv={sorted(selected_types_csv)}"
                ),
            )

        if not threshold_group.empty:
            top = threshold_group.assign(
                _rank=pd.to_numeric(
                    threshold_group[
                        "candidate_rank"
                    ],
                    errors="raise",
                )
            ).sort_values(
                "_rank",
                kind="stable",
            ).iloc[0]

            leading = str(
                row.get(
                    "leading_candidate",
                    "",
                )
            ).strip().casefold()

            if (
                leading
                != str(
                    top["cell_type"]
                ).strip().casefold()
            ):
                leading_mismatches += 1
                add_issue(
                    issues,
                    "breast",
                    cluster,
                    "SUMMARY_LEADING_CANDIDATE_MISMATCH",
                    (
                        f"summary={row.get('leading_candidate')}; "
                        f"threshold_rank1={top['cell_type']}"
                    ),
                )

        total = to_int(
            row.get(
                "total_input_genes"
            )
        )
        direct = to_int(
            row.get(
                "direct_marker_supported_genes"
            )
        )
        follow_up = to_int(
            row.get(
                "follow_up_gene_count"
            )
        )

        if (
            total is not None
            and direct is not None
            and follow_up is not None
            and direct + follow_up != total
        ):
            accounting_mismatches += 1
            add_issue(
                issues,
                "breast",
                cluster,
                "SUMMARY_GENE_ACCOUNTING_MISMATCH",
                (
                    f"direct={direct}; follow_up={follow_up}; "
                    f"total={total}"
                ),
            )

    return {
        "summary_rows": int(
            len(summary)
        ),
        "selected_candidate_rows": int(
            len(selected)
        ),
        "threshold_candidate_rows": int(
            len(threshold)
        ),
        "summary_count_mismatches": int(
            count_mismatches
        ),
        "summary_cell_type_list_mismatches": int(
            type_list_mismatches
        ),
        "leading_candidate_mismatches": int(
            leading_mismatches
        ),
        "gene_accounting_mismatches": int(
            accounting_mismatches
        ),
    }


def validate_lung(
    threshold: pd.DataFrame,
    summary: pd.DataFrame,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:

    leading_mismatches = 0
    support_count_mismatches = 0
    support_gene_mismatches = 0
    both_count_mismatches = 0
    both_gene_mismatches = 0
    accounting_mismatches = 0

    for _, row in summary.iterrows():
        cluster = row["cluster_id"]

        group = threshold.loc[
            threshold[
                "cluster_id"
            ].astype(str)
            == str(cluster)
        ].copy()

        if group.empty:
            add_issue(
                issues,
                "lung",
                cluster,
                "NO_THRESHOLD_CANDIDATES_FOR_SUMMARY_CLUSTER",
                "",
            )
            continue

        top = group.assign(
            _rank=pd.to_numeric(
                group["candidate_rank"],
                errors="raise",
            )
        ).sort_values(
            "_rank",
            kind="stable",
        ).iloc[0]

        leading_summary = str(
            row.get(
                "leading_candidate",
                "",
            )
        ).strip().casefold()

        leading_top = str(
            top["cell_type"]
        ).strip().casefold()

        if leading_summary != leading_top:
            leading_mismatches += 1
            add_issue(
                issues,
                "lung",
                cluster,
                "SUMMARY_LEADING_CANDIDATE_MISMATCH",
                (
                    f"summary={row.get('leading_candidate')}; "
                    f"threshold_rank1={top['cell_type']}"
                ),
            )

        summary_support_count = to_int(
            row.get(
                "supporting_gene_count"
            ),
            default=-1,
        )

        top_support_count = to_int(
            top[
                "unique_supporting_gene_count"
            ],
            default=-1,
        )

        if summary_support_count != top_support_count:
            support_count_mismatches += 1
            add_issue(
                issues,
                "lung",
                cluster,
                "SUMMARY_SUPPORT_COUNT_MISMATCH",
                (
                    f"summary={summary_support_count}; "
                    f"threshold_rank1={top_support_count}"
                ),
            )

        if norm_set(
            row.get(
                "supporting_markers"
            )
        ) != norm_set(
            top[
                "supporting_genes"
            ]
        ):
            support_gene_mismatches += 1
            add_issue(
                issues,
                "lung",
                cluster,
                "SUMMARY_SUPPORTING_GENES_MISMATCH",
                "",
            )

        summary_both_count = to_int(
            row.get(
                "both_database_supporting_gene_count"
            ),
            default=-1,
        )

        top_both_count = to_int(
            top[
                "both_source_supporting_gene_count"
            ],
            default=-1,
        )

        if summary_both_count != top_both_count:
            both_count_mismatches += 1
            add_issue(
                issues,
                "lung",
                cluster,
                "SUMMARY_BOTH_SOURCE_COUNT_MISMATCH",
                (
                    f"summary={summary_both_count}; "
                    f"threshold_rank1={top_both_count}"
                ),
            )

        if norm_set(
            row.get(
                "both_database_supporting_markers"
            )
        ) != norm_set(
            top[
                "both_source_supporting_genes"
            ]
        ):
            both_gene_mismatches += 1
            add_issue(
                issues,
                "lung",
                cluster,
                "SUMMARY_BOTH_SOURCE_GENES_MISMATCH",
                "",
            )

        total = to_int(
            row.get(
                "total_input_genes"
            )
        )
        direct = to_int(
            row.get(
                "direct_marker_supported_genes"
            )
        )
        follow_up = to_int(
            row.get(
                "follow_up_gene_count"
            )
        )

        if (
            total is not None
            and direct is not None
            and follow_up is not None
            and direct + follow_up != total
        ):
            accounting_mismatches += 1
            add_issue(
                issues,
                "lung",
                cluster,
                "SUMMARY_GENE_ACCOUNTING_MISMATCH",
                (
                    f"direct={direct}; follow_up={follow_up}; "
                    f"total={total}"
                ),
            )

    return {
        "summary_rows": int(
            len(summary)
        ),
        "threshold_candidate_rows": int(
            len(threshold)
        ),
        "leading_candidate_mismatches": int(
            leading_mismatches
        ),
        "support_count_mismatches": int(
            support_count_mismatches
        ),
        "support_gene_mismatches": int(
            support_gene_mismatches
        ),
        "both_source_count_mismatches": int(
            both_count_mismatches
        ),
        "both_source_gene_mismatches": int(
            both_gene_mismatches
        ),
        "gene_accounting_mismatches": int(
            accounting_mismatches
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--breast-threshold",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--breast-selected",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--breast-summary",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--lung-threshold",
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
        default=Path(
            "data/analysis/pipeline_validation/agent2_output_consistency"
        ),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    breast_threshold = pd.read_csv(
        args.breast_threshold
    )

    breast_selected = pd.read_csv(
        args.breast_selected
    )

    breast_summary = pd.read_csv(
        args.breast_summary
    )

    lung_threshold = pd.read_csv(
        args.lung_threshold
    )

    lung_summary = pd.read_csv(
        args.lung_summary
    )

    issues: list[
        dict[str, Any]
    ] = []

    breast_threshold_report = validate_threshold_table(
        "breast",
        breast_threshold,
        issues,
    )

    lung_threshold_report = validate_threshold_table(
        "lung",
        lung_threshold,
        issues,
    )

    breast_report = validate_breast(
        breast_threshold,
        breast_selected,
        breast_summary,
        issues,
    )

    lung_report = validate_lung(
        lung_threshold,
        lung_summary,
        issues,
    )

    breast_failures = sum(
        int(value)
        for key, value
        in {
            **breast_threshold_report,
            **breast_report,
        }.items()
        if (
            key.endswith(
                "_failures"
            )
            or key.endswith(
                "_mismatches"
            )
        )
    )

    lung_failures = sum(
        int(value)
        for key, value
        in {
            **lung_threshold_report,
            **lung_report,
        }.items()
        if (
            key.endswith(
                "_failures"
            )
            or key.endswith(
                "_mismatches"
            )
        )
    )

    breast_status = (
        "PASS"
        if breast_failures == 0
        else "FAIL"
    )

    lung_status = (
        "PASS"
        if lung_failures == 0
        else "FAIL"
    )

    overall_status = (
        "PASS"
        if (
            breast_status == "PASS"
            and lung_status == "PASS"
        )
        else "FAIL"
    )

    report = {
        "audit_role": (
            "READ_ONLY_AGENT2_PERSISTED_OUTPUT_CONSISTENCY"
        ),
        "breast": {
            "threshold_table": breast_threshold_report,
            "cross_file_consistency": breast_report,
            "status": breast_status,
        },
        "lung": {
            "threshold_table": lung_threshold_report,
            "cross_file_consistency": lung_report,
            "status": lung_status,
        },
        "overall_status": overall_status,
        "independent_agent2_reconstruction_performed": False,
        "biological_truth_evaluated": False,
        "existing_outputs_modified": False,
    }

    report_path = (
        args.output_dir
        / "agent2_output_consistency_audit_v1.json"
    )

    issues_path = (
        args.output_dir
        / "agent2_output_consistency_issues_v1.csv"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        issues,
        columns=[
            "dataset",
            "cluster_id",
            "issue",
            "detail",
        ],
    ).to_csv(
        issues_path,
        index=False,
    )

    print(
        f"Agent 2 output-consistency audit: {report_path}"
    )

    print(
        f"Agent 2 consistency issues: {issues_path}"
    )

    print()

    for dataset, threshold_report, cross_report, status in [
        (
            "BREAST",
            breast_threshold_report,
            breast_report,
            breast_status,
        ),
        (
            "LUNG",
            lung_threshold_report,
            lung_report,
            lung_status,
        ),
    ]:
        print(
            f"{dataset}:"
        )

        print(
            f"  Threshold candidate rows: "
            f"{threshold_report['row_count']}"
        )

        print(
            f"  Clusters: "
            f"{threshold_report['cluster_count']}"
        )

        print(
            f"  Supporting-gene count failures: "
            f"{threshold_report['supporting_gene_count_failures']}"
        )

        print(
            f"  Both-source count failures: "
            f"{threshold_report['both_source_count_failures']}"
        )

        print(
            f"  Both-source subset failures: "
            f"{threshold_report['both_source_subset_failures']}"
        )

        print(
            f"  Rank-contiguity failures: "
            f"{threshold_report['rank_contiguity_failures']}"
        )

        print(
            f"  Support monotonicity failures: "
            f"{threshold_report['support_monotonicity_failures']}"
        )

        print(
            f"  Minimum-support failures: "
            f"{threshold_report['minimum_support_failures']}"
        )

        print(
            f"  Cross-file report: "
            f"{cross_report}"
        )

        print(
            f"  STATUS: {status}"
        )

        print()

    print(
        "OVERALL AGENT 2 PERSISTED-OUTPUT CONSISTENCY STATUS: "
        f"{overall_status}"
    )

    print()
    print(
        "This step did NOT independently reconstruct Agent 2 from Agent 1."
    )

    print(
        "No Agent 1, Agent 2, Agent 3, Router, orchestrator, or annotation "
        "output was changed."
    )


if __name__ == "__main__":
    main()
