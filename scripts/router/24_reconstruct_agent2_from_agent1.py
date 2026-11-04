from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd


# READ-ONLY, tissue-agnostic audit.


REQUIRED_MANIFEST_COLUMNS = {
    "dataset",
    "marker_file",
    "cluster_column",
    "gene_column",
    "rank_column",
    "logfc_column",
    "agent1_json",
    "threshold_candidates_csv",
    "minimum_supporting_genes",
}


def clean_string(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, float) and math.isnan(value):
        return ""

    return " ".join(
        str(value).strip().split()
    )


def normalize_gene(value: Any) -> str:
    return clean_string(
        value
    ).upper()


def normalize_cell_type(
    value: Any,
) -> str:
    text = clean_string(
        value
    )

    if not text:
        raise ValueError(
            "Cell type cannot be empty."
        )

    return text


def split_pipe(value: Any) -> list[str]:
    text = clean_string(
        value
    )

    if not text:
        return []

    return [
        clean_string(part)
        for part in text.split("|")
        if clean_string(part)
    ]


def normalized_set(
    value: Any,
) -> set[str]:
    return {
        item.casefold()
        for item in split_pipe(
            value
        )
    }


def unique_sorted(
    values: list[str] | set[str],
) -> list[str]:
    return sorted(
        {
            clean_string(value)
            for value in values
            if clean_string(value)
        },
        key=str.casefold,
    )


def record_cell_type(
    record: dict[str, Any],
) -> str:
    value = (
        record.get(
            "normalized_cell_type"
        )
        or record.get(
            "cell_type"
        )
        or record.get(
            "cell_name"
        )
    )

    return clean_string(
        value
    )


def record_source(
    record: dict[str, Any],
) -> str:
    return clean_string(
        record.get(
            "source"
        )
        or record.get(
            "database"
        )
    )


def split_record_value(
    value: Any,
) -> list[str]:
    if value is None:
        return []

    if isinstance(
        value,
        str,
    ):
        pieces = (
            value
            .replace(",", "|")
            .split("|")
        )

        return unique_sorted(
            pieces
        )

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return unique_sorted(
            [
                str(item)
                for item in value
                if item is not None
            ]
        )

    return [
        clean_string(
            value
        )
    ] if clean_string(
        value
    ) else []


def context_sources(
    gene_record: dict[str, Any],
) -> list[str]:
    sources: list[str] = []

    context_records = (
        gene_record.get(
            "context_matched_database_evidence"
        )
        or []
    )

    for record in context_records:
        if not isinstance(
            record,
            dict,
        ):
            continue

        source = record_source(
            record
        )

        if source:
            sources.append(
                source
            )

    supported_cell_types = (
        gene_record.get(
            "context_matched_cell_types"
        )
        or []
    )

    if (
        not sources
        and supported_cell_types
    ):
        sources.extend(
            [
                clean_string(
                    source
                )
                for source in (
                    gene_record.get(
                        "evidence_sources"
                    )
                    or []
                )
                if clean_string(
                    source
                )
            ]
        )

    return unique_sorted(
        sources
    )


def matching_context_records(
    gene_record: dict[str, Any],
    candidate_cell_type: str,
) -> list[dict[str, Any]]:
    target = (
        normalize_cell_type(
            candidate_cell_type
        ).casefold()
    )

    matches: list[
        dict[str, Any]
    ] = []

    for record in (
        gene_record.get(
            "context_matched_database_evidence"
        )
        or []
    ):
        if not isinstance(
            record,
            dict,
        ):
            continue

        cell_type = record_cell_type(
            record
        )

        if (
            cell_type
            and cell_type.casefold()
            == target
        ):
            matches.append(
                record
            )

    return matches


def candidate_sort_key(
    candidate: dict[str, Any],
) -> tuple[
    int,
    int,
    float,
    float,
    float,
    int,
    str,
]:
    mean_logfc = candidate[
        "mean_supporting_log_fold_change"
    ]

    logfc_key = (
        -float(
            mean_logfc
        )
        if mean_logfc is not None
        else float(
            "inf"
        )
    )

    return (
        -int(
            candidate[
                "unique_supporting_gene_count"
            ]
        ),
        -int(
            candidate[
                "both_source_supporting_gene_count"
            ]
        ),
        -float(
            candidate[
                "specificity_weighted_support"
            ]
        ),
        float(
            candidate[
                "mean_supporting_gene_rank"
            ]
        ),
        logfc_key,
        -int(
            candidate[
                "evidence_count"
            ]
        ),
        str(
            candidate[
                "cell_type"
            ]
        ).casefold(),
    )


def build_candidates_for_cluster(
    cluster_markers: pd.DataFrame,
    *,
    gene_column: str,
    rank_column: str,
    logfc_column: str,
    agent1_map: dict[
        str,
        dict[str, Any],
    ],
) -> list[
    dict[str, Any]
]:

    genes = [
        normalize_gene(
            value
        )
        for value in cluster_markers[
            gene_column
        ].tolist()
    ]

    if len(
        genes
    ) != len(
        set(
            genes
        )
    ):
        raise ValueError(
            "Cluster markers contain duplicate genes."
        )

    aggregates: dict[
        str,
        dict[str, Any],
    ] = {}

    for _, marker_row in cluster_markers.iterrows():
        gene = normalize_gene(
            marker_row[
                gene_column
            ]
        )

        if gene not in agent1_map:
            raise ValueError(
                f"Gene missing from Agent 1 catalog: {gene}"
            )

        gene_record = agent1_map[
            gene
        ]

        supported_cell_types = unique_sorted(
            [
                normalize_cell_type(
                    value
                )
                for value in (
                    gene_record.get(
                        "context_matched_cell_types"
                    )
                    or []
                )
                if clean_string(
                    value
                )
            ]
        )

        direct_marker_supported = bool(
            supported_cell_types
        )

        if not direct_marker_supported:
            continue

        sources = context_sources(
            gene_record
        )

        specificity_score = (
            1.0
            / float(
                len(
                    set(
                        supported_cell_types
                    )
                )
            )
        )

        rank = int(
            float(
                marker_row[
                    rank_column
                ]
            )
        )

        raw_logfc = marker_row[
            logfc_column
        ]

        logfc: float | None

        if pd.isna(
            raw_logfc
        ):
            logfc = None
        else:
            logfc = float(
                raw_logfc
            )

        for cell_type in supported_cell_types:
            display_name = normalize_cell_type(
                cell_type
            )

            key = display_name.casefold()

            if key not in aggregates:
                aggregates[
                    key
                ] = {
                    "display_name": display_name,
                    "supporting_genes": set(),
                    "both_source_genes": set(),
                    "specificity_scores": {},
                    "gene_ranks": {},
                    "log_fold_changes": {},
                    "evidence_count": 0,
                    "evidence_sources": set(),
                    "tissue_contexts": set(),
                    "disease_contexts": set(),
                    "cell_ontology_ids": set(),
                }

            aggregate = aggregates[
                key
            ]

            aggregate[
                "supporting_genes"
            ].add(
                gene
            )

            aggregate[
                "specificity_scores"
            ][
                gene
            ] = specificity_score

            aggregate[
                "gene_ranks"
            ][
                gene
            ] = rank

            if logfc is not None:
                aggregate[
                    "log_fold_changes"
                ][
                    gene
                ] = logfc

            aggregate[
                "evidence_sources"
            ].update(
                sources
            )

            if len(
                set(
                    sources
                )
            ) >= 2:
                aggregate[
                    "both_source_genes"
                ].add(
                    gene
                )

            matches = matching_context_records(
                gene_record,
                display_name,
            )

            for record in matches:
                aggregate[
                    "evidence_count"
                ] += 1

                source = record_source(
                    record
                )

                if source:
                    aggregate[
                        "evidence_sources"
                    ].add(
                        source
                    )

                for field in (
                    "tissue",
                    "tissue_type",
                    "tissue_context",
                ):
                    aggregate[
                        "tissue_contexts"
                    ].update(
                        split_record_value(
                            record.get(
                                field
                            )
                        )
                    )

                for field in (
                    "disease",
                    "disease_context",
                    "cancer_type",
                ):
                    aggregate[
                        "disease_contexts"
                    ].update(
                        split_record_value(
                            record.get(
                                field
                            )
                        )
                    )

                for field in (
                    "cell_ontology_id",
                    "cell_ontology_ids",
                    "cl_id",
                ):
                    aggregate[
                        "cell_ontology_ids"
                    ].update(
                        split_record_value(
                            record.get(
                                field
                            )
                        )
                    )

    candidates: list[
        dict[str, Any]
    ] = []

    for aggregate in aggregates.values():
        supporting_genes = sorted(
            aggregate[
                "supporting_genes"
            ]
        )

        both_source_genes = sorted(
            aggregate[
                "both_source_genes"
            ]
        )

        logfc_values = list(
            aggregate[
                "log_fold_changes"
            ].values()
        )

        candidate = {
            "candidate_rank": 1,
            "cell_type": aggregate[
                "display_name"
            ],
            "unique_supporting_gene_count": len(
                supporting_genes
            ),
            "supporting_genes": supporting_genes,
            "both_source_supporting_gene_count": len(
                both_source_genes
            ),
            "both_source_supporting_genes": both_source_genes,
            "specificity_weighted_support": sum(
                aggregate[
                    "specificity_scores"
                ].values()
            ),
            "mean_supporting_gene_rank": mean(
                aggregate[
                    "gene_ranks"
                ].values()
            ),
            "mean_supporting_log_fold_change": (
                mean(
                    logfc_values
                )
                if logfc_values
                else None
            ),
            "evidence_count": int(
                aggregate[
                    "evidence_count"
                ]
            ),
            "evidence_sources": sorted(
                aggregate[
                    "evidence_sources"
                ],
                key=str.casefold,
            ),
            "tissue_contexts": sorted(
                aggregate[
                    "tissue_contexts"
                ],
                key=str.casefold,
            ),
            "disease_contexts": sorted(
                aggregate[
                    "disease_contexts"
                ],
                key=str.casefold,
            ),
            "cell_ontology_ids": sorted(
                aggregate[
                    "cell_ontology_ids"
                ],
                key=str.casefold,
            ),
        }

        candidates.append(
            candidate
        )

    candidates = sorted(
        candidates,
        key=candidate_sort_key,
    )

    for rank, candidate in enumerate(
        candidates,
        start=1,
    ):
        candidate[
            "candidate_rank"
        ] = rank

    return candidates


def close_float(
    actual: Any,
    expected: Any,
    *,
    abs_tol: float = 1e-9,
) -> bool:

    if (
        actual is None
        or (
            isinstance(
                actual,
                float,
            )
            and math.isnan(
                actual
            )
        )
    ):
        return (
            expected is None
            or (
                isinstance(
                    expected,
                    float,
                )
                and math.isnan(
                    expected
                )
            )
        )

    if (
        expected is None
        or (
            isinstance(
                expected,
                float,
            )
            and math.isnan(
                expected
            )
        )
    ):
        return False

    return math.isclose(
        float(
            actual
        ),
        float(
            expected
        ),
        rel_tol=1e-9,
        abs_tol=abs_tol,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/analysis/pipeline_validation/"
            "agent2_independent_reconstruction"
        ),
    )

    args = parser.parse_args()

    manifest = pd.read_csv(
        args.manifest
    )

    missing_manifest = (
        REQUIRED_MANIFEST_COLUMNS
        - set(
            manifest.columns
        )
    )

    if missing_manifest:
        raise ValueError(
            "Manifest missing columns: "
            + ", ".join(
                sorted(
                    missing_manifest
                )
            )
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_reports: list[
        dict[str, Any]
    ] = []

    mismatch_rows: list[
        dict[str, Any]
    ] = []

    reconstructed_rows: list[
        dict[str, Any]
    ] = []

    for _, config in manifest.iterrows():
        dataset = clean_string(
            config[
                "dataset"
            ]
        )

        marker_path = Path(
            str(
                config[
                    "marker_file"
                ]
            )
        )

        agent1_path = Path(
            str(
                config[
                    "agent1_json"
                ]
            )
        )

        stored_path = Path(
            str(
                config[
                    "threshold_candidates_csv"
                ]
            )
        )

        for name, path in (
            (
                "marker",
                marker_path,
            ),
            (
                "agent1",
                agent1_path,
            ),
            (
                "stored candidates",
                stored_path,
            ),
        ):
            if not path.is_absolute():
                path = (
                    args.project_root
                    / path
                )

            if not path.exists():
                raise FileNotFoundError(
                    f"{dataset}: {name} file not found: {path}"
                )

            if name == "marker":
                marker_path = path
            elif name == "agent1":
                agent1_path = path
            else:
                stored_path = path

        cluster_column = clean_string(
            config[
                "cluster_column"
            ]
        )

        gene_column = clean_string(
            config[
                "gene_column"
            ]
        )

        rank_column = clean_string(
            config[
                "rank_column"
            ]
        )

        logfc_column = clean_string(
            config[
                "logfc_column"
            ]
        )

        minimum_support = int(
            float(
                config[
                    "minimum_supporting_genes"
                ]
            )
        )

        markers = pd.read_csv(
            marker_path
        )

        required_marker_columns = {
            cluster_column,
            gene_column,
            rank_column,
            logfc_column,
        }

        missing_marker = (
            required_marker_columns
            - set(
                markers.columns
            )
        )

        if missing_marker:
            raise ValueError(
                f"{dataset}: marker file missing columns: "
                + ", ".join(
                    sorted(
                        missing_marker
                    )
                )
            )

        with agent1_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            agent1 = json.load(
                handle
            )

        gene_evidence = (
            agent1.get(
                "gene_evidence"
            )
            or []
        )

        agent1_map = {
            normalize_gene(
                record.get(
                    "gene"
                )
            ): record
            for record in gene_evidence
            if isinstance(
                record,
                dict,
            )
            and normalize_gene(
                record.get(
                    "gene"
                )
            )
        }

        stored = pd.read_csv(
            stored_path
        )

        reconstructed_threshold: list[
            dict[str, Any]
        ] = []

        full_candidate_count = 0

        for cluster_id, group in markers.groupby(
            cluster_column,
            sort=True,
        ):
            group = group.sort_values(
                rank_column,
                kind="stable",
            )

            candidates = build_candidates_for_cluster(
                group,
                gene_column=gene_column,
                rank_column=rank_column,
                logfc_column=logfc_column,
                agent1_map=agent1_map,
            )

            full_candidate_count += len(
                candidates
            )

            threshold_candidates = [
                candidate
                for candidate in candidates
                if candidate[
                    "unique_supporting_gene_count"
                ]
                >= minimum_support
            ]

            for candidate in threshold_candidates:
                reconstructed_threshold.append(
                    {
                        "dataset": dataset,
                        "cluster_id": cluster_id,
                        **candidate,
                    }
                )

        reconstructed_df = pd.DataFrame(
            reconstructed_threshold
        )

        reconstructed_rows.extend(
            reconstructed_threshold
        )

        stored_keyed = {
            (
                str(
                    row[
                        "cluster_id"
                    ]
                ),
                str(
                    row[
                        "cell_type"
                    ]
                ).strip().casefold(),
            ): row
            for _, row in stored.iterrows()
        }

        reconstructed_keyed = {
            (
                str(
                    row[
                        "cluster_id"
                    ]
                ),
                str(
                    row[
                        "cell_type"
                    ]
                ).strip().casefold(),
            ): row
            for _, row in reconstructed_df.iterrows()
        }

        stored_keys = set(
            stored_keyed
        )

        reconstructed_keys = set(
            reconstructed_keyed
        )

        missing_reconstructed = sorted(
            stored_keys
            - reconstructed_keys
        )

        extra_reconstructed = sorted(
            reconstructed_keys
            - stored_keys
        )

        for key in missing_reconstructed:
            mismatch_rows.append(
                {
                    "dataset": dataset,
                    "cluster_id": key[0],
                    "cell_type": stored_keyed[
                        key
                    ][
                        "cell_type"
                    ],
                    "field": "candidate_presence",
                    "stored": "present",
                    "reconstructed": "missing",
                }
            )

        for key in extra_reconstructed:
            mismatch_rows.append(
                {
                    "dataset": dataset,
                    "cluster_id": key[0],
                    "cell_type": reconstructed_keyed[
                        key
                    ][
                        "cell_type"
                    ],
                    "field": "candidate_presence",
                    "stored": "missing",
                    "reconstructed": "present",
                }
            )

        compared_candidates = 0
        exact_candidates = 0
        field_mismatch_count = 0

        for key in sorted(
            stored_keys
            & reconstructed_keys
        ):
            stored_row = stored_keyed[
                key
            ]

            recon_row = reconstructed_keyed[
                key
            ]

            compared_candidates += 1

            candidate_has_mismatch = False

            integer_fields = [
                "candidate_rank",
                "unique_supporting_gene_count",
                "both_source_supporting_gene_count",
            ]

            for field in integer_fields:
                stored_value = int(
                    float(
                        stored_row[
                            field
                        ]
                    )
                )

                recon_value = int(
                    recon_row[
                        field
                    ]
                )

                if stored_value != recon_value:
                    candidate_has_mismatch = True
                    field_mismatch_count += 1

                    mismatch_rows.append(
                        {
                            "dataset": dataset,
                            "cluster_id": key[0],
                            "cell_type": stored_row[
                                "cell_type"
                            ],
                            "field": field,
                            "stored": stored_value,
                            "reconstructed": recon_value,
                        }
                    )

            list_fields = [
                (
                    "supporting_genes",
                    "supporting_genes",
                ),
                (
                    "both_source_supporting_genes",
                    "both_source_supporting_genes",
                ),
                (
                    "evidence_sources",
                    "evidence_sources",
                ),
                (
                    "tissue_contexts",
                    "tissue_contexts",
                ),
                (
                    "disease_contexts",
                    "disease_contexts",
                ),
                (
                    "cell_ontology_ids",
                    "cell_ontology_ids",
                ),
            ]

            for stored_field, recon_field in list_fields:
                stored_set = normalized_set(
                    stored_row.get(
                        stored_field
                    )
                )

                recon_set = {
                    clean_string(
                        value
                    ).casefold()
                    for value in recon_row[
                        recon_field
                    ]
                    if clean_string(
                        value
                    )
                }

                if stored_set != recon_set:
                    candidate_has_mismatch = True
                    field_mismatch_count += 1

                    mismatch_rows.append(
                        {
                            "dataset": dataset,
                            "cluster_id": key[0],
                            "cell_type": stored_row[
                                "cell_type"
                            ],
                            "field": stored_field,
                            "stored": " | ".join(
                                sorted(
                                    stored_set
                                )
                            ),
                            "reconstructed": " | ".join(
                                sorted(
                                    recon_set
                                )
                            ),
                        }
                    )

            float_fields = [
                "specificity_weighted_support",
                "mean_supporting_gene_rank",
                "mean_supporting_log_fold_change",
            ]

            for field in float_fields:
                stored_value = stored_row.get(
                    field
                )

                recon_value = recon_row.get(
                    field
                )

                if not close_float(
                    stored_value,
                    recon_value,
                ):
                    candidate_has_mismatch = True
                    field_mismatch_count += 1

                    mismatch_rows.append(
                        {
                            "dataset": dataset,
                            "cluster_id": key[0],
                            "cell_type": stored_row[
                                "cell_type"
                            ],
                            "field": field,
                            "stored": stored_value,
                            "reconstructed": recon_value,
                        }
                    )

            if not candidate_has_mismatch:
                exact_candidates += 1

        dataset_pass = (
            not missing_reconstructed
            and not extra_reconstructed
            and field_mismatch_count == 0
        )

        dataset_reports.append(
            {
                "dataset": dataset,
                "cluster_count": int(
                    markers[
                        cluster_column
                    ].nunique()
                ),
                "full_reconstructed_candidate_count": int(
                    full_candidate_count
                ),
                "minimum_supporting_genes": int(
                    minimum_support
                ),
                "stored_threshold_candidate_count": int(
                    len(
                        stored
                    )
                ),
                "reconstructed_threshold_candidate_count": int(
                    len(
                        reconstructed_df
                    )
                ),
                "stored_candidates_missing_from_reconstruction": int(
                    len(
                        missing_reconstructed
                    )
                ),
                "extra_reconstructed_candidates": int(
                    len(
                        extra_reconstructed
                    )
                ),
                "compared_candidates": int(
                    compared_candidates
                ),
                "exact_candidates_all_compared_fields": int(
                    exact_candidates
                ),
                "field_mismatch_count": int(
                    field_mismatch_count
                ),
                "status": (
                    "PASS"
                    if dataset_pass
                    else "FAIL"
                ),
            }
        )

    overall_pass = all(
        report[
            "status"
        ]
        == "PASS"
        for report in dataset_reports
    )

    report = {
        "audit_role": (
            "READ_ONLY_INDEPENDENT_AGENT2_RECONSTRUCTION_FROM_AGENT1"
        ),
        "agent2_implementation_imported": False,
        "datasets": dataset_reports,
        "overall_status": (
            "PASS"
            if overall_pass
            else "FAIL"
        ),
        "biological_ground_truth_evaluated": False,
        "hpa_used_for_candidate_voting": False,
        "existing_outputs_modified": False,
    }

    report_path = (
        args.output_dir
        / "agent2_independent_reconstruction_audit_v1.json"
    )

    mismatch_path = (
        args.output_dir
        / "agent2_independent_reconstruction_mismatches_v1.csv"
    )

    reconstruction_path = (
        args.output_dir
        / "agent2_reconstructed_threshold_candidates_v1.csv"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        mismatch_rows,
        columns=[
            "dataset",
            "cluster_id",
            "cell_type",
            "field",
            "stored",
            "reconstructed",
        ],
    ).to_csv(
        mismatch_path,
        index=False,
    )

    pd.DataFrame(
        reconstructed_rows
    ).to_csv(
        reconstruction_path,
        index=False,
    )

    print(
        f"Agent 2 independent reconstruction audit: {report_path}"
    )

    print(
        f"Mismatch table: {mismatch_path}"
    )

    print(
        f"Reconstructed candidates: {reconstruction_path}"
    )

    print()

    for item in dataset_reports:
        print(
            f"{item['dataset'].upper()}:"
        )

        print(
            f"  Clusters: "
            f"{item['cluster_count']}"
        )

        print(
            f"  Full reconstructed candidates: "
            f"{item['full_reconstructed_candidate_count']}"
        )

        print(
            f"  Minimum supporting genes: "
            f"{item['minimum_supporting_genes']}"
        )

        print(
            f"  Stored threshold candidates: "
            f"{item['stored_threshold_candidate_count']}"
        )

        print(
            f"  Reconstructed threshold candidates: "
            f"{item['reconstructed_threshold_candidate_count']}"
        )

        print(
            f"  Missing from reconstruction: "
            f"{item['stored_candidates_missing_from_reconstruction']}"
        )

        print(
            f"  Extra reconstructed candidates: "
            f"{item['extra_reconstructed_candidates']}"
        )

        print(
            f"  Exact candidates across compared fields: "
            f"{item['exact_candidates_all_compared_fields']}/"
            f"{item['compared_candidates']}"
        )

        print(
            f"  Field mismatches: "
            f"{item['field_mismatch_count']}"
        )

        print(
            f"  STATUS: "
            f"{item['status']}"
        )

        print()

    print(
        "OVERALL AGENT 2 INDEPENDENT RECONSTRUCTION STATUS: "
        f"{report['overall_status']}"
    )

    print()
    print(
        "Agent 2 implementation code was NOT imported by this validator."
    )

    print(
        "No Agent 1, Agent 2, Agent 3, Router, orchestrator, or annotation "
        "output was changed."
    )


if __name__ == "__main__":
    main()
