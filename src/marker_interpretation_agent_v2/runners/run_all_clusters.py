
from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from ..agent import MarkerInterpretationAgent
from ..config import DEFAULT_CONFIG
from ..interpretation.candidate_selector import (
    CandidateSelector,
)
from ..panel.gene_panel_loader import (
    GenePanelLoader,
)
from ..panel.panel_evidence_loader import (
    PanelEvidenceLoader,
)
from ..paths import (
    AGENT2_V2_DATA_DIR,
    PANEL_CATALOG_JSON,
    get_marker_file,
    normalize_path,
)
from ..schemas import (
    CandidateCellType,
    MarkerEvidence,
    PanelEvidenceCatalog,
)


@dataclass(frozen=True, slots=True)
class RunAllClustersArgs:
    """Typed command-line arguments for all-cluster execution."""

    marker_file: Path | None
    panel_catalog: Path
    output_root: Path
    species: str
    tissue: str
    platform: str
    top_k: int
    hpa_enabled: bool
    write_reports: bool


def _normalize_cluster_id(
    value: str | int,
) -> str:
    """Normalize one cluster identifier."""

    cluster_id = str(value).strip()

    if not cluster_id:
        raise ValueError(
            "cluster_id cannot be empty."
        )

    return cluster_id


def _normalize_gene(
    value: object,
) -> str:
    """Normalize one gene symbol."""

    gene = str(value).strip().upper()

    if not gene:
        raise ValueError(
            "Gene symbol cannot be empty."
        )

    return gene


def _cluster_sort_key(
    value: str,
) -> tuple[int, int | str]:
    """Sort numeric cluster identifiers before text identifiers."""

    try:
        return 0, int(value)

    except ValueError:
        return 1, value.casefold()


def _write_csv(
    path: Path,
    *,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, Any]],
) -> None:
    """Write a CSV with headers even when no rows exist."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(fieldnames),
            extrasaction="raise",
        )

        writer.writeheader()
        writer.writerows(rows)


def _marker_panel_genes(
    panel_loader: GenePanelLoader,
) -> set[str]:
    """Collect all unique panel genes."""

    genes: set[str] = set()

    for cluster_id in panel_loader.get_cluster_ids():
        input_genes = (
            panel_loader.get_cluster_markers(
                cluster_id
            )
        )

        for input_gene in input_genes:
            genes.add(
                _normalize_gene(
                    input_gene.gene
                )
            )

    if not genes:
        raise ValueError(
            "The marker CSV contains no panel genes."
        )

    return genes


def _validate_catalog_panel(
    panel_loader: GenePanelLoader,
    catalog: PanelEvidenceCatalog,
) -> None:
    """Require marker panel and reusable catalog to match."""

    marker_genes = _marker_panel_genes(
        panel_loader
    )

    catalog_genes = {
        _normalize_gene(record.gene)
        for record in catalog.genes
    }

    missing = sorted(
        marker_genes - catalog_genes
    )

    extra = sorted(
        catalog_genes - marker_genes
    )

    if missing or extra:
        raise ValueError(
            "Marker CSV and panel evidence catalog "
            "do not cover the same genes. "
            f"Missing from catalog: {missing}; "
            f"extra in catalog: {extra}."
        )


def _threshold_candidates_by_cluster(
    results: Sequence[MarkerEvidence],
    selector: CandidateSelector,
) -> dict[str, list[CandidateCellType]]:
    """Build prospective candidates using the support threshold."""

    return {
        result.cluster_id: (
            selector.select_by_support_threshold(
                result.ranked_candidate_cell_types
            )
        )
        for result in results
    }


def _summary_rows(
    results: Sequence[MarkerEvidence],
    threshold_candidates: Mapping[
        str,
        Sequence[CandidateCellType],
    ],
    *,
    threshold: int,
) -> list[dict[str, Any]]:
    """Create one summary row per cluster."""

    rows: list[dict[str, Any]] = []

    for result in results:
        prospective = list(
            threshold_candidates.get(
                result.cluster_id,
                (),
            )
        )

        rows.append(
            {
                "cluster_id": (
                    result.cluster_id
                ),
                "total_input_genes": (
                    result.summary
                    .total_input_genes
                ),
                "direct_marker_supported_genes": (
                    result.summary
                    .direct_marker_supported_genes
                ),
                "hpa_queried_genes": (
                    result.summary
                    .hpa_queried_genes
                ),
                "hpa_gene_matches": (
                    result.summary
                    .hpa_gene_matches
                ),
                "hpa_breast_expression_genes": (
                    result.summary
                    .hpa_breast_expression_genes
                ),
                "ranked_marker_gene_count": (
                    result.summary
                    .ranked_marker_gene_count
                ),
                "ranked_candidate_count": (
                    result.summary
                    .ranked_candidate_count
                ),
                "selected_candidate_count": (
                    result.summary
                    .selected_candidate_count
                ),
                "selected_candidate_cell_types": (
                    " | ".join(
                        result.candidate_selection
                        .selected_candidate_cell_types
                    )
                ),
                "leading_candidate": (
                    result.summary.leading_candidate
                    or ""
                ),
                "candidate_cutoff_method": (
                    result.candidate_selection
                    .cutoff_method
                ),
                "threshold_min_supporting_genes": (
                    threshold
                ),
                "threshold_candidate_count": (
                    len(prospective)
                ),
                "threshold_candidate_cell_types": (
                    " | ".join(
                        candidate.cell_type
                        for candidate
                        in prospective
                    )
                ),
                "follow_up_gene_count": (
                    result.summary
                    .follow_up_gene_count
                ),
                "human_review_required": (
                    result.summary
                    .human_review_required
                ),
                "warning_flag_codes": (
                    " | ".join(
                        sorted(
                            flag.code
                            for flag
                            in result.review_flags
                            if (
                                flag.severity
                                == "warning"
                            )
                        )
                    )
                ),
            }
        )

    return rows


def _candidate_row(
    result: MarkerEvidence,
    candidate: CandidateCellType,
) -> dict[str, Any]:
    """Create one candidate report row."""

    return {
        "cluster_id": (
            result.cluster_id
        ),
        "candidate_rank": (
            candidate.candidate_rank
        ),
        "cell_type": (
            candidate.cell_type
        ),
        "unique_supporting_gene_count": (
            candidate
            .unique_supporting_gene_count
        ),
        "supporting_genes": (
            " | ".join(
                candidate.supporting_genes
            )
        ),
        "both_source_supporting_gene_count": (
            candidate
            .both_source_supporting_gene_count
        ),
        "both_source_supporting_genes": (
            " | ".join(
                candidate
                .both_source_supporting_genes
            )
        ),
        "specificity_weighted_support": (
            candidate
            .specificity_weighted_support
        ),
        "mean_supporting_gene_rank": (
            candidate
            .mean_supporting_gene_rank
        ),
        "mean_supporting_log_fold_change": (
            candidate
            .mean_supporting_log_fold_change
        ),
        "evidence_sources": (
            " | ".join(
                candidate.evidence_sources
            )
        ),
        "tissue_contexts": (
            " | ".join(
                candidate.tissue_contexts
            )
        ),
        "disease_contexts": (
            " | ".join(
                candidate.disease_contexts
            )
        ),
        "cell_ontology_ids": (
            " | ".join(
                candidate.cell_ontology_ids
            )
        ),
    }


def _selected_candidate_rows(
    results: Sequence[MarkerEvidence],
) -> list[dict[str, Any]]:
    """Create rows for support-drop-selected candidates."""

    return [
        _candidate_row(
            result,
            candidate,
        )
        for result in results
        for candidate
        in result.selected_candidate_cell_types
    ]


def _threshold_candidate_rows(
    results: Sequence[MarkerEvidence],
    threshold_candidates: Mapping[
        str,
        Sequence[CandidateCellType],
    ],
    *,
    threshold: int,
) -> list[dict[str, Any]]:
    """Create rows for support-threshold candidates."""

    rows: list[dict[str, Any]] = []

    for result in results:
        for candidate in threshold_candidates.get(
            result.cluster_id,
            (),
        ):
            row = _candidate_row(
                result,
                candidate,
            )

            row["selection_method"] = (
                "minimum_support_threshold"
            )

            row[
                "minimum_supporting_genes"
            ] = threshold

            rows.append(
                row
            )

    return rows


def _follow_up_rows(
    results: Sequence[MarkerEvidence],
) -> list[dict[str, Any]]:
    """Create one deduplicated follow-up row per gene."""

    aggregates: dict[
        str,
        dict[str, set[str]],
    ] = {}

    for result in results:
        for item in result.follow_up_gene_queue:
            gene = _normalize_gene(
                item.gene
            )

            aggregate = aggregates.setdefault(
                gene,
                {
                    "cluster_ids": set(),
                    "evidence_categories": set(),
                    "recommended_actions": set(),
                    "reasons": set(),
                    "source_coverage": set(),
                    "warnings": set(),
                },
            )

            aggregate[
                "cluster_ids"
            ].add(
                result.cluster_id
            )

            aggregate[
                "evidence_categories"
            ].add(
                item.evidence_category
            )

            aggregate[
                "recommended_actions"
            ].add(
                item.recommended_action
            )

            aggregate[
                "reasons"
            ].add(
                item.reason
            )

            aggregate[
                "source_coverage"
            ].update(
                item.source_coverage
            )

            aggregate[
                "warnings"
            ].update(
                item.warnings
            )

    rows: list[dict[str, Any]] = []

    for gene in sorted(aggregates):
        aggregate = aggregates[gene]

        cluster_ids = sorted(
            aggregate["cluster_ids"],
            key=_cluster_sort_key,
        )

        rows.append(
            {
                "gene": gene,
                "cluster_count": (
                    len(cluster_ids)
                ),
                "cluster_ids": (
                    " | ".join(
                        cluster_ids
                    )
                ),
                "evidence_categories": (
                    " | ".join(
                        sorted(
                            aggregate[
                                "evidence_categories"
                            ]
                        )
                    )
                ),
                "recommended_actions": (
                    " | ".join(
                        sorted(
                            aggregate[
                                "recommended_actions"
                            ]
                        )
                    )
                ),
                "reasons": (
                    " | ".join(
                        sorted(
                            aggregate[
                                "reasons"
                            ]
                        )
                    )
                ),
                "source_coverage": (
                    " | ".join(
                        sorted(
                            aggregate[
                                "source_coverage"
                            ]
                        )
                    )
                ),
                "warnings": (
                    " | ".join(
                        sorted(
                            aggregate[
                                "warnings"
                            ]
                        )
                    )
                ),
            }
        )

    return rows


def _write_all_cluster_reports(
    results: Sequence[MarkerEvidence],
    *,
    output_root: Path,
    marker_file: Path,
    panel_catalog: Path,
    candidate_selector: CandidateSelector,
) -> dict[str, Path]:
    """Write aggregate reports for all clusters."""

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_csv = (
        output_root
        / "all_clusters_summary.csv"
    )

    summary_json = (
        output_root
        / "all_clusters_summary.json"
    )

    selected_csv = (
        output_root
        / "all_clusters_selected_candidates.csv"
    )

    threshold_csv = (
        output_root
        / "all_clusters_threshold_candidates.csv"
    )

    follow_up_csv = (
        output_root
        / "all_clusters_follow_up_gene_queue.csv"
    )

    threshold = (
        candidate_selector.config
        .candidate_reporting_min_supporting_genes
    )

    threshold_candidates = (
        _threshold_candidates_by_cluster(
            results,
            candidate_selector,
        )
    )

    summary_rows = _summary_rows(
        results,
        threshold_candidates,
        threshold=threshold,
    )

    selected_rows = (
        _selected_candidate_rows(
            results
        )
    )

    threshold_rows = (
        _threshold_candidate_rows(
            results,
            threshold_candidates,
            threshold=threshold,
        )
    )

    follow_up_rows = (
        _follow_up_rows(
            results
        )
    )

    _write_csv(
        summary_csv,
        fieldnames=[
            "cluster_id",
            "total_input_genes",
            "direct_marker_supported_genes",
            "hpa_queried_genes",
            "hpa_gene_matches",
            "hpa_breast_expression_genes",
            "ranked_marker_gene_count",
            "ranked_candidate_count",
            "selected_candidate_count",
            "selected_candidate_cell_types",
            "leading_candidate",
            "candidate_cutoff_method",
            "threshold_min_supporting_genes",
            "threshold_candidate_count",
            "threshold_candidate_cell_types",
            "follow_up_gene_count",
            "human_review_required",
            "warning_flag_codes",
        ],
        rows=summary_rows,
    )

    candidate_fieldnames = [
        "cluster_id",
        "candidate_rank",
        "cell_type",
        "unique_supporting_gene_count",
        "supporting_genes",
        "both_source_supporting_gene_count",
        "both_source_supporting_genes",
        "specificity_weighted_support",
        "mean_supporting_gene_rank",
        "mean_supporting_log_fold_change",
        "evidence_sources",
        "tissue_contexts",
        "disease_contexts",
        "cell_ontology_ids",
    ]

    _write_csv(
        selected_csv,
        fieldnames=candidate_fieldnames,
        rows=selected_rows,
    )

    _write_csv(
        threshold_csv,
        fieldnames=[
            *candidate_fieldnames,
            "selection_method",
            "minimum_supporting_genes",
        ],
        rows=threshold_rows,
    )

    _write_csv(
        follow_up_csv,
        fieldnames=[
            "gene",
            "cluster_count",
            "cluster_ids",
            "evidence_categories",
            "recommended_actions",
            "reasons",
            "source_coverage",
            "warnings",
        ],
        rows=follow_up_rows,
    )

    summary_payload = {
        "schema_name": (
            "Agent2AllClustersSummary"
        ),
        "schema_version": "2.1",
        "marker_file": str(
            marker_file
        ),
        "panel_catalog": str(
            panel_catalog
        ),
        "cluster_count": len(
            results
        ),
        "cluster_ids": [
            result.cluster_id
            for result in results
        ],
        "support_threshold": {
            "minimum_supporting_genes": (
                threshold
            ),
            "method": (
                "minimum_support_threshold"
            ),
        },
        "unique_follow_up_gene_count": (
            len(follow_up_rows)
        ),
        "clusters": [
            {
                "cluster_id": (
                    result.cluster_id
                ),
                "summary": (
                    result.summary.model_dump()
                ),
                "candidate_selection": (
                    result.candidate_selection
                    .model_dump()
                ),
                "support_threshold_selection": {
                    "minimum_supporting_genes": (
                        threshold
                    ),
                    "selected_candidate_count": len(
                        threshold_candidates[
                            result.cluster_id
                        ]
                    ),
                    "selected_candidate_cell_types": [
                        candidate.cell_type
                        for candidate
                        in threshold_candidates[
                            result.cluster_id
                        ]
                    ],
                },
                "warning_flag_codes": [
                    flag.code
                    for flag
                    in result.review_flags
                    if (
                        flag.severity
                        == "warning"
                    )
                ],
                "explanation": (
                    result.explanation
                ),
            }
            for result in results
        ],
    }

    summary_json.write_text(
        json.dumps(
            summary_payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {
        "all_clusters_summary_csv": (
            summary_csv
        ),
        "all_clusters_summary_json": (
            summary_json
        ),
        "all_clusters_selected_candidates_csv": (
            selected_csv
        ),
        "all_clusters_threshold_candidates_csv": (
            threshold_csv
        ),
        "all_clusters_follow_up_gene_queue_csv": (
            follow_up_csv
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    """Create the all-cluster command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Run Marker Interpretation Agent V2 for "
            "every cluster using one reusable panel "
            "evidence catalog."
        )
    )

    parser.add_argument(
        "--marker-file",
        type=Path,
        default=None,
        help=(
            "Cluster marker CSV. When omitted, "
            "the configured marker file is used."
        ),
    )

    parser.add_argument(
        "--panel-catalog",
        type=Path,
        default=PANEL_CATALOG_JSON,
        help=(
            "Reusable panel evidence catalog JSON. "
            f"Default: {PANEL_CATALOG_JSON}"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=AGENT2_V2_DATA_DIR,
        help=(
            "Root directory for reports. "
            f"Default: {AGENT2_V2_DATA_DIR}"
        ),
    )

    parser.add_argument(
        "--species",
        default=DEFAULT_CONFIG.species,
        help=(
            "Species context for Agent 2. "
            f"Default: {DEFAULT_CONFIG.species}"
        ),
    )

    parser.add_argument(
        "--tissue",
        default=DEFAULT_CONFIG.tissue,
        help=(
            "Tissue context for Agent 2. "
            f"Default: {DEFAULT_CONFIG.tissue}"
        ),
    )

    parser.add_argument(
        "--platform",
        default=DEFAULT_CONFIG.platform,
        help=(
            "Platform context for Agent 2. "
            f"Default: {DEFAULT_CONFIG.platform}"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_CONFIG.top_k,
        help=(
            "Maximum marker genes interpreted "
            "per cluster. "
            f"Default: {DEFAULT_CONFIG.top_k}"
        ),
    )

    parser.add_argument(
        "--no-hpa",
        action="store_true",
        help=(
            "Run marker interpretation with HPA disabled. "
            "Use this with marker-only catalogs."
        ),
    )

    parser.add_argument(
        "--no-write-reports",
        action="store_true",
        help=(
            "Run interpretation without "
            "writing report files."
        ),
    )

    return parser


def run(
    args: RunAllClustersArgs,
) -> tuple[
    list[MarkerEvidence],
    dict[str, Path],
]:
    """Interpret every cluster and write reports."""

    if args.top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    marker_source = (
        normalize_path(
            args.marker_file
        )
        if args.marker_file is not None
        else get_marker_file()
    )

    catalog_source = normalize_path(
        args.panel_catalog
    )

    output_root = normalize_path(
        args.output_root
    )

    if not marker_source.exists():
        raise FileNotFoundError(
            "Marker CSV does not exist: "
            f"{marker_source}"
        )

    if not catalog_source.exists():
        raise FileNotFoundError(
            "Panel evidence catalog does not exist: "
            f"{catalog_source}. Build the real "
            "panel catalog first."
        )

    panel_loader = GenePanelLoader(
        marker_source
    )

    catalog = (
        PanelEvidenceLoader()
        .load_catalog(
            catalog_source
        )
    )

    _validate_catalog_panel(
        panel_loader,
        catalog,
    )

    cluster_ids = sorted(
        {
            _normalize_cluster_id(
                cluster_id
            )
            for cluster_id
            in panel_loader.get_cluster_ids()
        },
        key=_cluster_sort_key,
    )

    if not cluster_ids:
        raise ValueError(
            "The marker CSV contains no clusters."
        )

    config = replace(
        DEFAULT_CONFIG,
        species=args.species,
        tissue=args.tissue,
        platform=args.platform,
        top_k=args.top_k,
        hpa_enabled=args.hpa_enabled,
    )

    agent = MarkerInterpretationAgent(
        config
    )

    results: list[
        MarkerEvidence
    ] = []

    for cluster_id in cluster_ids:
        input_genes = (
            panel_loader
            .get_cluster_markers(
                cluster_id
            )
        )

        if not input_genes:
            raise ValueError(
                f"Cluster {cluster_id!r} "
                "contains no marker genes."
            )

        results.append(
            agent.interpret_cluster(
                cluster_id=cluster_id,
                input_genes=input_genes,
                catalog=catalog,
            )
        )

    if not args.write_reports:
        return results, {}

    for result in results:
        cluster_output_dir = (
            output_root
            / "clusters"
            / f"cluster_{result.cluster_id}"
        )

        agent.write_reports(
            result,
            output_dir=(
                cluster_output_dir
            ),
        )

    aggregate_paths = (
        _write_all_cluster_reports(
            results,
            output_root=output_root,
            marker_file=marker_source,
            panel_catalog=catalog_source,
            candidate_selector=(
                agent.candidate_selector
            ),
        )
    )

    return results, aggregate_paths


def _print_results(
    results: Sequence[MarkerEvidence],
    paths: dict[str, Path],
) -> None:
    """Print a concise execution summary."""

    threshold = (
        DEFAULT_CONFIG
        .candidate_reporting_min_supporting_genes
    )

    selector = CandidateSelector(
        DEFAULT_CONFIG
    )

    print(
        "Marker Interpretation Agent V2 "
        "completed all clusters."
    )

    print(
        f"Cluster count: {len(results)}"
    )

    print(
        "Prospective candidate support threshold: "
        f"{threshold}"
    )

    for result in results:
        sudden_drop_selected = ", ".join(
            result.candidate_selection
            .selected_candidate_cell_types
        ) or "None"

        threshold_selected = ", ".join(
            candidate.cell_type
            for candidate
            in selector.select_by_support_threshold(
                result.ranked_candidate_cell_types
            )
        ) or "None"

        print(
            f"Cluster {result.cluster_id}: "
            f"leading="
            f"{result.summary.leading_candidate}; "
            "sudden_drop_selected="
            f"{sudden_drop_selected}; "
            "threshold_selected="
            f"{threshold_selected}; "
            "direct_marker_genes="
            f"{result.summary.direct_marker_supported_genes}; "
            "follow_up_genes="
            f"{result.summary.follow_up_gene_count}; "
            "human_review_required="
            f"{result.summary.human_review_required}"
        )

    if paths:
        for label, path in paths.items():
            print(
                f"{label}: {path}"
            )
    else:
        print(
            "Reports written: no"
        )


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run the all-cluster command-line workflow."""

    parser = build_parser()

    namespace = parser.parse_args(
        argv
    )

    args = RunAllClustersArgs(
        marker_file=cast(
            Path | None,
            namespace.marker_file,
        ),
        panel_catalog=cast(
            Path,
            namespace.panel_catalog,
        ),
        output_root=cast(
            Path,
            namespace.output_root,
        ),
        species=cast(
            str,
            namespace.species,
        ),
        tissue=cast(
            str,
            namespace.tissue,
        ),
        platform=cast(
            str,
            namespace.platform,
        ),
        top_k=cast(
            int,
            namespace.top_k,
        ),
        hpa_enabled=not cast(
            bool,
            namespace.no_hpa,
        ),
        write_reports=not cast(
            bool,
            namespace.no_write_reports,
        ),
    )

    try:
        results, paths = run(
            args
        )

        _print_results(
            results,
            paths,
        )

    except (
        FileNotFoundError,
        ValueError,
    ) as error:
        parser.error(
            str(error)
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )

