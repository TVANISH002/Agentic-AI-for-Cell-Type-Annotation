from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from .hpa_validation import (
    HPAConfig,
    HPACandidate,
    HPAScanResult,
    MarkerGene,
    broad_lineage,
    label_key,
)


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
    columns: list[str] | None = None,
) -> None:
    """
    Write rows to a CSV file.

    The columns argument is required when rows may be empty,
    such as the selected-candidates report.
    """

    if not rows and columns is None:
        raise ValueError(
            f"Cannot write an empty CSV without column names: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = columns or list(rows[0].keys())

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="raise",
        )

        writer.writeheader()
        writer.writerows(rows)


def _candidate_row(
    candidate: HPACandidate,
    config: HPAConfig,
) -> dict[str, object]:
    """
    Convert one HPA candidate object into a CSV row.
    """

    supporting_gene_count = len(
        candidate.supporting_genes
    )

    return {
        "cluster_id": candidate.cluster_id,
        "candidate_rank": candidate.rank,
        "hpa_cell_type": candidate.cell_type,
        "hpa_broad_lineage": candidate.broad_lineage,
        "selected_for_downstream": (
            candidate.selected_for_downstream
        ),
        "minimum_supporting_genes": (
            config.minimum_supporting_genes
        ),
        "unique_supporting_gene_count": (
            supporting_gene_count
        ),
        "supporting_genes": " | ".join(
            candidate.supporting_genes
        ),
        "specificity_weighted_support": round(
            candidate.specificity_weighted_support,
            9,
        ),
        "mean_supporting_gene_rank": round(
            candidate.mean_marker_rank,
            6,
        ),
        "mean_supporting_log_fold_change": round(
            candidate.mean_log_fold_change,
            9,
        ),
        "mean_hpa_top_ncpm": round(
            candidate.mean_top_ncpm,
            6,
        ),
        "max_hpa_top_ncpm": round(
            candidate.max_top_ncpm,
            6,
        ),
        "evidence_source": config.source_name,
        "source_release": config.source_release,
        "evidence_scope": config.evidence_scope,
        "warnings": (
            "Candidate is supported by only one marker gene "
            "whose highest healthy-lung HPA expression occurs "
            "in this cell type."
            if supporting_gene_count == 1
            else ""
        ),
    }


def write_hpa_reports(
    *,
    detail_output_dir: str | Path,
    final_output_dir: str | Path,
    annotation_rows: list[dict[str, str]],
    markers_by_cluster: dict[
        str,
        list[MarkerGene],
    ],
    candidates: list[HPACandidate],
    gene_rows: list[dict[str, object]],
    scan: HPAScanResult,
    config: HPAConfig,
    reference_rows: list[dict[str, str]] | None = None,
) -> list[Path]:
    """
    Write detailed HPA evidence and final summary reports.

    Existing CellMarker/PanglaoDB annotation columns are
    preserved. HPA evidence is appended separately and does
    not overwrite the existing annotation.
    """

    detail_dir = Path(
        detail_output_dir
    )

    final_dir = Path(
        final_output_dir
    )

    detail_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not annotation_rows:
        raise ValueError(
            "No CellMarker/PanglaoDB annotation rows "
            "were provided."
        )

    if not markers_by_cluster:
        raise ValueError(
            "No processed marker genes were provided."
        )

    if not candidates:
        raise ValueError(
            "No HPA candidates were generated."
        )

    candidate_rows = [
        _candidate_row(
            candidate,
            config,
        )
        for candidate in candidates
    ]

    candidate_columns = list(
        candidate_rows[0].keys()
    )

  
    # 1. All ranked HPA candidates
    

    ranked_path = (
        detail_dir
        / "lung_hpa_ranked_candidate_celltypes.csv"
    )

    _write_csv(
        ranked_path,
        candidate_rows,
        candidate_columns,
    )

   
    # 2. Only selected HPA candidates
   

    selected_rows = [
        row
        for row in candidate_rows
        if row["selected_for_downstream"] is True
    ]

    selected_path = (
        detail_dir
        / "lung_hpa_selected_candidate_celltypes.csv"
    )

    _write_csv(
        selected_path,
        selected_rows,
        candidate_columns,
    )

  
    # 3. Gene-level HPA evidence
   

    gene_path = (
        detail_dir
        / "lung_hpa_gene_evidence.csv"
    )

    _write_csv(
        gene_path,
        gene_rows,
    )

    # Group candidates by input cluster.
    candidates_by_cluster: dict[
        str,
        list[HPACandidate],
    ] = defaultdict(list)

    for candidate in candidates:
        candidates_by_cluster[
            candidate.cluster_id
        ].append(candidate)

    # Ensure candidates are ordered correctly in every cluster.
    for cluster_candidates in candidates_by_cluster.values():
        cluster_candidates.sort(
            key=lambda candidate: candidate.rank
        )

    # Reference annotations are optional and used only
    # after HPA candidate generation and ranking.
    reference_by_cluster = {
        (row.get("cluster") or "").strip(): row
        for row in (reference_rows or [])
    }

    consolidated_rows: list[
        dict[str, object]
    ] = []

    
    # 4. Combine existing results with separate HPA columns
   

    for source_row in annotation_rows:
        cluster_id = (
            source_row.get("cluster_id") or ""
        ).strip()

        if not cluster_id:
            raise ValueError(
                "An annotation row contains an empty cluster_id."
            )

        if cluster_id not in markers_by_cluster:
            raise ValueError(
                f"Cluster {cluster_id} exists in the annotation "
                "CSV but not in the processed marker data."
            )

        cluster_candidates = (
            candidates_by_cluster.get(
                cluster_id,
                [],
            )
        )

        if not cluster_candidates:
            raise ValueError(
                "No HPA candidates were generated for "
                f"cluster {cluster_id}."
            )

        top_candidate = cluster_candidates[0]

        selected_candidates = [
            candidate
            for candidate in cluster_candidates
            if candidate.selected_for_downstream
        ]

        existing_leading_candidate = (
            source_row.get("leading_candidate") or ""
        ).strip()

        if not existing_leading_candidate:
            raise ValueError(
                "The existing leading_candidate is empty "
                f"for cluster {cluster_id}."
            )

        existing_broad_lineage = broad_lineage(
            existing_leading_candidate
        )

        lineage_agreement = (
            label_key(existing_broad_lineage)
            == label_key(
                top_candidate.broad_lineage
            )
        )

        positive_expression_gene_count = sum(
            1
            for row in gene_rows
            if (
                str(
                    row.get(
                        "cluster_id",
                        "",
                    )
                ).strip()
                == cluster_id
                and row.get("hpa_status")
                == "hpa_lung_expression_found"
            )
        )

        if lineage_agreement:
            review_note = (
                "HPA agrees with the existing prediction at "
                "the broad-lineage level. HPA provides "
                "healthy-lung expression evidence and does "
                "not replace the CellMarker/PanglaoDB "
                "annotation."
            )
        else:
            review_note = (
                "HPA's top expression-derived lineage differs "
                "from the existing prediction. Review the "
                "supporting genes in the detailed reports. "
                "HPA must not automatically override the "
                "CellMarker/PanglaoDB annotation."
            )

        # Copy every original annotation column and value.
        consolidated: dict[str, object] = dict(
            source_row
        )

        # Append independent HPA evidence.
        consolidated.update(
            {
                "hpa_top_cell_type": (
                    top_candidate.cell_type
                ),
                "hpa_top_broad_lineage": (
                    top_candidate.broad_lineage
                ),
                "hpa_top_supporting_gene_count": (
                    len(
                        top_candidate.supporting_genes
                    )
                ),
                "hpa_top_supporting_genes": (
                    " | ".join(
                        top_candidate.supporting_genes
                    )
                ),
                "hpa_selected_candidate_cell_types": (
                    " | ".join(
                        candidate.cell_type
                        for candidate
                        in selected_candidates
                    )
                ),
                "hpa_queried_marker_gene_count": (
                    len(
                        markers_by_cluster[
                            cluster_id
                        ]
                    )
                ),
                "hpa_positive_lung_expression_gene_count": (
                    positive_expression_gene_count
                ),
                "hpa_agrees_with_existing_broad_lineage": (
                    lineage_agreement
                ),
                "hpa_evidence_scope": (
                    config.evidence_scope
                ),
                "hpa_review_note": review_note,
            }
        )

       
        # Optional comparison with professor/reference label
      

        reference = reference_by_cluster.get(
            cluster_id
        )

        if reference is not None:
            reference_cell_type = (
                reference.get("cell_type") or ""
            ).strip()

            reference_subtype = (
                reference.get(
                    "subtype_or_state"
                )
                or ""
            ).strip()

            reference_confidence = (
                reference.get("confidence")
                or ""
            ).strip()

            exact_reference_candidate = next(
                (
                    candidate
                    for candidate
                    in cluster_candidates
                    if (
                        label_key(
                            candidate.cell_type
                        )
                        == label_key(
                            reference_cell_type
                        )
                    )
                ),
                None,
            )

            reference_broad_lineage = broad_lineage(
                reference_cell_type
            )

            top_matches_reference_lineage = (
                label_key(
                    top_candidate.broad_lineage
                )
                == label_key(
                    reference_broad_lineage
                )
            )

            if (
                exact_reference_candidate is not None
                and exact_reference_candidate.rank == 1
            ):
                comparison_note = (
                    "HPA's top cell type matches the "
                    "reference label."
                )

            elif exact_reference_candidate is not None:
                comparison_note = (
                    "The reference label is present among "
                    "the HPA candidates at rank "
                    f"{exact_reference_candidate.rank}, "
                    "supported by "
                    f"{len(exact_reference_candidate.supporting_genes)} "
                    "marker genes."
                )

            elif top_matches_reference_lineage:
                comparison_note = (
                    "HPA matches the reference at the "
                    "broad-lineage level but not at the "
                    "exact cell-type label level."
                )

            else:
                comparison_note = (
                    "HPA's top candidate does not match the "
                    "reference broad lineage. Review label "
                    "granularity and the detailed gene-level "
                    "evidence."
                )

            consolidated.update(
                {
                    "reference_cell_type": (
                        reference_cell_type
                    ),
                    "reference_subtype_or_state": (
                        reference_subtype
                    ),
                    "reference_confidence": (
                        reference_confidence
                    ),
                    "hpa_top_matches_reference_broad_lineage": (
                        top_matches_reference_lineage
                    ),
                    "hpa_reference_candidate_rank": (
                        exact_reference_candidate.rank
                        if exact_reference_candidate
                        is not None
                        else ""
                    ),
                    "hpa_reference_candidate_supporting_gene_count": (
                        len(
                            exact_reference_candidate
                            .supporting_genes
                        )
                        if exact_reference_candidate
                        is not None
                        else 0
                    ),
                    "hpa_reference_candidate_supporting_genes": (
                        " | ".join(
                            exact_reference_candidate
                            .supporting_genes
                        )
                        if exact_reference_candidate
                        is not None
                        else ""
                    ),
                    "reference_comparison_note": (
                        comparison_note
                    ),
                }
            )

        consolidated_rows.append(
            consolidated
        )

        # -------------------------------------------------
        # Per-cluster ranked candidate report
        # -------------------------------------------------

        cluster_path = (
            detail_dir
            / "clusters"
            / (
                f"cluster_{cluster_id}_"
                "hpa_ranked_candidate_celltypes.csv"
            )
        )

        cluster_candidate_rows = [
            _candidate_row(
                candidate,
                config,
            )
            for candidate in cluster_candidates
        ]

        _write_csv(
            cluster_path,
            cluster_candidate_rows,
            candidate_columns,
        )


    # 5. Full existing annotation plus HPA evidence
  

    consolidated_path = (
        final_dir
        / "lung_cluster_annotations_with_hpa.csv"
    )

    _write_csv(
        consolidated_path,
        consolidated_rows,
    )

    # 6. Concise professor-facing HPA summary
  

    professor_rows: list[
        dict[str, object]
    ] = []

    for row in consolidated_rows:
        professor_rows.append(
            {
                "cluster_id": (
                    row["cluster_id"]
                ),
                "existing_leading_candidate": (
                    row["leading_candidate"]
                ),
                "hpa_top_cell_type": (
                    row["hpa_top_cell_type"]
                ),
                "hpa_top_broad_lineage": (
                    row["hpa_top_broad_lineage"]
                ),
                "hpa_top_supporting_gene_count": (
                    row[
                        "hpa_top_supporting_gene_count"
                    ]
                ),
                "hpa_top_supporting_genes": (
                    row[
                        "hpa_top_supporting_genes"
                    ]
                ),
                "hpa_agrees_with_existing_broad_lineage": (
                    row[
                        "hpa_agrees_with_existing_broad_lineage"
                    ]
                ),
                "reference_cell_type": (
                    row.get(
                        "reference_cell_type",
                        "",
                    )
                ),
                "reference_subtype_or_state": (
                    row.get(
                        "reference_subtype_or_state",
                        "",
                    )
                ),
                "hpa_reference_candidate_rank": (
                    row.get(
                        "hpa_reference_candidate_rank",
                        "",
                    )
                ),
                "hpa_reference_candidate_supporting_gene_count": (
                    row.get(
                        "hpa_reference_candidate_supporting_gene_count",
                        "",
                    )
                ),
                "hpa_reference_candidate_supporting_genes": (
                    row.get(
                        "hpa_reference_candidate_supporting_genes",
                        "",
                    )
                ),
                "interpretation": (
                    row.get(
                        "reference_comparison_note"
                    )
                    or row["hpa_review_note"]
                ),
            }
        )

    professor_path = (
        final_dir
        / "lung_hpa_validation_summary.csv"
    )

    _write_csv(
        professor_path,
        professor_rows,
    )

    
    # 7. Reproducibility metadata
 

    metadata = {
        "schema_name": "LungHPAValidationRun",
        "schema_version": "1.0",
        "query_tissue": config.query_tissue,
        "source": config.source_name,
        "source_release": config.source_release,
        "evidence_scope": config.evidence_scope,
        "method": (
            "For each input marker gene, select the "
            "healthy-lung HPA cell type with the highest "
            "nCPM. Aggregate these gene-level assignments "
            "into ranked candidates for each input cluster."
        ),
        "candidate_ranking": [
            "unique supporting gene count descending",
            "specificity-weighted support descending",
            "mean existing marker rank ascending",
            "mean marker log-fold change descending",
            "cell-type label alphabetical as final tie-breaker",
        ],
        "selection_rule": (
            "selected_for_downstream when "
            "unique_supporting_gene_count >= "
            f"{config.minimum_supporting_genes}"
        ),
        "cluster_count": len(
            markers_by_cluster
        ),
        "unique_selected_gene_count": len(
            {
                marker.gene
                for markers
                in markers_by_cluster.values()
                for marker in markers
            }
        ),
        "reference_annotations_used_for_comparison_only": (
            bool(reference_rows)
        ),
        "hpa_archive_member": (
            scan.archive_member
        ),
        "hpa_source_sha256": (
            scan.source_sha256
        ),
        "hpa_rows_scanned": (
            scan.scanned_row_count
        ),
        "matched_selected_gene_rows_all_tissues": (
            scan.matched_selected_gene_row_count
        ),
        "positive_selected_gene_lung_rows": (
            scan.positive_lung_row_count
        ),
        "warnings": [
            (
                "HPA is an independent healthy-lung "
                "RNA-expression reference, not a curated "
                "cell-marker database."
            ),
            (
                "HPA Cluster values refer to HPA reference "
                "clusters and are not the input clusters 0-11."
            ),
            (
                "HPA evidence does not overwrite the existing "
                "CellMarker/PanglaoDB annotation."
            ),
            (
                "Optional reference annotations are joined "
                "only after HPA ranking and do not influence "
                "candidate generation."
            ),
        ],
    }

    metadata_path = (
        detail_dir
        / "lung_hpa_run_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    return [
        professor_path,
        consolidated_path,
        ranked_path,
        selected_path,
        gene_path,
        metadata_path,
    ]