from __future__ import annotations

import argparse
from pathlib import Path

from .hpa_report_writer import write_hpa_reports
from .hpa_validation import (
    HPAConfig,
    build_candidates,
    load_csv_rows,
    load_processed_markers,
    scan_hpa,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_hpa_validation(
    *,
    marker_csv: str | Path,
    annotation_csv: str | Path,
    hpa_zip: str | Path,
    detail_output_dir: str | Path,
    final_output_dir: str | Path,
    reference_csv: str | Path | None = None,
    config: HPAConfig = HPAConfig(),
) -> list[Path]:
    """
    Validate existing lung annotations using HPA.

    CellMarker/PanglaoDB annotations are preserved.
    HPA evidence is added independently and does not
    overwrite the existing annotation.
    """

    config.validate()

    print(
        "Loading the exact processed marker genes "
        "used by Agent 2..."
    )

    markers_by_cluster = load_processed_markers(
        marker_csv
    )

    annotation_rows = load_csv_rows(
        annotation_csv,
        {
            "cluster_id",
            "leading_candidate",
        },
        "CellMarker/PanglaoDB summary CSV",
    )

    reference_rows = (
        load_csv_rows(
            reference_csv,
            {
                "cluster",
                "cell_type",
                "subtype_or_state",
                "confidence",
            },
            "Reference annotation CSV",
        )
        if reference_csv is not None
        else None
    )

    marker_clusters = set(markers_by_cluster)

    annotation_clusters = {
        (row.get("cluster_id") or "").strip()
        for row in annotation_rows
    }

    if marker_clusters != annotation_clusters:
        raise ValueError(
            "Processed-marker and annotation cluster IDs "
            "do not match. "
            f"Only in markers: "
            f"{sorted(marker_clusters - annotation_clusters)}; "
            f"only in annotations: "
            f"{sorted(annotation_clusters - marker_clusters)}."
        )

    if reference_rows is not None:
        reference_clusters = {
            (row.get("cluster") or "").strip()
            for row in reference_rows
        }

        if marker_clusters != reference_clusters:
            raise ValueError(
                "Processed-marker and reference cluster IDs "
                "do not match. "
                f"Only in markers: "
                f"{sorted(marker_clusters - reference_clusters)}; "
                f"only in reference: "
                f"{sorted(reference_clusters - marker_clusters)}."
            )

    selected_genes = {
        marker.gene
        for markers in markers_by_cluster.values()
        for marker in markers
    }

    print(
        f"Clusters: {len(markers_by_cluster)} | "
        f"unique selected genes: {len(selected_genes)}"
    )

    print(
        "Streaming the HPA v25.1 ZIP for "
        "healthy-lung expression..."
    )

    scan = scan_hpa(
        hpa_zip,
        selected_genes,
        config,
    )

    print(
        f"HPA rows scanned: {scan.scanned_row_count:,} | "
        "selected genes with positive lung evidence: "
        f"{len(scan.evidence_by_gene)}"
    )

    candidates, gene_rows = build_candidates(
        markers_by_cluster,
        scan,
        config,
    )

    paths = write_hpa_reports(
        detail_output_dir=detail_output_dir,
        final_output_dir=final_output_dir,
        annotation_rows=annotation_rows,
        markers_by_cluster=markers_by_cluster,
        candidates=candidates,
        gene_rows=gene_rows,
        scan=scan,
        config=config,
        reference_rows=reference_rows,
    )

    print(
        "HPA validation completed. Existing "
        "CellMarker/PanglaoDB predictions were preserved."
    )

    print("Created output files:")

    for path in paths:
        print(path)

    return paths


def _parser() -> argparse.ArgumentParser:
    lung_root = (
        PROJECT_ROOT
        / "data"
        / "lung_run"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Add independent HPA healthy-lung expression "
            "validation to the existing "
            "CellMarker/PanglaoDB lung annotations."
        )
    )

    parser.add_argument(
        "--markers",
        type=Path,
        default=(
            lung_root
            / "processed"
            / "lung_cluster_markers_pipeline.csv"
        ),
        help=(
            "Processed lung marker CSV containing the exact "
            "marker genes and ranks used by Agent 2."
        ),
    )

    parser.add_argument(
        "--annotations",
        type=Path,
        default=(
            lung_root
            / "final"
            / "lung_cluster_annotations_pipeline.csv"
        ),
        help=(
            "Existing CellMarker/PanglaoDB annotation summary."
        ),
    )

    parser.add_argument(
        "--hpa",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "reference"
            / "rna_single_cell_cluster_hpa_v25_1.tsv.zip"
        ),
        help="HPA v25.1 single-cell RNA expression ZIP.",
    )

    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help=(
            "Optional professor/reference annotation CSV. "
            "It is used only for comparison after HPA ranking "
            "and does not influence candidate generation."
        ),
    )

    parser.add_argument(
        "--detail-output",
        type=Path,
        default=(
            lung_root
            / "hpa_output"
        ),
        help="Directory for detailed HPA evidence files.",
    )

    parser.add_argument(
        "--final-output",
        type=Path,
        default=(
            lung_root
            / "final"
        ),
        help="Directory for final professor-facing summaries.",
    )

    parser.add_argument(
        "--minimum-supporting-genes",
        type=int,
        default=2,
        help=(
            "Minimum number of supporting genes needed to "
            "select an HPA candidate for downstream review."
        ),
    )

    return parser


def main() -> int:
    args = _parser().parse_args()

    run_hpa_validation(
        marker_csv=args.markers,
        annotation_csv=args.annotations,
        hpa_zip=args.hpa,
        detail_output_dir=args.detail_output,
        final_output_dir=args.final_output,
        reference_csv=args.reference,
        config=HPAConfig(
            minimum_supporting_genes=(
                args.minimum_supporting_genes
            )
        ),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())