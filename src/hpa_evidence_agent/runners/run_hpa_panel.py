from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

from hpa_evidence_agent.cluster_projector import (
    project_hpa_to_clusters,
)
from hpa_evidence_agent.config import (
    DEFAULT_CONFIG,
    HPAEvidenceConfig,
)
from hpa_evidence_agent.evidence_builder import (
    build_hpa_evidence,
)
from hpa_evidence_agent.hpa_source_loader import (
    load_hpa_source,
)
from hpa_evidence_agent.panel_loader import (
    load_panel_genes,
)
from hpa_evidence_agent.report_writer import (
    write_hpa_reports,
)


def _format_seconds(
    seconds: float,
) -> str:
    """Display runtime in an easy-to-read format."""

    if seconds < 60.0:
        return f"{seconds:.2f} seconds"

    minutes = int(seconds // 60.0)
    remaining_seconds = (
        seconds
        - minutes * 60.0
    )

    return (
        f"{minutes} minute(s) "
        f"{remaining_seconds:.2f} seconds"
    )


def _print_heading(
    text: str,
) -> None:
    """Print a clear console section heading."""

    print()
    print("=" * 72)
    print(text)
    print("=" * 72)


def _validate_input_file(
    path: Path,
    *,
    description: str,
) -> None:
    """Stop early when a required input file is unavailable."""

    if not path.exists():
        raise FileNotFoundError(
            f"{description} was not found: "
            f"{path.resolve()}"
        )

    if not path.is_file():
        raise ValueError(
            f"{description} path is not a file: "
            f"{path.resolve()}"
        )

    if path.stat().st_size <= 0:
        raise ValueError(
            f"{description} is empty: "
            f"{path.resolve()}"
        )


def _validate_inputs(
    config: HPAEvidenceConfig,
) -> None:
    """Check all required files before scanning the large HPA source."""

    _validate_input_file(
        config.panel_gene_list_path,
        description="Panel gene-list CSV",
    )

    _validate_input_file(
        config.hpa_source_path,
        description="HPA source archive",
    )

    _validate_input_file(
        config.cluster_marker_path,
        description="Cluster marker CSV",
    )


def run(
    config: HPAEvidenceConfig = DEFAULT_CONFIG,
) -> list[Path]:
    """Run the complete independent HPA evidence workflow."""

    total_start = perf_counter()

    _print_heading(
        "Independent HPA Evidence Agent"
    )

    print(
        "Query tissue:",
        config.query_tissue,
    )

    print(
        "HPA release:",
        config.source_release,
    )

    print(
        "Panel gene file:",
        config.panel_gene_list_path,
    )

    print(
        "HPA source file:",
        config.hpa_source_path,
    )

    print(
        "Cluster marker file:",
        config.cluster_marker_path,
    )

    print(
        "Output directory:",
        config.output_dir,
    )

    print()
    print(
        "HPA evidence will remain independent "
        "from CellMarker and PanglaoDB."
    )

    config.validate()
    _validate_inputs(config)

 
    # Step 1: Load all canonical panel genes

    step_start = perf_counter()

    _print_heading(
        "Step 1 of 5: Loading panel genes"
    )

    selected_genes = load_panel_genes(
        config
    )

    panel_load_seconds = (
        perf_counter()
        - step_start
    )

    print(
        "Panel genes loaded:",
        len(selected_genes),
    )

    print(
        "First 10 genes:",
        " | ".join(
            selected_genes[:10]
        ),
    )

    print(
        "Runtime:",
        _format_seconds(
            panel_load_seconds
        ),
    )

  
    # Step 2: Scan the complete HPA source

    step_start = perf_counter()

    _print_heading(
        "Step 2 of 5: Scanning HPA source"
    )

    print(
        "This is the longest step because "
        "the complete HPA archive is scanned."
    )

    source_result = load_hpa_source(
        selected_genes,
        config,
    )

    hpa_scan_seconds = (
        perf_counter()
        - step_start
    )

    print(
        "Requested genes:",
        len(
            source_result
            .selected_genes
        ),
    )

    print(
        "Genes present in HPA:",
        len(
            source_result
            .genes_present_in_hpa
        ),
    )

    print(
        "Positive breast records:",
        len(
            source_result
            .breast_expression_records
        ),
    )

    print(
        "Scanned HPA rows:",
        source_result
        .scanned_row_count,
    )

    print(
        "Matched panel rows across all tissues:",
        source_result
        .matched_panel_row_count,
    )

    print(
        "Archive member:",
        source_result
        .archive_member_name,
    )

    print(
        "Runtime:",
        _format_seconds(
            hpa_scan_seconds
        ),
    )

 
    # Step 3: Build gene-level evidence

    step_start = perf_counter()

    _print_heading(
        "Step 3 of 5: Building gene-level HPA evidence"
    )

    panel_evidence = build_hpa_evidence(
        source_result,
        config,
    )

    evidence_build_seconds = (
        perf_counter()
        - step_start
    )

    print(
        "Requested genes:",
        panel_evidence
        .summary
        .requested_gene_count,
    )

    print(
        "Breast expression found:",
        panel_evidence
        .summary
        .breast_expression_found_count,
    )

    print(
        "Gene found without breast expression:",
        panel_evidence
        .summary
        .gene_found_without_breast_expression_count,
    )

    print(
        "Exact symbol not found in HPA:",
        panel_evidence
        .summary
        .gene_not_found_in_hpa_count,
    )

    print(
        "Preserved expression records:",
        panel_evidence
        .summary
        .expression_record_count,
    )

    missing_genes = [
        summary.gene
        for summary
        in panel_evidence.gene_summaries
        if (
            summary.source_status
            == "gene_not_found_in_hpa"
        )
    ]

    if missing_genes:
        print(
            "Symbols requiring review:",
            " | ".join(
                missing_genes
            ),
        )

    print(
        "Runtime:",
        _format_seconds(
            evidence_build_seconds
        ),
    )


    # Step 4: Project HPA evidence to clusters

    step_start = perf_counter()

    _print_heading(
        "Step 4 of 5: Projecting HPA evidence to clusters"
    )

    cluster_evidence = (
        project_hpa_to_clusters(
            panel_evidence,
            config,
        )
    )

    cluster_projection_seconds = (
        perf_counter()
        - step_start
    )

    print(
        "Clusters projected:",
        len(cluster_evidence),
    )

    for cluster in cluster_evidence:
        print(
            "Cluster "
            f"{cluster.cluster_id}: "
            f"{cluster.marker_gene_count} marker genes, "
            f"{cluster.breast_expression_found_count} "
            "with breast expression, "
            f"{cluster.gene_not_found_in_hpa_count} "
            "not found in HPA"
        )

    print(
        "Runtime:",
        _format_seconds(
            cluster_projection_seconds
        ),
    )

    # Step 5: Write all reports

    step_start = perf_counter()

    _print_heading(
        "Step 5 of 5: Writing HPA reports"
    )

    written_paths = write_hpa_reports(
        panel_evidence=panel_evidence,
        cluster_evidence=(
            cluster_evidence
        ),
        source_result=source_result,
        config=config,
    )

    report_write_seconds = (
        perf_counter()
        - step_start
    )

    for path in written_paths:
        print(
            "Written:",
            path.resolve(),
        )

    print(
        "Runtime:",
        _format_seconds(
            report_write_seconds
        ),
    )

    total_seconds = (
        perf_counter()
        - total_start
    )

    _print_heading(
        "Independent HPA workflow completed successfully"
    )

    print(
        "Panel gene loading:",
        _format_seconds(
            panel_load_seconds
        ),
    )

    print(
        "HPA source scanning:",
        _format_seconds(
            hpa_scan_seconds
        ),
    )

    print(
        "Evidence construction:",
        _format_seconds(
            evidence_build_seconds
        ),
    )

    print(
        "Cluster projection:",
        _format_seconds(
            cluster_projection_seconds
        ),
    )

    print(
        "Report writing:",
        _format_seconds(
            report_write_seconds
        ),
    )

    print(
        "Total runtime:",
        _format_seconds(
            total_seconds
        ),
    )

    print(
        "Files written:",
        len(written_paths),
    )

    print(
        "Output directory:",
        config.output_dir.resolve(),
    )

    print()
    print(
        "Important: these reports contain independent "
        "healthy-breast HPA expression evidence."
    )

    print(
        "They are not final annotations and are not "
        "combined marker-support votes."
    )

    return written_paths


def main() -> int:
    """Run the workflow and return a terminal status code."""

    try:
        run()
    except (
        FileNotFoundError,
        ValueError,
        OSError,
    ) as error:
        print()
        print(
            "HPA workflow failed:",
            error,
            file=sys.stderr,
        )

        return 1

    except Exception as error:
        print()
        print(
            "Unexpected HPA workflow failure:",
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )