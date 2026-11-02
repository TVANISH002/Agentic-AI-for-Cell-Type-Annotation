from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from marker_interpretation_agent_v2.agent import (
    MarkerInterpretationAgent,
)
from marker_interpretation_agent_v2.config import (
    DEFAULT_CONFIG,
)
from marker_interpretation_agent_v2.panel.gene_panel_loader import (
    GenePanelLoader,
)
from marker_interpretation_agent_v2.panel.panel_evidence_loader import (
    PanelEvidenceLoader,
)
from marker_interpretation_agent_v2.paths import (
    PANEL_CATALOG_JSON,
    get_marker_file,
    normalize_path,
)
from marker_interpretation_agent_v2.schemas import (
    MarkerEvidence,
)


@dataclass(frozen=True, slots=True)
class RunSingleClusterArgs:
    """Typed command-line arguments for one-cluster execution."""

    cluster_id: str
    marker_file: Path | None
    panel_catalog: Path
    output_dir: Path | None
    top_k: int
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


def build_parser() -> argparse.ArgumentParser:
    """Create the single-cluster command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Run Marker Interpretation Agent V2 for one "
            "cluster using the reusable panel evidence catalog."
        )
    )

    parser.add_argument(
        "--cluster-id",
        required=True,
        help="Cluster identifier to interpret, for example 0.",
    )

    parser.add_argument(
        "--marker-file",
        type=Path,
        default=None,
        help=(
            "Cluster marker CSV. When omitted, the configured "
            "project marker file is used."
        ),
    )

    parser.add_argument(
        "--panel-catalog",
        type=Path,
        default=PANEL_CATALOG_JSON,
        help=(
            "Reusable panel evidence catalog JSON. Default: "
            f"{PANEL_CATALOG_JSON}"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. When omitted, reports are "
            "written to the configured cluster output directory."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_CONFIG.top_k,
        help=(
            "Maximum number of ranked marker genes to interpret. "
            f"Default: {DEFAULT_CONFIG.top_k}"
        ),
    )

    parser.add_argument(
        "--no-write-reports",
        action="store_true",
        help="Run validation and interpretation without writing files.",
    )

    return parser


def run(
    args: RunSingleClusterArgs,
) -> tuple[MarkerEvidence, dict[str, Path]]:
    """Load one cluster, run Agent 2 V2, and optionally write reports."""

    cluster_id = _normalize_cluster_id(
        args.cluster_id
    )

    if args.top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    marker_source = (
        normalize_path(args.marker_file)
        if args.marker_file is not None
        else get_marker_file()
    )

    catalog_source = normalize_path(
        args.panel_catalog
    )

    if not marker_source.exists():
        raise FileNotFoundError(
            f"Marker CSV does not exist: {marker_source}"
        )

    if not catalog_source.exists():
        raise FileNotFoundError(
            "Panel evidence catalog does not exist: "
            f"{catalog_source}. Build the real panel catalog first."
        )

    panel_loader = GenePanelLoader(
        marker_source
    )

    available_cluster_ids = {
        _normalize_cluster_id(value)
        for value in panel_loader.get_cluster_ids()
    }

    if cluster_id not in available_cluster_ids:
        raise ValueError(
            f"Cluster {cluster_id!r} was not found in the marker CSV. "
            f"Available clusters: {sorted(available_cluster_ids)}"
        )

    input_genes = panel_loader.get_cluster_markers(
        cluster_id
    )

    if not input_genes:
        raise ValueError(
            f"Cluster {cluster_id!r} contains no marker genes."
        )

    catalog = PanelEvidenceLoader().load_catalog(
        catalog_source
    )

    config = replace(
        DEFAULT_CONFIG,
        top_k=args.top_k,
    )

    agent = MarkerInterpretationAgent(
        config
    )

    return agent.run(
        cluster_id=cluster_id,
        input_genes=input_genes,
        catalog=catalog,
        output_dir=args.output_dir,
        write_reports=args.write_reports,
    )


def _print_result(
    result: MarkerEvidence,
    paths: dict[str, Path],
) -> None:
    """Print a concise professor-facing execution summary."""

    print(
        "Marker Interpretation Agent V2 completed successfully."
    )

    print(
        f"Cluster: {result.cluster_id}"
    )

    print(
        f"Input genes: {result.summary.total_input_genes}"
    )

    print(
        "Direct marker-supported genes: "
        f"{result.summary.direct_marker_supported_genes}"
    )

    print(
        f"HPA queried genes: {result.summary.hpa_queried_genes}"
    )

    print(
        "Ranked candidate cell types: "
        f"{result.summary.ranked_candidate_count}"
    )

    print(
        "Selected candidate cell types: "
        f"{result.candidate_selection.selected_candidate_cell_types}"
    )

    print(
        f"Leading candidate: {result.summary.leading_candidate}"
    )

    print(
        f"Follow-up genes: {result.summary.follow_up_gene_count}"
    )

    print(
        "Human review required: "
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
    """Run the single-cluster command-line workflow."""

    parser = build_parser()
    namespace = parser.parse_args(argv)

    args = RunSingleClusterArgs(
        cluster_id=cast(
            str,
            namespace.cluster_id,
        ),
        marker_file=cast(
            Path | None,
            namespace.marker_file,
        ),
        panel_catalog=cast(
            Path,
            namespace.panel_catalog,
        ),
        output_dir=cast(
            Path | None,
            namespace.output_dir,
        ),
        top_k=cast(
            int,
            namespace.top_k,
        ),
        write_reports=not cast(
            bool,
            namespace.no_write_reports,
        ),
    )

    try:
        result, paths = run(args)
        _print_result(result, paths)

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