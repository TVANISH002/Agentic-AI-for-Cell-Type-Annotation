from __future__ import annotations

import argparse
import csv
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from marker_interpretation_agent_v2.panel.panel_evidence_builder import (
    PanelEvidenceBuilder,
)
from marker_interpretation_agent_v2.paths import (
    PANEL_OUTPUT_DIR,
    normalize_path,
)


DEFAULT_PANEL_GENE_LIST = PANEL_OUTPUT_DIR / "panel_gene_list.csv"


@dataclass(frozen=True, slots=True)
class BuildPanelCatalogArgs:
    """Typed command-line arguments for panel-catalog construction."""

    agent1_json: Path
    hpa_json: Path | None
    marker_only: bool
    panel_gene_list: Path
    output_json: Path | None
    output_csv: Path | None
    unresolved_csv: Path | None


def _normalize_gene(value: object) -> str:
    """Normalize one panel gene symbol."""

    gene = str(value).strip().upper()

    if not gene:
        raise ValueError("Panel gene symbol cannot be empty.")

    return gene


def _find_gene_column(fieldnames: Sequence[str] | None) -> str:
    """Find the gene-symbol column in the panel inventory CSV."""

    if not fieldnames:
        raise ValueError(
            "Panel gene inventory CSV has no header row."
        )

    normalized = {
        str(field).strip().casefold(): str(field)
        for field in fieldnames
        if str(field).strip()
    }

    for candidate in (
        "gene",
        "gene_symbol",
        "official_symbol",
    ):
        if candidate in normalized:
            return normalized[candidate]

    raise ValueError(
        "Panel gene inventory must contain a gene column. "
        f"Available columns: {list(fieldnames)}"
    )


def load_expected_genes(
    path: str | Path,
) -> list[str]:
    """Load unique expected genes from the panel inventory CSV."""

    source = normalize_path(path)

    if not source.exists():
        raise FileNotFoundError(
            f"Panel gene inventory does not exist: {source}"
        )

    with source.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        gene_column = _find_gene_column(
            reader.fieldnames
        )

        genes = [
            _normalize_gene(
                row.get(gene_column, "")
            )
            for row in reader
        ]

    if not genes:
        raise ValueError(
            "Panel gene inventory contains no genes: "
            f"{source}"
        )

    duplicates = sorted(
        gene
        for gene, count
        in Counter(genes).items()
        if count > 1
    )

    if duplicates:
        raise ValueError(
            "Panel gene inventory contains duplicate genes: "
            f"{duplicates}"
        )

    return genes


def build_parser() -> argparse.ArgumentParser:
    """Create the panel-catalog command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Build the reusable Agent 2 V2 panel evidence "
            "catalog from panel-wide Agent 1 and optional HPA JSON files."
        )
    )

    parser.add_argument(
        "--agent1-json",
        required=True,
        type=Path,
        help="Panel-wide Agent 1 evidence JSON.",
    )

    parser.add_argument(
        "--hpa-json",
        required=False,
        default=None,
        type=Path,
        help=(
            "Panel-wide HPA healthy-breast evidence JSON. "
            "Omit when --marker-only is used."
        ),
    )

    parser.add_argument(
        "--panel-gene-list",
        type=Path,
        default=DEFAULT_PANEL_GENE_LIST,
        help=(
            "Panel gene inventory CSV. Default: "
            f"{DEFAULT_PANEL_GENE_LIST}"
        ),
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional catalog JSON output path.",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional flattened catalog CSV output path.",
    )

    parser.add_argument(
        "--unresolved-csv",
        type=Path,
        default=None,
        help="Optional unresolved-gene CSV output path.",
    )

    parser.add_argument(
        "--marker-only",
        action="store_true",
        help=(
            "Build a marker-only catalog and mark HPA "
            "as intentionally not queried."
        ),
    )

    return parser


def run(
    args: BuildPanelCatalogArgs,
) -> dict[str, Path]:
    """Build, validate, and write the panel evidence catalog."""

    expected_genes = load_expected_genes(
        args.panel_gene_list
    )

    if args.marker_only and args.hpa_json is not None:
        raise ValueError(
            "--hpa-json should be omitted when --marker-only is used."
        )

    if not args.marker_only and args.hpa_json is None:
        raise ValueError(
            "--hpa-json is required unless --marker-only is used."
        )

    builder = PanelEvidenceBuilder()

    catalog = builder.build_from_files(
        args.agent1_json,
        None if args.marker_only else args.hpa_json,
        expected_genes=expected_genes,
        require_hpa_for_all_genes=(
            not args.marker_only
        ),
    )

    paths = builder.write(
        catalog,
        json_path=args.output_json,
        csv_path=args.output_csv,
        unresolved_path=args.unresolved_csv,
        require_hpa_for_all_genes=(
            not args.marker_only
        ),
    )

    direct_marker_count = sum(
        record.direct_marker_supported
        for record in catalog.genes
    )

    hpa_queried_count = sum(
        record.hpa.queried
        for record in catalog.genes
    )

    hpa_gene_match_count = sum(
        record.hpa.gene_present_in_hpa
        for record in catalog.genes
    )

    hpa_breast_expression_count = sum(
        record.hpa.breast_expression_found
        for record in catalog.genes
    )

    print(
        "Panel evidence catalog built successfully."
    )

    print(
        f"Panel genes: {catalog.panel_gene_count}"
    )

    print(
        f"Direct marker genes: {direct_marker_count}"
    )

    print(
        f"HPA queried genes: {hpa_queried_count}"
    )

    print(
        f"HPA gene matches: {hpa_gene_match_count}"
    )

    print(
        "HPA breast-expression genes: "
        f"{hpa_breast_expression_count}"
    )

    print(
        f"Unresolved genes: "
        f"{len(catalog.unresolved_genes)}"
    )

    for label, path in paths.items():
        print(
            f"{label}: {path}"
        )

    return paths


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run the panel-catalog command-line workflow."""

    parser = build_parser()

    namespace = parser.parse_args(
        argv
    )

    args = BuildPanelCatalogArgs(
        agent1_json=cast(
            Path,
            namespace.agent1_json,
        ),
        hpa_json=cast(
            Path | None,
            namespace.hpa_json,
        ),
        marker_only=cast(
            bool,
            namespace.marker_only,
        ),
        panel_gene_list=cast(
            Path,
            namespace.panel_gene_list,
        ),
        output_json=cast(
            Path | None,
            namespace.output_json,
        ),
        output_csv=cast(
            Path | None,
            namespace.output_csv,
        ),
        unresolved_csv=cast(
            Path | None,
            namespace.unresolved_csv,
        ),
    )

    try:
        run(args)

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