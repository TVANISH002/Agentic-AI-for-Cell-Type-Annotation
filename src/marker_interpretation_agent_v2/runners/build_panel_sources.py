from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from biological_knowledge_agent import (
    BiologicalKnowledgeRetrievalAgent,
    KnowledgeEvidence,
)
from supplemental_evidence_retrieval.hpa_report_writer import HPAReportWriter
from supplemental_evidence_retrieval.hpa_retriever import HPASingleCellRetriever
from supplemental_evidence_retrieval.hpa_schemas import HPARetrievalResult

from ..panel.panel_evidence_loader import PanelEvidenceLoader
from ..paths import PROJECT_ROOT


DEFAULT_PANEL_DIR = PROJECT_ROOT / "data" / "agent2_v2" / "panel"
DEFAULT_PANEL_GENE_LIST = DEFAULT_PANEL_DIR / "panel_gene_list.csv"
DEFAULT_HPA_REFERENCE = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "hpa"
    / "rna_single_cell_cluster_hpa_v25_1.tsv.zip"
)
DEFAULT_HPA_SOURCE_URL = (
    "https://www.proteinatlas.org/download/tsv/"
    "rna_single_cell_cluster.tsv.zip"
)
DEFAULT_HPA_SOURCE_RELEASE = "25.1"


def _normalize_gene(value: object) -> str:
    """Normalize one gene symbol."""

    gene = str(value).strip().upper()

    if not gene:
        raise ValueError("Gene symbol cannot be empty.")

    return gene


def _load_panel_genes(path: Path) -> list[str]:
    """Load and validate the unique panel genes."""

    if not path.exists():
        raise FileNotFoundError(f"Panel gene list not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None or "gene" not in reader.fieldnames:
            raise ValueError("Panel gene list must contain a 'gene' column.")

        genes = [_normalize_gene(row.get("gene", "")) for row in reader]

    if not genes:
        raise ValueError("Panel gene list contains no genes.")

    if len(genes) != len(set(genes)):
        duplicates = sorted({gene for gene in genes if genes.count(gene) > 1})
        raise ValueError(
            f"Panel gene list contains duplicate genes: {duplicates}"
        )

    return genes


def _write_agent1_transport_csv(path: Path, genes: list[str]) -> None:
    """Create transport-only input for the frozen cluster Agent 1."""

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "spatial_cluster",
        "rank_within_cluster",
        "names",
        "scores",
        "logfoldchanges",
        "pvals",
        "pvals_adj",
        "pct_nz_group",
        "pct_nz_reference",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for rank, gene in enumerate(genes, start=1):
            writer.writerow(
                {
                    "spatial_cluster": "panel",
                    "rank_within_cluster": rank,
                    "names": gene,
                    "scores": float(len(genes) - rank + 1),
                    "logfoldchanges": 1.0,
                    "pvals": 0.01,
                    "pvals_adj": 0.01,
                    "pct_nz_group": 1.0,
                    "pct_nz_reference": 0.0,
                }
            )


def _ensure_exact_gene_order(
    expected: list[str],
    actual: list[str],
    source_name: str,
) -> None:
    """Validate exact producer-consumer gene coverage and ordering."""

    normalized_actual = [_normalize_gene(gene) for gene in actual]

    if normalized_actual == expected:
        return

    expected_set = set(expected)
    actual_set = set(normalized_actual)
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)

    raise ValueError(
        f"{source_name} gene contract mismatch. "
        f"Expected={len(expected)}, actual={len(normalized_actual)}, "
        f"missing={missing}, unexpected={unexpected}, "
        f"exact_order_match={normalized_actual == expected}."
    )


def _run_agent1(
    *,
    transport_csv: Path,
    genes: list[str],
    species: str,
    tissue: str,
    platform: str,
) -> KnowledgeEvidence:
    """Run frozen Agent 1 for every unique panel gene."""

    gene_count = len(genes)

    agent = BiologicalKnowledgeRetrievalAgent(
        project_root=PROJECT_ROOT,
        marker_file=transport_csv,
        top_k=gene_count,
        candidate_pool_size=gene_count,
        min_log_fold_change=0.5,
        max_adjusted_p_value=0.05,
    )

    result = agent.run(
        cluster_id="panel",
        species=species,
        tissue=tissue,
        platform=platform,
    )

    _ensure_exact_gene_order(
        genes,
        [item.gene for item in result.input_genes],
        "Agent 1 input_genes",
    )

    _ensure_exact_gene_order(
        genes,
        [item.gene for item in result.gene_evidence],
        "Agent 1 gene_evidence",
    )

    return result


def _run_hpa(
    *,
    reference_path: Path,
    genes: list[str],
    tissue: str,
    source_url: str,
    source_release: str,
) -> HPARetrievalResult:
    """Run HPA once for every unique panel gene."""

    retriever = HPASingleCellRetriever(
        source_url=source_url,
        source_release=source_release,
    )

    result = retriever.retrieve(
        reference_path=reference_path,
        genes=genes,
        tissue=tissue,
        cluster_id="panel",
        run_mode="all",
        progress_callback=print,
    )

    _ensure_exact_gene_order(
        genes,
        result.selected_genes,
        "HPA selected_genes",
    )

    _ensure_exact_gene_order(
        genes,
        [item.gene for item in result.gene_summaries],
        "HPA gene_summaries",
    )

    return result


def build_panel_sources(
    *,
    panel_gene_list: Path,
    hpa_reference: Path,
    output_dir: Path,
    species: str = "Human",
    tissue: str = "Breast",
    platform: str = "Xenium",
    hpa_tissue: str = "breast",
    hpa_source_url: str = DEFAULT_HPA_SOURCE_URL,
    hpa_source_release: str = DEFAULT_HPA_SOURCE_RELEASE,
    include_hpa: bool = True,
) -> dict[str, Path]:
    """Build and validate panel-wide Agent 1 and HPA outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    genes = _load_panel_genes(panel_gene_list)

    transport_csv = output_dir / "panel_agent1_transport_input.csv"
    agent1_json = output_dir / "panel_agent1_knowledge_evidence.json"
    hpa_json = output_dir / "panel_hpa_evidence.json"
    validation_json = output_dir / "panel_source_validation.json"

    _write_agent1_transport_csv(transport_csv, genes)

    print(f"Running frozen Agent 1 for {len(genes)} panel genes...")

    agent1_result = _run_agent1(
        transport_csv=transport_csv,
        genes=genes,
        species=species,
        tissue=tissue,
        platform=platform,
    )

    agent1_json.write_text(
        agent1_result.model_dump_json(indent=2),
        encoding="utf-8",
    )

    if not include_hpa:
        validation = {
            "schema_name": "PanelSourceValidation",
            "schema_version": "1.0",
            "panel_gene_count": len(genes),
            "agent1_input_gene_count": len(agent1_result.input_genes),
            "agent1_gene_evidence_count": len(agent1_result.gene_evidence),
            "hpa_enabled": False,
            "hpa_selected_gene_count": 0,
            "hpa_gene_summary_count": 0,
            "species": species,
            "tissue": tissue,
            "platform": platform,
            "panel_gene_list": str(panel_gene_list.resolve()),
            "agent1_transport_input": str(transport_csv.resolve()),
            "agent1_output": str(agent1_json.resolve()),
            "note": (
                "HPA retrieval intentionally skipped. "
                "Use the separate HPA validation workflow."
            ),
        }

        validation_json.write_text(
            json.dumps(validation, indent=2),
            encoding="utf-8",
        )

        return {
            "panel_agent1_transport_input_csv": transport_csv,
            "panel_agent1_knowledge_evidence_json": agent1_json,
            "panel_source_validation_json": validation_json,
        }

    print(f"Running HPA retrieval for {len(genes)} panel genes...")

    hpa_result = _run_hpa(
        reference_path=hpa_reference,
        genes=genes,
        tissue=hpa_tissue,
        source_url=hpa_source_url,
        source_release=hpa_source_release,
    )

    hpa_paths = HPAReportWriter().write_all(hpa_result, output_dir)
    generated_hpa_json = hpa_paths["json"]

    if generated_hpa_json != hpa_json:
        generated_hpa_json.replace(hpa_json)
        hpa_paths["json"] = hpa_json

    loader = PanelEvidenceLoader()

    agent1_payload = loader.load_agent1(
        agent1_json,
        expected_genes=genes,
    )

    hpa_payload = loader.load_hpa(
        hpa_json,
        expected_genes=genes,
    )

    loader.validate_pair(agent1_payload, hpa_payload)

    validation = {
        "schema_name": "PanelSourceValidation",
        "schema_version": "1.0",
        "panel_gene_count": len(genes),
        "agent1_input_gene_count": len(agent1_result.input_genes),
        "agent1_gene_evidence_count": len(agent1_result.gene_evidence),
        "hpa_selected_gene_count": len(hpa_result.selected_genes),
        "hpa_gene_summary_count": len(hpa_result.gene_summaries),
        "exact_gene_order_match": True,
        "missing_genes": [],
        "unexpected_genes": [],
        "species": species,
        "tissue": tissue,
        "platform": platform,
        "hpa_query_tissue": hpa_tissue,
        "hpa_source_release": hpa_source_release,
        "hpa_source_url": hpa_source_url,
        "panel_gene_list": str(panel_gene_list.resolve()),
        "agent1_transport_input": str(transport_csv.resolve()),
        "agent1_output": str(agent1_json.resolve()),
        "hpa_reference": str(hpa_reference.resolve()),
        "hpa_output": str(hpa_json.resolve()),
        "transport_policy": (
            "Synthetic DEG values are transport-only placeholders used to run "
            "the frozen cluster-oriented Agent 1 once for every unique panel "
            "gene. Agent 2 reloads genuine cluster DEG statistics from the "
            "original marker CSV."
        ),
    }

    validation_json.write_text(
        json.dumps(validation, indent=2),
        encoding="utf-8",
    )

    return {
        "panel_agent1_transport_input_csv": transport_csv,
        "panel_agent1_knowledge_evidence_json": agent1_json,
        "panel_hpa_evidence_json": hpa_json,
        "panel_hpa_gene_summary_csv": hpa_paths["gene_summary_csv"],
        "panel_hpa_cell_type_summary_csv": hpa_paths[
            "cell_type_summary_csv"
        ],
        "panel_hpa_expression_records_csv": hpa_paths[
            "expression_records_csv"
        ],
        "panel_source_validation_json": validation_json,
    }


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Build panel-wide Agent 1 and HPA source files for Agent 2 V2."
        )
    )

    parser.add_argument(
        "--panel-gene-list",
        type=Path,
        default=DEFAULT_PANEL_GENE_LIST,
        help=f"Unique panel gene CSV. Default: {DEFAULT_PANEL_GENE_LIST}",
    )

    parser.add_argument(
        "--hpa-reference",
        type=Path,
        default=DEFAULT_HPA_REFERENCE,
        help=f"Local HPA ZIP. Default: {DEFAULT_HPA_REFERENCE}",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PANEL_DIR,
        help=f"Output directory. Default: {DEFAULT_PANEL_DIR}",
    )

    parser.add_argument("--species", default="Human")
    parser.add_argument("--tissue", default="Breast")
    parser.add_argument("--platform", default="Xenium")
    parser.add_argument("--hpa-tissue", default="breast")
    parser.add_argument("--hpa-source-url", default=DEFAULT_HPA_SOURCE_URL)
    parser.add_argument(
        "--hpa-source-release",
        default=DEFAULT_HPA_SOURCE_RELEASE,
    )

    parser.add_argument(
        "--no-hpa",
        action="store_true",
        help=(
            "Skip HPA retrieval and write only Agent 1 "
            "marker-database source files."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the panel-wide source builder."""

    args = _parse_args()

    paths = build_panel_sources(
        panel_gene_list=args.panel_gene_list.resolve(),
        hpa_reference=args.hpa_reference.resolve(),
        output_dir=args.output_dir.resolve(),
        species=args.species,
        tissue=args.tissue,
        platform=args.platform,
        hpa_tissue=args.hpa_tissue,
        hpa_source_url=args.hpa_source_url,
        hpa_source_release=args.hpa_source_release,
        include_hpa=not args.no_hpa,
    )

    print("Panel source build completed.")

    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()