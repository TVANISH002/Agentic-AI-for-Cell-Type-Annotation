from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from biological_knowledge_agent import (
    BiologicalKnowledgeRetrievalAgent,
)

from marker_interpretation_agent_v2.paths import (
    PROJECT_ROOT,
)


def _load_panel_genes(
    path: Path,
) -> list[str]:
    """Load the exact unique genes Agent 1 must retrieve."""

    if not path.exists():
        raise FileNotFoundError(
            f"Panel gene list not found: {path}"
        )

    dataframe = pd.read_csv(path)

    if "gene" not in dataframe.columns:
        raise ValueError(
            "Panel gene list must contain a 'gene' column."
        )

    genes = (
        dataframe["gene"]
        .astype("string")
        .str.strip()
        .str.upper()
        .tolist()
    )

    if not genes or any(not gene for gene in genes):
        raise ValueError(
            "Panel gene list contains a missing gene."
        )

    if len(genes) != len(set(genes)):
        raise ValueError(
            "Panel gene list contains duplicate genes."
        )

    return genes


def _write_transport_csv(
    path: Path,
    genes: list[str],
) -> None:
    """
    Create the panel transport file accepted by Agent 1.

    The synthetic numeric values are transport-only placeholders.
    Actual CosMX cluster DEG statistics remain in the processed
    marker file used by downstream stages.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for rank, gene in enumerate(
            genes,
            start=1,
        ):
            writer.writerow(
                {
                    "spatial_cluster": "panel",
                    "rank_within_cluster": rank,
                    "names": gene,
                    "scores": len(genes) - rank + 1,
                    "logfoldchanges": 1.0,
                    "pvals": 0.01,
                    "pvals_adj": 0.01,
                    "pct_nz_group": 1.0,
                    "pct_nz_reference": 0.0,
                }
            )


def run_agent1(
    panel_gene_list_csv: Path,
    output_dir: Path,
    *,
    species: str,
    tissue: str,
    platform: str,
) -> Path:
    """
    Run the existing Biological Knowledge Retrieval Agent.

    Biological context is supplied externally instead of being
    hard-coded inside this CosMX runner.
    """

    genes = _load_panel_genes(
        panel_gene_list_csv
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    transport_csv = (
        output_dir
        / "panel_agent1_transport_input.csv"
    )

    agent1_json = (
        output_dir
        / "panel_agent1_knowledge_evidence.json"
    )

    _write_transport_csv(
        transport_csv,
        genes,
    )

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

    input_genes = [
        record.gene.upper()
        for record in result.input_genes
    ]

    evidence_genes = [
        record.gene.upper()
        for record in result.gene_evidence
    ]

    if input_genes != genes:
        raise ValueError(
            "Agent 1 input genes do not exactly match "
            "the CosMX panel gene list."
        )

    if evidence_genes != genes:
        raise ValueError(
            "Agent 1 evidence genes do not exactly match "
            "the CosMX panel gene list."
        )

    agent1_json.write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )

    return agent1_json


def _parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Run CosMX Agent 1 with biological context "
            "supplied by the orchestration layer."
        )
    )

    parser.add_argument(
        "--panel-gene-list",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--species",
        required=True,
    )

    parser.add_argument(
        "--tissue",
        required=True,
    )

    parser.add_argument(
        "--platform",
        required=True,
    )

    return parser.parse_args()


def main() -> None:

    args = _parse_args()

    output = run_agent1(
        panel_gene_list_csv=args.panel_gene_list.resolve(),
        output_dir=args.output_dir.resolve(),
        species=args.species,
        tissue=args.tissue,
        platform=args.platform,
    )

    print(
        "CosMX orchestrated Agent 1 completed successfully."
    )
    print(f"Species: {args.species}")
    print(f"Tissue: {args.tissue}")
    print(f"Platform: {args.platform}")
    print(f"Agent 1 JSON: {output}")


if __name__ == "__main__":
    main()