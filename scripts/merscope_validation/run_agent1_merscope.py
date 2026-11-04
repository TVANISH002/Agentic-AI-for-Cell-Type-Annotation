
"""MERSCOPE breast cancer: frozen Agent 1 execution.

Uses the existing BiologicalKnowledgeRetrievalAgent.
Keeps MERSCOPE outputs separate from CosMX.

Panel transport statistics are interface placeholders,
NOT observed differential expression statistics.
The original MERSCOPE marker CSV remains the authoritative
source for cluster-level DEG statistics used by Agent 2.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from biological_knowledge_agent import (
    BiologicalKnowledgeRetrievalAgent,
)


DATASET = "merscope_brcr_validated_clean"

MARKERS_CSV = (
    PROJECT_ROOT
    / "data/merscope_brcr/processed"
    / "merscope_brcr_top50_filtered_markers_clean.csv"
)

PANEL_CSV = (
    PROJECT_ROOT
    / "data/merscope_brcr/processed"
    / "merscope_brcr_unique_panel_genes.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/pipeline_runs"
    / DATASET
    / "agent1_output_clean"
)

EXPECTED_GENES = 192
EXPECTED_MARKER_ROWS = 350
EXPECTED_CLUSTERS = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024), b""
        ):
            digest.update(block)

    return digest.hexdigest()


def load_inputs() -> list[str]:
    if not PANEL_CSV.is_file():
        raise FileNotFoundError(PANEL_CSV)

    if not MARKERS_CSV.is_file():
        raise FileNotFoundError(MARKERS_CSV)

    panel = pd.read_csv(PANEL_CSV)
    markers = pd.read_csv(MARKERS_CSV)

    if "gene" not in panel:
        raise ValueError("Panel CSV lacks gene column")

    required = {
        "gene", "cluster_id",
        "log_fold_change", "adjusted_p_value"
    }

    if not required.issubset(markers.columns):
        raise ValueError(
            "Processed marker CSV has missing columns"
        )

    genes = [
        str(gene).strip().upper()
        for gene in panel["gene"].tolist()
    ]

    if (
        not genes
        or any(not gene or gene == "NAN" for gene in genes)
        or len(set(genes)) != len(genes)
    ):
        raise ValueError(
            "Invalid or duplicate panel genes"
        )

    marker_genes = set(
        markers["gene"].astype(str).str.strip().str.upper()
    )

    if set(genes) != marker_genes:
        raise ValueError(
            "Panel genes do not match processed markers"
        )

    if len(genes) != EXPECTED_GENES:
        raise ValueError(
            f"Expected {EXPECTED_GENES} unique genes, "
            f"found {len(genes)}"
        )

    if len(markers) != EXPECTED_MARKER_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_MARKER_ROWS} marker rows, "
            f"found {len(markers)}"
        )

    if markers["cluster_id"].nunique() != EXPECTED_CLUSTERS:
        raise ValueError("Unexpected cluster count")

    p_adj = pd.to_numeric(
        markers["adjusted_p_value"],
        errors="raise",
    )

    logfc = pd.to_numeric(
        markers["log_fold_change"],
        errors="raise",
    )

    if not (
        (p_adj <= 0.05).all()
        and (p_adj >= 0).all()
        and (logfc >= 0.5).all()
    ):
        raise ValueError(
            "Processed markers violate frozen thresholds"
        )

    return genes


def write_transport(
    path: Path,
    genes: list[str],
) -> None:
    """Write schema-compatible, panel-only Agent 1 input.

    The frozen Agent 1 interface requires numeric DEG fields.
    These values are placeholders for transport, never
    biological measurements or model evidence scores.
    """

    columns = [
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
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
        )

        writer.writeheader()

        for rank, gene in enumerate(genes, start=1):
            writer.writerow({
                "spatial_cluster": "panel",
                "rank_within_cluster": rank,
                "names": gene,
                "scores": len(genes) - rank + 1,
                "logfoldchanges": 1.0,
                "pvals": 0.01,
                "pvals_adj": 0.01,
                "pct_nz_group": 1.0,
                "pct_nz_reference": 0.0,
            })


def main() -> None:
    genes = load_inputs()

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            "Agent 1 output directory is not empty. "
            "Existing outputs will not be overwritten:\n"
            f"{OUTPUT_DIR}"
        )

    references = [
        PROJECT_ROOT / "data/reference/Cell_marker_Human.xlsx",
        PROJECT_ROOT
        / "data/reference/PanglaoDB_markers_27_Mar_2020.tsv",
    ]

    for reference in references:
        if not reference.is_file():
            raise FileNotFoundError(reference)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    transport = (
        OUTPUT_DIR / "panel_agent1_transport_input.csv"
    )

    output_json = (
        OUTPUT_DIR / "panel_agent1_knowledge_evidence.json"
    )

    start = time.perf_counter()

    write_transport(transport, genes)

    agent = BiologicalKnowledgeRetrievalAgent(
        project_root=PROJECT_ROOT,
        marker_file=transport,
        top_k=len(genes),
        candidate_pool_size=len(genes),
        min_log_fold_change=0.5,
        max_adjusted_p_value=0.05,
    )

    result = agent.run(
        cluster_id="panel",
        species="Human",
        tissue="Breast",
        platform="MERSCOPE",
    )

    # Verify all input genes survive frozen Agent 1.
    input_genes = [
        record.gene.strip().upper()
        for record in result.input_genes
    ]

    evidence_genes = [
        record.gene.strip().upper()
        for record in result.gene_evidence
    ]

    for name, observed in (
        ("input_genes", input_genes),
        ("gene_evidence", evidence_genes),
    ):
        if (
            len(observed) != len(genes)
            or set(observed) != set(genes)
            or len(set(observed)) != len(observed)
        ):
            raise ValueError(
                f"Agent 1 {name} did not preserve "
                "the complete MERSCOPE panel"
            )

    unresolved = {
        gene.strip().upper()
        for gene in result.unresolved_genes
    }

    if not unresolved.issubset(set(genes)):
        raise ValueError(
            "Unresolved genes contain genes outside panel"
        )

    output_json.write_text(
        result.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - start
    matched = len(genes) - len(unresolved)

    manifest = {
        "dataset": DATASET,
        "status": "AGENT1_COMPLETED_VALIDATED",
        "species": "Human",
        "tissue": "Breast",
        "platform": "MERSCOPE",
        "input_unique_genes": len(genes),
        "matched_genes": matched,
        "unresolved_genes": len(unresolved),
        "coverage_percent": round(
            100 * matched / len(genes), 4
        ),
        "marker_rows": EXPECTED_MARKER_ROWS,
        "input_marker_sha256": sha256(MARKERS_CSV),
        "input_panel_sha256": sha256(PANEL_CSV),
        "reference_sha256": {
            path.name: sha256(path)
            for path in references
        },
        "agent1_json_sha256": sha256(output_json),
        "runtime_seconds": round(elapsed, 6),
        "reference_labels_used": False,
        "expert_labels_used": False,
        "transport_statistics": (
            "Synthetic schema-compatible placeholders. "
            "Not observed DEGs. Use original marker CSV "
            "for actual cluster-level statistics."
        ),
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    manifest_path = (
        OUTPUT_DIR / "agent1_execution_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("MERSCOPE AGENT 1 COMPLETED")
    print("=" * 60)
    print(f"Dataset            : {DATASET}")
    print(f"Input unique genes : {len(genes)}")
    print(f"Matched genes      : {matched}")
    print(f"Unresolved genes   : {len(unresolved)}")
    print(
        f"Coverage           : "
        f"{manifest['coverage_percent']:.2f}%"
    )
    print(f"Runtime            : {elapsed:.3f} seconds")
    print("Reference labels   : NOT USED")
    print("Input preservation : PASS")
    print(f"Agent 1 JSON       : {output_json}")
    print(f"Manifest           : {manifest_path}")
    print("CosMX outputs      : UNCHANGED")
    print("=" * 60)


if __name__ == "__main__":
    main()
