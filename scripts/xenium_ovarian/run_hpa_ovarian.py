from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hpa_evidence_agent.config import DEFAULT_CONFIG
from hpa_evidence_agent.runners.run_hpa_panel import run


def main() -> None:
    panel = ROOT / "data/xenium_ovarian/panel/panel_gene_list_clean.csv"

    markers = (
        ROOT
        / "data/xenium_ovarian/processed"
        / "ovarian_top50_filtered_markers_clean.csv"
    )

    hpa_source = (
        ROOT
        / "data/reference/hpa"
        / "rna_single_cell_cluster_hpa_v25_1.tsv.zip"
    )

    output = (
        ROOT
        / "data/pipeline_runs/xenium_ovarian_validated_clean"
        / "agent2_input/panel_sources/ovary_hpa_evidence"
    )

    for path in (panel, markers, hpa_source):
        if not path.is_file():
            raise FileNotFoundError(path)

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output}"
        )

    config = replace(
        DEFAULT_CONFIG,
        panel_gene_list_path=panel,
        cluster_marker_path=markers,
        hpa_source_path=hpa_source,
        output_dir=output,
        query_tissue="ovary",
        evidence_scope="healthy_ovary_reference",
        marker_gene_column="gene",
        marker_cluster_column="cluster_id",
        marker_score_column="score",
        marker_logfc_column="log_fold_change",
        marker_padj_column="adjusted_p_value",
        top_genes_per_cluster=50,
    )

    print("\nDATASET: xenium_ovarian_validated_clean")
    print("HPA query tissue:", config.query_tissue)
    print("Evidence scope:", config.evidence_scope)

    start = time.perf_counter()
    paths = run(config)
    elapsed = time.perf_counter() - start

    print("\n========== OVARIAN HPA EXECUTION ==========")
    print("Files generated:", len(paths))
    print(f"Runtime: {elapsed:.3f} seconds")
    print("Output directory:", output)
    print("Status: EXECUTION COMPLETED — VALIDATION REQUIRED")


if __name__ == "__main__":
    main()