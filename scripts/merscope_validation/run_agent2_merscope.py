
"""Run frozen Agent 2 on MERSCOPE breast cancer markers.

Reuses the existing HPA retriever, panel catalog builder,
marker interpretation agent and aggregate report writer.

Does not rerun Agent 1 or overwrite existing output folders.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from supplemental_evidence_retrieval.hpa_retriever import (
    HPASingleCellRetriever,
)
from supplemental_evidence_retrieval.hpa_report_writer import (
    HPAReportWriter,
)
from marker_interpretation_agent_v2.panel.panel_evidence_builder import (
    PanelEvidenceBuilder,
)
from marker_interpretation_agent_v2.panel.panel_evidence_loader import (
    PanelEvidenceLoader,
)
from marker_interpretation_agent_v2.panel.gene_panel_loader import (
    GenePanelLoader,
)
from marker_interpretation_agent_v2.config import DEFAULT_CONFIG
from marker_interpretation_agent_v2.agent import (
    MarkerInterpretationAgent,
)
from marker_interpretation_agent_v2.runners.run_all_clusters import (
    _validate_catalog_panel,
    _write_all_cluster_reports,
)

DATASET = "merscope_brcr_validated_clean"

MARKERS = ROOT / (
    "data/merscope_brcr/processed/"
    "merscope_brcr_top50_filtered_markers_clean.csv"
)

PANEL = ROOT / (
    "data/merscope_brcr/processed/"
    "merscope_brcr_unique_panel_genes.csv"
)

BASE = ROOT / "data/pipeline_runs" / DATASET

AGENT1_DIR = BASE / "agent1_output_clean"
AGENT1_JSON = AGENT1_DIR / "panel_agent1_knowledge_evidence.json"
AGENT1_TRANSPORT = AGENT1_DIR / "panel_agent1_transport_input.csv"

SOURCES = BASE / "agent2_input" / "panel_sources"
CATALOG_DIR = BASE / "agent2_input" / "panel_catalog"
OUTPUT = BASE / "agent2_output"

HPA_REFERENCE = ROOT / (
    "data/reference/hpa/"
    "rna_single_cell_cluster_hpa_v25_1.tsv.zip"
)

HPA_URL = (
    "https://www.proteinatlas.org/download/tsv/"
    "rna_single_cell_cluster.tsv.zip"
)

HPA_RELEASE = "25.1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_genes() -> list[str]:
    with PANEL.open(
        encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        if "gene" not in (reader.fieldnames or []):
            raise ValueError("Panel gene column is missing")

        genes = [
            str(row["gene"]).strip().upper()
            for row in reader
        ]

    if len(genes) != 192 or len(set(genes)) != 192:
        raise ValueError(
            "Expected exactly 192 unique panel genes"
        )

    return genes


def validate_agent1(genes: list[str]) -> None:
    payload = json.loads(
        AGENT1_JSON.read_text(encoding="utf-8")
    )

    for field in ("input_genes", "gene_evidence"):
        records = payload.get(field)

        if not isinstance(records, list):
            raise ValueError(f"Agent 1 missing {field}")

        actual = [
            str(item["gene"]).strip().upper()
            for item in records
        ]

        if actual != genes:
            raise ValueError(
                f"Agent 1 {field} gene order/coverage mismatch"
            )


def check_outputs_are_new() -> None:
    for directory in (SOURCES, CATALOG_DIR, OUTPUT):
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(
                "Refusing to overwrite existing outputs:\n"
                f"{directory}"
            )


def main() -> None:
    start = time.perf_counter()

    for path in (
        MARKERS,
        PANEL,
        AGENT1_JSON,
        AGENT1_TRANSPORT,
        HPA_REFERENCE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    genes = load_genes()
    validate_agent1(genes)
    check_outputs_are_new()

    for directory in (SOURCES, CATALOG_DIR, OUTPUT):
        directory.mkdir(parents=True, exist_ok=True)

    print("\nSTAGE 1: PREPARING AGENT 2 SOURCES")

    # Reuse completed Agent 1 rather than running it again.
    agent1_copy = SOURCES / AGENT1_JSON.name
    shutil.copyfile(AGENT1_JSON, agent1_copy)
    shutil.copyfile(
        AGENT1_TRANSPORT,
        SOURCES / AGENT1_TRANSPORT.name,
    )

    print("Existing MERSCOPE Agent 1 evidence reused.")

    # --------------------------------------------------------
    # HPA HEALTHY-BREAST EVIDENCE
    # --------------------------------------------------------

    print("\nSTAGE 2: FROZEN HPA RETRIEVAL")

    hpa_start = time.perf_counter()

    retriever = HPASingleCellRetriever(
        source_url=HPA_URL,
        source_release=HPA_RELEASE,
    )

    hpa = retriever.retrieve(
        reference_path=HPA_REFERENCE,
        genes=genes,
        tissue="breast",
        cluster_id="panel",
        run_mode="all",
        progress_callback=print,
    )

    hpa_genes = [
        str(gene).strip().upper()
        for gene in hpa.selected_genes
    ]

    summary_genes = [
        item.gene.strip().upper()
        for item in hpa.gene_summaries
    ]

    if hpa_genes != genes or summary_genes != genes:
        raise ValueError(
            "HPA did not preserve all 192 panel genes"
        )

    hpa_paths = HPAReportWriter().write_all(
        hpa, SOURCES
    )

    hpa_json = SOURCES / "panel_hpa_evidence.json"
    hpa_paths["json"].replace(hpa_json)

    hpa_seconds = time.perf_counter() - hpa_start

    print(f"HPA gene records: {len(summary_genes)}")
    print(f"HPA runtime: {hpa_seconds:.3f} seconds")

    # --------------------------------------------------------
    # FROZEN PANEL EVIDENCE CATALOG
    # --------------------------------------------------------

    print("\nSTAGE 3: BUILDING FROZEN PANEL CATALOG")

    catalog_start = time.perf_counter()

    builder = PanelEvidenceBuilder()

    catalog = builder.build_from_files(
        agent1_copy,
        hpa_json,
        expected_genes=genes,
        require_hpa_for_all_genes=True,
    )

    catalog_json = CATALOG_DIR / "panel_evidence_catalog.json"

    builder.write(
        catalog,
        json_path=catalog_json,
        csv_path=CATALOG_DIR / "panel_evidence_catalog.csv",
        unresolved_path=CATALOG_DIR / "panel_unresolved_genes.csv",
        require_hpa_for_all_genes=True,
    )

    if len(catalog.genes) != 192:
        raise ValueError("Catalog is missing panel genes")

    catalog_seconds = time.perf_counter() - catalog_start

    print(f"Catalog genes: {len(catalog.genes)}")
    print(f"Catalog runtime: {catalog_seconds:.3f} seconds")

    # --------------------------------------------------------
    # FROZEN AGENT 2 INTERPRETATION
    # --------------------------------------------------------

    print("\nSTAGE 4: INTERPRETING ALL 8 CLUSTERS")

    interpretation_start = time.perf_counter()

    panel_loader = GenePanelLoader(MARKERS)

    loaded_catalog = PanelEvidenceLoader().load_catalog(
        catalog_json
    )

    _validate_catalog_panel(
        panel_loader,
        loaded_catalog,
    )

    cluster_ids = sorted(
        [str(cluster) for cluster in panel_loader.get_cluster_ids()],
        key=lambda value: int(value),
    )

    if cluster_ids != [str(i) for i in range(8)]:
        raise ValueError(
            f"Unexpected MERSCOPE clusters: {cluster_ids}"
        )

    # Same biological parameters as frozen CosMX baseline.
    # Only platform metadata is adapted for this dataset.
    config = replace(
        DEFAULT_CONFIG,
        species="Human",
        tissue="Breast",
        platform="MERSCOPE",
        top_k=50,
        hpa_enabled=True,
    )

    agent = MarkerInterpretationAgent(config)
    results = []

    for cluster_id in cluster_ids:
        input_genes = panel_loader.get_cluster_markers(
            cluster_id
        )

        result = agent.interpret_cluster(
            cluster_id=cluster_id,
            input_genes=input_genes,
            catalog=loaded_catalog,
        )

        agent.write_reports(
            result,
            output_dir=OUTPUT / "clusters" / f"cluster_{cluster_id}",
        )

        results.append(result)

        print(
            f"Cluster {cluster_id}: "
            f"markers={result.summary.total_input_genes}, "
            f"leading={result.summary.leading_candidate}, "
            f"review={result.summary.human_review_required}"
        )

    # The same aggregate writer used by the frozen runner.
    aggregate_paths = _write_all_cluster_reports(
        results,
        output_root=OUTPUT,
        marker_file=MARKERS,
        panel_catalog=catalog_json,
        candidate_selector=agent.candidate_selector,
    )

    interpretation_seconds = (
        time.perf_counter() - interpretation_start
    )

    total_seconds = time.perf_counter() - start

    # --------------------------------------------------------
    # REPRODUCIBILITY MANIFEST
    # --------------------------------------------------------

    manifest = {
        "dataset": DATASET,
        "status": "AGENT2_COMPLETED_VALIDATED",
        "cluster_count": len(results),
        "cluster_ids": cluster_ids,
        "input_unique_genes": len(genes),
        "marker_file": str(MARKERS),
        "agent1_source": str(AGENT1_JSON),
        "hpa_reference": str(HPA_REFERENCE),
        "hpa_release": HPA_RELEASE,
        "hpa_context": (
            "healthy breast expression evidence; "
            "not tumor-specific validation"
        ),
        "configuration": config.to_dict(),
        "reference_labels_used": False,
        "expert_labels_used": False,
        "agent1_rerun": False,
        "source_hashes": {
            "markers_sha256": sha256(MARKERS),
            "panel_sha256": sha256(PANEL),
            "agent1_sha256": sha256(AGENT1_JSON),
            "hpa_reference_sha256": sha256(HPA_REFERENCE),
        },
        "timings_seconds": {
            "hpa": round(hpa_seconds, 6),
            "catalog": round(catalog_seconds, 6),
            "interpretation": round(interpretation_seconds, 6),
            "total": round(total_seconds, 6),
        },
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "aggregate_outputs": {
            key: str(path)
            for key, path in aggregate_paths.items()
        },
    }

    manifest_path = OUTPUT / "merscope_agent2_manifest.json"

    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("MERSCOPE AGENT 2 COMPLETED")
    print("=" * 60)
    print(f"Clusters processed : {len(results)}")
    print(f"Panel genes        : {len(genes)}")
    print(f"HPA runtime        : {hpa_seconds:.3f} s")
    print(f"Catalog runtime    : {catalog_seconds:.3f} s")
    print(
        f"Interpretation time: "
        f"{interpretation_seconds:.3f} s"
    )
    print(f"Total runtime      : {total_seconds:.3f} s")
    print(f"Output directory   : {OUTPUT}")
    print(f"Manifest           : {manifest_path}")
    print("CosMX outputs      : NOT MODIFIED")
    print("=" * 60)


if __name__ == "__main__":
    main()
