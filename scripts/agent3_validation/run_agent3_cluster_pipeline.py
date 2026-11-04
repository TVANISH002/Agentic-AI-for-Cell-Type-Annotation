from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(".").resolve()
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def run_command(
    label: str,
    cmd: list[str],
) -> None:
    print()
    print("=" * 90)
    print(label)
    print("=" * 90)
    print(" ".join(cmd))

    subprocess.run(
        cmd,
        check=True,
    )


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8-sig")
    )


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def find_required(
    root: Path,
    pattern: str,
) -> Path:
    hits = sorted(
        p for p in root.rglob(pattern)
        if p.is_file()
    )

    if not hits:
        raise FileNotFoundError(
            f"No file found under {root} for pattern: {pattern}"
        )

    if len(hits) > 1:
        raise RuntimeError(
            f"Multiple files found for pattern {pattern}:\n"
            + "\n".join(str(p) for p in hits)
        )

    return hits[0]


def build_gene_terminology(
    *,
    rescue_plan_path: Path,
    literature_request_path: Path,
    output_path: Path,
) -> None:
    from literature_retrieval_agent.rescue_gene_terminology import (
        RescueGeneTerminologyBuilder,
    )

    rescue_plan = load_json(
        rescue_plan_path
    )
    literature_request = load_json(
        literature_request_path
    )

    builder = RescueGeneTerminologyBuilder()

    terminology = builder.build(
        rescue_plan=rescue_plan,
        literature_request=literature_request,
    )

    write_json(
        output_path,
        terminology,
    )

    print()
    print("=" * 90)
    print("GENERIC RESCUE GENE TERMINOLOGY")
    print("=" * 90)
    print("Output:", output_path)
    print("Organism:", terminology.get("organism"))
    print("Marker count:", terminology.get("marker_count"))
    print("Reference label used:", terminology.get("reference_label_used"))
    print("Expert label used:", terminology.get("expert_label_used"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the standardized Agent 3 literature evidence pipeline "
            "for one cluster using an existing LiteratureRequest JSON."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name, e.g. lung or breast.",
    )
    parser.add_argument(
        "--cluster-id",
        required=True,
        help="Cluster identifier.",
    )
    parser.add_argument(
        "--request-json",
        required=True,
        type=Path,
        help="Cluster-specific Agent 3 LiteratureRequest JSON.",
    )
    parser.add_argument(
        "--cluster-root",
        required=True,
        type=Path,
        help="Output root for this cluster run.",
    )
    parser.add_argument(
        "--cell-ontology",
        default=Path(
            r"data\resources\cell_ontology\v2026-06-08\cl-base.json"
        ),
        type=Path,
    )

    parser.add_argument("--context-page-size", type=int, default=3)
    parser.add_argument("--rescue-page-size", type=int, default=10)
    parser.add_argument("--confirmation-page-size", type=int, default=10)
    parser.add_argument("--candidate-count", type=int, default=3)
    parser.add_argument("--top-k-passages", type=int, default=10)
    parser.add_argument("--top-marker-count", type=int, default=25)
    parser.add_argument("--rescue-top-marker-limit", type=int, default=5)
    parser.add_argument(
        "--min-paper-relevance",
        default="medium",
        choices=["high", "medium", "low"],
    )

    args = parser.parse_args()

    py = sys.executable

    cluster_id = str(args.cluster_id)
    request_json = args.request_json
    cluster_root = args.cluster_root

    if not request_json.exists():
        raise FileNotFoundError(
            f"LiteratureRequest JSON not found: {request_json}"
        )

    if not args.cell_ontology.exists():
        raise FileNotFoundError(
            f"Cell Ontology file not found: {args.cell_ontology}"
        )

    cluster_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    context_dir = cluster_root / "context_retrieval"
    pubmedbert_dir = cluster_root / "pubmedbert"
    rank_coverage_dir = cluster_root / "rank_coverage"

    rescue_dir = cluster_root / "marker_rescue"
    rescue_retrieval_dir = rescue_dir / "retrieval"
    discovery_dir = rescue_dir / "candidate_discovery"

    confirmation_dir = rescue_dir / "confirmation"
    confirmation_retrieval_dir = confirmation_dir / "retrieval"
    grounding_dir = confirmation_dir / "grounding"
    confirmation_pubmedbert_dir = confirmation_dir / "pubmedbert"

    provenance_dir = rescue_dir / "provenance"
    packet_dir = rescue_dir / "final_evidence_packet"

    started_at = datetime.now().isoformat(timespec="seconds")

    print("=" * 90)
    print("STANDARDIZED AGENT 3 CLUSTER PIPELINE")
    print("=" * 90)
    print("Dataset:", args.dataset)
    print("Cluster:", cluster_id)
    print("Request JSON:", request_json)
    print("Cluster root:", cluster_root)
    print("Started:", started_at)
    print()
    print("Important: this pipeline does not use reference/expert labels for inference.")

    # 1. Context retrieval.
    run_command(
        "1. CONTEXT RETRIEVAL",
        [
            py,
            "-m",
            "literature_retrieval_agent.runners.run_context_retrieval",
            "--cluster-id",
            cluster_id,
            "--request-json",
            str(request_json),
            "--output-directory",
            str(context_dir),
            "--page-size",
            str(args.context_page_size),
        ],
    )

    literature_request = find_required(
        context_dir,
        f"cluster_{cluster_id}_context_v2_literature_request.json",
    )
    context_pilot = find_required(
        context_dir,
        f"cluster_{cluster_id}_context_v2_retrieval_pilot.json",
    )

    # 2. Paper ranking.
    run_command(
        "2. PAPER RANKING",
        [
            py,
            "-m",
            "literature_retrieval_agent.runners.run_paper_ranking",
            "--cluster-id",
            cluster_id,
            "--pilot-json",
            str(context_pilot),
            "--output-directory",
            str(context_dir),
            "--show-top",
            "10",
        ],
    )

    ranked_papers = find_required(
        context_dir,
        f"cluster_{cluster_id}_ranked_papers.json",
    )

    # 3. Evidence corpus.
    run_command(
        "3. EVIDENCE CORPUS",
        [
            py,
            "-m",
            "literature_retrieval_agent.runners.run_evidence_corpus",
            "--cluster-id",
            cluster_id,
            "--pilot-json",
            str(context_pilot),
            "--ranking-json",
            str(ranked_papers),
            "--output-directory",
            str(context_dir),
            "--min-paper-relevance",
            args.min_paper_relevance,
        ],
    )

    evidence_corpus = find_required(
        context_dir,
        f"cluster_{cluster_id}_evidence_corpus.json",
    )
    evidence_passages_csv = find_required(
        context_dir,
        f"cluster_{cluster_id}_evidence_passages.csv",
    )

    # 4. Initial PubMedBERT over upstream candidates.
    run_command(
        "4. INITIAL PUBMEDBERT RERANKING",
        [
            py,
            "-m",
            "literature_retrieval_agent.runners.run_pubmedbert_reranker",
            "--cluster-id",
            cluster_id,
            "--corpus-json",
            str(evidence_corpus),
            "--output-directory",
            str(pubmedbert_dir),
            "--candidate-count",
            str(args.candidate_count),
            "--top-k-passages",
            str(args.top_k_passages),
            "--top-marker-count",
            str(args.top_marker_count),
        ],
    )

    # 5. Rank coverage.
    run_command(
        "5. RANK-AWARE COVERAGE AUDIT",
        [
            py,
            "scripts/agent3_validation/audit_pubmedbert_rank_coverage.py",
            "--literature-request",
            str(literature_request),
            "--evidence-passages-csv",
            str(evidence_passages_csv),
            "--output-directory",
            str(rank_coverage_dir),
        ],
    )

    # 6. Marker rescue query plan.
    run_command(
        "6. CANDIDATE-INDEPENDENT MARKER RESCUE QUERY PLAN",
        [
            py,
            "scripts/agent3_validation/build_marker_rescue_queries.py",
            "--literature-request",
            str(literature_request),
            "--evidence-passages-csv",
            str(evidence_passages_csv),
            "--output-directory",
            str(rescue_dir),
            "--top-marker-limit",
            str(args.rescue_top_marker_limit),
        ],
    )

    rescue_plan = find_required(
        rescue_dir,
        f"cluster_{cluster_id}_marker_rescue_query_plan.json",
    )

    # 7. Marker rescue retrieval.
    run_command(
        "7. CANDIDATE-INDEPENDENT MARKER RESCUE RETRIEVAL",
        [
            py,
            "scripts/agent3_validation/run_marker_rescue_retrieval.py",
            "--literature-request",
            str(literature_request),
            "--rescue-plan",
            str(rescue_plan),
            "--output-directory",
            str(rescue_retrieval_dir),
            "--page-size",
            str(args.rescue_page_size),
        ],
    )

    rescue_pilot = find_required(
        rescue_retrieval_dir,
        f"cluster_{cluster_id}_retrieval_pilot.json",
    )

    # 8. Ontology-grounded candidate discovery.
    run_command(
        "8. ONTOLOGY-GROUNDED RESCUE CANDIDATE DISCOVERY",
        [
            py,
            "scripts/agent3_validation/discover_rescue_candidates.py",
            "--retrieval-pilot",
            str(rescue_pilot),
            "--cell-ontology",
            str(args.cell_ontology),
            "--literature-request",
            str(literature_request),
            "--output-directory",
            str(discovery_dir),
        ],
    )

    discovery_json = find_required(
        discovery_dir,
        f"cluster_{cluster_id}_rescue_candidate_discovery.json",
    )

    # 9. Confirmation query plan.
    confirmation_plan = (
        confirmation_dir
        / f"cluster_{cluster_id}_rescue_confirmation_query_plan.json"
    )

    run_command(
        "9. TARGET-CONTEXT CONFIRMATION QUERY PLAN",
        [
            py,
            "scripts/agent3_validation/build_rescue_confirmation_queries.py",
            "--candidate-discovery",
            str(discovery_json),
            "--rescue-plan",
            str(rescue_plan),
            "--literature-request",
            str(literature_request),
            "--output-json",
            str(confirmation_plan),
        ],
    )

    # 10. Confirmation retrieval.
    run_command(
        "10. TARGET-CONTEXT CONFIRMATION RETRIEVAL",
        [
            py,
            "scripts/agent3_validation/run_rescue_confirmation_retrieval.py",
            "--literature-request",
            str(literature_request),
            "--confirmation-plan",
            str(confirmation_plan),
            "--output-directory",
            str(confirmation_retrieval_dir),
            "--page-size",
            str(args.confirmation_page_size),
        ],
    )

    confirmation_pilot = find_required(
        confirmation_retrieval_dir,
        f"cluster_{cluster_id}_retrieval_pilot.json",
    )

    # 11. Gene terminology.
    gene_terminology = (
        confirmation_dir
        / f"cluster_{cluster_id}_rescue_gene_terminology.json"
    )

    build_gene_terminology(
        rescue_plan_path=rescue_plan,
        literature_request_path=literature_request,
        output_path=gene_terminology,
    )

    # 12. Terminology-aware grounding.
    run_command(
        "12. TERMINOLOGY-AWARE CONFIRMATION GROUNDING",
        [
            py,
            "-m",
            "literature_retrieval_agent.rescue_confirmation_grounding",
            "--pilot-json",
            str(confirmation_pilot),
            "--confirmation-plan",
            str(confirmation_plan),
            "--literature-request",
            str(literature_request),
            "--gene-terminology",
            str(gene_terminology),
            "--output-directory",
            str(grounding_dir),
        ],
    )

    grounding_json = find_required(
        grounding_dir,
        f"cluster_{cluster_id}_rescue_grounding_audit.json",
    )

    # 13. Rescue confirmation PubMedBERT.
    run_command(
        "13. RESCUE CONFIRMATION PUBMEDBERT",
        [
            py,
            "scripts/agent3_validation/run_rescue_confirmation_pubmedbert.py",
            "--pilot-json",
            str(confirmation_pilot),
            "--confirmation-plan",
            str(confirmation_plan),
            "--literature-request",
            str(literature_request),
            "--output-directory",
            str(confirmation_pubmedbert_dir),
            "--top-marker-count",
            str(args.top_marker_count),
            "--top-k-passages",
            str(args.top_k_passages),
        ],
    )

    confirmation_pubmedbert = find_required(
        confirmation_pubmedbert_dir,
        f"cluster_{cluster_id}_rescue_confirmation_pubmedbert.json",
    )

    # 14. Candidate provenance audit.
    run_command(
        "14. RESCUE CANDIDATE PROVENANCE AUDIT",
        [
            py,
            "-m",
            "literature_retrieval_agent.rescue_candidate_provenance",
            "--candidate-discovery",
            str(discovery_json),
            "--retrieval-pilot",
            str(rescue_pilot),
            "--rescue-plan",
            str(rescue_plan),
            "--literature-request",
            str(literature_request),
            "--output-directory",
            str(provenance_dir),
        ],
    )

    provenance_json = find_required(
        provenance_dir,
        f"cluster_{cluster_id}_rescue_candidate_provenance_audit.json",
    )

    # 15. Final evidence packet.
    run_command(
        "15. FINAL RESCUE EVIDENCE PACKET",
        [
            py,
            "-m",
            "literature_retrieval_agent.rescue_evidence_packet",
            "--discovery-json",
            str(discovery_json),
            "--provenance-json",
            str(provenance_json),
            "--grounding-json",
            str(grounding_json),
            "--pubmedbert-json",
            str(confirmation_pubmedbert),
            "--output-directory",
            str(packet_dir),
        ],
    )

    final_packet = find_required(
        packet_dir,
        f"cluster_{cluster_id}_rescue_evidence_packet.json",
    )

    # 16. Standardized freeze/audit.
    standardized_script = Path(
        "scripts/agent3_validation/run_agent3_cluster_standardized.py"
    )

    if standardized_script.exists():
        run_command(
            "16. STANDARDIZED FREEZE AUDIT",
            [
                py,
                str(standardized_script),
                "--dataset",
                args.dataset,
                "--cluster-id",
                cluster_id,
                "--cluster-root",
                str(cluster_root),
            ],
        )
    else:
        print()
        print("WARNING: standardized freeze script not found; skipping final audit.")

    finished_at = datetime.now().isoformat(timespec="seconds")

    run_manifest = {
        "schema_name": "Agent3StandardizedClusterPipelineRun",
        "schema_version": "1.0",
        "dataset": args.dataset,
        "cluster_id": cluster_id,
        "request_json": str(request_json),
        "cluster_root": str(cluster_root),
        "started_at": started_at,
        "finished_at": finished_at,
        "final_evidence_packet": str(final_packet),
        "automatic_acceptance_policy": "NOT_CALIBRATED",
        "scientific_note": (
            "This pipeline executes the standardized Agent 3 evidence workflow. "
            "It does not convert PubMedBERT similarity into confidence and does "
            "not force final_annotation."
        ),
    }

    write_json(
        cluster_root / "agent3_pipeline_run_manifest.json",
        run_manifest,
    )

    print()
    print("=" * 90)
    print("STANDARDIZED AGENT 3 CLUSTER PIPELINE COMPLETE")
    print("=" * 90)
    print("Dataset:", args.dataset)
    print("Cluster:", cluster_id)
    print("Final packet:", final_packet)
    print("Run manifest:", cluster_root / "agent3_pipeline_run_manifest.json")
    print("Finished:", finished_at)
    print("=" * 90)


if __name__ == "__main__":
    main()
