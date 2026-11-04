from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(".").resolve()
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from literature_retrieval_agent.request_builder import (
    build_literature_request,
    load_agent2_handoff,
)
from literature_retrieval_agent.schemas import (
    LiteratureRequest,
)
from literature_retrieval_agent.rescue_query_builder import (
    MarkerRescueQueryBuilder,
)
from literature_retrieval_agent.rescue_gene_terminology import (
    RescueGeneTerminologyBuilder,
)


PUBMEDBERT_MODEL = "NeuML/pubmedbert-base-embeddings"


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def run_command(
    label: str,
    cmd: list[str],
) -> None:
    print()
    print("=" * 100)
    print(label)
    print("=" * 100)
    print(" ".join(cmd))

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    subprocess.run(
        cmd,
        check=True,
        env=env,
    )


def find_one(
    root: Path,
    pattern: str,
) -> Path:
    hits = sorted(
        path
        for path in root.rglob(pattern)
        if path.is_file()
    )

    if not hits:
        raise FileNotFoundError(
            f"No file found under {root} "
            f"for pattern {pattern}"
        )

    if len(hits) > 1:
        raise RuntimeError(
            f"Multiple files found for {pattern}:\n"
            + "\n".join(
                str(path)
                for path in hits
            )
        )

    return hits[0]


def infer_agent2_json(
    dataset: str,
    cluster_id: str,
) -> Path:
    dataset = dataset.casefold()

    if dataset == "lung":
        base = Path(
            r"data\pipeline_runs\lung_validated_clean"
        )

    elif dataset == "breast":
        base = Path(
            r"data\pipeline_runs\breast_xenium_validated_clean"
        )

    else:
        raise ValueError(
            "Automatic Agent 2 path discovery is available only for "
            "the existing lung and breast datasets. For any other "
            "dataset/tissue, provide --agent2-json explicitly."
        )

    return (
        base
        / "agent2_output"
        / "clusters"
        / f"cluster_{cluster_id}"
        / f"cluster_{cluster_id}_marker_evidence.json"
    )


def build_request(
    *,
    agent2_json: Path,
    disease_context: str,
    candidate_count: int,
    output_path: Path,
) -> LiteratureRequest:
    payload, source = load_agent2_handoff(
        agent2_json
    )

    request = build_literature_request(
        payload,
        source_file=source,
        disease_context=disease_context,
        candidate_count=candidate_count,
        marker_count=None,
    )

    serialized = request.model_dump_json(
        indent=2
    )

    # Strict frozen-schema validation.
    validated = LiteratureRequest.model_validate_json(
        serialized
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        serialized,
        encoding="utf-8",
    )

    return validated


def build_gene_terminology(
    *,
    rescue_plan_path: Path,
    literature_request_path: Path,
    output_path: Path,
) -> dict:
    builder = RescueGeneTerminologyBuilder()

    result = builder.build(
        rescue_plan=load_json(
            rescue_plan_path
        ),
        literature_request=load_json(
            literature_request_path
        ),
    )

    write_json(
        output_path,
        result,
    )

    return result


def build_rescue_plan(
    *,
    literature_request_path: Path,
    pubmedbert_evidence_passages: Path,
    output_path: Path,
    top_marker_limit: int,
) -> dict:
    request = load_json(
        literature_request_path
    )

    builder = MarkerRescueQueryBuilder(
        top_marker_limit=top_marker_limit,
        pairwise_marker_limit=3,
    )

    explained = builder.load_explained_markers(
        pubmedbert_evidence_passages
    )

    plan = builder.build(
        request=request,
        explained_markers=explained,
        source_literature_request=str(
            literature_request_path
        ),
        source_pubmedbert_evidence=str(
            pubmedbert_evidence_passages
        ),
    )

    write_json(
        output_path,
        plan,
    )

    return plan


def make_file_record(path: Path) -> dict:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_freeze_manifest(
    *,
    cluster_root: Path,
    result_path: Path,
    request_path: Path,
    dataset: str,
    cluster_id: str,
) -> Path:
    freeze_dir = (
        cluster_root
        / "freeze"
    )

    freeze_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = load_json(
        result_path
    )

    request = LiteratureRequest.model_validate_json(
        request_path.read_text(
            encoding="utf-8-sig"
        )
    )

    artifacts = {}

    for path in sorted(
        cluster_root.rglob("*")
    ):
        if not path.is_file():
            continue

        if freeze_dir in path.parents:
            continue

        if path.suffix.casefold() not in {
            ".json",
            ".csv",
            ".txt",
        }:
            continue

        rel = str(
            path.relative_to(
                cluster_root
            )
        ).replace("\\", "/")

        artifacts[rel] = (
            make_file_record(path)
        )

    invariants = {
        "request_schema_valid": True,

        "cluster_matches": (
            str(request.cluster_id)
            == str(cluster_id)
        ),

        "result_cluster_matches": (
            str(result.get("cluster_id"))
            == str(cluster_id)
        ),

        "reference_label_not_used": (
            result.get(
                "reference_label_used"
            )
            is False
        ),

        "expert_label_not_used": (
            result.get(
                "expert_label_used"
            )
            is False
        ),

        "automatic_acceptance_not_calibrated": (
            result.get(
                "automatic_acceptance"
            )
            == "NOT_CALIBRATED"
        ),

        "final_annotation_none": (
            result.get(
                "final_annotation"
            )
            is None
        ),

        "pubmedbert_not_called_confidence": (
            "semantic"
            in str(
                result.get(
                    "pubmedbert_interpretation",
                    "",
                )
            ).casefold()
        ),
    }

    overall_pass = all(
        invariants.values()
    )

    manifest = {
        "schema_name":
            "Agent3StandardizedFreezeManifest",

        "schema_version":
            "1.0",

        "created_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "dataset":
            dataset,

        "cluster_id":
            str(cluster_id),

        "overall_pass":
            overall_pass,

        "invariants":
            invariants,

        "artifact_count":
            len(artifacts),

        "artifacts":
            artifacts,

        "scientific_note": (
            "This freeze records a standardized "
            "Agent 3 evidence run. PubMedBERT "
            "similarities represent semantic relevance, "
            "not annotation probability or biological "
            "confidence. Automatic acceptance remains "
            "uncalibrated."
        ),
    }

    manifest_path = (
        freeze_dir
        / (
            f"agent3_{dataset}_cluster_"
            f"{cluster_id}_freeze_manifest.json"
        )
    )

    write_json(
        manifest_path,
        manifest,
    )

    print()
    print("=" * 100)
    print("STANDARDIZED FREEZE AUDIT")
    print("=" * 100)
    print(
        "Overall:",
        "PASS"
        if overall_pass
        else "FAIL",
    )

    failed = [
        key
        for key, value
        in invariants.items()
        if not value
    ]

    print(
        "Invariant failures:",
        "none"
        if not failed
        else ", ".join(failed),
    )

    print(
        "Manifest:",
        manifest_path,
    )

    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the complete standardized Agent 3 "
            "workflow for any Breast or Lung cluster."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "Dataset identifier. Existing lung/breast datasets support "
            "automatic Agent 2 path discovery. For any other dataset or "
            "tissue, provide --agent2-json explicitly."
        ),
    )

    parser.add_argument(
        "--cluster-id",
        required=True,
    )

    parser.add_argument(
        "--disease-context",
        required=True,
    )

    parser.add_argument(
        "--agent2-json",
        type=Path,
        default=None,
        help=(
            "Optional explicit Agent 2 MarkerEvidence "
            "JSON. Otherwise inferred from dataset "
            "and cluster."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--candidate-count",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--context-page-size",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--rescue-page-size",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--confirmation-page-size",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--top-k-passages",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--top-marker-count",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--rescue-top-marker-limit",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--cell-ontology",
        type=Path,
        default=Path(
            r"data\resources\cell_ontology"
            r"\v2026-06-08\cl-base.json"
        ),
    )

    parser.add_argument(
        "--plan-only",
        action="store_true",
        help=(
            "Validate inputs and print the standardized "
            "execution plan without retrieving papers."
        ),
    )

    args = parser.parse_args()

    py = sys.executable

    dataset = args.dataset
    cluster_id = str(
        args.cluster_id
    )

    if args.agent2_json is not None:
        agent2_json = args.agent2_json
    else:
        agent2_json = infer_agent2_json(
            dataset,
            cluster_id,
        )

    if not agent2_json.exists():
        raise FileNotFoundError(
            f"Agent 2 MarkerEvidence JSON not found: "
            f"{agent2_json}"
        )

    if not args.cell_ontology.exists():
        raise FileNotFoundError(
            f"Cell Ontology file not found: "
            f"{args.cell_ontology}"
        )

    cluster_root = (
        args.output_root
        if args.output_root
        else (
            Path(
                "data/analysis/"
                "agent3_validation/"
                "standardized_agent3_runs"
            )
            / dataset
            / f"cluster_{cluster_id}"
        )
    )

    cluster_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    request_dir = (
        cluster_root
        / "request"
    )

    context_dir = (
        cluster_root
        / "context_retrieval"
    )

    pubmedbert_dir = (
        cluster_root
        / "pubmedbert"
    )

    audit_dir = (
        cluster_root
        / "pubmedbert_audit"
    )

    coverage_dir = (
        cluster_root
        / "rank_coverage"
    )

    rescue_dir = (
        cluster_root
        / "marker_rescue"
    )

    rescue_retrieval_dir = (
        rescue_dir
        / "retrieval"
    )

    discovery_dir = (
        rescue_dir
        / "candidate_discovery"
    )

    confirmation_dir = (
        rescue_dir
        / "confirmation"
    )

    confirmation_retrieval_dir = (
        confirmation_dir
        / "retrieval"
    )

    grounding_dir = (
        confirmation_dir
        / "grounding"
    )

    confirmation_pubmedbert_dir = (
        confirmation_dir
        / "pubmedbert"
    )

    provenance_dir = (
        rescue_dir
        / "provenance"
    )

    rescue_packet_dir = (
        rescue_dir
        / "final_evidence_packet"
    )

    result_dir = (
        cluster_root
        / "result"
    )

    request_path = (
        request_dir
        / f"cluster_{cluster_id}_literature_request.json"
    )

    request = build_request(
        agent2_json=agent2_json,
        disease_context=args.disease_context,
        candidate_count=args.candidate_count,
        output_path=request_path,
    )

    print()
    print("=" * 100)
    print("STANDARDIZED AGENT 3 EXECUTION PLAN")
    print("=" * 100)
    print("Dataset:", dataset)
    print("Cluster:", cluster_id)
    print("Species:", request.species)
    print("Tissue:", request.tissue)
    print(
        "Disease:",
        request.disease_context,
    )
    print("Platform:", request.platform)
    print(
        "Marker count:",
        len(request.marker_genes),
    )
    print(
        "Candidate count:",
        len(
            request.candidate_annotations
        ),
    )

    for candidate in (
        request.candidate_annotations
    ):
        print(
            f"  {candidate.upstream_rank}. "
            f"{candidate.name}"
        )

    print()
    print(
        "Agent 2 source:",
        request.source_agent2_file,
    )

    print()
    print("STANDARDIZED STAGES:")
    print(" 01 request")
    print(" 02 fresh context retrieval")
    print(" 03 paper ranking")
    print(" 04 evidence corpus")
    print(" 05 PubMedBERT reranking")
    print(" 06 PubMedBERT evidence audit")
    print(" 07 rank-aware evidence coverage")
    print(" 08 production marker-rescue decision")
    print(" 09 conditional rescue retrieval")
    print(" 10 ontology discovery")
    print(" 11 confirmation query generation")
    print(" 12 fresh confirmation retrieval")
    print(" 13 NCBI gene terminology")
    print(" 14 terminology-aware grounding")
    print(" 15 confirmation PubMedBERT")
    print(" 16 provenance")
    print(" 17 standardized evidence result")
    print(" 18 freeze audit")

    if args.plan_only:
        print()
        print("=" * 100)
        print("PLAN-ONLY VALIDATION: PASS")
        print("No literature retrieval was executed.")
        print("=" * 100)
        return

    #
    # 02 Context retrieval
    #
    run_command(
        "02 FRESH CONTEXT RETRIEVAL",
        [
            py,
            "-m",
            "literature_retrieval_agent.runners.run_context_retrieval",
            "--cluster-id",
            cluster_id,
            "--request-json",
            str(request_path),
            "--output-directory",
            str(context_dir),
            "--page-size",
            str(
                args.context_page_size
            ),
        ],
    )

    context_request = find_one(
        context_dir,
        f"cluster_{cluster_id}_context_v2_literature_request.json",
    )

    context_pilot = find_one(
        context_dir,
        f"cluster_{cluster_id}_context_v2_retrieval_pilot.json",
    )

    #
    # 03 Paper ranking
    #
    run_command(
        "03 PAPER RANKING",
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

    ranked_papers = find_one(
        context_dir,
        f"cluster_{cluster_id}_ranked_papers.json",
    )

    #
    # 04 Evidence corpus
    #
    run_command(
        "04 EVIDENCE CORPUS",
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
            "medium",
        ],
    )

    evidence_corpus = find_one(
        context_dir,
        f"cluster_{cluster_id}_evidence_corpus.json",
    )

    #
    # 05 Initial PubMedBERT
    #
    run_command(
        "05 PUBMEDBERT RERANKING",
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
            "--model-id",
            PUBMEDBERT_MODEL,
            "--candidate-count",
            str(args.candidate_count),
            "--top-k-passages",
            str(args.top_k_passages),
            "--top-marker-count",
            str(args.top_marker_count),
        ],
    )

    pubmedbert_json = find_one(
        pubmedbert_dir,
        f"cluster_{cluster_id}_pubmedbert_results.json",
    )

    #
    # 06 PubMedBERT evidence audit
    #
    run_command(
        "06 PUBMEDBERT EVIDENCE AUDIT",
        [
            py,
            "scripts/agent3_validation/audit_pubmedbert_evidence.py",
            "--pubmedbert-json",
            str(pubmedbert_json),
            "--output-directory",
            str(audit_dir),
        ],
    )

    audit_json = find_one(
        audit_dir,
        f"cluster_{cluster_id}_pubmedbert_evidence_audit.json",
    )

    audit_passages = find_one(
        audit_dir,
        f"cluster_{cluster_id}_pubmedbert_evidence_passages.csv",
    )

    #
    # 07 Rank-aware coverage
    #
    run_command(
        "07 RANK-AWARE EVIDENCE COVERAGE",
        [
            py,
            "scripts/agent3_validation/audit_pubmedbert_rank_coverage.py",
            "--literature-request",
            str(context_request),
            "--evidence-passages-csv",
            str(audit_passages),
            "--output-directory",
            str(coverage_dir),
        ],
    )

    coverage_json = find_one(
        coverage_dir,
        f"cluster_{cluster_id}_pubmedbert_rank_coverage.json",
    )

    #
    # 08 Production marker rescue plan
    #
    rescue_plan_path = (
        rescue_dir
        / f"cluster_{cluster_id}_marker_rescue_query_plan.json"
    )

    rescue_plan = build_rescue_plan(
        literature_request_path=context_request,
        pubmedbert_evidence_passages=audit_passages,
        output_path=rescue_plan_path,
        top_marker_limit=args.rescue_top_marker_limit,
    )

    rescue_markers = (
        rescue_plan.get(
            "rescue_markers",
            [],
        )
        or []
    )

    rescue_triggered = bool(
        rescue_markers
    )

    print()
    print("=" * 100)
    print("08 MARKER-RESCUE DECISION")
    print("=" * 100)
    print(
        "Top markers:",
        ", ".join(
            str(item.get("gene"))
            for item in rescue_plan.get(
                "top_markers",
                [],
            )
        )
        or "none",
    )

    print(
        "Already explained:",
        ", ".join(
            rescue_plan.get(
                "already_explained_top_markers",
                [],
            )
        )
        or "none",
    )

    print(
        "Rescue markers:",
        ", ".join(
            str(item.get("gene"))
            for item in rescue_markers
        )
        or "none",
    )

    print(
        "Rescue triggered:",
        rescue_triggered,
    )

    rescue_packet_path = None

    if rescue_triggered:

        #
        # 09 Rescue retrieval
        #
        run_command(
            "09 CANDIDATE-INDEPENDENT RESCUE RETRIEVAL",
            [
                py,
                "scripts/agent3_validation/run_marker_rescue_retrieval.py",
                "--literature-request",
                str(context_request),
                "--rescue-plan",
                str(rescue_plan_path),
                "--output-directory",
                str(
                    rescue_retrieval_dir
                ),
                "--page-size",
                str(
                    args.rescue_page_size
                ),
            ],
        )

        rescue_pilot = find_one(
            rescue_retrieval_dir,
            f"cluster_{cluster_id}_retrieval_pilot.json",
        )

        #
        # 10 Ontology discovery
        #
        run_command(
            "10 ONTOLOGY-GROUNDED CANDIDATE DISCOVERY",
            [
                py,
                "scripts/agent3_validation/discover_rescue_candidates.py",
                "--retrieval-pilot",
                str(rescue_pilot),
                "--cell-ontology",
                str(args.cell_ontology),
                "--literature-request",
                str(context_request),
                "--output-directory",
                str(discovery_dir),
            ],
        )

        discovery_json = find_one(
            discovery_dir,
            f"cluster_{cluster_id}_rescue_candidate_discovery.json",
        )

        #
        # 11 Confirmation queries
        #
        confirmation_plan = (
            confirmation_dir
            / (
                f"cluster_{cluster_id}_"
                "rescue_confirmation_query_plan.json"
            )
        )

        run_command(
            "11 TARGET-CONTEXT CONFIRMATION QUERY PLAN",
            [
                py,
                "scripts/agent3_validation/build_rescue_confirmation_queries.py",
                "--candidate-discovery",
                str(discovery_json),
                "--rescue-plan",
                str(rescue_plan_path),
                "--literature-request",
                str(context_request),
                "--output-json",
                str(confirmation_plan),
            ],
        )

        #
        # 12 Confirmation retrieval
        #
        run_command(
            "12 FRESH TARGET-CONTEXT CONFIRMATION RETRIEVAL",
            [
                py,
                "scripts/agent3_validation/run_rescue_confirmation_retrieval.py",
                "--literature-request",
                str(context_request),
                "--confirmation-plan",
                str(confirmation_plan),
                "--output-directory",
                str(
                    confirmation_retrieval_dir
                ),
                "--page-size",
                str(
                    args.confirmation_page_size
                ),
            ],
        )

        confirmation_pilot = find_one(
            confirmation_retrieval_dir,
            f"cluster_{cluster_id}_retrieval_pilot.json",
        )

        #
        # 13 Gene terminology
        #
        terminology_path = (
            confirmation_dir
            / (
                f"cluster_{cluster_id}_"
                "rescue_gene_terminology.json"
            )
        )

        terminology = (
            build_gene_terminology(
                rescue_plan_path=
                    rescue_plan_path,
                literature_request_path=
                    context_request,
                output_path=
                    terminology_path,
            )
        )

        print()
        print("=" * 100)
        print("13 NCBI GENE TERMINOLOGY")
        print("=" * 100)
        print(
            "Organism:",
            terminology.get(
                "organism"
            ),
        )
        print(
            "Marker count:",
            terminology.get(
                "marker_count"
            ),
        )

        #
        # 14 Grounding
        #
        run_command(
            "14 TERMINOLOGY-AWARE GROUNDING",
            [
                py,
                "-m",
                "literature_retrieval_agent.rescue_confirmation_grounding",
                "--pilot-json",
                str(confirmation_pilot),
                "--confirmation-plan",
                str(confirmation_plan),
                "--literature-request",
                str(context_request),
                "--gene-terminology",
                str(terminology_path),
                "--output-directory",
                str(grounding_dir),
            ],
        )

        grounding_json = find_one(
            grounding_dir,
            f"cluster_{cluster_id}_rescue_grounding_audit.json",
        )

        #
        # 15 Confirmation PubMedBERT
        #
        run_command(
            "15 CONFIRMATION PUBMEDBERT",
            [
                py,
                "scripts/agent3_validation/run_rescue_confirmation_pubmedbert.py",
                "--pilot-json",
                str(confirmation_pilot),
                "--confirmation-plan",
                str(confirmation_plan),
                "--literature-request",
                str(context_request),
                "--output-directory",
                str(
                    confirmation_pubmedbert_dir
                ),
                "--model-id",
                PUBMEDBERT_MODEL,
                "--top-marker-count",
                str(
                    args.top_marker_count
                ),
                "--top-k-passages",
                str(
                    args.top_k_passages
                ),
            ],
        )

        confirmation_pubmedbert = (
            find_one(
                confirmation_pubmedbert_dir,
                (
                    f"cluster_{cluster_id}_"
                    "rescue_confirmation_pubmedbert_results.json"
                ),
            )
        )

        #
        # 16 Provenance
        #
        run_command(
            "16 CANDIDATE PROVENANCE",
            [
                py,
                "-m",
                "literature_retrieval_agent.rescue_candidate_provenance",
                "--candidate-discovery",
                str(discovery_json),
                "--retrieval-pilot",
                str(rescue_pilot),
                "--rescue-plan",
                str(rescue_plan_path),
                "--literature-request",
                str(context_request),
                "--output-directory",
                str(provenance_dir),
            ],
        )

        provenance_json = find_one(
            provenance_dir,
            f"cluster_{cluster_id}_rescue_candidate_provenance_audit.json",
        )

        #
        # Final rescue evidence packet
        #
        run_command(
            "17 RESCUE EVIDENCE PACKET",
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
                str(
                    confirmation_pubmedbert
                ),
                "--output-directory",
                str(rescue_packet_dir),
            ],
        )

        rescue_packet_path = find_one(
            rescue_packet_dir,
            f"cluster_{cluster_id}_rescue_evidence_packet.json",
        )

    #
    # Standard result exists whether rescue
    # ran or not.
    #
    result_path = (
        result_dir
        / (
            f"cluster_{cluster_id}_"
            "agent3_standardized_result.json"
        )
    )

    result = {
        "schema_name":
            "Agent3ClusterEvidenceResult",

        "schema_version":
            "1.0",

        "dataset":
            dataset,

        "cluster_id":
            cluster_id,

        "species":
            request.species,

        "tissue":
            request.tissue,

        "disease_context":
            request.disease_context,

        "platform":
            request.platform,

        "source_agent2_file":
            request.source_agent2_file,

        "candidate_count":
            len(
                request.candidate_annotations
            ),

        "candidate_annotations": [
            {
                "name":
                    candidate.name,

                "upstream_rank":
                    candidate.upstream_rank,

                "supporting_genes":
                    candidate.supporting_genes,

                "evidence_sources":
                    candidate.evidence_sources,
            }
            for candidate
            in request.candidate_annotations
        ],

        "initial_pubmedbert_results":
            str(pubmedbert_json),

        "initial_evidence_audit":
            str(audit_json),

        "rank_coverage":
            str(coverage_json),

        "rescue_trigger_rule": (
            "Development-derived operational rule: "
            "run candidate-independent rescue when "
            "one or more markers in the predefined "
            "top-marker window are not represented "
            "in candidate-associated PubMedBERT "
            "evidence. This rule is not independently "
            "calibrated."
        ),

        "rescue_top_marker_limit":
            args.rescue_top_marker_limit,

        "rescue_triggered":
            rescue_triggered,

        "rescue_markers":
            rescue_markers,

        "rescue_evidence_packet": (
            str(rescue_packet_path)
            if rescue_packet_path
            else None
        ),

        "reference_label_used":
            False,

        "expert_label_used":
            False,

        "automatic_acceptance":
            "NOT_CALIBRATED",

        "final_annotation":
            None,

        "pubmedbert_interpretation": (
            "PubMedBERT values are semantic "
            "retrieval relevance measures only; "
            "they are not annotation confidence, "
            "probability, or biological correctness."
        ),
    }

    write_json(
        result_path,
        result,
    )

    manifest_path = (
        write_freeze_manifest(
            cluster_root=
                cluster_root,
            result_path=
                result_path,
            request_path=
                request_path,
            dataset=
                dataset,
            cluster_id=
                cluster_id,
        )
    )

    print()
    print("=" * 100)
    print("STANDARDIZED AGENT 3 COMPLETE")
    print("=" * 100)
    print(
        "Dataset:",
        dataset,
    )
    print(
        "Cluster:",
        cluster_id,
    )
    print(
        "Rescue triggered:",
        rescue_triggered,
    )
    print(
        "Automatic acceptance:",
        "NOT_CALIBRATED",
    )
    print(
        "Final annotation:",
        None,
    )
    print(
        "Result:",
        result_path,
    )
    print(
        "Freeze manifest:",
        manifest_path,
    )
    print("=" * 100)


if __name__ == "__main__":
    main()








