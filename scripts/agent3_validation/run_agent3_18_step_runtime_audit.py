from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(".").resolve()
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


STEP_NAMES = {
    1: "Agent 2 handoff validation",
    2: "Literature request construction",
    3: "Context-aware Europe PMC retrieval",
    4: "Paper ranking",
    5: "Evidence-corpus construction",
    6: "PubMedBERT semantic reranking",
    7: "Evidence audit",
    8: "Rank-aware marker coverage check",
    9: "Conditional marker-rescue trigger",
    10: "Candidate-independent rescue retrieval",
    11: "Cell Ontology candidate discovery",
    12: "Target-context confirmation retrieval",
    13: "Terminology-aware grounding",
    14: "Confirmation PubMedBERT reranking",
    15: "Evidence provenance packet",
    16: "Decision summary construction",
    17: "Possible-label output",
    18: "Decision-state assignment",
}


class RuntimeAudit:
    def __init__(
        self,
        *,
        dataset: str,
        tissue: str,
        cluster_id: str,
        output_csv: Path,
    ) -> None:
        self.dataset = dataset
        self.tissue = tissue
        self.cluster_id = cluster_id
        self.output_csv = output_csv
        self.rows: list[dict[str, Any]] = []

    def add(
        self,
        *,
        step_number: int,
        status: str,
        elapsed_seconds: float,
        command: list[str] | None = None,
        output_artifact: Path | None = None,
        notes: str = "",
        queries_executed: int | None = None,
        papers_retrieved: int | None = None,
        passages_selected: int | None = None,
        rescue_triggered: bool | None = None,
        final_label: str | None = None,
        decision_state: str | None = None,
    ) -> None:
        self.rows.append(
            {
                "dataset": self.dataset,
                "tissue": self.tissue,
                "cluster_id": self.cluster_id,
                "step_number": step_number,
                "step_name": STEP_NAMES[step_number],
                "status": status,
                "elapsed_seconds": round(elapsed_seconds, 6),
                "queries_executed": queries_executed,
                "papers_retrieved": papers_retrieved,
                "passages_selected": passages_selected,
                "rescue_triggered": rescue_triggered,
                "final_label": final_label,
                "decision_state": decision_state,
                "output_artifact": str(output_artifact) if output_artifact else "",
                "command": " ".join(command) if command else "",
                "notes": notes,
            }
        )
        self.write()

    def write(self) -> None:
        self.output_csv.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = [
            "dataset",
            "tissue",
            "cluster_id",
            "step_number",
            "step_name",
            "status",
            "elapsed_seconds",
            "queries_executed",
            "papers_retrieved",
            "passages_selected",
            "rescue_triggered",
            "final_label",
            "decision_state",
            "output_artifact",
            "command",
            "notes",
        ]

        with self.output_csv.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(self.rows)


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8-sig",
        )
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
        path
        for path in root.rglob(pattern)
        if path.is_file()
    )

    if not hits:
        raise FileNotFoundError(
            f"No file found under {root} for pattern: {pattern}"
        )

    if len(hits) > 1:
        raise RuntimeError(
            f"Multiple files found for {pattern}:\n"
            + "\n".join(str(path) for path in hits)
        )

    return hits[0]


def run_timed(
    *,
    audit: RuntimeAudit,
    step_number: int,
    cmd: list[str],
    output_artifact: Path | None = None,
    notes: str = "",
    extra_metrics: dict[str, Any] | None = None,
) -> None:
    print()
    print("=" * 100)
    print(f"STEP {step_number}: {STEP_NAMES[step_number]}")
    print("=" * 100)
    print(" ".join(cmd))

    started = time.perf_counter()
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(SRC)
        if not existing_pythonpath
        else f"{SRC}{os.pathsep}{existing_pythonpath}"
    )

    try:
        subprocess.run(
            cmd,
            check=True,
            env=env,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        audit.add(
            step_number=step_number,
            status="failed",
            elapsed_seconds=elapsed,
            command=cmd,
            output_artifact=output_artifact,
            notes=f"{notes} Error: {exc}".strip(),
            **(extra_metrics or {}),
        )
        raise

    elapsed = time.perf_counter() - started
    audit.add(
        step_number=step_number,
        status="completed",
        elapsed_seconds=elapsed,
        command=cmd,
        output_artifact=output_artifact,
        notes=notes,
        **(extra_metrics or {}),
    )


def run_commands_timed(
    *,
    audit: RuntimeAudit,
    step_number: int,
    commands: list[list[str]],
    output_artifact: Path | None = None,
    notes: str = "",
    extra_metrics: dict[str, Any] | None = None,
) -> None:
    print()
    print("=" * 100)
    print(f"STEP {step_number}: {STEP_NAMES[step_number]}")
    print("=" * 100)

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(SRC)
        if not existing_pythonpath
        else f"{SRC}{os.pathsep}{existing_pythonpath}"
    )

    started = time.perf_counter()
    command_texts = []

    try:
        for cmd in commands:
            print(" ".join(cmd))
            command_texts.append(" ".join(cmd))
            subprocess.run(
                cmd,
                check=True,
                env=env,
            )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        audit.add(
            step_number=step_number,
            status="failed",
            elapsed_seconds=elapsed,
            command=[" && ".join(command_texts)],
            output_artifact=output_artifact,
            notes=f"{notes} Error: {exc}".strip(),
            **(extra_metrics or {}),
        )
        raise

    elapsed = time.perf_counter() - started
    audit.add(
        step_number=step_number,
        status="completed",
        elapsed_seconds=elapsed,
        command=[" && ".join(command_texts)],
        output_artifact=output_artifact,
        notes=notes,
        **(extra_metrics or {}),
    )


def count_context_metrics(pilot_json: Path) -> dict[str, int]:
    data = load_json(pilot_json)
    papers = data.get("papers") or data.get("retrieved_papers") or []
    queries = data.get("queries") or data.get("query_results") or []

    return {
        "queries_executed": int(
            data.get("queries_executed")
            or data.get("query_count")
            or len(queries)
            or 0
        ),
        "papers_retrieved": int(
            data.get("unique_papers_retrieved")
            or data.get("unique_paper_count")
            or len(papers)
            or 0
        ),
    }


def count_passages_from_csv(path: Path) -> int:
    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    return max(
        0,
        len(rows) - 1,
    )


def validate_handoff(
    *,
    marker_file: Path,
    candidate_file: Path,
    cluster_id: str,
) -> str:
    import pandas as pd

    markers = pd.read_csv(marker_file)
    candidates = pd.read_csv(candidate_file)

    marker_cluster_col = (
        "cluster_id"
        if "cluster_id" in markers.columns
        else "cluster"
        if "cluster" in markers.columns
        else "spatial_cluster"
        if "spatial_cluster" in markers.columns
        else None
    )

    if marker_cluster_col is None:
        raise ValueError(
            f"Marker file has no cluster column: {marker_file}"
        )

    marker_rows = markers[
        markers[marker_cluster_col].astype(str) == str(cluster_id)
    ]

    if marker_rows.empty:
        raise ValueError(
            f"No marker rows found for cluster {cluster_id}: {marker_file}"
        )

    if "cluster_id" in candidates.columns:
        candidate_rows = candidates[
            candidates["cluster_id"].astype(str) == str(cluster_id)
        ]
    elif "cluster" in candidates.columns:
        candidate_rows = candidates[
            candidates["cluster"].astype(str) == str(cluster_id)
        ]
    else:
        candidate_rows = candidates

    if candidate_rows.empty:
        raise ValueError(
            f"No candidate rows found for cluster {cluster_id}: {candidate_file}"
        )

    return (
        f"validated marker_rows={len(marker_rows)}; "
        f"candidate_rows={len(candidate_rows)}"
    )


def build_decision_outputs(
    *,
    packet_json: Path,
    dataset: str,
    cluster_id: str,
    ontology_path: Path,
    result_dir: Path,
) -> Path:
    from literature_retrieval_agent.agent3_decision_summary import (
        build_decision_summary,
        write_decision_outputs,
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = build_decision_summary(
        load_json(packet_json),
        dataset=dataset,
        cluster_id=cluster_id,
        ontology_path=ontology_path,
    )

    output_json = (
        result_dir
        / f"cluster_{cluster_id}_agent3_decision_summary.json"
    )

    write_decision_outputs(
        summary,
        output_json,
    )

    return output_json


def write_possible_labels(
    *,
    decision_json: Path,
    output_csv: Path,
) -> tuple[str, str]:
    summary = load_json(decision_json)
    labels = summary.get("possible_labels") or []

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "rank",
        "label",
        "source",
        "evidence_state",
    ]

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for index, item in enumerate(
            labels,
            start=1,
        ):
            writer.writerow(
                {
                    "rank": index,
                    "label": item.get("label", ""),
                    "source": item.get("source", ""),
                    "evidence_state": item.get("evidence_state", ""),
                }
            )

    final_label = "; ".join(
        str(item.get("label", ""))
        for item in labels
        if item.get("label")
    )

    return final_label, str(summary.get("decision_state", ""))


def write_decision_state(
    *,
    decision_json: Path,
    output_json: Path,
) -> str:
    summary = load_json(decision_json)
    decision_state = str(
        summary.get("decision_state", "")
    )

    write_json(
        output_json,
        {
            "cluster_id": summary.get("cluster_id"),
            "decision_state": decision_state,
            "ambiguity_subtype": summary.get("ambiguity_subtype"),
            "possible_label_count": summary.get("possible_label_count"),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    return decision_state


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Agent 3 and write an exactly 18-row step-level runtime audit."
        )
    )

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cluster-id", required=True)
    parser.add_argument("--species", required=True)
    parser.add_argument("--tissue", required=True)
    parser.add_argument("--disease-context", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--marker-file", required=True, type=Path)
    parser.add_argument("--candidate-file", required=True, type=Path)
    parser.add_argument("--router-file", type=Path)
    parser.add_argument("--cluster-root", required=True, type=Path)
    parser.add_argument(
        "--cell-ontology",
        type=Path,
        default=Path("data/resources/cell_ontology/v2026-06-08/cl-base.json"),
    )
    parser.add_argument("--candidate-count", type=int, default=0)
    parser.add_argument("--max-markers", type=int, default=50)
    parser.add_argument("--context-page-size", type=int, default=3)
    parser.add_argument("--rescue-page-size", type=int, default=10)
    parser.add_argument("--confirmation-page-size", type=int, default=10)
    parser.add_argument("--pubmedbert-candidate-count", type=int, default=3)
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
    cluster_root = args.cluster_root
    cluster_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    runtime_dir = cluster_root / "runtime_audit"
    runtime_csv = (
        runtime_dir
        / f"cluster_{cluster_id}_agent3_18_step_runtime.csv"
    )

    audit = RuntimeAudit(
        dataset=args.dataset,
        tissue=args.tissue,
        cluster_id=cluster_id,
        output_csv=runtime_csv,
    )

    request_dir = cluster_root / "request"
    context_dir = cluster_root / "context_retrieval"
    pubmedbert_dir = cluster_root / "pubmedbert"
    pubmedbert_audit_dir = cluster_root / "pubmedbert_audit"
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
    result_dir = cluster_root / "result"

    request_json = (
        request_dir
        / f"cluster_{cluster_id}_literature_request.json"
    )

    started = time.perf_counter()
    note = validate_handoff(
        marker_file=args.marker_file,
        candidate_file=args.candidate_file,
        cluster_id=cluster_id,
    )
    audit.add(
        step_number=1,
        status="completed",
        elapsed_seconds=time.perf_counter() - started,
        notes=note,
    )

    request_cmd = [
        py,
        "scripts/agent3_validation/build_agent3_literature_request_from_agent2.py",
        "--dataset",
        args.dataset,
        "--cluster-id",
        cluster_id,
        "--species",
        args.species,
        "--tissue",
        args.tissue,
        "--disease-context",
        args.disease_context,
        "--platform",
        args.platform,
        "--marker-file",
        str(args.marker_file),
        "--candidate-file",
        str(args.candidate_file),
        "--candidate-count",
        str(args.candidate_count),
        "--max-markers",
        str(args.max_markers),
        "--output-json",
        str(request_json),
    ]

    if args.router_file:
        request_cmd.extend(
            [
                "--router-file",
                str(args.router_file),
            ]
        )

    run_timed(
        audit=audit,
        step_number=2,
        cmd=request_cmd,
        output_artifact=request_json,
    )

    run_timed(
        audit=audit,
        step_number=3,
        cmd=[
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
        output_artifact=context_dir,
    )

    literature_request = find_required(
        context_dir,
        f"cluster_{cluster_id}_context_v2_literature_request.json",
    )
    context_pilot = find_required(
        context_dir,
        f"cluster_{cluster_id}_context_v2_retrieval_pilot.json",
    )

    context_metrics = count_context_metrics(
        context_pilot
    )
    audit.rows[-1].update(context_metrics)
    audit.write()

    run_timed(
        audit=audit,
        step_number=4,
        cmd=[
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
        output_artifact=context_dir,
    )

    ranked_papers = find_required(
        context_dir,
        f"cluster_{cluster_id}_ranked_papers.json",
    )

    run_timed(
        audit=audit,
        step_number=5,
        cmd=[
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
        output_artifact=context_dir,
    )

    evidence_corpus = find_required(
        context_dir,
        f"cluster_{cluster_id}_evidence_corpus.json",
    )

    run_timed(
        audit=audit,
        step_number=6,
        cmd=[
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
            str(args.pubmedbert_candidate_count),
            "--top-k-passages",
            str(args.top_k_passages),
            "--top-marker-count",
            str(args.top_marker_count),
        ],
        output_artifact=pubmedbert_dir,
    )

    pubmedbert_json = find_required(
        pubmedbert_dir,
        f"cluster_{cluster_id}_pubmedbert_results.json",
    )

    run_timed(
        audit=audit,
        step_number=7,
        cmd=[
            py,
            "scripts/agent3_validation/audit_pubmedbert_evidence.py",
            "--pubmedbert-json",
            str(pubmedbert_json),
            "--output-directory",
            str(pubmedbert_audit_dir),
        ],
        output_artifact=pubmedbert_audit_dir,
    )

    pubmedbert_passages_csv = find_required(
        pubmedbert_audit_dir,
        f"cluster_{cluster_id}_pubmedbert_evidence_passages.csv",
    )
    passages_selected = count_passages_from_csv(
        pubmedbert_passages_csv
    )
    audit.rows[-1]["passages_selected"] = passages_selected
    audit.write()

    run_timed(
        audit=audit,
        step_number=8,
        cmd=[
            py,
            "scripts/agent3_validation/audit_pubmedbert_rank_coverage.py",
            "--literature-request",
            str(literature_request),
            "--evidence-passages-csv",
            str(pubmedbert_passages_csv),
            "--output-directory",
            str(rank_coverage_dir),
        ],
        output_artifact=rank_coverage_dir,
    )

    run_timed(
        audit=audit,
        step_number=9,
        cmd=[
            py,
            "scripts/agent3_validation/build_marker_rescue_queries.py",
            "--literature-request",
            str(literature_request),
            "--evidence-passages-csv",
            str(pubmedbert_passages_csv),
            "--output-directory",
            str(rescue_dir),
            "--top-marker-limit",
            str(args.rescue_top_marker_limit),
        ],
        output_artifact=rescue_dir,
    )

    rescue_plan = find_required(
        rescue_dir,
        f"cluster_{cluster_id}_marker_rescue_query_plan.json",
    )
    rescue_payload = load_json(
        rescue_plan
    )
    rescue_markers = (
        rescue_payload.get("rescue_markers")
        or rescue_payload.get("unexplained_markers")
        or []
    )
    rescue_triggered = bool(rescue_markers)
    audit.rows[-1]["rescue_triggered"] = rescue_triggered
    audit.write()

    if not rescue_triggered:
        for step in range(10, 16):
            audit.add(
                step_number=step,
                status="skipped",
                elapsed_seconds=0.0,
                rescue_triggered=False,
                notes="Marker rescue was not triggered by Step 9.",
            )
        print("Marker rescue was not triggered; stopping before final decision packet.")
        print("Runtime CSV:", runtime_csv)
        return

    run_timed(
        audit=audit,
        step_number=10,
        cmd=[
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
        output_artifact=rescue_retrieval_dir,
        extra_metrics={"rescue_triggered": True},
    )

    rescue_pilot = find_required(
        rescue_retrieval_dir,
        f"cluster_{cluster_id}_retrieval_pilot.json",
    )
    rescue_metrics = count_context_metrics(
        rescue_pilot
    )
    audit.rows[-1].update(rescue_metrics)
    audit.write()

    run_timed(
        audit=audit,
        step_number=11,
        cmd=[
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
        output_artifact=discovery_dir,
        extra_metrics={"rescue_triggered": True},
    )

    discovery_json = find_required(
        discovery_dir,
        f"cluster_{cluster_id}_rescue_candidate_discovery.json",
    )
    confirmation_plan = (
        confirmation_dir
        / f"cluster_{cluster_id}_rescue_confirmation_query_plan.json"
    )

    run_commands_timed(
        audit=audit,
        step_number=12,
        commands=[
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
        ],
        output_artifact=confirmation_retrieval_dir,
        notes="Includes confirmation query-plan construction and retrieval.",
        extra_metrics={"rescue_triggered": True},
    )

    confirmation_pilot = find_required(
        confirmation_retrieval_dir,
        f"cluster_{cluster_id}_retrieval_pilot.json",
    )
    confirmation_metrics = count_context_metrics(
        confirmation_pilot
    )
    audit.rows[-1].update(confirmation_metrics)
    audit.write()

    gene_terminology = (
        confirmation_dir
        / f"cluster_{cluster_id}_rescue_gene_terminology.json"
    )

    run_commands_timed(
        audit=audit,
        step_number=13,
        commands=[
            [
                py,
                "scripts/agent3_validation/build_rescue_gene_terminology.py",
                "--rescue-plan",
                str(rescue_plan),
                "--output-json",
                str(gene_terminology),
            ],
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
        ],
        output_artifact=grounding_dir,
        notes="Includes gene terminology retrieval and grounding execution.",
        extra_metrics={"rescue_triggered": True},
    )

    grounding_json = find_required(
        grounding_dir,
        f"cluster_{cluster_id}_rescue_grounding_audit.json",
    )

    run_timed(
        audit=audit,
        step_number=14,
        cmd=[
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
        output_artifact=confirmation_pubmedbert_dir,
        extra_metrics={"rescue_triggered": True},
    )

    confirmation_pubmedbert = find_required(
        confirmation_pubmedbert_dir,
        f"cluster_{cluster_id}_rescue_confirmation_pubmedbert.json",
    )

    run_commands_timed(
        audit=audit,
        step_number=15,
        commands=[
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
            [
                py,
                "-m",
                "literature_retrieval_agent.rescue_evidence_packet",
                "--discovery-json",
                str(discovery_json),
                "--provenance-json",
                str(
                    provenance_dir
                    / f"cluster_{cluster_id}_rescue_candidate_provenance_audit.json"
                ),
                "--grounding-json",
                str(grounding_json),
                "--pubmedbert-json",
                str(confirmation_pubmedbert),
                "--output-directory",
                str(packet_dir),
            ],
        ],
        output_artifact=packet_dir,
        notes="Includes candidate provenance audit and final evidence-packet construction.",
        extra_metrics={"rescue_triggered": True},
    )

    final_packet = find_required(
        packet_dir,
        f"cluster_{cluster_id}_rescue_evidence_packet.json",
    )

    started = time.perf_counter()
    decision_json = build_decision_outputs(
        packet_json=final_packet,
        dataset=args.dataset,
        cluster_id=cluster_id,
        ontology_path=args.cell_ontology,
        result_dir=result_dir,
    )
    audit.add(
        step_number=16,
        status="completed",
        elapsed_seconds=time.perf_counter() - started,
        output_artifact=decision_json,
        rescue_triggered=True,
    )

    possible_labels_csv = (
        result_dir
        / f"cluster_{cluster_id}_possible_labels.csv"
    )
    started = time.perf_counter()
    final_label, decision_state = write_possible_labels(
        decision_json=decision_json,
        output_csv=possible_labels_csv,
    )
    audit.add(
        step_number=17,
        status="completed",
        elapsed_seconds=time.perf_counter() - started,
        output_artifact=possible_labels_csv,
        rescue_triggered=True,
        final_label=final_label,
        decision_state=decision_state,
    )

    decision_state_json = (
        result_dir
        / f"cluster_{cluster_id}_decision_state.json"
    )
    started = time.perf_counter()
    decision_state = write_decision_state(
        decision_json=decision_json,
        output_json=decision_state_json,
    )
    audit.add(
        step_number=18,
        status="completed",
        elapsed_seconds=time.perf_counter() - started,
        output_artifact=decision_state_json,
        rescue_triggered=True,
        final_label=final_label,
        decision_state=decision_state,
    )

    print()
    print("=" * 100)
    print("AGENT 3 18-STEP RUNTIME AUDIT COMPLETE")
    print("=" * 100)
    print("Runtime CSV:", runtime_csv)
    print("Decision JSON:", decision_json)
    print("Possible labels CSV:", possible_labels_csv)
    print("Decision state JSON:", decision_state_json)
    print("=" * 100)


if __name__ == "__main__":
    main()
