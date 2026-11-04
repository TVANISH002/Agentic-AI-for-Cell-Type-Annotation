from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(".").resolve()
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from literature_retrieval_agent.agent3_decision_summary import (
    build_decision_summary,
    write_decision_outputs,
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def run(cmd: list[str], allow_fail: bool = False) -> int:
    print()
    print("RUNNING:")
    print(" ".join(cmd))
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0 and not allow_fail:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result.returncode


def default_cluster_root(dataset: str, cluster_id: str) -> Path:
    return (
        ROOT
        / "data"
        / "analysis"
        / "agent3_validation"
        / "standardized_agent3_runs"
        / dataset
        / f"cluster_{cluster_id}"
    )


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def resolve_existing_path(value: str | None, cluster_root: Path) -> Path | None:
    if not value:
        return None

    candidate = Path(value)

    if candidate.exists():
        return candidate

    rooted = ROOT / candidate
    if rooted.exists():
        return rooted

    cluster_relative = cluster_root / candidate
    if cluster_relative.exists():
        return cluster_relative

    return None


def write_initial_evidence_packet_for_decision(
    *,
    standardized_result_json: Path,
    cluster_root: Path,
    cluster_id: str,
) -> Path:
    result = load_json(standardized_result_json)

    audit_path = resolve_existing_path(
        result.get("initial_evidence_audit"),
        cluster_root,
    )

    if audit_path is None:
        audit_path = (
            cluster_root
            / "pubmedbert_audit"
            / f"cluster_{cluster_id}_pubmedbert_evidence_audit.json"
        )

    audit_path = require(audit_path)
    audit = load_json(audit_path)

    packet = {
        "schema_name": "Agent3InitialEvidenceAudit",
        "schema_version": "1.0",
        "cluster_id": str(cluster_id),
        "source_pubmedbert_evidence_audit": str(audit_path),
        "rescue_triggered": False,
        "decision_input_note": (
            "Marker rescue was not triggered because high-ranked markers were "
            "adequately explained by upstream/literature/HPA-supported evidence. "
            "Decision summary was therefore generated from the initial Agent 3 "
            "evidence audit rather than from a rescue evidence packet."
        ),
        "candidates": audit.get("results", []) or [],
    }

    output_path = (
        cluster_root
        / "result"
        / f"cluster_{cluster_id}_initial_evidence_packet_for_decision.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(packet, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return output_path


def run_recovery_packaging(cluster_root: Path, cluster_id: str) -> Path:
    marker_root = cluster_root / "marker_rescue"

    discovery_json = require(
        marker_root
        / "candidate_discovery"
        / f"cluster_{cluster_id}_rescue_candidate_discovery.json"
    )

    retrieval_pilot = require(
        marker_root
        / "retrieval"
        / f"cluster_{cluster_id}_retrieval_pilot.json"
    )

    rescue_plan = require(
        marker_root
        / f"cluster_{cluster_id}_marker_rescue_query_plan.json"
    )

    literature_request = require(
        marker_root
        / "retrieval"
        / f"cluster_{cluster_id}_literature_request.json"
    )

    grounding_json = require(
        marker_root
        / "confirmation"
        / "grounding"
        / f"cluster_{cluster_id}_rescue_grounding_audit.json"
    )

    pubmedbert_json = require(
        marker_root
        / "confirmation"
        / "pubmedbert"
        / f"cluster_{cluster_id}_rescue_confirmation_pubmedbert_results.json"
    )

    provenance_dir = marker_root / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)

    provenance_json = (
        provenance_dir
        / f"cluster_{cluster_id}_rescue_candidate_provenance_audit.json"
    )

    if not provenance_json.exists():
        run(
            [
                sys.executable,
                "-m",
                "literature_retrieval_agent.rescue_candidate_provenance",
                "--candidate-discovery",
                str(discovery_json),
                "--retrieval-pilot",
                str(retrieval_pilot),
                "--rescue-plan",
                str(rescue_plan),
                "--literature-request",
                str(literature_request),
                "--output-directory",
                str(provenance_dir),
            ]
        )

    final_packet_dir = marker_root / "final_evidence_packet"
    final_packet_dir.mkdir(parents=True, exist_ok=True)

    packet_json = (
        final_packet_dir
        / f"cluster_{cluster_id}_rescue_evidence_packet.json"
    )

    if not packet_json.exists():
        run(
            [
                sys.executable,
                "-m",
                "literature_retrieval_agent.rescue_evidence_packet",
                "--discovery-json",
                str(discovery_json),
                "--provenance-json",
                str(provenance_json),
                "--grounding-json",
                str(grounding_json),
                "--pubmedbert-json",
                str(pubmedbert_json),
                "--output-directory",
                str(final_packet_dir),
            ]
        )

    return packet_json


def write_decision(
    *,
    packet_json: Path,
    cluster_root: Path,
    dataset: str,
    cluster_id: str,
    ontology_path: Path,
) -> Path:
    result_dir = cluster_root / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    summary_path = (
        result_dir
        / f"cluster_{cluster_id}_agent3_decision_summary.json"
    )

    summary = build_decision_summary(
        load_json(packet_json),
        dataset=dataset,
        cluster_id=cluster_id,
        ontology_path=ontology_path,
    )

    write_decision_outputs(summary, summary_path)

    print()
    print("Agent 3 decision summary added.")
    print("Decision:", summary["decision_state"])
    print("Ambiguity subtype:", summary["ambiguity_subtype"])
    print(
        "Possible labels:",
        "; ".join(
            item["label"]
            for item in summary["possible_labels"]
        ),
    )
    print("Decision JSON:", summary_path)
    print("Decision CSV:", summary_path.with_suffix(".csv"))

    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run standardized Agent 3 and always try to complete "
            "provenance, final evidence packet, and decision summary."
        )
    )

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cluster-id", required=True)
    parser.add_argument("--disease-context", required=True)
    parser.add_argument("--agent2-json", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--candidate-count", type=int, default=3)
    parser.add_argument("--context-page-size", type=int, default=3)
    parser.add_argument("--rescue-page-size", type=int, default=10)
    parser.add_argument("--confirmation-page-size", type=int, default=10)
    parser.add_argument("--top-k-passages", type=int, default=10)
    parser.add_argument("--top-marker-count", type=int, default=25)
    parser.add_argument("--rescue-top-marker-limit", type=int, default=5)
    parser.add_argument(
        "--cell-ontology",
        type=Path,
        default=Path(
            "data/resources/cell_ontology/v2026-06-08/cl-base.json"
        ),
    )

    args = parser.parse_args()

    dataset = args.dataset
    cluster_id = str(args.cluster_id)

    cluster_root = (
        args.output_root
        if args.output_root
        else default_cluster_root(dataset, cluster_id)
    )

    cmd = [
        sys.executable,
        "scripts/agent3_validation/run_agent3_standardized_full.py",
        "--dataset",
        dataset,
        "--cluster-id",
        cluster_id,
        "--disease-context",
        args.disease_context,
        "--candidate-count",
        str(args.candidate_count),
        "--context-page-size",
        str(args.context_page_size),
        "--rescue-page-size",
        str(args.rescue_page_size),
        "--confirmation-page-size",
        str(args.confirmation_page_size),
        "--top-k-passages",
        str(args.top_k_passages),
        "--top-marker-count",
        str(args.top_marker_count),
        "--rescue-top-marker-limit",
        str(args.rescue_top_marker_limit),
        "--cell-ontology",
        str(args.cell_ontology),
    ]

    if args.agent2_json:
        cmd.extend(["--agent2-json", str(args.agent2_json)])

    if args.output_root:
        cmd.extend(["--output-root", str(args.output_root)])

    print("Running standardized Agent 3...")
    status = run(cmd, allow_fail=True)

    if status != 0:
        print()
        print(
            "Standardized runner stopped before final packaging. "
            "Trying recovery packaging from completed artifacts..."
        )

        packet_json = run_recovery_packaging(
            cluster_root,
            cluster_id,
        )
    else:
        standardized_result_json = (
            cluster_root
            / "result"
            / f"cluster_{cluster_id}_agent3_standardized_result.json"
        )
        standardized_result_json = require(standardized_result_json)
        standardized_result = load_json(standardized_result_json)

        rescue_triggered = bool(
            standardized_result.get("rescue_triggered")
        )

        if rescue_triggered:
            rescue_packet = resolve_existing_path(
                standardized_result.get("rescue_evidence_packet"),
                cluster_root,
            )

            if rescue_packet is not None:
                packet_json = rescue_packet
            else:
                packet_json = run_recovery_packaging(
                    cluster_root,
                    cluster_id,
                )
        else:
            print()
            print(
                "Marker rescue was not triggered; writing decision summary "
                "from initial Agent 3 evidence audit."
            )
            packet_json = write_initial_evidence_packet_for_decision(
                standardized_result_json=standardized_result_json,
                cluster_root=cluster_root,
                cluster_id=cluster_id,
            )

    write_decision(
        packet_json=packet_json,
        cluster_root=cluster_root,
        dataset=dataset,
        cluster_id=cluster_id,
        ontology_path=args.cell_ontology,
    )

    print()
    print("STATUS: COMPLETE")


if __name__ == "__main__":
    main()

