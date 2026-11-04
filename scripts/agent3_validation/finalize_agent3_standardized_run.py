from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from literature_retrieval_agent.schemas import LiteratureRequest


def load_json(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8-sig")
    )


def write_json(path: Path, obj):
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


def file_record(path: Path):
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Finalize an already-completed standardized "
            "Agent 3 run without repeating literature retrieval."
        )
    )

    ap.add_argument(
        "--dataset",
        required=True,
    )
    ap.add_argument(
        "--cluster-id",
        required=True,
    )
    ap.add_argument(
        "--cluster-root",
        required=True,
        type=Path,
    )

    args = ap.parse_args()

    root = args.cluster_root
    cluster_id = str(args.cluster_id)
    dataset = args.dataset

    request_path = require(
        root
        / "request"
        / f"cluster_{cluster_id}_literature_request.json"
    )

    pubmedbert_path = require(
        root
        / "pubmedbert"
        / f"cluster_{cluster_id}_pubmedbert_results.json"
    )

    audit_path = require(
        root
        / "pubmedbert_audit"
        / f"cluster_{cluster_id}_pubmedbert_evidence_audit.json"
    )

    coverage_path = require(
        root
        / "rank_coverage"
        / f"cluster_{cluster_id}_pubmedbert_rank_coverage.json"
    )

    rescue_plan_path = require(
        root
        / "marker_rescue"
        / f"cluster_{cluster_id}_marker_rescue_query_plan.json"
    )

    rescue_packet_path = require(
        root
        / "marker_rescue"
        / "final_evidence_packet"
        / f"cluster_{cluster_id}_rescue_evidence_packet.json"
    )

    request = LiteratureRequest.model_validate_json(
        request_path.read_text(
            encoding="utf-8-sig"
        )
    )

    rescue_plan = load_json(
        rescue_plan_path
    )

    rescue_packet = load_json(
        rescue_packet_path
    )

    rescue_markers = (
        rescue_plan.get(
            "rescue_markers",
            [],
        )
        or []
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
                    c.name,

                "upstream_rank":
                    c.upstream_rank,

                "supporting_genes":
                    c.supporting_genes,

                "evidence_sources":
                    c.evidence_sources,
            }
            for c in request.candidate_annotations
        ],

        "initial_pubmedbert_results":
            str(pubmedbert_path),

        "initial_evidence_audit":
            str(audit_path),

        "rank_coverage":
            str(coverage_path),

        "rescue_trigger_rule": (
            "Development-derived operational rule: "
            "candidate-independent marker rescue is "
            "executed when one or more markers in the "
            "predefined top-marker window remain "
            "unexplained by candidate-associated "
            "literature evidence. The rule is not "
            "independently calibrated."
        ),

        "rescue_top_marker_limit":
            rescue_plan.get(
                "top_marker_limit"
            ),

        "rescue_triggered":
            bool(rescue_markers),

        "rescue_markers":
            rescue_markers,

        "rescue_evidence_packet":
            str(rescue_packet_path),

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
            "they are not biological confidence, "
            "annotation probability, or correctness."
        ),
    }

    result_dir = (
        root
        / "result"
    )

    result_path = (
        result_dir
        / f"cluster_{cluster_id}_agent3_standardized_result.json"
    )

    write_json(
        result_path,
        result,
    )

    packet_candidates = (
        rescue_packet.get(
            "candidates",
            [],
        )
        or []
    )

    invariants = {
        "literature_request_schema_valid":
            True,

        "cluster_id_matches_request":
            str(request.cluster_id)
            == cluster_id,

        "packet_cluster_matches":
            str(
                rescue_packet.get(
                    "cluster_id"
                )
            )
            == cluster_id,

        "result_cluster_matches":
            result["cluster_id"]
            == cluster_id,

        "reference_label_not_used":
            result[
                "reference_label_used"
            ]
            is False,

        "expert_label_not_used":
            result[
                "expert_label_used"
            ]
            is False,

        "packet_reference_label_not_used":
            rescue_packet.get(
                "reference_label_used",
                False,
            )
            is False,

        "automatic_acceptance_not_calibrated":
            result[
                "automatic_acceptance"
            ]
            == "NOT_CALIBRATED",

        "packet_not_calibrated":
            rescue_packet.get(
                "automatic_acceptance"
            )
            == "NOT_CALIBRATED",

        "result_final_annotation_none":
            result[
                "final_annotation"
            ]
            is None,

        "packet_candidates_not_auto_accepted":
            all(
                c.get(
                    "automatic_acceptance"
                )
                == "NOT_CALIBRATED"
                for c in packet_candidates
            ),

        "packet_candidates_have_no_final_annotation":
            all(
                c.get(
                    "final_annotation"
                )
                is None
                for c in packet_candidates
            ),

        "rescue_was_triggered":
            bool(
                rescue_markers
            ),

        "pubmedbert_is_semantic_only":
            "semantic"
            in result[
                "pubmedbert_interpretation"
            ].casefold(),
    }

    overall_pass = all(
        invariants.values()
    )

    freeze_dir = (
        root
        / "freeze"
    )

    freeze_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifacts = {}

    for path in sorted(
        root.rglob("*")
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
                root
            )
        ).replace(
            "\\",
            "/",
        )

        artifacts[
            rel
        ] = file_record(
            path
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
            cluster_id,

        "overall_pass":
            overall_pass,

        "invariants":
            invariants,

        "artifact_count":
            len(artifacts),

        "artifacts":
            artifacts,

        "standardized_result":
            str(result_path),

        "scientific_note": (
            "This manifest freezes a standardized "
            "Agent 3 evidence run. PubMedBERT "
            "similarities represent semantic relevance "
            "only. Automatic acceptance remains "
            "uncalibrated and no final annotation "
            "is forced."
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

    print("=" * 100)
    print("FINAL STANDARDIZED AGENT 3 FREEZE AUDIT")
    print("=" * 100)
    print(
        "Overall:",
        "PASS"
        if overall_pass
        else "FAIL",
    )

    print()
    print("Invariant failures:")

    failed = [
        name
        for name, ok
        in invariants.items()
        if not ok
    ]

    if failed:
        for name in failed:
            print(" -", name)
    else:
        print(" none")

    print()
    print(
        "Cluster:",
        cluster_id,
    )
    print(
        "Upstream candidates:",
        len(
            request.candidate_annotations
        ),
    )
    print(
        "Rescue candidates:",
        len(
            packet_candidates
        ),
    )
    print(
        "Rescue triggered:",
        bool(
            rescue_markers
        ),
    )
    print(
        "Automatic acceptance:",
        result[
            "automatic_acceptance"
        ],
    )
    print(
        "Final annotation:",
        result[
            "final_annotation"
        ],
    )

    print()
    print(
        "Result:",
        result_path,
    )
    print(
        "Manifest:",
        manifest_path,
    )
    print(
        "Artifacts frozen:",
        len(artifacts),
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
