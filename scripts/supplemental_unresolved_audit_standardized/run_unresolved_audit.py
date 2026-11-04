
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping



# Make src/ importable when executed directly from repository root.


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from supplemental_unresolved_audit_standardized.request_builder import (
    build_request,
    write_request,
)

from supplemental_unresolved_audit_standardized.evidence_manifest import (
    build_evidence_manifest,
    write_evidence_manifest,
)

from supplemental_unresolved_audit_standardized.evidence_extractor import (
    build_evidence_packet,
    write_evidence_packet,
)


def clean(value: str | None) -> str | None:
    if value is None:
        return None

    value = " ".join(str(value).split())
    return value or None


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected JSON object in existing file: {path}"
        )

    return payload


def ensure_output_is_separate(
    *,
    core_decision_json: Path,
    output_root: Path,
) -> None:

    core_cluster_root = core_decision_json.resolve().parent.parent
    resolved_output = output_root.resolve()

    if resolved_output == core_cluster_root:
        raise RuntimeError(
            "Supplemental output root cannot be the frozen "
            "core Agent 3 cluster directory."
        )

    if resolved_output.is_relative_to(core_cluster_root):
        raise RuntimeError(
            "Supplemental outputs must be stored outside the "
            "frozen core Agent 3 cluster directory. "
            f"Core directory: {core_cluster_root} "
            f"Requested output: {resolved_output}"
        )


def verify_or_write_request(
    request,
    output_json: Path,
) -> tuple[Path, str]:

    if not output_json.exists():

        written = write_request(
            request,
            output_json,
            overwrite=False,
        )

        return written, "CREATED"

    existing = load_json(
        output_json
    )

    expected = asdict(
        request
    )

    invariant_keys = (
        "schema_name",
        "schema_version",
        "dataset_id",
        "cluster_id",
        "species",
        "tissue",
        "platform",
        "disease_context",
        "trigger_source",
        "core_agent3_decision",
        "human_trigger_required",
        "human_trigger_confirmed",
        "supplemental_audit_invoked",
        "core_agent3_output_changed",
        "reference_label_used",
        "expert_label_used",
        "core_agent3_decision_json",
        "core_agent3_decision_sha256",
        "audit_status",
    )

    mismatches: list[str] = []

    for key in invariant_keys:

        if existing.get(key) != expected.get(key):
            mismatches.append(
                key
            )

    if mismatches:
        raise RuntimeError(
            "An existing supplemental audit request was found, "
            "but it does not match the current invocation. "
            "Refusing to overwrite it. "
            "Mismatched fields: "
            + ", ".join(mismatches)
        )

    return (
        output_json,
        "REUSED_EXISTING_VERIFIED",
    )


def verify_or_write_manifest(
    manifest,
    output_json: Path,
) -> tuple[Path, str]:

    expected = asdict(
        manifest
    )

    if not output_json.exists():

        written = write_evidence_manifest(
            manifest,
            output_json,
            overwrite=False,
        )

        return written, "CREATED"

    existing = load_json(
        output_json
    )

    if existing != expected:
        raise RuntimeError(
            "An evidence manifest already exists but does not "
            "match the currently supplied frozen inputs. "
            "Refusing to overwrite it."
        )

    return (
        output_json,
        "REUSED_EXISTING_VERIFIED",
    )


def verify_or_write_packet(
    packet: Mapping[str, Any],
    output_json: Path,
) -> tuple[Path, str]:

    if not output_json.exists():

        written = write_evidence_packet(
            packet,
            output_json,
            overwrite=False,
        )

        return written, "CREATED"

    existing = load_json(
        output_json
    )

    if existing != dict(packet):
        raise RuntimeError(
            "A standardized evidence packet already exists "
            "but does not match the currently reconstructed "
            "packet. Refusing to overwrite it."
        )

    return (
        output_json,
        "REUSED_EXISTING_VERIFIED",
    )


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Human-triggered standardized supplemental audit "
            "entry point for frozen core Agent 3 "
            "UNRESOLVED cases."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset identifier.",
    )

    parser.add_argument(
        "--cluster-id",
        required=True,
        help="Cluster identifier.",
    )

    parser.add_argument(
        "--core-agent3-decision-json",
        required=True,
        help=(
            "Frozen core Agent 3 decision-summary JSON. "
            "Decision must be UNRESOLVED."
        ),
    )

    parser.add_argument(
        "--output-root",
        required=True,
        help=(
            "Separate output root for supplemental "
            "unresolved-audit artifacts."
        ),
    )

    parser.add_argument(
        "--agent2-evidence",
        action="append",
        required=True,
        help=(
            "Frozen Agent 2 evidence artifact. "
            "Repeat this argument for multiple Agent 2 "
            "artifacts."
        ),
    )

    parser.add_argument(
        "--marker-evidence",
        required=True,
        help=(
            "Frozen standardized marker/DEG evidence file."
        ),
    )

    parser.add_argument(
        "--agent1-evidence",
        default=None,
        help=(
            "Optional frozen Agent 1 evidence artifact."
        ),
    )

    parser.add_argument(
        "--hpa-evidence",
        default=None,
        help=(
            "Optional HPA evidence artifact. "
            "Omit when HPA is unavailable or disabled."
        ),
    )

    parser.add_argument(
        "--species",
        default=None,
    )

    parser.add_argument(
        "--tissue",
        default=None,
    )

    parser.add_argument(
        "--platform",
        default=None,
    )

    parser.add_argument(
        "--disease-context",
        default=None,
    )

    return parser


def main() -> None:

    parser = build_parser()
    args = parser.parse_args()

    dataset_id = clean(
        args.dataset
    )

    cluster_id = clean(
        args.cluster_id
    )

    if not dataset_id:
        parser.error(
            "--dataset must not be empty."
        )

    if not cluster_id:
        parser.error(
            "--cluster-id must not be empty."
        )

    decision_json = Path(
        args.core_agent3_decision_json
    )

    output_root = Path(
        args.output_root
    )

    if not decision_json.exists():
        raise FileNotFoundError(
            "Core Agent 3 decision JSON does not exist: "
            f"{decision_json}"
        )



    ensure_output_is_separate(
        core_decision_json=decision_json,
        output_root=output_root,
    )

    cluster_root = (
        output_root
        / f"cluster_{cluster_id}"
    )

  
    # STEP 1
    # Manual-trigger request

    request = build_request(
        dataset_id=dataset_id,
        cluster_id=cluster_id,
        core_agent3_decision_json=decision_json,
        species=clean(
            args.species
        ),
        tissue=clean(
            args.tissue
        ),
        platform=clean(
            args.platform
        ),
        disease_context=clean(
            args.disease_context
        ),
    )

    request_json = (
        cluster_root
        / "request"
        / (
            f"cluster_{cluster_id}_"
            "supplemental_audit_request.json"
        )
    )

    (
        request_path,
        request_action,
    ) = verify_or_write_request(
        request,
        request_json,
    )

    # STEP 2
    # Freeze evidence manifest


    manifest = build_evidence_manifest(
        dataset_id=dataset_id,
        cluster_id=cluster_id,

        species=clean(
            args.species
        ),

        tissue=clean(
            args.tissue
        ),

        platform=clean(
            args.platform
        ),

        disease_context=clean(
            args.disease_context
        ),

        agent1_evidence=(
            Path(
                args.agent1_evidence
            )
            if args.agent1_evidence
            else None
        ),

        agent2_evidence=[
            Path(path)
            for path in args.agent2_evidence
        ],

        marker_evidence=Path(
            args.marker_evidence
        ),

        hpa_evidence=(
            Path(
                args.hpa_evidence
            )
            if args.hpa_evidence
            else None
        ),

        core_agent3_decision=(
            decision_json
        ),
    )

    manifest_json = (
        cluster_root
        / "manifest"
        / (
            f"cluster_{cluster_id}_"
            "supplemental_evidence_manifest.json"
        )
    )

    (
        manifest_path,
        manifest_action,
    ) = verify_or_write_manifest(
        manifest,
        manifest_json,
    )

 
    # STEP 3
    # Build standardized evidence packet

    packet = build_evidence_packet(
        request_json=request_path,
        manifest_json=manifest_path,
    )

    packet_json = (
        cluster_root
        / "evidence"
        / (
            f"cluster_{cluster_id}_"
            "standardized_evidence_packet.json"
        )
    )

    (
        packet_path,
        packet_action,
    ) = verify_or_write_packet(
        packet,
        packet_json,
    )

    # TERMINAL REPORT
 

    print()
    print("=" * 82)

    print(
        "STANDARDIZED SUPPLEMENTAL UNRESOLVED AUDIT "
        "— EVIDENCE PREPARATION"
    )

    print("=" * 82)

    print(
        f"Dataset: {dataset_id}"
    )

    print(
        f"Cluster: {cluster_id}"
    )

    print(
        "Core Agent 3 decision: "
        f"{request.core_agent3_decision}"
    )

    print(
        "Trigger source: "
        f"{request.trigger_source}"
    )

    print(
        "Human trigger confirmed: "
        f"{request.human_trigger_confirmed}"
    )

    print()

    print(
        "Reference labels allowed: "
        f"{manifest.reference_labels_allowed}"
    )

    print(
        "Expert labels allowed: "
        f"{manifest.expert_labels_allowed}"
    )

    print(
        "Core Agent 3 output changed: "
        f"{request.core_agent3_output_changed}"
    )

    print()

    marker_set = packet.get(
        "marker_set",
        {},
    )

    agent1 = packet.get(
        "agent1",
        {},
    )

    agent2 = packet.get(
        "agent2",
        {},
    )

    hpa = packet.get(
        "hpa",
        {},
    )

    print("Evidence packet summary:")

    print(
        "  Marker count: "
        f"{marker_set.get('marker_count')}"
    )

    print(
        "  Marker padding used: "
        f"{marker_set.get('padding_used')}"
    )

    print(
        "  Agent 1 matched markers: "
        f"{agent1.get('matched_marker_count')}"
    )

    print(
        "  Agent 1 missing markers: "
        f"{agent1.get('missing_marker_count')}"
    )

    print(
        "  Agent 2 gene evidence count: "
        f"{agent2.get('gene_evidence_count')}"
    )

    print(
        "  Agent 2 candidate count: "
        f"{agent2.get('candidate_count')}"
    )

    print(
        "  HPA enabled: "
        f"{hpa.get('enabled')}"
    )

    print(
        "  HPA matched markers: "
        f"{hpa.get('matched_marker_count')}"
    )

    print(
        "  HPA missing markers: "
        f"{hpa.get('missing_marker_count')}"
    )

    print()

    print(
        f"Request: {request_action}"
    )

    print(
        request_path
    )

    print()

    print(
        f"Evidence manifest: {manifest_action}"
    )

    print(
        manifest_path
    )

    print()

    print(
        f"Evidence packet: {packet_action}"
    )

    print(
        packet_path
    )

    print()

    print(
        "Packet status: "
        f"{packet.get('status')}"
    )

    print()

    print(
        "BIOLOGICAL ADJUDICATION PERFORMED: "
        f"{packet.get('validation', {}).get('biological_adjudication_performed')}"
    )

    print()

    print(
        "The frozen core Agent 3 output remains unchanged."
    )


if __name__ == "__main__":
    main()