from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from supplemental_unresolved_audit_standardized.candidate_evidence_audit import (
    build_candidate_evidence_audit,
    write_candidate_evidence_audit,
    write_candidate_evidence_csv,
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected JSON object: {path}"
        )

    return payload


def verify_or_write_json(
    audit: Mapping[str, Any],
    output_path: Path,
) -> tuple[Path, str]:

    if not output_path.exists():

        written = write_candidate_evidence_audit(
            audit,
            output_path,
            overwrite=False,
        )

        return written, "CREATED"

    existing = load_json(
        output_path
    )

    if existing != dict(audit):
        raise RuntimeError(
            "Existing candidate evidence audit differs "
            "from the newly reconstructed audit. "
            "Refusing to overwrite."
        )

    return (
        output_path,
        "REUSED_EXISTING_VERIFIED",
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Build the standardized candidate-level evidence "
            "audit for a human-triggered supplemental "
            "UNRESOLVED case."
        )
    )

    parser.add_argument(
        "--evidence-packet",
        required=True,
        help=(
            "Standardized supplemental evidence packet JSON."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Separate candidate-audit output directory."
        ),
    )

    args = parser.parse_args()

    evidence_packet = Path(
        args.evidence_packet
    )

    output_dir = Path(
        args.output_dir
    )

    if not evidence_packet.exists():
        raise FileNotFoundError(
            f"Evidence packet not found: {evidence_packet}"
        )

    audit = build_candidate_evidence_audit(
        evidence_packet_json=evidence_packet
    )

    cluster_id = str(
        audit.get("cluster_id", "")
    ).strip()

    if not cluster_id:
        raise ValueError(
            "Candidate audit did not contain cluster_id."
        )

    json_path = (
        output_dir
        / (
            f"cluster_{cluster_id}_"
            "candidate_evidence_audit.json"
        )
    )

    csv_path = (
        output_dir
        / (
            f"cluster_{cluster_id}_"
            "candidate_evidence_audit.csv"
        )
    )

    (
        written_json,
        json_action,
    ) = verify_or_write_json(
        audit,
        json_path,
    )

    if not csv_path.exists():

        written_csv = write_candidate_evidence_csv(
            audit,
            csv_path,
            overwrite=False,
        )

        csv_action = "CREATED"

    else:

        # Do not overwrite existing CSV.
        written_csv = csv_path
        csv_action = "EXISTING_NOT_OVERWRITTEN"

    summary = audit.get(
        "summary",
        {},
    )

    decision = audit.get(
        "decision",
        {},
    )

    print()
    print("=" * 82)
    print(
        "STANDARDIZED SUPPLEMENTAL CANDIDATE EVIDENCE AUDIT"
    )
    print("=" * 82)

    print(
        f"Dataset: {audit.get('dataset_id')}"
    )

    print(
        f"Cluster: {audit.get('cluster_id')}"
    )

    print(
        "Core Agent 3 decision: "
        f"{audit.get('core_agent3_decision')}"
    )

    print()

    print(
        "Marker count: "
        f"{summary.get('marker_count')}"
    )

    print(
        "Candidate count: "
        f"{summary.get('candidate_count')}"
    )

    print(
        "Selected downstream candidate count: "
        f"{summary.get('selected_downstream_candidate_count')}"
    )

    print()

    print(
        "Selected downstream candidates:"
    )

    selected = audit.get(
        "selected_downstream_candidates",
        [],
    )

    for candidate in selected:
        print(
            f"  - {candidate}"
        )

    print()

    print(
        f"JSON audit: {json_action}"
    )
    print(
        written_json
    )

    print()

    print(
        f"CSV audit: {csv_action}"
    )
    print(
        written_csv
    )

    print()

    print(
        "Supplemental decision: "
        f"{decision.get('supplemental_decision')}"
    )

    print(
        "Core Agent 3 output changed: "
        f"{decision.get('core_agent3_output_changed')}"
    )

    print()

    print(
        "STATUS: "
        f"{audit.get('status')}"
    )

    print()

    print(
        "NO RESOLVED / AMBIGUOUS / UNRESOLVED "
        "SUPPLEMENTAL DECISION HAS BEEN MADE YET."
    )


if __name__ == "__main__":
    main()