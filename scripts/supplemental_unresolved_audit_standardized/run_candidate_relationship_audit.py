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


from supplemental_unresolved_audit_standardized.candidate_relationship_audit import (
    build_candidate_relationship_audit,
    write_candidate_relationship_audit,
)


def load_json(
    path: Path,
) -> dict[str, Any]:

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        payload = json.load(
            handle
        )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            f"Expected JSON object: {path}"
        )

    return payload


def verify_or_write_audit(
    audit: Mapping[str, Any],
    output_json: Path,
) -> tuple[Path, str]:

    if not output_json.exists():

        written = write_candidate_relationship_audit(
            audit,
            output_json,
            overwrite=False,
        )

        return (
            written,
            "CREATED",
        )

    existing = load_json(
        output_json
    )

    if existing != dict(
        audit
    ):

        raise RuntimeError(
            "An existing relationship audit differs "
            "from the newly reconstructed audit. "
            "Refusing to overwrite."
        )

    return (
        output_json,
        "REUSED_EXISTING_VERIFIED",
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Build standardized Cell Ontology relationship "
            "analysis for frozen downstream-selected candidates "
            "from a supplemental UNRESOLVED-case audit."
        )
    )

    parser.add_argument(
        "--candidate-audit",
        required=True,
        help=(
            "Candidate evidence audit JSON."
        ),
    )

    parser.add_argument(
        "--cell-ontology-json",
        required=True,
        help=(
            "Frozen Cell Ontology JSON."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Separate supplemental relationship-audit "
            "output directory."
        ),
    )

    args = parser.parse_args()

    candidate_audit_path = Path(
        args.candidate_audit
    )

    ontology_path = Path(
        args.cell_ontology_json
    )

    output_dir = Path(
        args.output_dir
    )

    if not candidate_audit_path.exists():

        raise FileNotFoundError(
            "Candidate audit does not exist: "
            f"{candidate_audit_path}"
        )

    if not ontology_path.exists():

        raise FileNotFoundError(
            "Cell Ontology JSON does not exist: "
            f"{ontology_path}"
        )

    audit = build_candidate_relationship_audit(
        candidate_audit_json=candidate_audit_path,
        cell_ontology_json=ontology_path,
    )

    cluster_id = str(
        audit.get(
            "cluster_id",
            "",
        )
    ).strip()

    if not cluster_id:

        raise ValueError(
            "Relationship audit is missing cluster_id."
        )

    output_json = (
        output_dir
        / (
            f"cluster_{cluster_id}_"
            "candidate_relationship_audit.json"
        )
    )

    (
        written_path,
        action,
    ) = verify_or_write_audit(
        audit,
        output_json,
    )

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
        "STANDARDIZED SUPPLEMENTAL "
        "CANDIDATE RELATIONSHIP AUDIT"
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
        "Selected candidates: "
        f"{summary.get('selected_candidate_count')}"
    )

    print(
        "Ontology-resolved candidates: "
        f"{summary.get('ontology_resolved_candidate_count')}"
    )

    print(
        "Ontology-unresolved candidates: "
        f"{summary.get('ontology_unresolved_candidate_count')}"
    )

    print()

    print(
        "Total candidate pairs: "
        f"{summary.get('total_candidate_pairs')}"
    )

    print(
        "Evaluable pairs: "
        f"{summary.get('evaluable_candidate_pairs')}"
    )

    print(
        "Closely related pairs: "
        f"{summary.get('closely_related_pair_count')}"
    )

    print(
        "Non-related pairs: "
        f"{summary.get('nonrelated_pair_count')}"
    )

    print()

    print(
        "Relationship components: "
        f"{summary.get('relationship_component_count')}"
    )

    print(
        "Relationship structure: "
        f"{summary.get('relationship_structure')}"
    )

    print()

    print(
        "Components:"
    )

    for index, component in enumerate(
        audit.get(
            "relationship_components",
            [],
        ),
        start=1,
    ):

        print(
            f"  Component {index}: "
            + " | ".join(
                component
            )
        )

    print()

    print(
        "Pairwise relationship checks:"
    )

    for row in audit.get(
        "pairwise_relationship_checks",
        [],
    ):

        left = row.get(
            "left_label"
        )

        right = row.get(
            "right_label"
        )

        related = row.get(
            "closely_related"
        )

        evaluable = row.get(
            "relationship_evaluable"
        )

        evidence = row.get(
            "relationship_evidence",
            {},
        )

        ancestor = (
            evidence.get(
                "common_ancestor_label"
            )
            if isinstance(
                evidence,
                Mapping,
            )
            else None
        )

        distance = (
            evidence.get(
                "ontology_distance"
            )
            if isinstance(
                evidence,
                Mapping,
            )
            else None
        )

        print(
            f"  {left} <-> {right}"
        )

        print(
            f"      evaluable: {evaluable}"
        )

        print(
            f"      closely_related: {related}"
        )

        print(
            f"      common_ancestor: {ancestor}"
        )

        print(
            f"      ontology_distance: {distance}"
        )

    print()

    print(
        f"Relationship audit: {action}"
    )

    print(
        written_path
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
        "NO FINAL SUPPLEMENTAL BIOLOGICAL "
        "DECISION HAS BEEN MADE."
    )


if __name__ == "__main__":
    main()