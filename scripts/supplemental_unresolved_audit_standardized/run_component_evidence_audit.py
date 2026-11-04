from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from supplemental_unresolved_audit_standardized.component_evidence_audit import (
    build_component_evidence_audit,
    write_component_evidence_audit,
    write_component_evidence_csv,
)


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Build standardized component-level evidence "
            "for a supplemental UNRESOLVED-case audit."
        )
    )

    parser.add_argument(
        "--candidate-audit",
        required=True,
    )

    parser.add_argument(
        "--relationship-audit",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    candidate_path = Path(
        args.candidate_audit
    )

    relationship_path = Path(
        args.relationship_audit
    )

    output_dir = Path(
        args.output_dir
    )

    audit = build_component_evidence_audit(
        candidate_audit_json=candidate_path,
        relationship_audit_json=relationship_path,
    )

    cluster_id = str(
        audit.get(
            "cluster_id",
            "",
        )
    ).strip()

    if not cluster_id:
        raise ValueError(
            "Component audit missing cluster_id."
        )

    json_path = (
        output_dir
        / (
            f"cluster_{cluster_id}_"
            "component_evidence_audit.json"
        )
    )

    csv_path = (
        output_dir
        / (
            f"cluster_{cluster_id}_"
            "component_evidence_audit.csv"
        )
    )

    written_json = (
        write_component_evidence_audit(
            audit,
            json_path,
            overwrite=False,
        )
    )

    written_csv = (
        write_component_evidence_csv(
            audit,
            csv_path,
            overwrite=False,
        )
    )

    print()
    print("=" * 82)

    print(
        "STANDARDIZED SUPPLEMENTAL "
        "COMPONENT EVIDENCE AUDIT"
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

    summary = audit.get(
        "summary",
        {},
    )

    print(
        "Selected candidates: "
        f"{summary.get('selected_candidate_count')}"
    )

    print(
        "Relationship components: "
        f"{summary.get('relationship_component_count')}"
    )

    print()

    print(
        "Component evidence:"
    )

    for component in audit.get(
        "components",
        [],
    ):

        print()

        print(
            f"  {component.get('component_id')}"
        )

        print(
            "    Candidates: "
            + " | ".join(
                component.get(
                    "member_candidate_labels",
                    [],
                )
            )
        )

        print(
            "    Unique supporting genes: "
            f"{component.get('unique_supporting_gene_count')}"
        )

        print(
            "    Repeated memberships removed: "
            f"{component.get('repeated_gene_membership_count_removed')}"
        )

        print(
            "    Best marker rank: "
            f"{component.get('best_supporting_marker_rank')}"
        )

        print(
            "    Median marker rank: "
            f"{component.get('median_supporting_marker_rank')}"
        )

        print(
            "    Supporting genes: "
            + " | ".join(
                component.get(
                    "unique_supporting_genes",
                    [],
                )
            )
        )

    print()

    print(
        "JSON:"
    )

    print(
        written_json
    )

    print()

    print(
        "CSV:"
    )

    print(
        written_csv
    )

    print()

    print(
        "Supplemental decision: "
        f"{audit.get('decision', {}).get('supplemental_decision')}"
    )

    print(
        "Core Agent 3 output changed: "
        f"{audit.get('decision', {}).get('core_agent3_output_changed')}"
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