from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_ROOT),
    )


from supplemental_unresolved_audit_standardized.supplemental_adjudicator import (
    build_supplemental_adjudication,
    write_supplemental_adjudication,
)


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Apply the standardized supplemental "
            "evidence-state adjudication policy to an "
            "Agent 3 UNRESOLVED case."
        )
    )

    parser.add_argument(
        "--component-audit",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    component_path = Path(
        args.component_audit
    )

    output_dir = Path(
        args.output_dir
    )

    if not component_path.exists():

        raise FileNotFoundError(
            "Component evidence audit not found: "
            f"{component_path}"
        )

    result = build_supplemental_adjudication(
        component_audit_json=component_path
    )

    cluster_id = str(
        result.get(
            "cluster_id",
            "",
        )
    ).strip()

    if not cluster_id:

        raise ValueError(
            "Supplemental adjudication missing cluster_id."
        )

    output_json = (
        output_dir
        / (
            f"cluster_{cluster_id}_"
            "supplemental_adjudication.json"
        )
    )

    written = write_supplemental_adjudication(
        result,
        output_json,
        overwrite=False,
    )

    evaluation = result.get(
        "component_evidence_evaluation",
        {},
    )

    print()
    print("=" * 82)

    print(
        "STANDARDIZED SUPPLEMENTAL "
        "UNRESOLVED-CASE ADJUDICATION"
    )

    print("=" * 82)

    print(
        f"Dataset: {result.get('dataset_id')}"
    )

    print(
        f"Cluster: {result.get('cluster_id')}"
    )

    print()

    print(
        "Frozen core Agent 3 decision: "
        f"{result.get('core_agent3_decision')}"
    )

    print(
        "Core Agent 3 output changed: "
        f"{result.get('core_agent3_output_changed')}"
    )

    print()

    print(
        "Total relationship components: "
        f"{evaluation.get('total_component_count')}"
    )

    print(
        "Qualifying components: "
        f"{evaluation.get('qualifying_component_count')}"
    )

    print()

    for component in evaluation.get(
        "components",
        [],
    ):

        print(
            f"{component.get('component_id')}: "
            + " | ".join(
                component.get(
                    "member_candidate_labels",
                    [],
                )
            )
        )

        print(
            "  unique genes: "
            f"{component.get('unique_supporting_gene_count')}"
        )

        print(
            "  ranked marker hits: "
            f"{component.get('unique_ranked_supporting_marker_count')}"
        )

        print(
            "  qualifies: "
            f"{component.get('qualifies_minimum_evidence')}"
        )

    print()

    print(
        "SUPPLEMENTAL DECISION: "
        f"{result.get('supplemental_decision_state')}"
    )

    print(
        "AMBIGUITY SUBTYPE: "
        f"{result.get('supplemental_ambiguity_subtype')}"
    )

    print()

    print(
        "POSSIBLE LABELS:"
    )

    labels = result.get(
        "supplemental_possible_labels",
        [],
    )

    if labels:

        for label in labels:

            print(
                f"  - {label}"
            )

    else:

        print(
            "  <none>"
        )

    print()

    print(
        "REASON:"
    )

    print(
        result.get(
            "supplemental_decision_reason"
        )
    )

    print()

    print(
        "Automatic acceptance: "
        f"{result.get('adjudication_policy', {}).get('automatic_acceptance')}"
    )

    print(
        "Reference label used: "
        f"{result.get('adjudication_policy', {}).get('reference_label_used')}"
    )

    print(
        "Expert label used: "
        f"{result.get('adjudication_policy', {}).get('expert_label_used')}"
    )

    print()

    print(
        "Saved:"
    )

    print(
        written
    )

    print()

    print(
        "STATUS: "
        f"{result.get('status')}"
    )


if __name__ == "__main__":
    main()