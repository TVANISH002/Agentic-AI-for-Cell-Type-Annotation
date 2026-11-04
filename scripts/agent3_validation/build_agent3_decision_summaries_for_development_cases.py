from __future__ import annotations

import csv
import json
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


ONTOLOGY = (
    ROOT
    / "data"
    / "resources"
    / "cell_ontology"
    / "v2026-06-08"
    / "cl-base.json"
)


CASES = [
    {
        "case_id": "breast_cluster_6",
        "dataset": "breast",
        "cluster_id": "6",
        "packet": (
            ROOT
            / "data"
            / "analysis"
            / "agent3_validation"
            / "cluster6_adjudication_final"
            / "cluster_6_adjudication_evidence_packet.json"
        ),
    },
    {
        "case_id": "breast_cluster_7",
        "dataset": "breast",
        "cluster_id": "7",
        "packet": (
            ROOT
            / "data"
            / "analysis"
            / "agent3_validation"
            / "cluster7_rescue_final"
            / "cluster_7_rescue_evidence_packet.json"
        ),
    },
    {
        "case_id": "lung_cluster_6",
        "dataset": "lung",
        "cluster_id": "6",
        "packet": (
            ROOT
            / "data"
            / "analysis"
            / "agent3_validation"
            / "lung_cross_tissue"
            / "cluster_6"
            / "marker_rescue"
            / "final_evidence_packet"
            / "cluster_6_rescue_evidence_packet.json"
        ),
    },
    {
        "case_id": "lung_cluster_9",
        "dataset": "lung",
        "cluster_id": "9",
        "packet": (
            ROOT
            / "data"
            / "analysis"
            / "agent3_validation"
            / "standardized_agent3_runs"
            / "lung"
            / "cluster_9"
            / "marker_rescue"
            / "final_evidence_packet"
            / "cluster_9_rescue_evidence_packet.json"
        ),
    },
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    output_dir = (
        ROOT
        / "data"
        / "analysis"
        / "agent3_validation"
        / "decision_summaries_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for case in CASES:
        packet_path = case["packet"]

        if not packet_path.exists():
            raise FileNotFoundError(packet_path)

        summary = build_decision_summary(
            load_json(packet_path),
            dataset=case["dataset"],
            cluster_id=case["cluster_id"],
            ontology_path=ONTOLOGY,
        )

        output_json = output_dir / f"{case['case_id']}_agent3_decision_summary.json"

        write_decision_outputs(summary, output_json)

        rows.append(
            {
                "case_id": case["case_id"],
                "dataset": case["dataset"],
                "cluster_id": case["cluster_id"],
                "source_packet": str(packet_path),
                "decision_state": summary["decision_state"],
                "ambiguity_subtype": summary["ambiguity_subtype"],
                "possible_label_count": summary["possible_label_count"],
                "possible_labels": "; ".join(
                    item["label"] for item in summary["possible_labels"]
                ),
                "decision_summary_json": str(output_json),
            }
        )

    table_path = output_dir / "agent3_development_case_decision_summary_table.csv"

    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Agent 3 development-case decision summaries completed.")
    print("Output directory:", output_dir)
    print("Summary table:", table_path)
    print()

    for row in rows:
        print(
            f"{row['case_id']}: "
            f"{row['decision_state']} | "
            f"{row['ambiguity_subtype']} | "
            f"{row['possible_labels']}"
        )


if __name__ == "__main__":
    main()
