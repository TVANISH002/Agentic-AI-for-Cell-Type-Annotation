from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from literature_retrieval_agent.unresolved_case_audit.report_writer import write_audit_reports
from literature_retrieval_agent.unresolved_case_audit.schemas import ClusterAuditResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HPA-aware unresolved-case audit without modifying Agent 3 outputs."
    )
    parser.add_argument(
        "--locked-summary",
        default="data/analysis/agent3_locked_standard_results/summary_tables/agent3_locked_standard_cluster_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="data/analysis/agent3_unresolved_case_audit/summary_tables",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.locked_summary)
    output_dir = Path(args.output_dir)

    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    summary = pd.read_csv(summary_path)
    unresolved = summary[
        summary["Decision"].astype(str).str.upper().eq("UNRESOLVED")
    ].copy()

    results: list[ClusterAuditResult] = []

    for _, row in unresolved.iterrows():
        dataset = str(row["Dataset"])
        cluster = str(row["Cluster"])
        labels = "" if pd.isna(row.get("Labels", "")) else str(row.get("Labels", ""))

        results.append(
            ClusterAuditResult(
                dataset=dataset,
                cluster=cluster,
                original_decision="UNRESOLVED",
                original_labels=labels,
                projected_decision="PENDING_AUDIT",
                projected_labels="PENDING_AUDIT",
                rescue_trigger_changed=False,
                final_decision_changed=False,
                reason=(
                    "Cluster identified as unresolved. Marker-level HPA/structured/literature "
                    "support audit should be run next."
                ),
            )
        )

    write_audit_reports(results, output_dir)

    print("Unresolved-case audit scaffold completed.")
    print(f"Unresolved clusters found: {len(results)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
