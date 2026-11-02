from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import csv
import json

from .schemas import ClusterAuditResult


def write_audit_reports(results: list[ClusterAuditResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "unresolved_case_audit_summary.csv"
    json_path = output_dir / "unresolved_case_audit_summary.json"

    rows = [asdict(result) for result in results]

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text(
            "dataset,cluster,original_decision,original_labels,projected_decision,projected_labels,rescue_trigger_changed,final_decision_changed,reason\n",
            encoding="utf-8",
        )

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
