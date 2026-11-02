from __future__ import annotations

import csv
from pathlib import Path

from .schemas import (
    SupplementalEvidenceQueue,
)


class SupplementalEvidenceQueueReportWriter:
    def write(
        self,
        queue: SupplementalEvidenceQueue,
        output_dir: Path | str,
    ) -> dict[str, Path]:
        folder = Path(
            output_dir
        ).resolve()

        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        prefix = (
            f"cluster_{queue.cluster_id}"
            "_supplemental_retrieval_queue"
        )

        json_path = (
            folder
            / f"{prefix}.json"
        )

        csv_path = (
            folder
            / f"{prefix}.csv"
        )

        json_path.write_text(
            queue.model_dump_json(
                indent=2
            ),
            encoding="utf-8",
        )

        fieldnames = [
            "priority",
            "gene",
            "rank_within_cluster",
            "score",
            "log_fold_change",
            "agent1_database_evidence_found",
            "agent1_unresolved",
            "agent1_evidence_status",
            "queue_reason",
            "pilot_selected",
            "retrieval_status",
        ]

        with csv_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for item in queue.items:
                writer.writerow(
                    item.model_dump()
                )

        return {
            "queue_json": json_path,
            "queue_csv": csv_path,
        }