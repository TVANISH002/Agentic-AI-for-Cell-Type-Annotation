from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ReviewDecision:
    context_name: str
    cluster_id: str
    predicted_label: str
    reference_label: str
    match_status: str
    confidence: str
    support_gap: str
    route_to_literature: bool
    reason: str


class ReviewRouter:
    """
    Compares Agent 2 cluster-level annotations against LLM/reference annotations.
    Flags mismatched or low-confidence clusters for Literature Retrieval Agent.
    """

    PREDICTED_LABEL_COLUMNS = [
        "predicted_cell_type",
        "top_cell_type",
        "cell_type",
        "candidate_cell_type",
        "annotation",
    ]

    REFERENCE_LABEL_COLUMNS = [
        "reference_label",
        "llm_label",
        "expert_label",
        "cell_type",
        "annotation",
    ]

    CLUSTER_COLUMNS = ["cluster_id", "cluster", "spatial_cluster"]

    CONFIDENCE_COLUMNS = ["confidence", "confidence_score", "score"]
    SUPPORT_GAP_COLUMNS = ["support_gap", "gap", "support_difference"]

    def __init__(
        self,
        context_name: str,
        agent2_output_csv: str | Path,
        reference_labels_csv: str | Path,
        output_dir: str | Path,
        low_confidence_threshold: float = 0.60,
        min_support_gap: float = 1.0,
    ) -> None:
        self.context_name = context_name
        self.agent2_output_csv = Path(agent2_output_csv)
        self.reference_labels_csv = Path(reference_labels_csv)
        self.output_dir = Path(output_dir)
        self.low_confidence_threshold = low_confidence_threshold
        self.min_support_gap = min_support_gap

    def run(self) -> dict[str, Any]:
        start = time.perf_counter()

        predictions = self._read_rows(self.agent2_output_csv)
        references = self._read_rows(self.reference_labels_csv)

        pred_cluster_col = self._pick_column(predictions, self.CLUSTER_COLUMNS)
        pred_label_col = self._pick_column(predictions, self.PREDICTED_LABEL_COLUMNS)

        ref_cluster_col = self._pick_column(references, self.CLUSTER_COLUMNS)
        ref_label_col = self._pick_column(references, self.REFERENCE_LABEL_COLUMNS)

        confidence_col = self._pick_optional_column(predictions, self.CONFIDENCE_COLUMNS)
        support_gap_col = self._pick_optional_column(predictions, self.SUPPORT_GAP_COLUMNS)

        reference_by_cluster = {
            str(row[ref_cluster_col]).strip(): str(row[ref_label_col]).strip()
            for row in references
        }

        decisions: list[ReviewDecision] = []

        for row in predictions:
            cluster_id = str(row[pred_cluster_col]).strip()
            predicted_label = str(row[pred_label_col]).strip()
            reference_label = reference_by_cluster.get(cluster_id, "")

            confidence = str(row.get(confidence_col, "")).strip() if confidence_col else ""
            support_gap = str(row.get(support_gap_col, "")).strip() if support_gap_col else ""

            decision = self._decide(
                predicted_label=predicted_label,
                reference_label=reference_label,
                confidence=confidence,
                support_gap=support_gap,
            )

            decisions.append(
                ReviewDecision(
                    context_name=self.context_name,
                    cluster_id=cluster_id,
                    predicted_label=predicted_label,
                    reference_label=reference_label,
                    match_status=decision["match_status"],
                    confidence=confidence,
                    support_gap=support_gap,
                    route_to_literature=decision["route_to_literature"],
                    reason=decision["reason"],
                )
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        decisions_csv = self.output_dir / "review_decisions.csv"
        summary_json = self.output_dir / "review_router_summary.json"

        self._write_decisions(decisions_csv, decisions)

        elapsed = time.perf_counter() - start
        routed = [d for d in decisions if d.route_to_literature]

        summary = {
            "context_name": self.context_name,
            "agent": "review_router",
            "total_clusters_compared": len(decisions),
            "matched_clusters": sum(d.match_status == "matched" for d in decisions),
            "mismatched_clusters": sum(d.match_status == "mismatched" for d in decisions),
            "missing_reference_clusters": sum(d.match_status == "missing_reference" for d in decisions),
            "clusters_routed_to_literature": len(routed),
            "routed_cluster_ids": [d.cluster_id for d in routed],
            "low_confidence_threshold": self.low_confidence_threshold,
            "min_support_gap": self.min_support_gap,
            "elapsed_seconds": elapsed,
            "outputs": {
                "review_decisions_csv": str(decisions_csv),
                "review_router_summary_json": str(summary_json),
            },
        }

        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _decide(
        self,
        predicted_label: str,
        reference_label: str,
        confidence: str,
        support_gap: str,
    ) -> dict[str, Any]:
        if not reference_label:
            return {
                "match_status": "missing_reference",
                "route_to_literature": True,
                "reason": "No reference/LLM label available for this cluster.",
            }

        normalized_pred = self._normalize_label(predicted_label)
        normalized_ref = self._normalize_label(reference_label)

        is_match = normalized_pred == normalized_ref

        low_confidence = self._is_low_confidence(confidence)
        low_support_gap = self._is_low_support_gap(support_gap)

        if is_match and not low_confidence and not low_support_gap:
            return {
                "match_status": "matched",
                "route_to_literature": False,
                "reason": "Agent 2 label matches reference label with acceptable confidence/support.",
            }

        if not is_match:
            return {
                "match_status": "mismatched",
                "route_to_literature": True,
                "reason": "Agent 2 label does not match reference/LLM label.",
            }

        return {
            "match_status": "matched",
            "route_to_literature": True,
            "reason": "Label matches, but confidence/support gap is low.",
        }

    def _is_low_confidence(self, value: str) -> bool:
        if not value:
            return False
        try:
            return float(value) < self.low_confidence_threshold
        except ValueError:
            return value.lower() in {"low", "low-medium", "medium-low"}

    def _is_low_support_gap(self, value: str) -> bool:
        if not value:
            return False
        try:
            return float(value) < self.min_support_gap
        except ValueError:
            return False

    @staticmethod
    def _normalize_label(label: str) -> str:
        return (
            label.lower()
            .replace("-", " ")
            .replace("_", " ")
            .replace("cells", "cell")
            .strip()
        )

    @staticmethod
    def _read_rows(path: Path) -> list[dict[str, str]]:
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

        if not rows:
            raise ValueError(f"CSV file is empty: {path}")

        return rows

    @staticmethod
    def _pick_column(rows: list[dict[str, str]], candidates: list[str]) -> str:
        columns = set(rows[0].keys())
        for candidate in candidates:
            if candidate in columns:
                return candidate
        raise ValueError(f"Could not find any of {candidates}. Available columns: {sorted(columns)}")

    @staticmethod
    def _pick_optional_column(rows: list[dict[str, str]], candidates: list[str]) -> str | None:
        columns = set(rows[0].keys())
        for candidate in candidates:
            if candidate in columns:
                return candidate
        return None

    @staticmethod
    def _write_decisions(path: Path, decisions: list[ReviewDecision]) -> None:
        fieldnames = list(asdict(decisions[0]).keys()) if decisions else []

        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for decision in decisions:
                writer.writerow(asdict(decision))