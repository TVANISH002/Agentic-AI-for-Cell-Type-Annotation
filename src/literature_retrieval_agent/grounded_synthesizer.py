from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .evidence_appraiser import build_literature_evidence_packet


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {source}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _candidate_rows(packet: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in packet.get("candidate_evidence", []):
        rows.append(
            {
                "cluster_id": packet.get("cluster_id"),
                "candidate_cell_type": candidate.get("candidate_cell_type"),
                "supporting_markers": "; ".join(
                    candidate.get("supporting_markers", [])
                ),
                "support_summary": candidate.get("support_summary"),
                "direct_supporting_claim_count": candidate.get(
                    "direct_supporting_claim_count"
                ),
                "direct_contradictory_claim_count": candidate.get(
                    "direct_contradictory_claim_count"
                ),
                "context_relevance": candidate.get("context_relevance"),
                "evidence_strength": candidate.get("evidence_strength"),
                "confidence_contribution": candidate.get(
                    "confidence_contribution"
                ),
                "reference_count": len(candidate.get("references", [])),
                "unsupported_upstream_markers": "; ".join(
                    candidate.get("unsupported_upstream_markers", [])
                ),
                "final_annotation_selected": packet.get(
                    "final_annotation_selected"
                ),
            }
        )
    return rows


def _evidence_rows(packet: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in packet.get("candidate_evidence", []):
        candidate_name = candidate.get("candidate_cell_type")
        for reference in candidate.get("references", []):
            rows.append(
                {
                    "cluster_id": packet.get("cluster_id"),
                    "candidate_cell_type": candidate_name,
                    "evidence_role": reference.get("evidence_role"),
                    "supported_claim": reference.get("supported_claim"),
                    "context_relevance": reference.get("context_relevance"),
                    "title": reference.get("title"),
                    "authors": "; ".join(reference.get("authors", [])),
                    "journal": reference.get("journal"),
                    "year": reference.get("year"),
                    "doi": reference.get("doi"),
                    "pmid": reference.get("pmid"),
                    "pmcid": reference.get("pmcid"),
                    "passage_id": reference.get("passage_id"),
                    "evidence_quote": reference.get("evidence_quote"),
                }
            )
    return rows


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    columns: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def synthesize_literature_evidence_packet(
    semantic_audit_file: str | Path,
    corpus_file: str | Path,
    output_json: str | Path,
    candidate_csv: str | Path,
    evidence_csv: str | Path,
) -> dict[str, Any]:
    semantic_audit = _read_json(semantic_audit_file)
    corpus = _read_json(corpus_file)
    packet = build_literature_evidence_packet(
        semantic_audit=semantic_audit,
        corpus=corpus,
    )

    json_path = Path(output_json).expanduser().resolve()
    candidate_path = Path(candidate_csv).expanduser().resolve()
    evidence_path = Path(evidence_csv).expanduser().resolve()

    _write_json(json_path, packet)
    _write_csv(
        candidate_path,
        _candidate_rows(packet),
        [
            "cluster_id",
            "candidate_cell_type",
            "supporting_markers",
            "support_summary",
            "direct_supporting_claim_count",
            "direct_contradictory_claim_count",
            "context_relevance",
            "evidence_strength",
            "confidence_contribution",
            "reference_count",
            "unsupported_upstream_markers",
            "final_annotation_selected",
        ],
    )
    _write_csv(
        evidence_path,
        _evidence_rows(packet),
        [
            "cluster_id",
            "candidate_cell_type",
            "evidence_role",
            "supported_claim",
            "context_relevance",
            "title",
            "authors",
            "journal",
            "year",
            "doi",
            "pmid",
            "pmcid",
            "passage_id",
            "evidence_quote",
        ],
    )

    return packet
