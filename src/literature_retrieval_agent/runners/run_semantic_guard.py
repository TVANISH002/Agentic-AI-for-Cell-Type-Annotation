from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ..semantic_guard import audit_semantics


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")

    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_csv(path: Path, audit: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_cell_type",
        "claim_number",
        "claim_text",
        "original_stance",
        "corrected_stance",
        "semantically_valid_as_returned",
        "accepted_for_appraisal",
        "context_relevance",
        "context_scope",
        "observed_markers_in_quote",
        "reported_markers_outside_observed",
        "passage_id",
        "title",
        "year",
        "doi",
        "pmid",
        "reasons",
        "evidence_quote",
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for candidate in audit.get("candidate_audits", []):
            for claim in candidate.get("claims", []):
                reference = claim.get("reference", {})
                writer.writerow(
                    {
                        "candidate_cell_type": candidate.get(
                            "candidate_cell_type", ""
                        ),
                        "claim_number": claim.get("claim_number", ""),
                        "claim_text": claim.get("claim_text", ""),
                        "original_stance": claim.get("original_stance", ""),
                        "corrected_stance": claim.get("corrected_stance", ""),
                        "semantically_valid_as_returned": claim.get(
                            "semantically_valid_as_returned", False
                        ),
                        "accepted_for_appraisal": claim.get(
                            "accepted_for_appraisal", False
                        ),
                        "context_relevance": claim.get(
                            "context_relevance", ""
                        ),
                        "context_scope": claim.get("context_scope", ""),
                        "observed_markers_in_quote": "; ".join(
                            claim.get("observed_markers_in_quote", [])
                        ),
                        "reported_markers_outside_observed": "; ".join(
                            claim.get(
                                "reported_markers_outside_observed", []
                            )
                        ),
                        "passage_id": claim.get("passage_id", ""),
                        "title": reference.get("title", ""),
                        "year": reference.get("year", ""),
                        "doi": reference.get("doi", ""),
                        "pmid": reference.get("pmid", ""),
                        "reasons": " | ".join(claim.get("reasons", [])),
                        "evidence_quote": claim.get("evidence_quote", ""),
                    }
                )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the semantic interpretation of grounded claims."
    )
    parser.add_argument("--benchmark-json", type=Path, required=True)
    parser.add_argument("--corpus-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()
    benchmark = _read_json(args.benchmark_json.resolve())
    corpus = _read_json(args.corpus_json.resolve())
    audit = audit_semantics(benchmark=benchmark, corpus=corpus)

    output_json = args.output_json.resolve()
    output_csv = args.output_csv.resolve()
    _write_json(output_json, audit)
    _write_csv(output_csv, audit)

    print("Semantic claim audit completed successfully.")
    print(f"Cluster: {audit['cluster_id']}")
    print(f"Claims audited: {audit['total_claims']}")
    print(f"Unchanged claims: {audit['unchanged_claims']}")
    print(f"Reclassified claims: {audit['reclassified_claims']}")
    print(f"Rejected claims: {audit['rejected_claims']}")
    print(f"JSON: {output_json}")
    print(f"CSV: {output_csv}")


if __name__ == "__main__":
    main()
