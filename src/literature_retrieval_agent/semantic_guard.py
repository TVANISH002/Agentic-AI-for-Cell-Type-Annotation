from __future__ import annotations

import re
from collections import Counter
from typing import Any


_RESTRICTIVE_TERMS = (
    "restricted to",
    "specific to",
    "specific for",
    "exclusive to",
    "exclusively",
    "only in",
)

_MARKER_TO_CELL_RELATIONS = (
    "expressed in",
    "expression in",
    "expressing",
    "marker of",
    "marker for",
    "marks",
    "positive",
    "specific to",
    "restricted to",
    "associated with",
    "enriched in",
    "characteristic of",
    "signature of",
)

_CELL_TO_MARKER_RELATIONS = (
    "expressing",
    "expressed",
    "contain",
    "contains",
    "containing",
    "include",
    "includes",
    "including",
    "positive for",
    "marker",
    "markers",
    "characterized by",
    "defined by",
    "signature",
    "specific gene",
    "restricted gene",
    "enriched for",
)


def _normalize(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[-_/]+", " ", text)
    text = re.sub(r"[^a-z0-9+]+", " ", text)
    return " ".join(text.split())


def _contains_gene(text: str, gene: str) -> bool:
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9]){re.escape(gene)}(?![A-Za-z0-9])",
            text,
            flags=re.IGNORECASE,
        )
    )


def _mentions_candidate(text: str, candidate: str) -> bool:
    """Detect a cell-type mention while excluding terms such as BCR."""

    normalized_text = _normalize(text)
    normalized_candidate = _normalize(candidate)

    if not normalized_candidate:
        return False

    if normalized_candidate.endswith(" cell"):
        plural = f"{normalized_candidate}s"
        if re.search(
            rf"(?<![a-z0-9]){re.escape(plural)}(?![a-z0-9])",
            normalized_text,
        ):
            return True

        singular_pattern = (
            rf"(?<![a-z0-9]){re.escape(normalized_candidate)}"
            r"(?![a-z0-9])"
            r"(?!\s+(?:receptor|signaling|activation|associated|"
            r"derived|module|target|class|lineage))"
        )
        return bool(
            re.search(
                singular_pattern,
                normalized_text,
            )
        )

    return bool(
        re.search(
            rf"(?<![a-z0-9]){re.escape(normalized_candidate)}"
            rf"(?![a-z0-9])",
            normalized_text,
        )
    )


def _directly_links_candidate_and_marker(
    text: str,
    candidate: str,
    markers: list[str],
) -> bool:
    """Require relational wording, not candidate/marker co-occurrence."""

    normalized_text = _normalize(text)
    normalized_candidate = _normalize(candidate)
    if not normalized_text or not normalized_candidate or not markers:
        return False

    candidate_pattern = (
        rf"(?<![a-z0-9]){re.escape(normalized_candidate)}s?"
        rf"(?![a-z0-9])"
    )
    marker_to_cell = "|".join(
        re.escape(value)
        for value in _MARKER_TO_CELL_RELATIONS
    )
    cell_to_marker = "|".join(
        re.escape(value)
        for value in _CELL_TO_MARKER_RELATIONS
    )

    for marker in markers:
        normalized_marker = _normalize(marker)
        if not normalized_marker:
            continue

        marker_pattern = (
            rf"(?<![a-z0-9]){re.escape(normalized_marker)}"
            rf"(?![a-z0-9])"
        )
        marker_first = (
            marker_pattern
            + rf".{{0,260}}(?:{marker_to_cell}).{{0,140}}"
            + candidate_pattern
        )
        candidate_first = (
            candidate_pattern
            + rf".{{0,140}}(?:{cell_to_marker}).{{0,180}}"
            + marker_pattern
        )

        if re.search(marker_first, normalized_text):
            return True
        if re.search(candidate_first, normalized_text):
            return True

    return False


def _publication_dict(passage: dict[str, Any]) -> dict[str, Any]:
    publication = passage.get("publication", {})
    if not isinstance(publication, dict):
        publication = {}

    return {
        "title": publication.get("title"),
        "authors": publication.get("authors", []),
        "journal": publication.get("journal"),
        "year": publication.get("publication_year"),
        "doi": publication.get("doi"),
        "pmid": publication.get("pmid"),
        "pmcid": publication.get("pmcid"),
    }


def _context_assessment(
    quote: str,
    passage: dict[str, Any],
    tissue: str,
    disease_context: str,
) -> tuple[str, str, list[str]]:
    normalized_quote = _normalize(quote)
    quote_matches: list[str] = []

    normalized_tissue = _normalize(tissue)
    normalized_disease = _normalize(disease_context)

    if normalized_tissue and normalized_tissue in normalized_quote:
        quote_matches.append("tissue")

    if normalized_disease and normalized_disease in normalized_quote:
        quote_matches.append("disease")

    passage_matches = {
        str(value)
        for value in passage.get("context_matches", [])
    }
    source_matches = sorted(
        passage_matches.intersection({"tissue", "disease"})
    )

    if quote_matches:
        return "high", "quote_explicit", sorted(quote_matches)

    if len(source_matches) == 2:
        return "high", "paper_level", source_matches

    if len(source_matches) == 1:
        return "moderate", "paper_level", source_matches

    return "low", "out_of_context", []


def _corrected_stance(
    candidate: str,
    quote: str,
    mentioned_candidates: list[str],
    linked_candidates: list[str],
    observed_quote_markers: list[str],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    candidate_is_mentioned = candidate in mentioned_candidates
    candidate_is_linked = candidate in linked_candidates
    other_linked_candidates = [
        value
        for value in linked_candidates
        if value != candidate
    ]
    other_mentioned_candidates = [
        value
        for value in mentioned_candidates
        if value != candidate
    ]

    if candidate_is_linked and observed_quote_markers:
        if other_linked_candidates:
            reasons.append(
                "The quotation links the marker to multiple supplied "
                "candidate cell types."
            )
            return "ambiguous", reasons

        return "supporting", reasons

    if candidate_is_mentioned and observed_quote_markers:
        reasons.append(
            "The quotation co-mentions the candidate and an observed "
            "marker but does not explicitly connect them."
        )
        return "context_only", reasons

    if candidate_is_mentioned and not observed_quote_markers:
        reasons.append(
            "The quotation mentions the candidate but no observed "
            "cluster marker."
        )
        return "context_only", reasons

    if other_linked_candidates and observed_quote_markers:
        normalized_quote = _normalize(quote)
        if any(
            _normalize(term) in normalized_quote
            for term in _RESTRICTIVE_TERMS
        ):
            reasons.append(
                "The quotation restricts marker evidence to another "
                "candidate cell type."
            )
            return "contradictory", reasons

        reasons.append(
            "The quotation links the marker to another candidate but "
            "does not establish exclusivity."
        )
        return "ambiguous", reasons

    if other_mentioned_candidates and observed_quote_markers:
        reasons.append(
            "The quotation co-mentions another candidate and an observed "
            "marker but does not explicitly connect them."
        )
        return "context_only", reasons

    reasons.append(
        "The quotation does not directly connect an observed marker "
        "to the candidate."
    )
    return "context_only", reasons


def audit_semantics(
    benchmark: dict[str, Any],
    corpus: dict[str, Any],
) -> dict[str, Any]:
    """Audit model claims without using an expert or ground-truth label."""

    request = corpus.get("literature_request", {})
    if not isinstance(request, dict):
        raise ValueError("The evidence corpus has no literature_request.")

    cluster_id = str(request.get("cluster_id", corpus.get("cluster_id", "")))
    if cluster_id != str(benchmark.get("cluster_id", "")):
        raise ValueError("Benchmark and corpus cluster IDs do not match.")

    observed_markers = [
        str(item.get("gene", "")).upper()
        for item in request.get("marker_genes", [])
        if isinstance(item, dict) and item.get("gene")
    ]
    observed_marker_set = set(observed_markers)

    candidates = [
        str(item.get("name", ""))
        for item in request.get("candidate_annotations", [])
        if isinstance(item, dict) and item.get("name")
    ]

    passage_by_id = {
        str(passage.get("passage_id")): passage
        for passage in corpus.get("passages", [])
        if isinstance(passage, dict) and passage.get("passage_id")
    }

    grounding_by_candidate: dict[str, dict[int, bool]] = {}
    for result in benchmark.get("results", []):
        if not isinstance(result, dict):
            continue
        candidate = str(result.get("candidate_cell_type", ""))
        grounding_by_candidate[candidate] = {
            int(check.get("claim_number", 0)): bool(check.get("grounded"))
            for check in result.get("grounding_checks", [])
            if isinstance(check, dict) and check.get("claim_number")
        }

    candidate_audits: list[dict[str, Any]] = []
    total_claims = 0
    unchanged_claims = 0
    reclassified_claims = 0
    rejected_claims = 0

    for result in benchmark.get("results", []):
        if not isinstance(result, dict):
            continue

        candidate = str(result.get("candidate_cell_type", ""))
        response = result.get("response", {})
        if not isinstance(response, dict):
            response = {}

        claims = response.get("claims", [])
        if not isinstance(claims, list):
            claims = []

        audited_claims: list[dict[str, Any]] = []
        stance_counts: Counter[str] = Counter()

        for claim_number, claim in enumerate(claims, start=1):
            if not isinstance(claim, dict):
                continue

            total_claims += 1
            passage_id = str(claim.get("passage_id", ""))
            quote = str(claim.get("evidence_quote", ""))
            original_stance = str(claim.get("stance", ""))
            passage = passage_by_id.get(passage_id, {})

            reported_markers = [
                str(marker)
                for marker in claim.get("markers", [])
            ]
            reported_observed_markers = sorted({
                marker.upper()
                for marker in reported_markers
                if marker.upper() in observed_marker_set
            })
            reported_outside_observed = sorted({
                marker
                for marker in reported_markers
                if marker.upper() not in observed_marker_set
            })
            observed_quote_markers = [
                marker
                for marker in observed_markers
                if _contains_gene(quote, marker)
            ]
            mentioned_candidates = [
                name
                for name in candidates
                if _mentions_candidate(quote, name)
            ]
            linked_candidates = [
                name
                for name in candidates
                if _directly_links_candidate_and_marker(
                    text=quote,
                    candidate=name,
                    markers=observed_quote_markers,
                )
            ]

            corrected_stance, reasons = _corrected_stance(
                candidate=candidate,
                quote=quote,
                mentioned_candidates=mentioned_candidates,
                linked_candidates=linked_candidates,
                observed_quote_markers=observed_quote_markers,
            )

            grounded = grounding_by_candidate.get(candidate, {}).get(
                claim_number,
                False,
            )

            if not passage:
                reasons.append("The cited passage does not exist in the corpus.")

            if not grounded:
                reasons.append("The quotation failed deterministic grounding.")

            if reported_outside_observed:
                reasons.append(
                    "Reported markers are outside the observed marker list: "
                    + ", ".join(reported_outside_observed)
                    + "."
                )

            context_relevance, context_scope, context_matches = (
                _context_assessment(
                    quote=quote,
                    passage=passage,
                    tissue=str(request.get("tissue", "")),
                    disease_context=str(request.get("disease_context", "")),
                )
            )

            accepted = bool(passage) and grounded
            stance_unchanged = corrected_stance == original_stance
            semantically_valid_as_returned = (
                accepted
                and stance_unchanged
                and not reported_outside_observed
            )

            if not accepted:
                rejected_claims += 1
            elif stance_unchanged and not reported_outside_observed:
                unchanged_claims += 1
            else:
                reclassified_claims += 1

            stance_counts[corrected_stance] += 1

            audited_claims.append(
                {
                    "claim_number": claim_number,
                    "claim_text": claim.get("claim_text", ""),
                    "passage_id": passage_id,
                    "evidence_quote": quote,
                    "original_stance": original_stance,
                    "corrected_stance": corrected_stance,
                    "grounded": grounded,
                    "accepted_for_appraisal": accepted,
                    "semantically_valid_as_returned": (
                        semantically_valid_as_returned
                    ),
                    "reported_markers": reported_markers,
                    "reported_observed_markers": (
                        reported_observed_markers
                    ),
                    "reported_markers_outside_observed": (
                        reported_outside_observed
                    ),
                    "observed_markers_in_quote": (
                        observed_quote_markers
                    ),
                    "candidate_cell_types_in_quote": (
                        mentioned_candidates
                    ),
                    "directly_linked_candidate_cell_types": (
                        linked_candidates
                    ),
                    "context_relevance": context_relevance,
                    "context_scope": context_scope,
                    "context_matches": context_matches,
                    "reasons": reasons,
                    "reference": _publication_dict(passage),
                }
            )

        candidate_audits.append(
            {
                "candidate_cell_type": candidate,
                "model_claim_count": len(claims),
                "model_insufficient_evidence": response.get(
                    "insufficient_evidence"
                ),
                "corrected_stance_counts": dict(stance_counts),
                "claims": audited_claims,
            }
        )

    return {
        "semantic_guard_version": "1.1",
        "cluster_id": cluster_id,
        "model_id": benchmark.get("model_id"),
        "prompt_version": benchmark.get("prompt_version"),
        "expert_label_used": False,
        "total_claims": total_claims,
        "unchanged_claims": unchanged_claims,
        "reclassified_claims": reclassified_claims,
        "rejected_claims": rejected_claims,
        "all_claims_semantically_valid_as_returned": (
            total_claims > 0
            and unchanged_claims == total_claims
        ),
        "candidate_audits": candidate_audits,
    }
