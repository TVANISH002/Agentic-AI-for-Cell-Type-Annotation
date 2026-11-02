from __future__ import annotations

from collections import defaultdict
from typing import Any


_CONTEXT_WEIGHTS = {
    "high": 1.0,
    "moderate": 0.6,
    "low": 0.25,
}

_RESTRICTIVE_TERMS = (
    "restricted to",
    "specific to",
    "specific for",
    "exclusive to",
    "exclusively",
    "only in",
)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _candidate_data(
    request: dict[str, Any],
) -> tuple[list[str], dict[str, set[str]]]:
    names: list[str] = []
    supporting_genes: dict[str, set[str]] = {}

    for item in request.get("candidate_annotations", []):
        if not isinstance(item, dict):
            continue

        name = _clean(item.get("name"))
        if not name:
            continue

        names.append(name)
        supporting_genes[name] = {
            _clean(gene).upper()
            for gene in item.get("supporting_genes", [])
            if _clean(gene)
        }

    return names, supporting_genes


def _deduplicate_claims(
    claims: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}

    for claim in claims:
        if not claim.get("accepted_for_appraisal"):
            continue

        key = (
            _clean(claim.get("passage_id")),
            _clean(claim.get("evidence_quote")),
            _clean(claim.get("corrected_stance")),
        )

        if key not in grouped:
            grouped[key] = {
                "stance": key[2],
                "claim_texts": [],
                "evidence_quote": key[1],
                "passage_id": key[0],
                "markers": set(),
                "context_relevance": claim.get(
                    "context_relevance", "low"
                ),
                "context_scope": claim.get(
                    "context_scope", "out_of_context"
                ),
                "reasons": set(),
                "reference": claim.get("reference", {}),
            }

        row = grouped[key]
        claim_text = _clean(claim.get("claim_text"))
        if claim_text:
            row["claim_texts"].append(claim_text)

        row["markers"].update(
            _clean(marker).upper()
            for marker in claim.get("observed_markers_in_quote", [])
            if _clean(marker)
        )
        row["reasons"].update(
            _clean(reason)
            for reason in claim.get("reasons", [])
            if _clean(reason)
        )

    records: list[dict[str, Any]] = []
    for row in grouped.values():
        records.append(
            {
                **row,
                "claim_texts": sorted(set(row["claim_texts"])),
                "markers": sorted(row["markers"]),
                "reasons": sorted(row["reasons"]),
            }
        )

    return sorted(
        records,
        key=lambda item: (
            item["stance"],
            item["passage_id"],
            item["evidence_quote"],
        ),
    )


def _reference_record(
    evidence: dict[str, Any],
    evidence_role: str,
) -> dict[str, Any]:
    reference = evidence.get("reference", {})
    if not isinstance(reference, dict):
        reference = {}

    claim_texts = evidence.get("claim_texts", [])
    supported_claim = (
        evidence.get("derived_claim", "")
        if evidence_role == "contradictory"
        and evidence.get("derived_claim")
        else "; ".join(claim_texts)
    )

    return {
        "title": reference.get("title"),
        "authors": reference.get("authors", []),
        "journal": reference.get("journal"),
        "year": reference.get("year"),
        "doi": reference.get("doi"),
        "pmid": reference.get("pmid"),
        "pmcid": reference.get("pmcid"),
        "supported_claim": supported_claim,
        "evidence_role": evidence_role,
        "context_relevance": evidence.get("context_relevance"),
        "passage_id": evidence.get("passage_id"),
        "evidence_quote": evidence.get("evidence_quote"),
    }


def _score_candidate(
    supporting: list[dict[str, Any]],
    ambiguous: list[dict[str, Any]],
) -> tuple[float, str]:
    if not supporting:
        return 0.0, "insufficient"

    markers = {
        marker
        for claim in supporting
        for marker in claim.get("markers", [])
    }
    papers = {
        _clean((claim.get("reference") or {}).get("doi"))
        or _clean(claim.get("passage_id"))
        for claim in supporting
    }
    context_weights = [
        _CONTEXT_WEIGHTS.get(
            _clean(claim.get("context_relevance")),
            0.0,
        )
        for claim in supporting
    ]
    average_context = (
        sum(context_weights) / len(context_weights)
        if context_weights
        else 0.0
    )

    score = (
        min(0.45, 0.15 * len(markers))
        + min(0.20, 0.10 * len(papers))
        + 0.25 * average_context
        - min(0.15, 0.05 * len(ambiguous))
    )

    if not any(
        claim.get("context_relevance") in {"high", "moderate"}
        for claim in supporting
    ):
        score = min(score, 0.59)

    score = round(max(0.0, min(1.0, score)), 2)

    if score >= 0.75:
        strength = "strong"
    elif score >= 0.45:
        strength = "moderate"
    else:
        strength = "weak"

    return score, strength


def _context_summary(
    supporting: list[dict[str, Any]],
    context_only: list[dict[str, Any]],
    tissue: str,
    disease_context: str,
) -> str:
    if not supporting and not context_only:
        return "Insufficient: no validated candidate-specific evidence."

    direct_levels = {
        claim.get("context_relevance")
        for claim in supporting
    }
    context_levels = {
        claim.get("context_relevance")
        for claim in context_only
    }

    if not supporting and context_only:
        if "high" in context_levels:
            return (
                "Context only: in-context literature mentions the "
                "candidate, but no observed cluster marker is directly "
                "linked to that candidate."
            )

        return (
            "Context only: literature mentions the candidate, but no "
            "observed cluster marker is directly linked to that candidate."
        )

    if supporting and direct_levels.issubset({"low"}):
        if "high" in context_levels:
            return (
                "Mixed: direct marker evidence is outside the supplied "
                f"{tissue}/{disease_context} context, while in-context "
                "papers provide only context-level evidence."
            )

        return (
            "Low: direct marker evidence is outside the supplied "
            f"{tissue}/{disease_context} context."
        )

    if "high" in direct_levels:
        return (
            "High: at least one direct marker claim matches the supplied "
            f"{tissue}/{disease_context} context."
        )

    return "Partial: some, but not all, direct evidence matches the context."


def _support_summary(
    candidate: str,
    supporting: list[dict[str, Any]],
    ambiguous: list[dict[str, Any]],
    context_only: list[dict[str, Any]],
) -> str:
    if not supporting:
        return (
            "No validated direct marker evidence supporting "
            f"{candidate} was found in the supplied passages."
        )

    markers = sorted({
        marker
        for claim in supporting
        for marker in claim.get("markers", [])
    })
    papers = {
        _clean((claim.get("reference") or {}).get("doi"))
        or _clean(claim.get("passage_id"))
        for claim in supporting
    }
    low_context = all(
        claim.get("context_relevance") == "low"
        for claim in supporting
    )

    paper_word = "paper" if len(papers) == 1 else "papers"
    summary = (
        f"Validated literature directly links {', '.join(markers)} to "
        f"{candidate} across {len(papers)} unique {paper_word}."
    )

    if low_context:
        summary += (
            " These direct marker claims come from disease/tissue contexts "
            "outside the supplied dataset."
        )

    if ambiguous:
        if len(ambiguous) == 1:
            summary += " 1 additional claim is ambiguous."
        else:
            summary += f" {len(ambiguous)} additional claims are ambiguous."

    if context_only:
        if len(context_only) == 1:
            summary += " 1 additional claim provides context only."
        else:
            summary += (
                f" {len(context_only)} additional claims provide context only."
            )

    return summary


def _cross_candidate_contradictions(
    evidence_by_candidate: dict[str, list[dict[str, Any]]],
    supporting_genes: dict[str, set[str]],
) -> dict[str, list[dict[str, Any]]]:
    contradictions: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for source_candidate, records in evidence_by_candidate.items():
        for evidence in records:
            if evidence.get("stance") != "supporting":
                continue

            quote = _clean(evidence.get("evidence_quote")).casefold()
            if not any(term in quote for term in _RESTRICTIVE_TERMS):
                continue

            markers = set(evidence.get("markers", []))
            for target_candidate, genes in supporting_genes.items():
                if target_candidate == source_candidate:
                    continue

                overlap = sorted(markers.intersection(genes))
                if not overlap:
                    continue

                derived = {
                    **evidence,
                    "stance": "contradictory",
                    "source_candidate": source_candidate,
                    "markers": overlap,
                    "derived_claim": (
                        f"The quotation restricts a module containing "
                        f"{', '.join(overlap)} to {source_candidate}, "
                        f"weakening marker-only support for {target_candidate}."
                    ),
                }
                contradictions[target_candidate].append(derived)

    return contradictions


def build_literature_evidence_packet(
    semantic_audit: dict[str, Any],
    corpus: dict[str, Any],
) -> dict[str, Any]:
    """Create the professor-aligned literature evidence packet."""

    request = corpus.get("literature_request", {})
    if not isinstance(request, dict):
        raise ValueError("Corpus does not contain a literature_request object.")

    cluster_id = _clean(request.get("cluster_id", corpus.get("cluster_id")))
    if cluster_id != _clean(semantic_audit.get("cluster_id")):
        raise ValueError("Semantic audit and corpus cluster IDs do not match.")

    candidates, supporting_genes = _candidate_data(request)
    audit_by_candidate = {
        _clean(item.get("candidate_cell_type")): item
        for item in semantic_audit.get("candidate_audits", [])
        if isinstance(item, dict)
    }

    evidence_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        audit = audit_by_candidate.get(candidate, {})
        claims = audit.get("claims", [])
        evidence_by_candidate[candidate] = _deduplicate_claims(
            claims if isinstance(claims, list) else []
        )

    cross_contradictions = _cross_candidate_contradictions(
        evidence_by_candidate=evidence_by_candidate,
        supporting_genes=supporting_genes,
    )

    candidate_packets: list[dict[str, Any]] = []
    global_ambiguities: list[str] = []

    for candidate in candidates:
        records = evidence_by_candidate[candidate]
        supporting = [
            row for row in records if row["stance"] == "supporting"
        ]
        ambiguous = [
            row for row in records if row["stance"] == "ambiguous"
        ]
        context_only = [
            row for row in records if row["stance"] == "context_only"
        ]
        contradictory = cross_contradictions.get(candidate, []) + [
            row for row in records if row["stance"] == "contradictory"
        ]

        supporting_markers = sorted({
            marker
            for row in supporting
            for marker in row.get("markers", [])
        })
        for row in ambiguous:
            marker_text = ", ".join(row.get("markers", [])) or "The claim"
            global_ambiguities.append(
                f"{marker_text} evidence for {candidate} is ambiguous: "
                + "; ".join(row.get("reasons", []))
            )

        for row in context_only:
            global_ambiguities.append(
                f"A {candidate} paper provides context-only evidence "
                "without directly linking an observed marker."
            )

        confidence, strength = _score_candidate(
            supporting=supporting,
            ambiguous=ambiguous,
        )

        if not supporting and contradictory:
            strength = "weak"

        context_summary = _context_summary(
            supporting,
            context_only,
            _clean(request.get("tissue")),
            _clean(request.get("disease_context")),
        )

        if not supporting and contradictory:
            contradiction_levels = {
                row.get("context_relevance")
                for row in contradictory
            }
            if contradiction_levels.issubset({"low"}):
                context_summary = (
                    "Low: no direct supporting evidence was found, and "
                    "the available contradictory evidence is outside the "
                    "supplied tissue/disease context."
                )
            else:
                context_summary = (
                    "Contradictory: no direct supporting evidence was found, "
                    "but candidate-specific contradictory evidence exists."
                )

        references: list[dict[str, Any]] = []
        references.extend(
            _reference_record(row, "supporting")
            for row in supporting
        )
        references.extend(
            _reference_record(row, "contradictory")
            for row in contradictory
        )
        references.extend(
            _reference_record(row, "ambiguous")
            for row in ambiguous
        )
        references.extend(
            _reference_record(row, "context_only")
            for row in context_only
        )

        candidate_packets.append(
            {
                "candidate_cell_type": candidate,
                "supporting_markers": supporting_markers,
                "support_summary": _support_summary(
                    candidate,
                    supporting,
                    ambiguous,
                    context_only,
                ),
                "contradictory_evidence": [
                    {
                        "claim": row.get("derived_claim")
                        or "; ".join(row.get("claim_texts", [])),
                        "markers": row.get("markers", []),
                        "evidence_quote": row.get("evidence_quote"),
                        "passage_id": row.get("passage_id"),
                        "context_relevance": row.get("context_relevance"),
                        "reference": row.get("reference", {}),
                    }
                    for row in contradictory
                ],
                "context_relevance": context_summary,
                "evidence_strength": strength,
                "references": references,
                "direct_supporting_claim_count": len(supporting),
                "direct_contradictory_claim_count": len(contradictory),
                "confidence_contribution": confidence,
                "unsupported_upstream_markers": sorted(
                    supporting_genes.get(candidate, set())
                    - set(supporting_markers)
                ),
            }
        )

    ranked = sorted(
        candidate_packets,
        key=lambda item: item["confidence_contribution"],
        reverse=True,
    )
    top = ranked[0] if ranked else None
    second_score = (
        ranked[1]["confidence_contribution"]
        if len(ranked) > 1
        else 0.0
    )

    if (
        top
        and top["confidence_contribution"] >= 0.45
        and top["confidence_contribution"] - second_score >= 0.15
    ):
        conclusion = (
            "Validated literature evidence provisionally favors "
            f"{top['candidate_cell_type']} over the other supplied "
            "candidates. Direct marker evidence is not fully matched to "
            "the supplied tissue/disease context, so this is a literature "
            "contribution rather than a final annotation."
        )
        confidence_contribution = top["confidence_contribution"]
    else:
        conclusion = (
            "The current validated literature evidence is insufficient "
            "to distinguish the supplied candidate annotations."
        )
        confidence_contribution = (
            top["confidence_contribution"] if top else 0.0
        )

    return {
        "schema_name": "LiteratureEvidencePacket",
        "schema_version": "1.0",
        "evidence_appraiser_version": "1.1",
        "semantic_guard_version": semantic_audit.get(
            "semantic_guard_version"
        ),
        "cluster_id": cluster_id,
        "candidate_evidence": candidate_packets,
        "ambiguities": sorted(set(global_ambiguities)),
        "missing_discriminating_markers": [],
        "missing_marker_definition": (
            "No additional discriminating markers were proposed by the "
            "validated model evidence. Candidate-specific upstream markers "
            "that lack direct literature support are reported separately "
            "under unsupported_upstream_markers."
        ),
        "literature_based_conclusion": conclusion,
        "confidence_contribution": confidence_contribution,
        "confidence_interpretation": (
            "Heuristic literature-evidence strength only; it is not the "
            "final annotation confidence and requires later calibration."
        ),
        "confidence_method": {
            "marker_diversity_weight": 0.45,
            "independent_paper_weight": 0.20,
            "context_relevance_weight": 0.25,
            "ambiguity_penalty_per_claim": 0.05,
            "out_of_context_direct_evidence_cap": 0.59,
        },
        "final_annotation_selected": False,
        "expert_label_used_as_agent_input": False,
        "model_id": semantic_audit.get("model_id"),
        "prompt_version": semantic_audit.get("prompt_version"),
        "warnings": [],
    }
