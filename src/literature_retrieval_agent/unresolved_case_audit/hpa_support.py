from __future__ import annotations

from pathlib import Path


def marker_has_hpa_support(agent2_json: Path, gene: str, candidate_terms: list[str]) -> bool:
    """Check whether HPA supplemental evidence links a marker to candidate lineage.

    This is intentionally separate from Agent 2 voting. It only tells Agent 3 audit
    whether the marker should be considered completely unexplained.
    """
    if not agent2_json.exists():
        return False

    text = agent2_json.read_text(encoding="utf-8", errors="ignore").lower()
    gene_hit = gene.lower() in text
    hpa_hit = "hpa" in text or "human protein atlas" in text
    candidate_hit = any(term.lower() in text for term in candidate_terms)

    return gene_hit and hpa_hit and candidate_hit
