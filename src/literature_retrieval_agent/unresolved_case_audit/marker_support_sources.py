from __future__ import annotations

from pathlib import Path
from typing import Any
import json


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def marker_has_structured_support(agent2_json: Path, gene: str, candidate_terms: list[str]) -> bool:
    """Check whether Agent 2 structured sources support gene for candidate terms.

    This is a conservative placeholder. We will make this schema-aware after inspecting
    the exact Agent 2 JSON structure for breast/lung.
    """
    if not agent2_json.exists():
        return False

    text = agent2_json.read_text(encoding="utf-8", errors="ignore").lower()
    gene_hit = gene.lower() in text
    candidate_hit = any(term.lower() in text for term in candidate_terms)
    source_hit = ("cellmarker" in text) or ("panglaodb" in text)

    return gene_hit and candidate_hit and source_hit
