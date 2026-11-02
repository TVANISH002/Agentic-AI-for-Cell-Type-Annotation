from __future__ import annotations

from pathlib import Path
from typing import Iterable
import json


def _walk_values(obj):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _walk_values(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_values(value)
    else:
        yield obj


def marker_has_literature_support(rank_coverage_json: Path, gene: str) -> bool:
    """Check whether Agent 3 rank-aware coverage already explained the marker."""
    if not rank_coverage_json.exists():
        return False

    data = json.loads(rank_coverage_json.read_text(encoding="utf-8"))
    gene_upper = gene.upper()

    for value in _walk_values(data):
        if isinstance(value, str) and value.upper() == gene_upper:
            return True
        if isinstance(value, list) and gene_upper in {str(x).upper() for x in value}:
            return True

    return False
