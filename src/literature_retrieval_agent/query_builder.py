from __future__ import annotations

from literature_retrieval_agent.config import (
    DEFAULT_CONFIG,
    LiteratureRetrievalConfig,
)
from literature_retrieval_agent.schemas import (
    LiteratureQuery,
    LiteratureQueryPlan,
    LiteratureRequest,
)


def _escape(value: str) -> str:
    """Escape text before placing it in a query."""

    return (
        str(value)
        .replace("\\", " ")
        .replace('"', " ")
        .strip()
    )


def _title_abstract(value: str) -> str:
    """Create one Europe PMC title/abstract term."""

    return f'TITLE_ABS:"{_escape(value)}"'


def _or_clause(
    values: list[str],
) -> str:
    """Combine several title/abstract terms using OR."""

    unique: list[str] = []
    seen: set[str] = set()

    for value in values:
        cleaned = _escape(value)
        key = cleaned.casefold()

        if cleaned and key not in seen:
            seen.add(key)
            unique.append(
                _title_abstract(cleaned)
            )

    if not unique:
        raise ValueError(
            "Cannot build an OR clause "
            "from empty values."
        )

    if len(unique) == 1:
        return unique[0]

    return (
        "("
        + " OR ".join(unique)
        + ")"
    )


def _species_terms(
    species: str,
) -> list[str]:
    """Return common literature terms for a species."""

    normalized = " ".join(
        species.split()
    ).casefold()

    if normalized in {
        "human",
        "humans",
        "homo sapiens",
    }:
        return [
            "human",
            "Homo sapiens",
        ]

    if normalized in {
        "mouse",
        "mice",
        "mus musculus",
    }:
        return [
            "mouse",
            "mice",
            "Mus musculus",
        ]

    return [species]


def _merge_genes(
    primary: list[str],
    secondary: list[str],
    *,
    limit: int,
) -> list[str]:
    """
    Merge candidate-supporting genes with the
    highest-ranked cluster markers.
    """

    merged: list[str] = []
    seen: set[str] = set()

    for gene in [
        *primary,
        *secondary,
    ]:
        cleaned = gene.strip().upper()

        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            merged.append(cleaned)

        if len(merged) == limit:
            break

    return merged


class LiteratureQueryBuilder:
    """
    Build deterministic, context-specific,
    auditable Europe PMC queries.
    """

    def __init__(
        self,
        config: LiteratureRetrievalConfig = DEFAULT_CONFIG,
    ) -> None:
        self.config = config

    def build(
        self,
        request: LiteratureRequest,
    ) -> LiteratureQueryPlan:
        """Build the complete query plan."""

        context_clause = _or_clause(
            [
                request.disease_context,
                request.tissue,
            ]
        )

        candidate_names = [
            candidate.name
            for candidate
            in request.candidate_annotations
        ]

        candidate_clause = _or_clause(
            candidate_names
        )

        species_clause = _or_clause(
            _species_terms(request.species)
        )

        platform_clause = _or_clause(
            [
                request.platform,
                "spatial transcriptomics",
                "single-cell",
            ]
        )

        top_markers = [
            marker.gene
            for marker
            in request.marker_genes[
                : self.config.max_marker_genes_per_query
            ]
        ]

        top_marker_clause = _or_clause(
            top_markers
        )

        queries: list[LiteratureQuery] = []

        queries.append(
            LiteratureQuery(
                query_id="Q01_context_overview",
                purpose="context_overview",
                query=(
                    f"{species_clause} "
                    f"AND {context_clause} "
                    f"AND {candidate_clause} "
                    f"AND {platform_clause}"
                ),
                candidate_annotations=(
                    candidate_names
                ),
                marker_genes=[],
            )
        )

        next_number = 2

        for candidate in (
            request.candidate_annotations
        ):
            candidate_genes = _merge_genes(
                candidate.supporting_genes[
                    : self.config
                    .max_candidate_supporting_genes
                ],
                top_markers,
                limit=(
                    self.config
                    .max_marker_genes_per_query
                ),
            )

            queries.append(
                LiteratureQuery(
                    query_id=(
                        f"Q{next_number:02d}"
                        "_candidate_support"
                    ),
                    purpose="candidate_support",
                    query=(
                        f"{_title_abstract(candidate.name)} "
                        f"AND {_or_clause(candidate_genes)}"
                    ),
                    candidate_annotations=[
                        candidate.name
                    ],
                    marker_genes=candidate_genes,
                )
            )

            next_number += 1

            queries.append(
                LiteratureQuery(
                    query_id=(
                        f"Q{next_number:02d}"
                        "_candidate_context"
                    ),
                    purpose="candidate_context",
                    query=(
                        f"{context_clause} "
                        f"AND {_title_abstract(candidate.name)} "
                        f"AND {platform_clause}"
                    ),
                    candidate_annotations=[
                        candidate.name
                    ],
                    marker_genes=[],
                )
            )

            next_number += 1

        queries.append(
            LiteratureQuery(
                query_id=(
                    f"Q{next_number:02d}"
                    "_cluster_marker_context"
                ),
                purpose="cluster_marker_context",
                query=(
                    f"{context_clause} "
                    f"AND {top_marker_clause}"
                ),
                candidate_annotations=[],
                marker_genes=top_markers,
            )
        )

        next_number += 1

        queries.append(
            LiteratureQuery(
                query_id=(
                    f"Q{next_number:02d}"
                    "_candidate_comparison"
                ),
                purpose="candidate_comparison",
                query=(
                    f"{context_clause} "
                    f"AND {candidate_clause} "
                    f"AND {top_marker_clause}"
                ),
                candidate_annotations=(
                    candidate_names
                ),
                marker_genes=top_markers,
            )
        )

        next_number += 1

        queries.append(
            LiteratureQuery(
                query_id=(
                    f"Q{next_number:02d}"
                    "_platform_context"
                ),
                purpose="platform_context",
                query=(
                    f"{context_clause} "
                    f"AND {_title_abstract(request.platform)} "
                    'AND (TITLE_ABS:"cell annotation" '
                    'OR TITLE_ABS:"cell type")'
                ),
                candidate_annotations=[],
                marker_genes=[],
            )
        )

        unique_queries: list[
            LiteratureQuery
        ] = []

        seen_queries: set[str] = set()

        for query in queries:
            key = query.query.casefold()

            if key not in seen_queries:
                seen_queries.add(key)
                unique_queries.append(query)

        return LiteratureQueryPlan(
            schema_name=(
                self.config.query_plan_schema_name
            ),
            schema_version=(
                self.config.query_plan_schema_version
            ),
            cluster_id=request.cluster_id,
            queries=unique_queries,
        )