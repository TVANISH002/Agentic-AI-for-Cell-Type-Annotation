from __future__ import annotations

from literature_retrieval_agent.context_expansion import (
    ContextExpansionTerms,
    ContextQueryAudit,
    ContextQueryConfig,
)
from literature_retrieval_agent.schemas import (
    CandidateAnnotation,
    LiteratureQuery,
    LiteratureQueryPlan,
    LiteratureRequest,
)


def _escape(value: str) -> str:
    return (
        " ".join(str(value).split())
        .replace("\\", " ")
        .replace('"', " ")
        .strip()
    )


def _title_abstract(value: str) -> str:
    return f'TITLE_ABS:"{_escape(value)}"'


def _unique(values: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()

    for value in values:
        cleaned = _escape(value)
        key = cleaned.casefold()

        if cleaned and key not in seen:
            seen.add(key)
            unique.append(cleaned)

    return unique


def _or_clause(values: list[str]) -> str:
    terms = [_title_abstract(value) for value in _unique(values)]

    if not terms:
        raise ValueError("Cannot build an OR clause from empty values.")

    if len(terms) == 1:
        return terms[0]

    return "(" + " OR ".join(terms) + ")"


def _species_terms(species: str) -> list[str]:
    normalized = " ".join(species.split()).casefold()

    if normalized in {"human", "humans", "homo sapiens"}:
        return ["human", "Homo sapiens", "patient", "patients"]

    if normalized in {"mouse", "mice", "mus musculus"}:
        return ["mouse", "mice", "Mus musculus"]

    return [species]


def _marker_pool(
    candidate: CandidateAnnotation,
    observed_markers: list[str],
) -> list[str]:
    """Prioritize upstream candidate genes, then observed marker rank."""

    return _unique([*candidate.supporting_genes, *observed_markers])


def _batches(
    values: list[str],
    *,
    batch_size: int,
    maximum_batches: int,
) -> list[list[str]]:
    maximum_values = batch_size * maximum_batches
    selected = values[:maximum_values]

    return [
        selected[index : index + batch_size]
        for index in range(0, len(selected), batch_size)
    ]


class ContextExpandedQueryBuilder:
    """
    Build a balanced, context-first Europe PMC plan for any cluster.

    The builder accepts only LiteratureRequest and external vocabulary. It
    has no parameter for an expert label or expected final annotation.
    """

    def __init__(
        self,
        query_config: ContextQueryConfig | None = None,
        expansion_terms: ContextExpansionTerms | None = None,
    ) -> None:
        self.query_config = query_config or ContextQueryConfig()
        self.expansion_terms = expansion_terms or ContextExpansionTerms()

    def _context_clauses(
        self,
        request: LiteratureRequest,
    ) -> tuple[str, str, str, str]:
        species_clause = _or_clause(_species_terms(request.species))
        tissue_clause = _or_clause(
            [request.tissue, *self.expansion_terms.tissue_aliases]
        )
        disease_clause = _or_clause(
            [
                request.disease_context,
                *self.expansion_terms.disease_aliases,
            ]
        )
        platform_clause = _or_clause(
            [
                request.platform,
                *self.expansion_terms.platform_aliases,
                "single-cell",
                "single cell RNA sequencing",
                "spatial transcriptomics",
            ]
        )

        return (
            species_clause,
            tissue_clause,
            disease_clause,
            platform_clause,
        )

    def build(
        self,
        request: LiteratureRequest,
    ) -> LiteratureQueryPlan:
        if not request.candidate_annotations:
            raise ValueError(
                "Context-expanded retrieval requires at least one candidate."
            )

        observed_markers = [marker.gene for marker in request.marker_genes]

        if not observed_markers:
            raise ValueError(
                "Context-expanded retrieval requires observed marker genes."
            )

        (
            species_clause,
            tissue_clause,
            disease_clause,
            platform_clause,
        ) = self._context_clauses(request)

        context_prefix = (
            f"{species_clause} AND {tissue_clause} AND {disease_clause}"
        )

        all_candidate_terms: list[str] = []

        for candidate in request.candidate_annotations:
            all_candidate_terms.extend(
                [
                    candidate.name,
                    *self.expansion_terms.aliases_for_candidate(
                        candidate.name
                    ),
                ]
            )

        queries: list[LiteratureQuery] = []
        next_number = 1

        queries.append(
            LiteratureQuery(
                query_id=f"V2_Q{next_number:02d}_context_overview",
                purpose="context_overview",
                query=(
                    f"{context_prefix} AND {_or_clause(all_candidate_terms)} "
                    f"AND {platform_clause}"
                ),
                candidate_annotations=[
                    candidate.name for candidate in request.candidate_annotations
                ],
                marker_genes=[],
            )
        )
        next_number += 1

        for candidate_index, candidate in enumerate(
            request.candidate_annotations,
            start=1,
        ):
            candidate_terms = [
                candidate.name,
                *self.expansion_terms.aliases_for_candidate(candidate.name),
            ]
            candidate_clause = _or_clause(candidate_terms)

            queries.append(
                LiteratureQuery(
                    query_id=(
                        f"V2_Q{next_number:02d}_candidate_"
                        f"{candidate_index:02d}_context"
                    ),
                    purpose="candidate_context",
                    query=f"{context_prefix} AND {candidate_clause}",
                    candidate_annotations=[candidate.name],
                    marker_genes=[],
                )
            )
            next_number += 1

            marker_batches = _batches(
                _marker_pool(candidate, observed_markers),
                batch_size=self.query_config.marker_batch_size,
                maximum_batches=self.query_config.max_marker_batches,
            )

            for batch_index, marker_batch in enumerate(
                marker_batches,
                start=1,
            ):
                queries.append(
                    LiteratureQuery(
                        query_id=(
                            f"V2_Q{next_number:02d}_candidate_"
                            f"{candidate_index:02d}_marker_batch_"
                            f"{batch_index:02d}"
                        ),
                        purpose="candidate_support",
                        query=(
                            f"{context_prefix} AND {candidate_clause} "
                            f"AND {_or_clause(marker_batch)}"
                        ),
                        candidate_annotations=[candidate.name],
                        marker_genes=marker_batch,
                    )
                )
                next_number += 1

        if self.query_config.include_cluster_marker_queries:
            global_batches = _batches(
                observed_markers,
                batch_size=self.query_config.marker_batch_size,
                maximum_batches=self.query_config.max_marker_batches,
            )

            for batch_index, marker_batch in enumerate(
                global_batches,
                start=1,
            ):
                queries.append(
                    LiteratureQuery(
                        query_id=(
                            f"V2_Q{next_number:02d}_cluster_marker_batch_"
                            f"{batch_index:02d}"
                        ),
                        purpose="cluster_marker_context",
                        query=(
                            f"{context_prefix} AND {_or_clause(marker_batch)}"
                        ),
                        candidate_annotations=[],
                        marker_genes=marker_batch,
                    )
                )
                next_number += 1

        if self.query_config.include_platform_context_query:
            queries.append(
                LiteratureQuery(
                    query_id=f"V2_Q{next_number:02d}_platform_context",
                    purpose="platform_context",
                    query=(
                        f"{context_prefix} AND {platform_clause} AND "
                        f"{_or_clause(all_candidate_terms)}"
                    ),
                    candidate_annotations=[
                        candidate.name
                        for candidate in request.candidate_annotations
                    ],
                    marker_genes=[],
                )
            )

        return LiteratureQueryPlan(
            schema_name="LiteratureQueryPlan",
            schema_version=self.query_config.query_plan_schema_version,
            cluster_id=request.cluster_id,
            queries=queries,
        )

    def audit(
        self,
        request: LiteratureRequest,
        plan: LiteratureQueryPlan,
    ) -> ContextQueryAudit:
        candidate_query_counts: dict[str, int] = {}
        candidate_marker_batch_counts: dict[str, int] = {}
        candidate_alias_counts: dict[str, int] = {}
        all_context_constrained = True

        _, tissue_clause, disease_clause, _ = self._context_clauses(request)

        for candidate in request.candidate_annotations:
            candidate_queries = [
                query
                for query in plan.queries
                if query.candidate_annotations == [candidate.name]
            ]
            marker_queries = [
                query for query in candidate_queries if query.marker_genes
            ]

            candidate_query_counts[candidate.name] = len(candidate_queries)
            candidate_marker_batch_counts[candidate.name] = len(marker_queries)
            candidate_alias_counts[candidate.name] = len(
                self.expansion_terms.aliases_for_candidate(candidate.name)
            )

            all_context_constrained = all_context_constrained and all(
                tissue_clause in query.query and disease_clause in query.query
                for query in candidate_queries
            )

        query_count_values = set(candidate_query_counts.values())
        marker_count_values = set(candidate_marker_batch_counts.values())
        equal_query_counts = len(query_count_values) <= 1
        equal_marker_counts = len(marker_count_values) <= 1

        request_candidate_keys = {
            candidate.name.casefold()
            for candidate in request.candidate_annotations
        }
        unused_alias_candidates = sorted(
            configured_name
            for configured_name in self.expansion_terms.candidate_aliases
            if configured_name.casefold() not in request_candidate_keys
        )

        errors: list[str] = []
        warnings: list[str] = []

        if not equal_query_counts:
            errors.append("Candidates do not have equal query counts.")

        if not equal_marker_counts:
            errors.append("Candidates do not have equal marker-batch counts.")

        if not all_context_constrained:
            errors.append(
                "At least one candidate query lacks tissue or disease context."
            )

        if unused_alias_candidates:
            warnings.append(
                "Some configured candidate aliases were not used because the "
                "canonical candidates were absent from this request."
            )

        accepted_input_fields = [
            "cluster_id",
            "species",
            "tissue",
            "disease_context",
            "platform",
            "marker_genes",
            "candidate_annotations",
            "review_flags",
            "spatial_context",
            "question",
            "upstream_schema_name",
            "upstream_schema_version",
            "source_agent2_file",
        ]

        return ContextQueryAudit(
            strategy_name=self.query_config.strategy_name,
            cluster_id=request.cluster_id,
            valid=not errors,
            total_query_count=len(plan.queries),
            candidate_query_counts=candidate_query_counts,
            candidate_marker_batch_counts=candidate_marker_batch_counts,
            candidate_alias_counts=candidate_alias_counts,
            equal_candidate_query_counts=equal_query_counts,
            equal_candidate_marker_batch_counts=equal_marker_counts,
            all_candidate_queries_context_constrained=all_context_constrained,
            expert_label_used_as_query_input=False,
            accepted_input_fields=accepted_input_fields,
            unused_alias_config_candidates=unused_alias_candidates,
            errors=errors,
            warnings=warnings,
        )

    def build_with_audit(
        self,
        request: LiteratureRequest,
    ) -> tuple[LiteratureQueryPlan, ContextQueryAudit]:
        plan = self.build(request)
        return plan, self.audit(request, plan)
