from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from literature_retrieval_agent.runtime_context import (
    Agent3RuntimeContext,
)


def _clean(value: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(value or ""),
    ).strip()


def _quote(value: str) -> str:
    value = _clean(value).replace(
        '"',
        "",
    )
    return f'"{value}"'


def _unique(values: Sequence[str]) -> list[str]:
    seen = set()
    output = []

    for value in values:
        value = _clean(value)

        if not value:
            continue

        key = value.casefold()

        if key in seen:
            continue

        seen.add(key)
        output.append(value)

    return output


def _or_clause(values: Sequence[str]) -> str:
    values = _unique(values)

    if not values:
        return ""

    return (
        "("
        + " OR ".join(
            _quote(value)
            for value in values
        )
        + ")"
    )


def _extract_marker_gene(value: Any) -> str:
    """
    Accept either a plain marker-gene string
    or a structured marker record.
    """

    if isinstance(
        value,
        Mapping,
    ):
        for key in (
            "gene",
            "GENE",
            "symbol",
            "SYMBOL",
            "marker",
            "MARKER",
        ):
            candidate = _clean(
                value.get(
                    key,
                    "",
                )
            )

            if candidate:
                return (
                    candidate.upper()
                )

        return ""

    return _clean(
        value
    ).upper()


class RescueConfirmationQueryBuilder:
    """
    Build target-context literature queries for
    ontology-discovered rescue candidates.

    Cluster, species, tissue, disease context,
    candidates, and rescue markers are supplied
    dynamically at runtime.

    No expert/reference annotation is used.
    """

    def build(
        self,
        candidate_discovery: Mapping[str, Any],
        rescue_plan: Mapping[str, Any],
        literature_request: Mapping[str, Any],
        *,
        source_candidate_discovery: str = "",
        source_rescue_plan: str = "",
        source_literature_request: str = "",
    ) -> dict:

        context = (
            Agent3RuntimeContext
            .from_mapping(
                literature_request
            )
        )

        # Cluster consistency
        

        cluster_ids = {
            str(
                candidate_discovery.get(
                    "cluster_id"
                )
            ),
            str(
                rescue_plan.get(
                    "cluster_id"
                )
            ),
            str(
                literature_request.get(
                    "cluster_id"
                )
            ),
        }

        if len(cluster_ids) != 1:
            raise ValueError(
                "Cluster mismatch across "
                "Agent 3 input artifacts."
            )

       
        # Rescue markers
       

        rescue_markers = []

        for marker in rescue_plan.get(
            "rescue_markers",
            [],
        ):
            gene = _extract_marker_gene(
                marker
            )

            if (
                gene
                and gene not in rescue_markers
            ):
                rescue_markers.append(
                    gene
                )

        
        # Compatibility fallback for plans where markers exist only inside query records.
        
        if not rescue_markers:

            for query in rescue_plan.get(
                "queries",
                [],
            ):

                raw_markers = (
                    query.get(
                        "markers"
                    )
                    or query.get(
                        "marker_genes"
                    )
                    or []
                )

                for marker in raw_markers:

                    gene = (
                        _extract_marker_gene(
                            marker
                        )
                    )

                    if (
                        gene
                        and gene
                        not in rescue_markers
                    ):
                        rescue_markers.append(
                            gene
                        )

        if not rescue_markers:
            raise ValueError(
                "No rescue markers found."
            )

        
        # Ontology-discovered candidates
        

        candidates = []

        for item in (
            candidate_discovery.get(
                "results",
                [],
            )
        ):

            name = _clean(
                item.get(
                    "cell_type",
                    "",
                )
            )

            ontology_id = _clean(
                item.get(
                    "ontology_id",
                    "",
                )
            )

            if not name:
                continue

            candidates.append(
                {
                    "name":
                        name,

                    "ontology_id":
                        ontology_id,

                    "is_novel_to_agent2":
                        bool(
                            item.get(
                                "is_novel_to_agent2",
                                True,
                            )
                        ),

                    "independent_paper_count":
                        int(
                            item.get(
                                "independent_paper_count",
                                0,
                            )
                            or 0
                        ),
                }
            )

        if not candidates:
            raise ValueError(
                "No rescue candidates found."
            )

        
        # Generic biological context
        

        marker_clause = (
            _or_clause(
                rescue_markers
            )
        )

        context_clause = (
            _or_clause(
                context.context_terms()
            )
        )

        
        # Species aliases come from RuntimeContext.
        # Human additionally receives patient/patients; non-human species do not.
        
        species_terms = list(
            context.species_search_terms
        )

        species_terms.extend(
            context.subject_search_terms
        )

        species_clause = (
            _or_clause(
                species_terms
            )
        )

        modality_clause = (
            '("single cell" OR '
            '"single-cell" OR '
            '"spatial transcriptomics" OR '
            '"spatial gene expression")'
        )

     
        # One confirmation query per discovered candidate
        

        queries = []

        for index, candidate in enumerate(
            candidates,
            start=1,
        ):

            parts = [
                _quote(
                    candidate["name"]
                )
            ]

            if context_clause:
                parts.append(
                    context_clause
                )

            if species_clause:
                parts.append(
                    species_clause
                )

            parts.append(
                modality_clause
            )

            parts.append(
                marker_clause
            )

            queries.append(
                {
                    "query_id":
                        f"CR{index:03d}",

                    "query_family":
                        (
                            "rescue_candidate_"
                            "context_confirmation"
                        ),

                    "candidate":
                        candidate["name"],

                    "ontology_id":
                        candidate[
                            "ontology_id"
                        ],

                    "markers":
                        rescue_markers,

                    "query":
                        " AND ".join(
                            parts
                        ),
                }
            )

        return {
            "schema_name":
                "RescueCandidateConfirmationQueryPlan",

            "schema_version":
                "2.0",

            **context.to_dict(),

            "expert_label_used":
                False,

            "reference_label_used":
                False,

            "source_candidate_discovery":
                source_candidate_discovery,

            "source_rescue_plan":
                source_rescue_plan,

            "source_literature_request":
                source_literature_request,

            "rescue_markers":
                rescue_markers,

            "candidate_count":
                len(candidates),

            "query_count":
                len(queries),

            "queries":
                queries,
        }







