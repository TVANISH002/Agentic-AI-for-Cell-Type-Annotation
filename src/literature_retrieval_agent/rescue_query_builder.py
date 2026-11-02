from __future__ import annotations

import csv
import itertools
from pathlib import Path
from typing import Any, Mapping, Sequence

from literature_retrieval_agent.runtime_context import (
    Agent3RuntimeContext,
)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    return str(value).strip().casefold() in {
        "true",
        "1",
        "yes",
    }


def _split_semicolon(
    value: Any,
) -> list[str]:

    if value is None:
        return []

    text = str(value).strip()

    if (
        not text
        or text.casefold() == "nan"
    ):
        return []

    return [
        item.strip().upper()
        for item in text.split(";")
        if item.strip()
    ]


def _quote(
    value: str,
) -> str:

    value = str(value).strip()

    return (
        '"'
        + value.replace(
            '"',
            '\\"',
        )
        + '"'
    )


def _or_clause(
    values: Sequence[str],
) -> str:

    cleaned = []
    seen = set()

    for value in values:

        value = str(
            value
        ).strip()

        if not value:
            continue

        key = value.casefold()

        if key in seen:
            continue

        seen.add(key)

        cleaned.append(
            _quote(value)
        )

    if not cleaned:
        return ""

    return (
        "("
        + " OR ".join(cleaned)
        + ")"
    )


def _context_clause(
    context: Agent3RuntimeContext,
) -> str:

    return _or_clause(
        context.context_terms()
    )


def _species_clause(
    context: Agent3RuntimeContext,
    *,
    include_subject_terms: bool,
) -> str:

    values = list(
        context.species_search_terms
    )

    if include_subject_terms:
        values.extend(
            context.subject_search_terms
        )

    return _or_clause(
        values
    )


class MarkerRescueQueryBuilder:
    """
    Candidate-independent marker-rescue query builder.

    Biological context, species, cluster identity,
    and marker genes are supplied at runtime.
    No expert/reference annotation is used.
    """

    def __init__(
        self,
        top_marker_limit: int = 5,
        pairwise_marker_limit: int = 3,
    ):

        if top_marker_limit < 1:
            raise ValueError(
                "top_marker_limit must be >= 1."
            )

        if pairwise_marker_limit < 2:
            raise ValueError(
                "pairwise_marker_limit must be >= 2."
            )

        self.top_marker_limit = (
            top_marker_limit
        )

        self.pairwise_marker_limit = (
            pairwise_marker_limit
        )

    @staticmethod
    def load_explained_markers(
        evidence_passages_csv: Path,
    ) -> set[str]:

        explained = set()

        with evidence_passages_csv.open(
            newline="",
            encoding="utf-8",
        ) as handle:

            reader = csv.DictReader(
                handle
            )

            for row in reader:

                if not _truthy(
                    row.get(
                        "candidate_present",
                        False,
                    )
                ):
                    continue

                explained.update(
                    _split_semicolon(
                        row.get(
                            "observed_markers_in_passage",
                            "",
                        )
                    )
                )

        return explained

    def build(
        self,
        request: Mapping[str, Any],
        explained_markers: set[str],
        *,
        source_literature_request: str = "",
        source_pubmedbert_evidence: str = "",
    ) -> dict:

        context = (
            Agent3RuntimeContext
            .from_mapping(
                request
            )
        )

        marker_records = request.get(
            "marker_genes",
            []
        )

        if not marker_records:
            raise ValueError(
                "LiteratureRequest contains "
                "no marker_genes."
            )

        ranked_markers = sorted(
            marker_records,
            key=lambda item: int(
                item[
                    "rank_within_cluster"
                ]
            ),
        )

        top_markers = ranked_markers[
            :self.top_marker_limit
        ]

        top_marker_names = [
            str(
                item["gene"]
            ).upper()
            for item in top_markers
        ]

        normalized_explained = {
            str(gene).upper()
            for gene in explained_markers
        }

        rescue_markers = [
            marker
            for marker in top_markers
            if str(
                marker["gene"]
            ).upper()
            not in normalized_explained
        ]

        context_clause = (
            _context_clause(
                context
            )
        )

        species_context_clause = (
            _species_clause(
                context,
                include_subject_terms=True,
            )
        )

        species_identity_clause = (
            _species_clause(
                context,
                include_subject_terms=False,
            )
        )

        queries = []
        query_index = 1

       
        # FAMILY 1
        # Context-specific single-marker retrieval
      

        for marker in rescue_markers:

            gene = str(
                marker["gene"]
            ).upper()

            pieces = [
                _quote(gene),
            ]

            if context_clause:
                pieces.append(
                    context_clause
                )

            if species_context_clause:
                pieces.append(
                    species_context_clause
                )

            pieces.append(
                "("
                '"single cell" OR '
                '"single-cell" OR '
                '"spatial transcriptomics" OR '
                '"spatial gene expression"'
                ")"
            )

            queries.append(
                {
                    "query_id":
                        f"MR{query_index:02d}",

                    "query_family":
                        "context_specific_marker",

                    "markers":
                        [gene],

                    "query":
                        " AND ".join(
                            pieces
                        ),
                }
            )

            query_index += 1

      
        # FAMILY 2
        # Candidate-independent biological identity
        

        for marker in rescue_markers:

            gene = str(
                marker["gene"]
            ).upper()

            pieces = [
                _quote(gene),

                "("
                '"cell type" OR '
                '"cell identity" OR '
                '"cell marker" OR '
                '"marker gene"'
                ")",
            ]

            if species_identity_clause:
                pieces.append(
                    species_identity_clause
                )

            queries.append(
                {
                    "query_id":
                        f"MR{query_index:02d}",

                    "query_family":
                        "biological_identity",

                    "markers":
                        [gene],

                    "query":
                        " AND ".join(
                            pieces
                        ),
                }
            )

            query_index += 1

     
        # FAMILY 3
        # Pairwise highest-ranked rescue markers
       

        pair_markers = rescue_markers[
            :self.pairwise_marker_limit
        ]

        for left, right in (
            itertools.combinations(
                pair_markers,
                2,
            )
        ):

            gene_left = str(
                left["gene"]
            ).upper()

            gene_right = str(
                right["gene"]
            ).upper()

            pieces = [
                (
                    "("
                    f"{_quote(gene_left)}"
                    " AND "
                    f"{_quote(gene_right)}"
                    ")"
                )
            ]

            if context_clause:
                pieces.append(
                    context_clause
                )

            if species_context_clause:
                pieces.append(
                    species_context_clause
                )

            pieces.append(
                "("
                '"single cell" OR '
                '"single-cell" OR '
                '"spatial transcriptomics" OR '
                '"cell type"'
                ")"
            )

            queries.append(
                {
                    "query_id":
                        f"MR{query_index:02d}",

                    "query_family":
                        "marker_pair",

                    "markers":
                        [
                            gene_left,
                            gene_right,
                        ],

                    "query":
                        " AND ".join(
                            pieces
                        ),
                }
            )

            query_index += 1

        candidates = []

        for candidate in request.get(
            "candidate_annotations",
            [],
        ):

            if isinstance(
                candidate,
                Mapping,
            ):
                name = candidate.get(
                    "name"
                )
            else:
                name = str(
                    candidate
                )

            if name:
                candidates.append(
                    str(name)
                )

        return {
            "schema_name":
                "MarkerRescueQueryPlan",

            "schema_version":
                "2.0",

            **context.to_dict(),

            "expert_label_used":
                False,

            "reference_label_used":
                False,

            "source_literature_request":
                source_literature_request,

            "source_pubmedbert_evidence":
                source_pubmedbert_evidence,

            "selection_rule": (
                "Among the first N ranked cluster "
                "markers, select markers not "
                "represented in any candidate-associated "
                "observed-marker passage from the "
                "PubMedBERT evidence set."
            ),

            "top_marker_limit":
                self.top_marker_limit,

            "pairwise_marker_limit":
                self.pairwise_marker_limit,

            "top_markers": [
                {
                    "gene":
                        marker["gene"],

                    "rank_within_cluster":
                        int(
                            marker[
                                "rank_within_cluster"
                            ]
                        ),
                }
                for marker in top_markers
            ],

            "already_explained_top_markers": [
                gene
                for gene in top_marker_names
                if gene
                in normalized_explained
            ],

            "rescue_markers": [
                {
                    "gene":
                        marker["gene"],

                    "rank_within_cluster":
                        int(
                            marker[
                                "rank_within_cluster"
                            ]
                        ),
                }
                for marker in rescue_markers
            ],

            
            # Retained explicitly as a leakage audit.
            # These names are NOT inserted into queries.
            
            "candidate_names_not_used_in_queries":
                candidates,

            "candidate_names_used_in_queries":
                False,

            "query_count":
                len(queries),

            "queries":
                queries,
        }
