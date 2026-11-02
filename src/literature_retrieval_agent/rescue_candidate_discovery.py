from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from literature_retrieval_agent.runtime_context import (
    Agent3RuntimeContext,
)


CELLULAR_SUFFIXES = (
    "cyte",
    "blast",
    "phage",
)


def normalize_space(
    value: Any,
) -> str:

    return re.sub(
        r"\s+",
        " ",
        str(value or ""),
    ).strip()


def normalize_term(
    value: Any,
) -> str:

    return normalize_space(
        value
    ).casefold()


def extract_cl_id(
    node_id: str,
) -> str:

    match = re.search(
        r"/CL_(\d+)$",
        str(node_id),
    )

    if not match:
        return ""

    return (
        f"CL:{match.group(1)}"
    )


def phrase_pattern(
    term: str,
):

    """
    Conservative whole-phrase matcher.

    Canonical labels ending in "cell" also receive
    an automatic plural "cells" form.
    """

    term = normalize_space(
        term
    )

    forms = [
        term,
    ]

    if term.casefold().endswith(
        " cell"
    ):
        forms.append(
            term[:-4]
            + "cells"
        )

    escaped = [
        re.escape(value)
        for value in forms
    ]

    return re.compile(
        r"(?<![A-Za-z0-9])(?:"
        + "|".join(
            escaped
        )
        + r")(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )


def gene_pattern(
    gene: str,
):

    return re.compile(
        rf"(?<![A-Za-z0-9])"
        rf"{re.escape(gene)}"
        rf"(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )


def safe_synonym(
    synonym: str,
    canonical_labels: dict[str, str],
    current_ontology_id: str,
) -> bool:

    """
    Reject ambiguous or overly broad synonyms.

    If a synonym is itself the canonical label of
    another Cell Ontology concept, the synonym is
    not used as an alias.
    """

    synonym = normalize_space(
        synonym
    )

    normalized = normalize_term(
        synonym
    )

    if len(normalized) < 5:
        return False

    
    # Avoid short acronym-like synonyms.
    
    if (
        synonym.isupper()
        and len(synonym) <= 10
    ):
        return False

    
    # Reject aliases colliding with another canonical ontology label.
    
    other_id = canonical_labels.get(
        normalized
    )

    if (
        other_id
        and other_id
        != current_ontology_id
    ):
        return False

    
    # Reject generic single-word aliases unless their morphology resembles a cellular term.
    
    words = re.findall(
        r"[A-Za-z]+",
        synonym,
    )

    if len(words) == 1:

        word = (
            words[0]
            .casefold()
        )

        if not (
            word.endswith(
                CELLULAR_SUFFIXES
            )
            or word.endswith(
                "cell"
            )
        ):
            return False

    return True


def load_cell_ontology(
    path: Path,
) -> list[dict]:

    import json

    ontology = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    graphs = ontology.get(
        "graphs",
        [],
    )

    if not graphs:
        raise ValueError(
            "Cell Ontology JSON "
            "contains no graph."
        )

    nodes = graphs[0].get(
        "nodes",
        [],
    )

    
    # First pass:
    # all canonical labels are collected so synonym collisions can be detected.
    
    canonical_labels = {}

    for node in nodes:

        ontology_id = (
            extract_cl_id(
                node.get(
                    "id",
                    "",
                )
            )
        )

        label = normalize_space(
            node.get(
                "lbl",
                "",
            )
        )

        if (
            ontology_id
            and label
        ):
            canonical_labels[
                normalize_term(
                    label
                )
            ] = ontology_id

    concepts = []

    for node in nodes:

        ontology_id = (
            extract_cl_id(
                node.get(
                    "id",
                    "",
                )
            )
        )

        label = normalize_space(
            node.get(
                "lbl",
                "",
            )
        )

        if (
            not ontology_id
            or not label
        ):
            continue

        
        # Very generic roots do not represent useful discovered candidate cell types.
        
        if normalize_term(
            label
        ) in {
            "cell",
            "native cell",
        }:
            continue

        meta = (
            node.get(
                "meta",
                {},
            )
            or {}
        )

        synonyms = []

        for synonym_record in (
            meta.get(
                "synonyms",
                [],
            )
            or []
        ):

            if isinstance(
                synonym_record,
                dict,
            ):
                value = normalize_space(
                    synonym_record.get(
                        "val",
                        synonym_record.get(
                            "value",
                            "",
                        ),
                    )
                )
            else:
                value = normalize_space(
                    synonym_record
                )

            if not value:
                continue

            if (
                normalize_term(
                    value
                )
                == normalize_term(
                    label
                )
            ):
                continue

            if safe_synonym(
                value,
                canonical_labels,
                ontology_id,
            ):
                synonyms.append(
                    value
                )

        terms = [
            {
                "term":
                    label,

                "source":
                    "canonical_label",

                "pattern":
                    phrase_pattern(
                        label
                    ),
            }
        ]

        for synonym in sorted(
            set(
                synonyms
            ),
            key=str.casefold,
        ):

            terms.append(
                {
                    "term":
                        synonym,

                    "source":
                        "synonym",

                    "pattern":
                        phrase_pattern(
                            synonym
                        ),
                }
            )

        concepts.append(
            {
                "ontology_id":
                    ontology_id,

                "label":
                    label,

                "terms":
                    terms,
            }
        )

    return concepts


def sentence_split(
    text: str,
) -> list[str]:

    """
    Lightweight deterministic sentence segmentation.

    Used for evidence co-occurrence auditing;
    not treated as a biomedical NLP parser.
    """

    text = normalize_space(
        text
    )

    if not text:
        return []

    return [
        value.strip()
        for value in re.split(
            r"(?<=[.!?])\s+",
            text,
        )
        if value.strip()
    ]


def paper_key(
    paper: Mapping[str, Any],
) -> str:

    for field in (
        "record_key",
        "pmid",
        "doi",
        "pmcid",
        "title",
    ):

        value = normalize_space(
            paper.get(
                field
            )
        )

        if value:
            return (
                f"{field}:{value}"
            )

    return ""


class OntologyRescueCandidateDiscoverer:

    """
    Conservative, candidate-independent,
    ontology-grounded candidate discovery.

    Cell types are discovered from literature
    retrieved through unexplained cluster markers.

    No expert/reference label is used and no
    weighted candidate-confidence score is produced.
    """

    def __init__(
        self,
        cell_ontology: Path,
    ):

        self.cell_ontology = Path(
            cell_ontology
        )

        if not self.cell_ontology.exists():
            raise FileNotFoundError(
                self.cell_ontology
            )

        self.concepts = (
            load_cell_ontology(
                self.cell_ontology
            )
        )

    def discover(
        self,
        retrieval_pilot: Mapping[
            str,
            Any,
        ],
        literature_request: Mapping[
            str,
            Any,
        ],
        *,
        source_retrieval_pilot: str = "",
        source_literature_request: str = "",
    ) -> dict:

        context = (
            Agent3RuntimeContext
            .from_mapping(
                literature_request
            )
        )

        upstream_candidates = {
            normalize_term(
                candidate.get(
                    "name",
                    "",
                )
            )
            for candidate
            in literature_request.get(
                "candidate_annotations",
                [],
            )
            if isinstance(
                candidate,
                Mapping,
            )
        }

        evidence = defaultdict(
            lambda: {
                "papers":
                    set(),

                "queries":
                    set(),

                "canonical_papers":
                    set(),

                "marker_linked_papers":
                    set(),

                "marker_linked_queries":
                    set(),

                "title_papers":
                    set(),

                "linked_markers":
                    set(),

                "matched_terms":
                    set(),

                "matched_term_sources":
                    set(),

                "paper_provenance":
                    {},
            }
        )

        for query_result in (
            retrieval_pilot.get(
                "query_results",
                [],
            )
        ):

            query = (
                query_result.get(
                    "query",
                    {},
                )
                or {}
            )

            query_id = normalize_space(
                query.get(
                    "query_id",
                    query_result.get(
                        "query_id",
                        "",
                    ),
                )
            )

            #
            # Retrieval artifacts normally expose
            # marker_genes. "markers" is accepted
            # for compatibility with the generic
            # MarkerRescueQueryPlan before conversion
            # into LiteratureQuery.
            #
            raw_markers = (
                query.get(
                    "marker_genes"
                )
                or query.get(
                    "markers"
                )
                or []
            )

            rescue_markers = [
                normalize_space(
                    value
                ).upper()
                for value
                in raw_markers
                if normalize_space(
                    value
                )
            ]

            marker_patterns = {
                gene:
                    gene_pattern(
                        gene
                    )
                for gene
                in rescue_markers
            }

            papers = (
                query_result.get(
                    "papers"
                )
                or query_result.get(
                    "publications"
                )
                or query_result.get(
                    "results"
                )
                or query_result.get(
                    "records"
                )
                or []
            )

            for paper in papers:

                key = paper_key(
                    paper
                )

                if not key:
                    continue

                title = normalize_space(
                    paper.get(
                        "title",
                        "",
                    )
                )

                abstract = normalize_space(
                    paper.get(
                        "abstract",
                        paper.get(
                            "abstract_text",
                            "",
                        ),
                    )
                )

                full_text = (
                    normalize_space(
                        f"{title}. "
                        f"{abstract}"
                    )
                )

                if not full_text:
                    continue

                sentences = (
                    sentence_split(
                        full_text
                    )
                )

                for concept in (
                    self.concepts
                ):

                    concept_matches = []
                    canonical_found = False

                    for term_info in (
                        concept[
                            "terms"
                        ]
                    ):

                        if (
                            term_info[
                                "pattern"
                            ].search(
                                full_text
                            )
                        ):
                            concept_matches.append(
                                term_info
                            )

                            if (
                                term_info[
                                    "source"
                                ]
                                == "canonical_label"
                            ):
                                canonical_found = (
                                    True
                                )

                    if not concept_matches:
                        continue

                    ontology_id = concept[
                        "ontology_id"
                    ]

                    item = evidence[
                        ontology_id
                    ]

                    item[
                        "papers"
                    ].add(
                        key
                    )

                    if query_id:
                        item[
                            "queries"
                        ].add(
                            query_id
                        )

                    if canonical_found:
                        item[
                            "canonical_papers"
                        ].add(
                            key
                        )

                    if any(
                        term_info[
                            "pattern"
                        ].search(
                            title
                        )
                        for term_info
                        in concept_matches
                    ):
                        item[
                            "title_papers"
                        ].add(
                            key
                        )

                    for term_info in (
                        concept_matches
                    ):

                        item[
                            "matched_terms"
                        ].add(
                            term_info[
                                "term"
                            ]
                        )

                        item[
                            "matched_term_sources"
                        ].add(
                            term_info[
                                "source"
                            ]
                        )

                    
                    # Candidate concept and rescue marker must occur in the same sentence for marker-linked descriptive evidence.
                    
                    same_sentence_markers = (
                        set()
                    )

                    for sentence in (
                        sentences
                    ):

                        concept_in_sentence = any(
                            term_info[
                                "pattern"
                            ].search(
                                sentence
                            )
                            for term_info
                            in concept_matches
                        )

                        if not (
                            concept_in_sentence
                        ):
                            continue

                        for gene, pattern in (
                            marker_patterns.items()
                        ):

                            if pattern.search(
                                sentence
                            ):
                                same_sentence_markers.add(
                                    gene
                                )

                    if same_sentence_markers:

                        item[
                            "marker_linked_papers"
                        ].add(
                            key
                        )

                        if query_id:
                            item[
                                "marker_linked_queries"
                            ].add(
                                query_id
                            )

                        item[
                            "linked_markers"
                        ].update(
                            same_sentence_markers
                        )

                    provenance = (
                        item[
                            "paper_provenance"
                        ].setdefault(
                            key,
                            {
                                "title":
                                    title,

                                "pmid":
                                    paper.get(
                                        "pmid"
                                    ),

                                "doi":
                                    paper.get(
                                        "doi"
                                    ),

                                "query_ids":
                                    set(),

                                "same_sentence_markers":
                                    set(),
                            },
                        )
                    )

                    if query_id:
                        provenance[
                            "query_ids"
                        ].add(
                            query_id
                        )

                    provenance[
                        "same_sentence_markers"
                    ].update(
                        same_sentence_markers
                    )

        lookup = {
            concept[
                "ontology_id"
            ]:
                concept
            for concept
            in self.concepts
        }

        rows = []

        for ontology_id, item in (
            evidence.items()
        ):

            concept = lookup[
                ontology_id
            ]

            label = concept[
                "label"
            ]

            rows.append(
                {
                    "ontology_id":
                        ontology_id,

                    "cell_type":
                        label,

                    "is_upstream_candidate":
                        normalize_term(
                            label
                        )
                        in upstream_candidates,

                    "is_novel_to_agent2":
                        normalize_term(
                            label
                        )
                        not in upstream_candidates,

                    "independent_paper_count":
                        len(
                            item[
                                "papers"
                            ]
                        ),

                    "canonical_label_paper_count":
                        len(
                            item[
                                "canonical_papers"
                            ]
                        ),

                    "title_mention_paper_count":
                        len(
                            item[
                                "title_papers"
                            ]
                        ),

                    "rescue_query_count":
                        len(
                            item[
                                "queries"
                            ]
                        ),

                    "marker_linked_paper_count":
                        len(
                            item[
                                "marker_linked_papers"
                            ]
                        ),

                    "marker_linked_query_count":
                        len(
                            item[
                                "marker_linked_queries"
                            ]
                        ),

                    "distinct_linked_marker_count":
                        len(
                            item[
                                "linked_markers"
                            ]
                        ),

                    "linked_markers":
                        "; ".join(
                            sorted(
                                item[
                                    "linked_markers"
                                ]
                            )
                        ),

                    "rescue_queries":
                        "; ".join(
                            sorted(
                                item[
                                    "queries"
                                ]
                            )
                        ),

                    "marker_linked_queries":
                        "; ".join(
                            sorted(
                                item[
                                    "marker_linked_queries"
                                ]
                            )
                        ),

                    "matched_terms":
                        "; ".join(
                            sorted(
                                item[
                                    "matched_terms"
                                ],
                                key=str.casefold,
                            )
                        ),

                    "matched_term_sources":
                        "; ".join(
                            sorted(
                                item[
                                    "matched_term_sources"
                                ]
                            )
                        ),
                }
            )

        
        # Descriptive ordering only.
        # This is NOT confidence/probability.
        
        rows.sort(
            key=lambda row: (
                -row[
                    "marker_linked_paper_count"
                ],
                -row[
                    "distinct_linked_marker_count"
                ],
                -row[
                    "marker_linked_query_count"
                ],
                -row[
                    "canonical_label_paper_count"
                ],
                -row[
                    "independent_paper_count"
                ],
                row[
                    "cell_type"
                ].casefold(),
            )
        )

        detailed_results = []

        for row in rows:

            ontology_id = row[
                "ontology_id"
            ]

            provenance = []

            for record in (
                evidence[
                    ontology_id
                ][
                    "paper_provenance"
                ].values()
            ):

                provenance.append(
                    {
                        "title":
                            record[
                                "title"
                            ],

                        "pmid":
                            record[
                                "pmid"
                            ],

                        "doi":
                            record[
                                "doi"
                            ],

                        "query_ids":
                            sorted(
                                record[
                                    "query_ids"
                                ]
                            ),

                        "same_sentence_markers":
                            sorted(
                                record[
                                    "same_sentence_markers"
                                ]
                            ),
                    }
                )

            detailed_results.append(
                {
                    **row,

                    "paper_provenance":
                        provenance,
                }
            )

        return {
            "schema_name":
                "OntologyGroundedRescueCandidateDiscovery",

            "schema_version":
                "2.0",

            **context.to_dict(),

            "expert_label_used":
                False,

            "reference_label_used":
                False,

            "source_retrieval_pilot":
                source_retrieval_pilot,

            "source_literature_request":
                source_literature_request,

            "cell_ontology":
                str(
                    self.cell_ontology
                ),

            "ontology_concept_count":
                len(
                    self.concepts
                ),

            "candidate_ordering_rule": (
                "Candidates are ordered "
                "descriptively by same-sentence "
                "marker-linked paper count, distinct "
                "linked rescue markers, marker-linked "
                "query count, canonical-label paper "
                "count, and total independent paper "
                "count. No weighted score is calculated."
            ),

            "important_note": (
                "Literature mention and marker "
                "co-occurrence are evidence-retrieval "
                "descriptors, not annotation "
                "probabilities, confidence scores, "
                "or proof of biological support."
            ),

            "candidate_count":
                len(
                    detailed_results
                ),

            "results":
                detailed_results,
        }
