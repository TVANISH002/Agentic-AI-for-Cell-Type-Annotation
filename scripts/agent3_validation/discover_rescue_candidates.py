from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


CELLULAR_SUFFIXES = (
    "cyte",
    "blast",
    "phage",
)


def normalize_space(value) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(value or ""),
    ).strip()


def normalize_term(value) -> str:
    return normalize_space(value).casefold()


def extract_cl_id(node_id: str) -> str:
    match = re.search(
        r"/CL_(\d+)$",
        str(node_id),
    )

    if not match:
        return ""

    return f"CL:{match.group(1)}"


def phrase_pattern(term: str):
    """
    Conservative whole-phrase matcher.

    Canonical labels ending in 'cell' also receive
    an automatic plural 'cells' form.
    """
    term = normalize_space(term)

    forms = [term]

    if term.casefold().endswith(" cell"):
        forms.append(
            term[:-4] + "cells"
        )

    escaped = [
        re.escape(x)
        for x in forms
    ]

    return re.compile(
        r"(?<![A-Za-z0-9])(?:"
        + "|".join(escaped)
        + r")(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )


def gene_pattern(gene: str):
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

    Important rule:
    if a synonym is itself the canonical label of
    another CL concept, it is not used as an alias.

    Example:
    a synonym 'macrophage' must not cause another
    ontology concept to inherit macrophage evidence.
    """
    synonym = normalize_space(
        synonym
    )

    normalized = normalize_term(
        synonym
    )

    if len(normalized) < 5:
        return False

    #
    # Avoid short acronym-like synonyms.
    #
    if (
        synonym.isupper()
        and len(synonym) <= 10
    ):
        return False

    #
    # Reject aliases that collide with the canonical
    # label of another ontology concept.
    #
    other_id = canonical_labels.get(
        normalized
    )

    if (
        other_id
        and other_id != current_ontology_id
    ):
        return False

    #
    # Reject generic single-word aliases such as
    # adjectives ('basal', 'stromal', etc.).
    #
    words = re.findall(
        r"[A-Za-z]+",
        synonym,
    )

    if len(words) == 1:
        word = words[0].casefold()

        if not (
            word.endswith(CELLULAR_SUFFIXES)
            or word.endswith("cell")
        ):
            return False

    return True


def load_cell_ontology(path: Path):
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
            "Cell Ontology JSON contains no graph."
        )

    nodes = graphs[0].get(
        "nodes",
        [],
    )

    #
    # First pass:
    # collect all canonical CL labels so synonym
    # collisions can be detected globally.
    #
    canonical_labels = {}

    for node in nodes:
        oid = extract_cl_id(
            node.get("id", "")
        )

        label = normalize_space(
            node.get("lbl", "")
        )

        if oid and label:
            canonical_labels[
                normalize_term(label)
            ] = oid

    concepts = []

    for node in nodes:
        oid = extract_cl_id(
            node.get("id", "")
        )

        label = normalize_space(
            node.get("lbl", "")
        )

        if not oid or not label:
            continue

        #
        # Extremely generic ontology roots do not
        # constitute meaningful discovered candidates.
        #
        if normalize_term(label) in {
            "cell",
            "native cell",
        }:
            continue

        synonyms = []

        meta = node.get(
            "meta",
            {},
        ) or {}

        for syn in (
            meta.get(
                "synonyms",
                [],
            )
            or []
        ):
            if isinstance(
                syn,
                dict,
            ):
                value = normalize_space(
                    syn.get(
                        "val",
                        syn.get(
                            "value",
                            "",
                        ),
                    )
                )
            else:
                value = normalize_space(
                    syn
                )

            if not value:
                continue

            if (
                normalize_term(value)
                == normalize_term(label)
            ):
                continue

            if safe_synonym(
                value,
                canonical_labels,
                oid,
            ):
                synonyms.append(
                    value
                )

        terms = [
            {
                "term": label,
                "source": "canonical_label",
                "pattern":
                    phrase_pattern(label),
            }
        ]

        for synonym in sorted(
            set(synonyms),
            key=str.casefold,
        ):
            terms.append(
                {
                    "term": synonym,
                    "source": "synonym",
                    "pattern":
                        phrase_pattern(
                            synonym
                        ),
                }
            )

        concepts.append(
            {
                "ontology_id": oid,
                "label": label,
                "terms": terms,
            }
        )

    return concepts


def sentence_split(text: str):
    """
    Lightweight deterministic sentence segmentation.
    Sufficient for an evidence-co-occurrence audit;
    not treated as a biomedical NLP parser.
    """
    text = normalize_space(
        text
    )

    if not text:
        return []

    return [
        x.strip()
        for x in re.split(
            r"(?<=[.!?])\s+",
            text,
        )
        if x.strip()
    ]


def paper_key(paper):
    for field in (
        "record_key",
        "pmid",
        "doi",
        "pmcid",
        "title",
    ):
        value = normalize_space(
            paper.get(field)
        )

        if value:
            return (
                f"{field}:{value}"
            )

    return ""


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Conservative ontology-grounded discovery "
            "of candidate cell types from marker-rescue "
            "literature."
        )
    )

    parser.add_argument(
        "--retrieval-pilot",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--cell-ontology",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--literature-request",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-directory",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    pilot = json.loads(
        args.retrieval_pilot.read_text(
            encoding="utf-8-sig"
        )
    )

    request = json.loads(
        args.literature_request.read_text(
            encoding="utf-8-sig"
        )
    )

    concepts = load_cell_ontology(
        args.cell_ontology
    )

    upstream_candidates = {
        normalize_term(
            c.get("name", "")
        )
        for c in request.get(
            "candidate_annotations",
            [],
        )
        if isinstance(c, dict)
    }

    evidence = defaultdict(
        lambda: {
            "papers": set(),
            "queries": set(),

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

    for query_result in pilot.get(
        "query_results",
        [],
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

        rescue_markers = [
            normalize_space(x).upper()
            for x in query.get(
                "marker_genes",
                [],
            )
            if normalize_space(x)
        ]

        marker_patterns = {
            gene:
                gene_pattern(gene)
            for gene in rescue_markers
        }

        papers = (
            query_result.get("papers")
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

            full_text = normalize_space(
                f"{title}. {abstract}"
            )

            if not full_text:
                continue

            sentences = sentence_split(
                full_text
            )

            for concept in concepts:
                concept_matches = []

                canonical_found = False

                for term_info in concept[
                    "terms"
                ]:
                    if term_info[
                        "pattern"
                    ].search(
                        full_text
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
                            canonical_found = True

                if not concept_matches:
                    continue

                oid = concept[
                    "ontology_id"
                ]

                item = evidence[
                    oid
                ]

                item[
                    "papers"
                ].add(key)

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
                    ].search(title)
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

                #
                # Stronger descriptive evidence:
                # candidate cell-type term and rescue
                # marker appear in the same sentence.
                #
                same_sentence_markers = set()

                for sentence in sentences:
                    concept_in_sentence = any(
                        term_info[
                            "pattern"
                        ].search(
                            sentence
                        )
                        for term_info
                        in concept_matches
                    )

                    if not concept_in_sentence:
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

                provenance = item[
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
        ]: concept
        for concept in concepts
    }

    rows = []

    for oid, item in (
        evidence.items()
    ):
        concept = lookup[
            oid
        ]

        label = concept[
            "label"
        ]

        rows.append(
            {
                "ontology_id":
                    oid,

                "cell_type":
                    label,

                "is_upstream_candidate":
                    normalize_term(label)
                    in upstream_candidates,

                "is_novel_to_agent2":
                    normalize_term(label)
                    not in upstream_candidates,

                "independent_paper_count":
                    len(
                        item["papers"]
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
                        item["queries"]
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

    #
    # Descriptive ordering, NOT a confidence score.
    #
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

    output_directory = (
        args.output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    cluster_id = str(
        request.get(
            "cluster_id",
            "",
        )
    )

    csv_path = (
        output_directory
        / (
            f"cluster_{cluster_id}_"
            "rescue_candidate_discovery.csv"
        )
    )

    json_path = (
        output_directory
        / (
            f"cluster_{cluster_id}_"
            "rescue_candidate_discovery.json"
        )
    )

    if rows:
        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(
                    rows[0].keys()
                ),
            )

            writer.writeheader()

            writer.writerows(
                rows
            )

    detailed_results = []

    for row in rows:
        oid = row[
            "ontology_id"
        ]

        provenance = []

        for record in evidence[
            oid
        ][
            "paper_provenance"
        ].values():
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

    output = {
        "schema_name":
            "OntologyGroundedRescueCandidateDiscovery",

        "schema_version":
            "1.1",

        "cluster_id":
            cluster_id,

        "expert_label_used":
            False,

        "source_retrieval_pilot":
            str(
                args.retrieval_pilot
            ),

        "source_literature_request":
            str(
                args.literature_request
            ),

        "cell_ontology":
            str(
                args.cell_ontology
            ),

        "candidate_ordering_rule": (
            "Candidates are ordered descriptively by "
            "same-sentence marker-linked paper count, "
            "distinct linked rescue markers, "
            "marker-linked query count, canonical-label "
            "paper count, and total independent paper count. "
            "No weighted score is calculated."
        ),

        "important_note": (
            "Literature mention and marker co-occurrence "
            "are evidence-retrieval descriptors, not "
            "annotation probabilities, confidence scores, "
            "or proof of biological support."
        ),

        "candidate_count":
            len(
                detailed_results
            ),

        "results":
            detailed_results,
    }

    json_path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "Conservative ontology-grounded rescue "
        "candidate discovery completed successfully."
    )

    print(
        "Cluster:",
        cluster_id,
    )

    print(
        "Ontology concepts loaded:",
        len(concepts),
    )

    print(
        "Cell types detected:",
        len(rows),
    )

    print()

    print(
        "TOP 25 CANDIDATES"
    )

    print(
        "=" * 120
    )

    for index, row in enumerate(
        rows[:25],
        1,
    ):
        status = (
            "UPSTREAM"
            if row[
                "is_upstream_candidate"
            ]
            else "NOVEL"
        )

        print(
            f"{index:02d}. "
            f"{row['cell_type']} "
            f"[{row['ontology_id']}] "
            f"| marker-linked papers="
            f"{row['marker_linked_paper_count']} "
            f"| linked markers="
            f"{row['distinct_linked_marker_count']} "
            f"| marker-linked queries="
            f"{row['marker_linked_query_count']} "
            f"| total papers="
            f"{row['independent_paper_count']} "
            f"| {status}"
        )

        print(
            "    linked markers:",
            row[
                "linked_markers"
            ] or "none",
        )

        print(
            "    matched terms:",
            row[
                "matched_terms"
            ],
        )

    print()

    print(
        "CSV:",
        csv_path,
    )

    print(
        "JSON:",
        json_path,
    )


if __name__ == "__main__":
    main()
