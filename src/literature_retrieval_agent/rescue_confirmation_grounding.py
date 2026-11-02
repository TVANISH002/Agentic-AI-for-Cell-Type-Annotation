from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def clean(value: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(value or ""),
    ).strip()


# Biomedical synonym expansion used for terminology-aware grounding.
# This is general logic, not cluster-specific logic.
CANDIDATE_SYNONYMS = {
    "t cell": [
        "T-cell",
        "T cells",
        "T-cells",
        "T lymphocyte",
        "T lymphocytes",
        "T lymphocytic cell",
    ],
    "cytotoxic t cell": [
        "cytotoxic T-cell",
        "cytotoxic T cells",
        "cytotoxic T-cells",
        "cytotoxic T lymphocyte",
        "cytotoxic T lymphocytes",
        "CD8 T cell",
        "CD8 T-cell",
        "CD8-positive T cell",
        "CD8-positive T-cell",
        "CD8+ T cell",
        "CD8+ T-cell",
        "CD8+ T cells",
        "CTL",
        "CTLs",
    ],
    "natural killer cell": [
        "NK cell",
        "NK cells",
        "natural killer cells",
    ],
    "b cell": [
        "B-cell",
        "B cells",
        "B-cells",
        "B lymphocyte",
        "B lymphocytes",
    ],
    "plasma cell": [
        "plasma cells",
        "antibody-secreting cell",
        "antibody secreting cell",
    ],
    "macrophage": [
        "macrophages",
        "tumor-associated macrophage",
        "tumor associated macrophage",
        "TAM",
        "TAMs",
    ],
}

GENE_ALIAS_OVERRIDES = {
    "PTPRC": ["CD45", "leukocyte common antigen"],
    "CTLA4": ["CTLA-4", "CD152"],
    "IL2RG": ["CD132", "common gamma chain", "IL-2 receptor gamma"],
    "CD247": ["CD3 zeta", "CD3?", "TCR zeta"],
    "TRAC": [
        "TCR alpha constant",
        "T-cell receptor alpha constant",
        "T cell receptor alpha constant",
    ],
    "CD3E": ["CD3 epsilon", "CD3e", "CD3"],
    "CD3G": ["CD3 gamma", "CD3g", "CD3"],
    "CD8A": ["CD8 alpha", "CD8?", "CD8"],
    "GZMA": ["granzyme A"],
    "PRF1": ["perforin", "perforin 1"],
    "FOXP3": ["FoxP3"],
    "NKG7": ["natural killer cell granule protein 7"],
}


def candidate_patterns(label: str):
    label = clean(label)
    if not label:
        return []

    labels = [label]
    labels.extend(
        CANDIDATE_SYNONYMS.get(
            label.casefold(),
            [],
        )
    )

    patterns = []
    seen = set()

    for value in labels:
        key = clean(value).casefold()
        if not key or key in seen:
            continue
        seen.add(key)

        pattern = canonical_pattern(value)
        if pattern is not None:
            patterns.append(pattern)

    return patterns


def canonical_pattern(label: str):
    """
    Conservative canonical candidate matching with
    simple plural handling.
    """
    label = clean(label)

    if not label:
        return None

    words = label.split()

    escaped = [
        re.escape(word)
        for word in words
    ]

    separator = r"(?:\s+|-)"

    if words[-1].casefold() == "cell":
        stem = separator.join(
            escaped[:-1]
        )

        phrase = (
            stem
            + separator
            + r"cells?"
            if stem
            else r"cells?"
        )

    else:
        stem = separator.join(
            escaped
        )

        
        # macrophage/macrophages,
        # fibroblast/fibroblasts, etc.
        
        if (
            words[-1].isalpha()
            and not words[-1].casefold().endswith("s")
        ):
            phrase = stem + r"s?"
        else:
            phrase = stem

    return re.compile(
        rf"(?<![A-Za-z0-9])"
        rf"{phrase}"
        rf"(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )


def term_pattern(term: str):
    term = clean(term)

    if not term:
        return None

    escaped = re.escape(term)

    
    # Permit ordinary spacing/hyphen variation.
    
    escaped = escaped.replace(
        r"\ ",
        r"(?:\s+|-)",
    )

    return re.compile(
        rf"(?<![A-Za-z0-9])"
        rf"{escaped}"
        rf"(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )


def sentence_split(text: str):
    return [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            clean(text),
        )
        if sentence.strip()
    ]


def extract_local_abbreviations(
    text: str,
    candidate_re,
) -> list[str]:
    """
    Learn abbreviations only when the CURRENT paper
    explicitly defines them.

    Example:
        Mast cells (MCs)

    No external/manual abbreviation dictionary.
    """
    if candidate_re is None:
        return []

    abbreviations = []

    
    # Candidate phrase followed by parenthetical abbreviation.
    
    pattern = re.compile(
        candidate_re.pattern
        + r"\s*\("
        + r"([A-Z][A-Za-z0-9-]{1,9})"
        + r"\)",
        flags=re.IGNORECASE,
    )

    for match in pattern.finditer(
        text
    ):
        abbreviation = clean(
            match.group(1)
        )

        
        # Require at least two actual uppercase characters in the abbreviation base.
        
        upper_count = sum(
            1
            for char in abbreviation
            if char.isupper()
        )

        if upper_count < 2:
            continue

        if abbreviation not in abbreviations:
            abbreviations.append(
                abbreviation
            )

        
        # MCs -> also permit MC inside the same paper.
        
        if (
            abbreviation.endswith("s")
            and len(abbreviation) > 2
        ):
            singular = abbreviation[:-1]

            if (
                sum(
                    1
                    for char in singular
                    if char.isupper()
                )
                >= 2
                and singular not in abbreviations
            ):
                abbreviations.append(
                    singular
                )

    return abbreviations


def abbreviation_pattern(
    abbreviation: str,
):
    return re.compile(
        rf"(?<![A-Za-z0-9])"
        rf"{re.escape(abbreviation)}"
        rf"(?![A-Za-z0-9])"
    )


def build_gene_terms(
    terminology: dict,
) -> dict[str, list[dict]]:
    output = {}

    for record in terminology.get(
        "genes",
        [],
    ):
        symbol = clean(
            record.get(
                "official_symbol",
                "",
            )
        ).upper()

        if not symbol:
            continue

        terms = []
        seen = set()

        def add_term(
            value,
            source_type,
            allow_short=False,
        ):
            value = clean(value)

            if not value:
                return

            
            # Avoid very short aliases unless they are the official symbol itself.
            
            if (
                not allow_short
                and len(
                    re.sub(
                        r"[^A-Za-z0-9]",
                        "",
                        value,
                    )
                )
                < 4
            ):
                return

            key = value.casefold()

            if key in seen:
                return

            seen.add(key)

            terms.append(
                {
                    "term": value,
                    "source_type":
                        source_type,
                    "pattern":
                        term_pattern(
                            value
                        ),
                }
            )

        add_term(
            symbol,
            "official_symbol",
            allow_short=True,
        )

        for alias in GENE_ALIAS_OVERRIDES.get(
            symbol,
            [],
        ):
            add_term(
                alias,
                "manual_biomedical_alias",
                allow_short=True,
            )

        add_term(
            record.get(
                "official_full_name"
            ),
            "official_full_name",
        )

        for value in record.get(
            "aliases",
            [],
        ):
            add_term(
                value,
                "ncbi_alias",
            )

        for value in record.get(
            "other_designations",
            [],
        ):
            add_term(
                value,
                "ncbi_other_designation",
            )

        output[symbol] = terms

    return output


def context_terms(
    request: dict,
):
    terms = []

    for field in (
        "tissue",
        "disease_context",
    ):
        value = clean(
            request.get(
                field,
                "",
            )
        )

        if value:
            terms.append(
                (
                    field,
                    value,
                    term_pattern(value),
                )
            )

    return terms


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pilot-json",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--confirmation-plan",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--literature-request",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--gene-terminology",
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
        args.pilot_json.read_text(
            encoding="utf-8-sig"
        )
    )

    plan = json.loads(
        args.confirmation_plan.read_text(
            encoding="utf-8-sig"
        )
    )

    request = json.loads(
        args.literature_request.read_text(
            encoding="utf-8-sig"
        )
    )

    terminology = json.loads(
        args.gene_terminology.read_text(
            encoding="utf-8-sig"
        )
    )

    
    # Provenance/leakage safeguards.
    
    if plan.get(
        "reference_label_used",
        False,
    ):
        raise ValueError(
            "Reference label usage detected."
        )

    if terminology.get(
        "reference_label_used",
        False,
    ):
        raise ValueError(
            "Reference label usage detected "
            "in terminology artifact."
        )

    gene_terms = build_gene_terms(
        terminology
    )

    target_context_terms = context_terms(
        request
    )

    plan_lookup = {
        item["query_id"]: item
        for item in plan.get(
            "queries",
            [],
        )
    }

    aggregate_rows = []
    detail_records = []

    for qr in pilot.get(
        "query_results",
        [],
    ):
        query_obj = (
            qr.get("query", {})
            or {}
        )

        query_id = clean(
            query_obj.get(
                "query_id",
                qr.get(
                    "query_id",
                    "",
                ),
            )
        )

        plan_item = plan_lookup.get(
            query_id,
        )

        if not plan_item:
            continue

        candidate = clean(
            plan_item.get(
                "candidate",
                "",
            )
        )

        candidate_res = candidate_patterns(
            candidate
        )

        candidate_re = (
            candidate_res[0]
            if candidate_res
            else None
        )

        papers = (
            qr.get("papers")
            or qr.get("publications")
            or qr.get("results")
            or qr.get("records")
            or []
        )

        candidate_papers = set()
        candidate_marker_papers = set()

        same_sentence_papers = set()
        adjacent_sentence_papers = set()
        within_three_sentence_papers = set()

        target_context_candidate_papers = set()
        target_context_candidate_marker_papers = set()

        linked_same = set()
        linked_adjacent = set()
        linked_three = set()

        paper_records = []

        for paper in papers:
            title = clean(
                paper.get(
                    "title",
                    "",
                )
            )

            abstract = clean(
                paper.get(
                    "abstract",
                    paper.get(
                        "abstract_text",
                        "",
                    ),
                )
            )

            full_text = clean(
                f"{title}. {abstract}"
            )

            paper_key = (
                clean(
                    paper.get(
                        "pmid"
                    )
                )
                or clean(
                    paper.get(
                        "doi"
                    )
                )
                or title
            )

            abbreviations = (
                extract_local_abbreviations(
                    full_text,
                    candidate_re,
                )
            )

            abbreviation_res = [
                abbreviation_pattern(
                    abbreviation
                )
                for abbreviation
                in abbreviations
            ]

            sentences = sentence_split(
                full_text
            )

            candidate_sentence_indices = []

            for index, sentence in enumerate(
                sentences
            ):
                canonical_hit = any(
                    pattern.search(sentence)
                    for pattern in candidate_res
                )

                abbreviation_hit = any(
                    pattern.search(
                        sentence
                    )
                    for pattern
                    in abbreviation_res
                )

                if (
                    canonical_hit
                    or abbreviation_hit
                ):
                    candidate_sentence_indices.append(
                        index
                    )

            candidate_found = bool(
                candidate_sentence_indices
            )

            if candidate_found:
                candidate_papers.add(
                    paper_key
                )

            context_matches = []

            for (
                context_type,
                context_value,
                pattern,
            ) in target_context_terms:
                if (
                    pattern
                    and pattern.search(
                        full_text
                    )
                ):
                    context_matches.append(
                        {
                            "type":
                                context_type,
                            "term":
                                context_value,
                        }
                    )

            target_context_found = bool(
                context_matches
            )

            if (
                candidate_found
                and target_context_found
            ):
                target_context_candidate_papers.add(
                    paper_key
                )

            marker_matches = {}

            for gene, terms in (
                gene_terms.items()
            ):
                matches = []

                marker_sentence_indices = set()

                for term_info in terms:
                    pattern = term_info[
                        "pattern"
                    ]

                    if pattern is None:
                        continue

                    for sentence_index, sentence in enumerate(
                        sentences
                    ):
                        if pattern.search(
                            sentence
                        ):
                            marker_sentence_indices.add(
                                sentence_index
                            )

                            matches.append(
                                {
                                    "term":
                                        term_info[
                                            "term"
                                        ],

                                    "source_type":
                                        term_info[
                                            "source_type"
                                        ],

                                    "sentence_index":
                                        sentence_index,
                                }
                            )

                if not matches:
                    continue

                
                # Deduplicate lexical matches.
                
                unique_matches = []
                seen_match = set()

                for match in matches:
                    key = (
                        match[
                            "term"
                        ].casefold(),
                        match[
                            "source_type"
                        ],
                        match[
                            "sentence_index"
                        ],
                    )

                    if key in seen_match:
                        continue

                    seen_match.add(
                        key
                    )

                    unique_matches.append(
                        match
                    )

                min_distance = None

                if (
                    candidate_sentence_indices
                    and marker_sentence_indices
                ):
                    min_distance = min(
                        abs(
                            candidate_index
                            - marker_index
                        )
                        for candidate_index
                        in candidate_sentence_indices
                        for marker_index
                        in marker_sentence_indices
                    )

                marker_matches[
                    gene
                ] = {
                    "matches":
                        unique_matches,

                    "minimum_candidate_sentence_distance":
                        min_distance,
                }

            if (
                candidate_found
                and marker_matches
            ):
                candidate_marker_papers.add(
                    paper_key
                )

                if target_context_found:
                    target_context_candidate_marker_papers.add(
                        paper_key
                    )

            for gene, evidence in (
                marker_matches.items()
            ):
                distance = evidence[
                    "minimum_candidate_sentence_distance"
                ]

                if distance is None:
                    continue

                if distance == 0:
                    same_sentence_papers.add(
                        paper_key
                    )

                    linked_same.add(
                        gene
                    )

                if distance <= 1:
                    adjacent_sentence_papers.add(
                        paper_key
                    )

                    linked_adjacent.add(
                        gene
                    )

                if distance <= 3:
                    within_three_sentence_papers.add(
                        paper_key
                    )

                    linked_three.add(
                        gene
                    )

            paper_records.append(
                {
                    "pmid":
                        paper.get(
                            "pmid"
                        ),

                    "doi":
                        paper.get(
                            "doi"
                        ),

                    "title":
                        title,

                    "candidate_mentioned":
                        candidate_found,

                    "candidate_local_abbreviations":
                        abbreviations,

                    "candidate_sentence_indices":
                        [
                            value + 1
                            for value
                            in candidate_sentence_indices
                        ],

                    "target_context_matches":
                        context_matches,

                    "marker_evidence":
                        marker_matches,
                }
            )

        row = {
            "query_id":
                query_id,

            "candidate":
                candidate,

            "returned_papers":
                len(papers),

            "candidate_mention_papers":
                len(
                    candidate_papers
                ),

            "candidate_marker_papers":
                len(
                    candidate_marker_papers
                ),

            "same_sentence_link_papers":
                len(
                    same_sentence_papers
                ),

            "adjacent_sentence_link_papers":
                len(
                    adjacent_sentence_papers
                ),

            "within_three_sentence_link_papers":
                len(
                    within_three_sentence_papers
                ),

            "target_context_candidate_papers":
                len(
                    target_context_candidate_papers
                ),

            "target_context_candidate_marker_papers":
                len(
                    target_context_candidate_marker_papers
                ),

            "same_sentence_linked_genes":
                "; ".join(
                    sorted(
                        linked_same
                    )
                ),

            "adjacent_sentence_linked_genes":
                "; ".join(
                    sorted(
                        linked_adjacent
                    )
                ),

            "within_three_sentence_linked_genes":
                "; ".join(
                    sorted(
                        linked_three
                    )
                ),
        }

        aggregate_rows.append(
            row
        )

        detail_records.append(
            {
                **row,
                "paper_evidence":
                    paper_records,
            }
        )

    
    # Descriptive ordering only.
    # No confidence score is created.
    
    aggregate_rows.sort(
        key=lambda row: (
            -row[
                "target_context_candidate_marker_papers"
            ],
            -row[
                "same_sentence_link_papers"
            ],
            -row[
                "adjacent_sentence_link_papers"
            ],
            -row[
                "within_three_sentence_link_papers"
            ],
            -row[
                "candidate_marker_papers"
            ],
            -row[
                "candidate_mention_papers"
            ],
            row[
                "candidate"
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

    csv_path = (
        output_directory
        / (
            f"cluster_{plan.get('cluster_id', '')}_"
            "rescue_grounding_audit.csv"
        )
    )

    json_path = (
        output_directory
        / (
            f"cluster_{plan.get('cluster_id', '')}_"
            "rescue_grounding_audit.json"
        )
    )

    if aggregate_rows:
        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(
                    aggregate_rows[0].keys()
                ),
            )

            writer.writeheader()

            writer.writerows(
                aggregate_rows
            )

    output = {
        "schema_name":
            "RescueConfirmationGroundingAudit",

        "schema_version":
            "1.0",

        "cluster_id":
            str(
                plan.get(
                    "cluster_id",
                    "",
                )
            ),

        "expert_label_used":
            False,

        "reference_label_used":
            False,

        "gene_terminology_source":
            "NCBI Gene",

        "candidate_abbreviation_rule": (
            "Candidate abbreviations are accepted "
            "only when explicitly defined within the "
            "same paper."
        ),

        "important_note": (
            "All counts are descriptive evidence "
            "properties. They are not probabilities, "
            "confidence scores, or calibrated "
            "annotation decisions."
        ),

        "results":
            detail_records,
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
        "Terminology-aware rescue grounding "
        "audit completed successfully."
    )

    print()

    print(
        "TOP 25 DESCRIPTIVE GROUNDING ROWS"
    )

    print("=" * 140)

    for index, row in enumerate(
        aggregate_rows[:25],
        start=1,
    ):
        print(
            f"{index:02d}. "
            f"{row['candidate']} "
            f"| candidate="
            f"{row['candidate_mention_papers']} "
            f"| cand+marker="
            f"{row['candidate_marker_papers']} "
            f"| same="
            f"{row['same_sentence_link_papers']} "
            f"| adjacent="
            f"{row['adjacent_sentence_link_papers']} "
            f"| within3="
            f"{row['within_three_sentence_link_papers']} "
            f"| target="
            f"{row['target_context_candidate_papers']} "
            f"| target+marker="
            f"{row['target_context_candidate_marker_papers']}"
        )

        print(
            "    same:",
            row[
                "same_sentence_linked_genes"
            ] or "none",
        )

        print(
            "    adjacent:",
            row[
                "adjacent_sentence_linked_genes"
            ] or "none",
        )

        print(
            "    within3:",
            row[
                "within_three_sentence_linked_genes"
            ] or "none",
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

