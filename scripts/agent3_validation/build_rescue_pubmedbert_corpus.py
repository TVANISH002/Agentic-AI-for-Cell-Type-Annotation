from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path


def normalize_space(value) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(value or ""),
    ).strip()


def candidate_pattern(label: str):
    """
    Conservative canonical-label matcher.

    Examples:
      mast cell  -> mast cell / mast cells
      T cell     -> T cell / T-cell / T cells / T-cells

    No ontology synonyms are used here.
    """
    label = normalize_space(label)

    words = label.split()

    if not words:
        return None

    escaped = [
        re.escape(word)
        for word in words
    ]

    separator = r"(?:\s+|-)"

    phrase = separator.join(
        escaped
    )

    #
    # Canonical labels ending in "cell"
    # also match the plural "cells".
    #
    if words[-1].casefold() == "cell":
        prefix = separator.join(
            escaped[:-1]
        )

        if prefix:
            phrase = (
                prefix
                + separator
                + r"cells?"
            )
        else:
            phrase = r"cells?"

    return re.compile(
        rf"(?<![A-Za-z0-9])"
        rf"{phrase}"
        rf"(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a rescue-candidate PubMedBERT corpus "
            "from the frozen marker-rescue evidence corpus."
        )
    )

    parser.add_argument(
        "--source-corpus",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--candidate-discovery",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-json",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    source = json.loads(
        args.source_corpus.read_text(
            encoding="utf-8-sig"
        )
    )

    discovery = json.loads(
        args.candidate_discovery.read_text(
            encoding="utf-8-sig"
        )
    )

    if (
        str(source.get("cluster_id"))
        != str(discovery.get("cluster_id"))
    ):
        raise ValueError(
            "Cluster mismatch between corpus "
            "and candidate-discovery artifact."
        )

    corpus = copy.deepcopy(
        source
    )

    request = corpus.get(
        "literature_request"
    )

    if not isinstance(
        request,
        dict,
    ):
        raise ValueError(
            "Missing literature_request."
        )

    results = discovery.get(
        "results",
        [],
    )

    if not results:
        raise ValueError(
            "No rescue candidates found."
        )

    #
    # Build ontology-derived candidate set.
    #
    # IMPORTANT:
    # supporting_genes is intentionally empty
    # for every rescue candidate so that old
    # Agent 2 candidates receive no advantage.
    #
    rescue_candidates = []

    patterns = {}

    seen_names = set()

    for index, result in enumerate(
        results,
        start=1,
    ):
        name = normalize_space(
            result.get(
                "cell_type",
                "",
            )
        )

        ontology_id = normalize_space(
            result.get(
                "ontology_id",
                "",
            )
        )

        if not name:
            continue

        key = name.casefold()

        if key in seen_names:
            continue

        seen_names.add(
            key
        )

        rescue_candidates.append(
            {
                "name":
                    name,

                "upstream_rank":
                    None,

                "supporting_genes":
                    [],

                "evidence_sources":
                    [],

                "ontology_id":
                    ontology_id,

                "rescue_discovery_rank":
                    index,

                "rescue_candidate":
                    True,

                "is_novel_to_agent2":
                    bool(
                        result.get(
                            "is_novel_to_agent2",
                            True,
                        )
                    ),
            }
        )

        patterns[name] = (
            candidate_pattern(
                name
            )
        )

    if not rescue_candidates:
        raise ValueError(
            "No usable rescue candidates."
        )

    #
    # Replace ONLY the candidate list in this copy.
    #
    request[
        "candidate_annotations"
    ] = rescue_candidates

    #
    # Recompute candidate mentions from canonical
    # labels over the frozen global passages.
    #
    passages = corpus.get(
        "passages",
        [],
    )

    if not isinstance(
        passages,
        list,
    ) or not passages:
        raise ValueError(
            "No global passages found."
        )

    mention_counts = {
        candidate["name"]: 0
        for candidate
        in rescue_candidates
    }

    for passage in passages:
        text = normalize_space(
            passage.get(
                "text",
                "",
            )
        )

        publication = passage.get(
            "publication",
            {},
        )

        if not isinstance(
            publication,
            dict,
        ):
            publication = {}

        title = normalize_space(
            publication.get(
                "title",
                "",
            )
        )

        searchable = normalize_space(
            f"{title}. {text}"
        )

        mentions = []

        for candidate in rescue_candidates:
            name = candidate[
                "name"
            ]

            pattern = patterns[
                name
            ]

            if (
                pattern is not None
                and pattern.search(
                    searchable
                )
            ):
                mentions.append(
                    name
                )

                mention_counts[
                    name
                ] += 1

        #
        # Remove stale Agent 2 candidate fields.
        #
        passage[
            "candidate_mentions"
        ] = mentions

        if (
            "candidate_scores"
            in passage
        ):
            passage[
                "candidate_scores"
            ] = {}

    #
    # The old candidate passage sets belong to
    # Fibroblast/Mesenchymal/Perivascular and are
    # intentionally invalidated in this rescue copy.
    #
    corpus[
        "candidate_passage_sets"
    ] = []

    corpus[
        "rescue_candidate_adapter"
    ] = {
        "schema_name":
            "RescueCandidatePubMedBERTAdapter",

        "schema_version":
            "1.0",

        "expert_label_used":
            False,

        "reference_label_used":
            False,

        "source_corpus":
            str(
                args.source_corpus
            ),

        "source_candidate_discovery":
            str(
                args.candidate_discovery
            ),

        "candidate_count":
            len(
                rescue_candidates
            ),

        "global_passage_count":
            len(
                passages
            ),

        "candidate_mention_rule": (
            "Canonical Cell Ontology label "
            "matched in title or passage text; "
            "simple cell/cells and space/hyphen "
            "variants permitted; ontology "
            "synonyms not used."
        ),

        "supporting_gene_policy": (
            "supporting_genes intentionally empty "
            "for all rescue candidates"
        ),

        "pubmedbert_eligibility_after_adapter": (
            "explicit rescue-candidate mention only"
        ),
    }

    args.output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output_json.write_text(
        json.dumps(
            corpus,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    nonzero = [
        (
            name,
            count,
        )
        for name, count
        in mention_counts.items()
        if count > 0
    ]

    nonzero.sort(
        key=lambda item: (
            -item[1],
            item[0].casefold(),
        )
    )

    print(
        "Rescue PubMedBERT corpus adapter "
        "completed successfully."
    )

    print(
        "Cluster:",
        corpus.get(
            "cluster_id"
        ),
    )

    print(
        "Global passages:",
        len(passages),
    )

    print(
        "Rescue candidates:",
        len(
            rescue_candidates
        ),
    )

    print(
        "Candidates with >=1 eligible passage:",
        len(nonzero),
    )

    print()

    print(
        "CANDIDATE ELIGIBLE-POOL AUDIT"
    )

    print(
        "=" * 100
    )

    for index, (
        name,
        count,
    ) in enumerate(
        nonzero,
        start=1,
    ):
        print(
            f"{index:02d}. "
            f"{name}: "
            f"{count} passages"
        )

    print()

    zero = [
        candidate["name"]
        for candidate
        in rescue_candidates
        if mention_counts[
            candidate["name"]
        ] == 0
    ]

    print(
        "Candidates with zero eligible passages:",
        len(zero),
    )

    if zero:
        print(
            "; ".join(
                zero
            )
        )

    print()

    print(
        "Saved:",
        args.output_json,
    )


if __name__ == "__main__":
    main()
