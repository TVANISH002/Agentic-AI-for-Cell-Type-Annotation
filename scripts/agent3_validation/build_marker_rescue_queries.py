from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path


def truthy(value):
    if isinstance(value, bool):
        return value

    return str(value).strip().casefold() in {
        "true", "1", "yes"
    }


def split_semicolon(value):
    if value is None:
        return []

    text = str(value).strip()

    if not text or text.casefold() == "nan":
        return []

    return [
        x.strip().upper()
        for x in text.split(";")
        if x.strip()
    ]


def build_context_expression(tissue, disease):
    terms = []

    if tissue:
        terms.append(f'"{tissue}"')

    if disease:
        terms.append(f'"{disease}"')

    return " OR ".join(terms)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate candidate-independent marker-rescue "
            "literature queries for unexplained high-ranked "
            "cluster markers."
        )
    )

    parser.add_argument(
        "--literature-request",
        required=True,
    )

    parser.add_argument(
        "--evidence-passages-csv",
        required=True,
    )

    parser.add_argument(
        "--output-directory",
        required=True,
    )

    parser.add_argument(
        "--top-marker-limit",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    request_path = Path(args.literature_request)
    passage_path = Path(args.evidence_passages_csv)
    outdir = Path(args.output_directory)

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    request = json.loads(
        request_path.read_text(
            encoding="utf-8"
        )
    )

    ranked_markers = sorted(
        request["marker_genes"],
        key=lambda x: int(
            x["rank_within_cluster"]
        ),
    )

    top_markers = ranked_markers[
        :args.top_marker_limit
    ]

    top_marker_names = [
        x["gene"].upper()
        for x in top_markers
    ]

    explained_markers = set()

    with passage_path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            if not truthy(
                row.get(
                    "candidate_present",
                    False,
                )
            ):
                continue

            observed = split_semicolon(
                row.get(
                    "observed_markers_in_passage",
                    "",
                )
            )

            explained_markers.update(
                observed
            )

    rescue_markers = [
        marker
        for marker in top_markers
        if marker["gene"].upper()
        not in explained_markers
    ]

    species = str(
        request.get("species", "")
    ).strip()

    tissue = str(
        request.get("tissue", "")
    ).strip()

    disease = str(
        request.get(
            "disease_context",
            "",
        )
    ).strip()

    platform = str(
        request.get("platform", "")
    ).strip()

    context_expression = (
        build_context_expression(
            tissue,
            disease,
        )
    )

    queries = []
    query_index = 1

    #
    # FAMILY 1:
    # Context-specific marker query.
    #
    for marker in rescue_markers:
        gene = marker["gene"].upper()

        pieces = [
            f'"{gene}"',
        ]

        if context_expression:
            pieces.append(
                f"({context_expression})"
            )

        if species:
            pieces.append(
                f'("{species}" OR patient)'
            )

        pieces.append(
            '('
            '"single cell" OR '
            '"single-cell" OR '
            '"spatial transcriptomics" OR '
            '"spatial gene expression"'
            ')'
        )

        queries.append(
            {
                "query_id":
                    f"MR{query_index:02d}",
                "query_family":
                    "context_specific_marker",
                "markers": [gene],
                "query":
                    " AND ".join(pieces),
            }
        )

        query_index += 1

    #
    # FAMILY 2:
    # Candidate-independent biological
    # identity query.
    #
    for marker in rescue_markers:
        gene = marker["gene"].upper()

        pieces = [
            f'"{gene}"',
            '('
            '"cell type" OR '
            '"cell identity" OR '
            '"cell marker" OR '
            '"marker gene"'
            ')',
        ]

        if species:
            pieces.append(
                f'("{species}" OR human)'
            )

        queries.append(
            {
                "query_id":
                    f"MR{query_index:02d}",
                "query_family":
                    "biological_identity",
                "markers": [gene],
                "query":
                    " AND ".join(pieces),
            }
        )

        query_index += 1

    #
    # FAMILY 3:
    # Pairwise queries among the highest
    # ranked rescue markers.
    #
    pair_markers = rescue_markers[:3]

    for left, right in itertools.combinations(
        pair_markers,
        2,
    ):
        g1 = left["gene"].upper()
        g2 = right["gene"].upper()

        pieces = [
            f'("{g1}" AND "{g2}")',
        ]

        if context_expression:
            pieces.append(
                f"({context_expression})"
            )

        pieces.append(
            '('
            '"single cell" OR '
            '"single-cell" OR '
            '"spatial transcriptomics" OR '
            '"cell type"'
            ')'
        )

        queries.append(
            {
                "query_id":
                    f"MR{query_index:02d}",
                "query_family":
                    "marker_pair",
                "markers": [g1, g2],
                "query":
                    " AND ".join(pieces),
            }
        )

        query_index += 1

    output = {
        "schema_name":
            "MarkerRescueQueryPlan",
        "schema_version":
            "1.0",

        "cluster_id":
            str(request["cluster_id"]),

        "species":
            species,

        "tissue":
            tissue,

        "disease_context":
            disease,

        "platform":
            platform,

        "expert_label_used":
            False,

        "source_literature_request":
            str(request_path),

        "source_pubmedbert_evidence":
            str(passage_path),

        "selection_rule": (
            "Among the first N ranked cluster markers, "
            "select markers not represented in any "
            "candidate-associated observed-marker passage "
            "from the frozen PubMedBERT evidence set."
        ),

        "top_marker_limit":
            args.top_marker_limit,

        "top_markers": [
            {
                "gene":
                    x["gene"],
                "rank_within_cluster":
                    int(
                        x[
                            "rank_within_cluster"
                        ]
                    ),
            }
            for x in top_markers
        ],

        "already_explained_top_markers": [
            g
            for g in top_marker_names
            if g in explained_markers
        ],

        "rescue_markers": [
            {
                "gene":
                    x["gene"],
                "rank_within_cluster":
                    int(
                        x[
                            "rank_within_cluster"
                        ]
                    ),
            }
            for x in rescue_markers
        ],

        "candidate_names_not_used_in_queries": [
            (
                c.get("name")
                if isinstance(c, dict)
                else str(c)
            )
            for c in request.get(
                "candidate_annotations",
                []
            )
        ],

        "query_count":
            len(queries),

        "queries":
            queries,
    }

    output_path = (
        outdir
        / (
            f"cluster_{request['cluster_id']}_"
            "marker_rescue_query_plan.json"
        )
    )

    output_path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "Marker-rescue query plan "
        "generated successfully."
    )

    print(
        "Cluster:",
        request["cluster_id"],
    )

    print(
        "Top-marker window:",
        args.top_marker_limit,
    )

    print(
        "Already represented:",
        ", ".join(
            output[
                "already_explained_top_markers"
            ]
        ) or "none",
    )

    print(
        "Rescue markers:",
        ", ".join(
            x["gene"]
            for x in output[
                "rescue_markers"
            ]
        ) or "none",
    )

    print(
        "Queries generated:",
        len(queries),
    )

    print()

    for q in queries:
        print(
            f"{q['query_id']} "
            f"[{q['query_family']}]"
        )
        print(
            " ",
            q["query"],
        )

    print()
    print(
        "Saved:",
        output_path,
    )


if __name__ == "__main__":
    main()
