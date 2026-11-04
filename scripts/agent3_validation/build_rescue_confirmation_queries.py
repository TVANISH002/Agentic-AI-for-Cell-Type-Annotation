from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def clean(value) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(value or ""),
    ).strip()


def quote(value: str) -> str:
    value = clean(value).replace('"', "")
    return f'"{value}"'


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build target-context confirmation queries "
            "for ontology-derived marker-rescue candidates."
        )
    )

    parser.add_argument(
        "--candidate-discovery",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--rescue-plan",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--literature-request",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-json",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    discovery = json.loads(
        args.candidate_discovery.read_text(
            encoding="utf-8-sig"
        )
    )

    rescue = json.loads(
        args.rescue_plan.read_text(
            encoding="utf-8-sig"
        )
    )

    request = json.loads(
        args.literature_request.read_text(
            encoding="utf-8-sig"
        )
    )

    cluster_ids = {
        str(discovery.get("cluster_id")),
        str(rescue.get("cluster_id")),
        str(request.get("cluster_id")),
    }

    if len(cluster_ids) != 1:
        raise ValueError(
            "Cluster mismatch across input artifacts."
        )

    #
    # Rescue markers come only from the marker-rescue
    # planner. No reference annotation is consulted.
    #
    def extract_marker_gene(value) -> str:
        """
        Accept either a plain gene-symbol string or a
        structured marker record such as:
        {"GENE": "CPA3", "RANK_WITHIN_CLUSTER": 1}.
        """
        if isinstance(value, dict):
            for key in (
                "gene",
                "GENE",
                "symbol",
                "SYMBOL",
                "marker",
                "MARKER",
            ):
                candidate = clean(
                    value.get(key, "")
                )

                if candidate:
                    return candidate.upper()

            return ""

        return clean(value).upper()


    rescue_markers = []

    for marker in rescue.get(
        "rescue_markers",
        [],
    ):
        gene = extract_marker_gene(
            marker
        )

        if (
            gene
            and gene not in rescue_markers
        ):
            rescue_markers.append(
                gene
            )

    #
    # Fallback for plan schemas where markers are
    # recorded only inside each query object.
    #
    if not rescue_markers:
        for query in rescue.get(
            "queries",
            [],
        ):
            for marker in query.get(
                "markers",
                [],
            ):
                gene = extract_marker_gene(
                    marker
                )

                if (
                    gene
                    and gene not in rescue_markers
                ):
                    rescue_markers.append(
                        gene
                    )

    if not rescue_markers:
        raise ValueError(
            "No rescue markers found."
        )

    species = clean(
        request.get(
            "species",
            "Human",
        )
    )

    tissue = clean(
        request.get(
            "tissue",
            "",
        )
    )

    disease = clean(
        request.get(
            "disease_context",
            "",
        )
    )

    candidates = []

    for item in discovery.get(
        "results",
        [],
    ):
        name = clean(
            item.get(
                "cell_type",
                "",
            )
        )

        ontology_id = clean(
            item.get(
                "ontology_id",
                "",
            )
        )

        if not name:
            continue

        candidates.append(
            {
                "name": name,
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

    marker_clause = (
        "("
        + " OR ".join(
            quote(marker)
            for marker
            in rescue_markers
        )
        + ")"
    )

    context_terms = []

    if tissue:
        context_terms.append(
            quote(tissue)
        )

    if disease:
        context_terms.append(
            quote(disease)
        )

    context_clause = (
        "("
        + " OR ".join(
            context_terms
        )
        + ")"
        if context_terms
        else ""
    )

    species_clause = (
        f"({quote(species)} OR patient)"
    )

    modality_clause = (
        '("single cell" OR "single-cell" '
        'OR "spatial transcriptomics" '
        'OR "spatial gene expression")'
    )

    queries = []

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):
        parts = [
            quote(
                candidate["name"]
            )
        ]

        if context_clause:
            parts.append(
                context_clause
            )

        parts.extend(
            [
                species_clause,
                modality_clause,
                marker_clause,
            ]
        )

        queries.append(
            {
                "query_id":
                    f"CR{index:03d}",

                "query_family":
                    "rescue_candidate_context_confirmation",

                "candidate":
                    candidate["name"],

                "ontology_id":
                    candidate["ontology_id"],

                "markers":
                    rescue_markers,

                "query":
                    " AND ".join(
                        parts
                    ),
            }
        )

    output = {
        "schema_name":
            "RescueCandidateConfirmationQueryPlan",

        "schema_version":
            "1.0",

        "cluster_id":
            str(request.get(
                "cluster_id"
            )),

        "expert_label_used":
            False,

        "reference_label_used":
            False,

        "source_candidate_discovery":
            str(
                args.candidate_discovery
            ),

        "source_rescue_plan":
            str(
                args.rescue_plan
            ),

        "source_literature_request":
            str(
                args.literature_request
            ),

        "species":
            species,

        "tissue":
            tissue,

        "disease_context":
            disease,

        "rescue_markers":
            rescue_markers,

        "candidate_count":
            len(candidates),

        "query_count":
            len(queries),

        "queries":
            queries,
    }

    args.output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output_json.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "Rescue candidate confirmation "
        "query plan completed successfully."
    )

    print(
        "Cluster:",
        output["cluster_id"],
    )

    print(
        "Candidates:",
        len(candidates),
    )

    print(
        "Queries:",
        len(queries),
    )

    print(
        "Reference label used:",
        False,
    )

    print(
        "Rescue markers:",
        ", ".join(
            rescue_markers
        ),
    )

    print()

    print(
        "FIRST 20 CONFIRMATION QUERIES"
    )

    print("=" * 120)

    for item in queries[:20]:
        print(
            f"\n{item['query_id']} "
            f"[{item['candidate']}]"
        )

        print(
            item["query"]
        )

    print()

    print(
        "Saved:",
        args.output_json,
    )


if __name__ == "__main__":
    main()
