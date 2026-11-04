from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def norm(value):
    return re.sub(
        r"\s+",
        " ",
        str(value or ""),
    ).strip()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--candidate-discovery",
        required=True,
        type=Path,
    )

    ap.add_argument(
        "--retrieval-pilot",
        required=True,
        type=Path,
    )

    ap.add_argument(
        "--rescue-plan",
        required=True,
        type=Path,
    )

    ap.add_argument(
        "--output-directory",
        required=True,
        type=Path,
    )

    args = ap.parse_args()

    discovery = json.loads(
        args.candidate_discovery.read_text(
            encoding="utf-8-sig"
        )
    )

    pilot = json.loads(
        args.retrieval_pilot.read_text(
            encoding="utf-8-sig"
        )
    )

    rescue = json.loads(
        args.rescue_plan.read_text(
            encoding="utf-8-sig"
        )
    )

    query_family = {
        q["query_id"]: q["query_family"]
        for q in rescue.get("queries", [])
    }

    query_markers = {
        q["query_id"]: [
            str(x).upper()
            for x in q.get("markers", [])
        ]
        for q in rescue.get("queries", [])
    }

    #
    # Build paper lookup from retrieval result.
    #
    paper_lookup = {}

    for qr in pilot.get(
        "query_results",
        [],
    ):
        q = qr.get("query", {}) or {}

        qid = norm(
            q.get(
                "query_id",
                qr.get("query_id", ""),
            )
        )

        papers = (
            qr.get("papers")
            or qr.get("publications")
            or qr.get("results")
            or qr.get("records")
            or []
        )

        for paper in papers:
            key = (
                norm(paper.get("record_key"))
                or norm(paper.get("pmid"))
                or norm(paper.get("doi"))
                or norm(paper.get("title"))
            )

            if not key:
                continue

            entry = paper_lookup.setdefault(
                key,
                {
                    "title": norm(
                        paper.get("title")
                    ),
                    "abstract": norm(
                        paper.get("abstract")
                    ),
                    "pmid": paper.get("pmid"),
                    "doi": paper.get("doi"),
                    "query_ids": set(),
                },
            )

            if qid:
                entry["query_ids"].add(qid)

    rows = []

    for result in discovery.get(
        "results",
        [],
    ):
        label = result["cell_type"]

        total_papers = 0
        breast_context_papers = 0
        title_mention_papers = 0

        pairwise_papers = set()
        pairwise_queries = set()
        single_marker_queries = set()
        biological_identity_queries = set()

        rescue_markers = set()

        for prov in result.get(
            "paper_provenance",
            [],
        ):
            total_papers += 1

            title = norm(
                prov.get("title")
            )

            key = (
                norm(prov.get("pmid"))
                or norm(prov.get("doi"))
                or title
            )

            #
            # Locate matching retrieved paper.
            #
            paper = None

            for candidate_key, record in (
                paper_lookup.items()
            ):
                if (
                    norm(record.get("pmid"))
                    == norm(prov.get("pmid"))
                    and prov.get("pmid")
                ):
                    paper = record
                    break

                if (
                    norm(record.get("doi"))
                    == norm(prov.get("doi"))
                    and prov.get("doi")
                ):
                    paper = record
                    break

            text = title

            if paper:
                text += " " + norm(
                    paper.get("abstract")
                )

            lower = text.casefold()

            if (
                "breast" in lower
                or "mammary" in lower
            ):
                breast_context_papers += 1

            if label.casefold() in title.casefold():
                title_mention_papers += 1

            for qid in prov.get(
                "query_ids",
                [],
            ):
                family = query_family.get(
                    qid,
                    "unknown",
                )

                if family == "marker_pair":
                    pairwise_queries.add(
                        qid
                    )

                    pairwise_papers.add(
                        (
                            prov.get("pmid")
                            or prov.get("doi")
                            or title
                        )
                    )

                elif (
                    family
                    == "context_specific_marker"
                ):
                    single_marker_queries.add(
                        qid
                    )

                elif (
                    family
                    == "biological_identity"
                ):
                    biological_identity_queries.add(
                        qid
                    )

                rescue_markers.update(
                    query_markers.get(
                        qid,
                        [],
                    )
                )

        rows.append(
            {
                "ontology_id":
                    result["ontology_id"],

                "cell_type":
                    label,

                "is_novel_to_agent2":
                    result[
                        "is_novel_to_agent2"
                    ],

                "total_papers":
                    total_papers,

                "breast_context_papers":
                    breast_context_papers,

                "title_mention_papers":
                    title_mention_papers,

                "pairwise_marker_papers":
                    len(pairwise_papers),

                "distinct_pairwise_queries":
                    len(pairwise_queries),

                "pairwise_query_ids":
                    sorted(pairwise_queries),

                "single_marker_query_count":
                    len(single_marker_queries),

                "biological_identity_query_count":
                    len(
                        biological_identity_queries
                    ),

                "same_sentence_marker_papers":
                    result[
                        "marker_linked_paper_count"
                    ],

                "same_sentence_markers":
                    result[
                        "linked_markers"
                    ],

                "all_rescue_markers":
                    sorted(rescue_markers),
            }
        )

    #
    # Descriptive ordering only.
    #
    rows.sort(
        key=lambda r: (
            -r["distinct_pairwise_queries"],
            -r["pairwise_marker_papers"],
            -r["breast_context_papers"],
            -r["title_mention_papers"],
            -r["same_sentence_marker_papers"],
            -r["total_papers"],
            r["cell_type"].casefold(),
        )
    )

    outdir = args.output_directory
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out = {
        "schema_name":
            "RescueCandidateProvenanceAudit",
        "schema_version":
            "1.0",
        "cluster_id":
            discovery.get("cluster_id"),
        "important_note": (
            "Candidate ordering is descriptive. "
            "No confidence score or annotation "
            "probability is calculated."
        ),
        "results":
            rows,
    }

    path = (
        outdir
        / "cluster_7_rescue_candidate_provenance_audit.json"
    )

    path.write_text(
        json.dumps(
            out,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "Rescue candidate provenance audit "
        "completed successfully."
    )
    print()

    for i, r in enumerate(
        rows[:25],
        1,
    ):
        status = (
            "NOVEL"
            if r["is_novel_to_agent2"]
            else "UPSTREAM"
        )

        print(
            f"{i:02d}. "
            f"{r['cell_type']} "
            f"[{r['ontology_id']}] "
            f"| pairwise_queries="
            f"{r['distinct_pairwise_queries']} "
            f"| pairwise_papers="
            f"{r['pairwise_marker_papers']} "
            f"| breast_papers="
            f"{r['breast_context_papers']} "
            f"| title_papers="
            f"{r['title_mention_papers']} "
            f"| same_sentence="
            f"{r['same_sentence_marker_papers']} "
            f"| total="
            f"{r['total_papers']} "
            f"| {status}"
        )

        print(
            "    pairwise IDs:",
            ", ".join(
                r["pairwise_query_ids"]
            ) or "none",
        )

        print(
            "    rescue markers:",
            ", ".join(
                r["all_rescue_markers"]
            ) or "none",
        )

    print()
    print("Saved:", path)


if __name__ == "__main__":
    main()
