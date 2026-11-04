from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def norm(value) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(value or ""),
    ).strip()


def phrase_pattern(term: str):
    term = norm(term)

    words = term.split()

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

    return re.compile(
        rf"(?<![A-Za-z0-9])"
        rf"{phrase}"
        rf"(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )


def gene_pattern(gene: str):
    return re.compile(
        rf"(?<![A-Za-z0-9])"
        rf"{re.escape(gene)}"
        rf"(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )


def sentence_split(text: str):
    text = norm(text)

    if not text:
        return []

    return [
        item.strip()
        for item in re.split(
            r"(?<=[.!?])\s+",
            text,
        )
        if item.strip()
    ]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--pilot-json",
        required=True,
        type=Path,
    )

    ap.add_argument(
        "--confirmation-plan",
        required=True,
        type=Path,
    )

    ap.add_argument(
        "--output-directory",
        required=True,
        type=Path,
    )

    args = ap.parse_args()

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

    plan_lookup = {
        item["query_id"]: item
        for item in plan.get(
            "queries",
            [],
        )
    }

    rows = []
    detailed = []

    for qr in pilot.get(
        "query_results",
        [],
    ):
        query_obj = (
            qr.get("query", {})
            or {}
        )

        qid = norm(
            query_obj.get(
                "query_id",
                qr.get(
                    "query_id",
                    "",
                ),
            )
        )

        plan_item = plan_lookup.get(
            qid,
            {},
        )

        candidate = norm(
            plan_item.get(
                "candidate",
                "",
            )
        )

        markers = [
            norm(x).upper()
            for x in plan_item.get(
                "markers",
                [],
            )
            if norm(x)
        ]

        candidate_re = (
            phrase_pattern(candidate)
        )

        marker_res = {
            gene: gene_pattern(gene)
            for gene in markers
        }

        papers = (
            qr.get("papers")
            or qr.get("publications")
            or qr.get("results")
            or qr.get("records")
            or []
        )

        candidate_mention_papers = set()
        marker_mention_papers = set()
        candidate_marker_papers = set()
        same_sentence_papers = set()

        linked_markers = set()

        provenance = []

        for paper in papers:
            title = norm(
                paper.get("title", "")
            )

            abstract = norm(
                paper.get(
                    "abstract",
                    paper.get(
                        "abstract_text",
                        "",
                    ),
                )
            )

            text = norm(
                f"{title}. {abstract}"
            )

            paper_key = (
                norm(paper.get("pmid"))
                or norm(paper.get("doi"))
                or title
            )

            candidate_found = bool(
                candidate_re
                and candidate_re.search(
                    text
                )
            )

            present_markers = {
                gene
                for gene, pattern
                in marker_res.items()
                if pattern.search(text)
            }

            same_sentence_markers = set()

            for sentence in sentence_split(
                text
            ):
                if not (
                    candidate_re
                    and candidate_re.search(
                        sentence
                    )
                ):
                    continue

                for gene, pattern in (
                    marker_res.items()
                ):
                    if pattern.search(
                        sentence
                    ):
                        same_sentence_markers.add(
                            gene
                        )

            if candidate_found:
                candidate_mention_papers.add(
                    paper_key
                )

            if present_markers:
                marker_mention_papers.add(
                    paper_key
                )

            if (
                candidate_found
                and present_markers
            ):
                candidate_marker_papers.add(
                    paper_key
                )

            if same_sentence_markers:
                same_sentence_papers.add(
                    paper_key
                )

            linked_markers.update(
                same_sentence_markers
            )

            provenance.append(
                {
                    "title": title,
                    "pmid":
                        paper.get("pmid"),
                    "doi":
                        paper.get("doi"),
                    "candidate_mentioned":
                        candidate_found,
                    "markers_mentioned":
                        sorted(
                            present_markers
                        ),
                    "same_sentence_markers":
                        sorted(
                            same_sentence_markers
                        ),
                }
            )

        row = {
            "query_id":
                qid,

            "candidate":
                candidate,

            "hit_count":
                qr.get(
                    "hit_count",
                    0,
                ),

            "returned_paper_count":
                len(papers),

            "candidate_mention_papers":
                len(
                    candidate_mention_papers
                ),

            "marker_mention_papers":
                len(
                    marker_mention_papers
                ),

            "candidate_and_marker_papers":
                len(
                    candidate_marker_papers
                ),

            "same_sentence_candidate_marker_papers":
                len(
                    same_sentence_papers
                ),

            "distinct_same_sentence_markers":
                len(
                    linked_markers
                ),

            "same_sentence_markers":
                "; ".join(
                    sorted(
                        linked_markers
                    )
                ),
        }

        rows.append(
            row
        )

        detailed.append(
            {
                **row,
                "paper_provenance":
                    provenance,
            }
        )

    #
    # Descriptive ordering only.
    # This is NOT a confidence ranking.
    #
    rows.sort(
        key=lambda row: (
            -row[
                "candidate_and_marker_papers"
            ],
            -row[
                "same_sentence_candidate_marker_papers"
            ],
            -row[
                "distinct_same_sentence_markers"
            ],
            -row[
                "candidate_mention_papers"
            ],
            row[
                "candidate"
            ].casefold(),
        )
    )

    outdir = (
        args.output_directory
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        outdir
        / (
            "cluster_7_rescue_"
            "confirmation_audit.csv"
        )
    )

    json_path = (
        outdir
        / (
            "cluster_7_rescue_"
            "confirmation_audit.json"
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

    output = {
        "schema_name":
            "RescueConfirmationEvidenceAudit",

        "schema_version":
            "1.0",

        "cluster_id":
            str(
                pilot.get(
                    "cluster_id",
                    plan.get(
                        "cluster_id",
                        "",
                    ),
                )
            ),

        "expert_label_used":
            False,

        "reference_label_used":
            False,

        "important_note": (
            "Counts describe retrieved abstract evidence. "
            "They are not annotation probabilities or "
            "confidence scores."
        ),

        "results":
            detailed,
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
        "Rescue confirmation evidence audit "
        "completed successfully."
    )

    print()

    print(
        "TOP 25 DESCRIPTIVE EVIDENCE ROWS"
    )

    print("=" * 125)

    for i, row in enumerate(
        rows[:25],
        1,
    ):
        print(
            f"{i:02d}. "
            f"{row['candidate']} "
            f"| returned="
            f"{row['returned_paper_count']} "
            f"| candidate="
            f"{row['candidate_mention_papers']} "
            f"| marker="
            f"{row['marker_mention_papers']} "
            f"| candidate+marker="
            f"{row['candidate_and_marker_papers']} "
            f"| same_sentence="
            f"{row['same_sentence_candidate_marker_papers']} "
            f"| linked_markers="
            f"{row['distinct_same_sentence_markers']}"
        )

        print(
            "    same-sentence markers:",
            row[
                "same_sentence_markers"
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
