from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def as_bool(value):
    if isinstance(value, bool):
        return value

    return str(value).strip().casefold() in {
        "true", "1", "yes"
    }


def split_semicolon(value):
    if pd.isna(value) or not str(value).strip():
        return []

    return [
        x.strip().upper()
        for x in str(value).split(";")
        if x.strip()
    ]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute rank-stratified marker coverage from "
            "candidate-specific PubMedBERT evidence."
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

    passages = pd.read_csv(
        passage_path
    )

    ranked_markers = sorted(
        request["marker_genes"],
        key=lambda x: x["rank_within_cluster"],
    )

    marker_rank = {
        x["gene"].upper():
            int(x["rank_within_cluster"])
        for x in ranked_markers
    }

    ranked_gene_names = [
        x["gene"].upper()
        for x in ranked_markers
    ]

    results = []

    for candidate, group in passages.groupby(
        "candidate",
        sort=False,
    ):
        direct_markers = set()

        for _, row in group.iterrows():
            if not as_bool(
                row.get(
                    "candidate_and_supporting_marker",
                    False,
                )
            ):
                continue

            direct_markers.update(
                split_semicolon(
                    row.get(
                        "supporting_markers_in_passage",
                        "",
                    )
                )
            )

        candidate_result = {
            "candidate": candidate,
            "direct_candidate_markers": sorted(
                direct_markers,
                key=lambda g: marker_rank.get(
                    g,
                    10**9,
                ),
            ),
        }

        for cutoff in (5, 10, 25):
            top = ranked_gene_names[:cutoff]

            explained = [
                g for g in top
                if g in direct_markers
            ]

            unexplained = [
                g for g in top
                if g not in direct_markers
            ]

            candidate_result[
                f"top_{cutoff}_explained_count"
            ] = len(explained)

            candidate_result[
                f"top_{cutoff}_coverage"
            ] = len(explained) / len(top)

            candidate_result[
                f"top_{cutoff}_explained_markers"
            ] = explained

            candidate_result[
                f"top_{cutoff}_unexplained_markers"
            ] = unexplained

        results.append(
            candidate_result
        )

    output = {
        "schema_name":
            "pubmedbert_rank_coverage_audit",
        "schema_version": "1.0",
        "cluster_id":
            str(request["cluster_id"]),
        "important_note": (
            "Coverage reflects candidate-associated "
            "marker evidence in PubMedBERT-selected "
            "literature passages. It is not biological "
            "confidence or annotation probability."
        ),
        "results": results,
    }

    json_path = (
        outdir
        / f"cluster_{request['cluster_id']}_"
          "pubmedbert_rank_coverage.json"
    )

    json_path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "Rank-aware PubMedBERT evidence audit "
        "completed successfully."
    )
    print(
        "Cluster:",
        request["cluster_id"],
    )
    print()

    for r in results:
        print(r["candidate"])

        for cutoff in (5, 10, 25):
            print(
                f"  Top-{cutoff}: "
                f"{r[f'top_{cutoff}_explained_count']}"
                f"/{cutoff} "
                f"({r[f'top_{cutoff}_coverage']:.1%})"
            )

            print(
                "    explained:",
                ", ".join(
                    r[
                        f"top_{cutoff}_explained_markers"
                    ]
                ) or "none",
            )

        print()

    print(
        "JSON:",
        json_path,
    )


if __name__ == "__main__":
    main()
