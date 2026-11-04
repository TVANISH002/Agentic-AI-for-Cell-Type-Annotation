from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def split_semicolon(value):
    if not value:
        return []
    return [x.strip() for x in str(value).split(";") if x.strip()]


def parse_query_list(query: str, label: str):
    pattern = rf"{re.escape(label)}:\s*(.*?)(?:\.\s+[A-Z][A-Za-z ]+:|\.\s+Find passages|$)"
    m = re.search(pattern, query or "")
    if not m:
        return []

    text = m.group(1).strip().rstrip(".")
    return [x.strip() for x in text.split(",") if x.strip()]


def paper_key(passage):
    for key in ("pmid", "doi", "pmcid", "title"):
        value = str(passage.get(key, "") or "").strip()
        if value:
            return f"{key}:{value}"
    return str(passage.get("passage_id", ""))


def main():
    ap = argparse.ArgumentParser(
        description="Audit PubMedBERT candidate-specific literature evidence without assigning a cell type."
    )
    ap.add_argument("--pubmedbert-json", required=True)
    ap.add_argument("--output-directory", required=True)
    args = ap.parse_args()

    src = Path(args.pubmedbert_json)
    outdir = Path(args.output_directory)
    outdir.mkdir(parents=True, exist_ok=True)

    data = json.loads(src.read_text(encoding="utf-8"))

    cluster_id = str(data.get("cluster_id", ""))
    summary_rows = []
    passage_rows = []
    audit_results = []

    for result in data.get("results", []):
        candidate = result["candidate"]
        query = result.get("candidate_query", "")

        supporting_genes = parse_query_list(
            query,
            "Candidate supporting genes",
        )

        observed_markers = parse_query_list(
            query,
            "Observed cluster markers",
        )

        supporting_set = {x.upper() for x in supporting_genes}
        observed_set = {x.upper() for x in observed_markers}

        top = result.get("top_passages", [])

        candidate_mention_passages = 0
        marker_mention_passages = 0
        supporting_marker_passages = 0
        candidate_supporting_marker_passages = 0
        context_passages = 0
        tissue_disease_passages = 0

        seen_supporting = set()
        seen_observed = set()
        unique_papers = set()

        for p in top:
            candidate_mentions = split_semicolon(
                p.get("candidate_mentions", "")
            )
            marker_mentions = split_semicolon(
                p.get("marker_mentions", "")
            )
            context_matches = split_semicolon(
                p.get("context_matches", "")
            )

            candidate_present = any(
                x.casefold() == candidate.casefold()
                for x in candidate_mentions
            )

            marker_upper = {x.upper() for x in marker_mentions}

            supporting_here = sorted(
                marker_upper & supporting_set
            )
            observed_here = sorted(
                marker_upper & observed_set
            )

            candidate_and_supporting = bool(
                candidate_present and supporting_here
            )

            if candidate_present:
                candidate_mention_passages += 1

            if observed_here:
                marker_mention_passages += 1

            if supporting_here:
                supporting_marker_passages += 1

            if candidate_and_supporting:
                candidate_supporting_marker_passages += 1

            if context_matches:
                context_passages += 1

            context_lower = {x.casefold() for x in context_matches}
            if "tissue" in context_lower and "disease" in context_lower:
                tissue_disease_passages += 1

            seen_supporting.update(supporting_here)
            seen_observed.update(observed_here)
            unique_papers.add(paper_key(p))

            passage_rows.append(
                {
                    "cluster_id": cluster_id,
                    "candidate": candidate,
                    "rank_within_candidate": p.get("rank_within_candidate"),
                    "similarity": p.get("similarity"),
                    "passage_id": p.get("passage_id"),
                    "title": p.get("title"),
                    "pmid": p.get("pmid"),
                    "doi": p.get("doi"),
                    "candidate_mentions": "; ".join(candidate_mentions),
                    "marker_mentions": "; ".join(marker_mentions),
                    "supporting_markers_in_passage": "; ".join(supporting_here),
                    "observed_markers_in_passage": "; ".join(observed_here),
                    "context_matches": "; ".join(context_matches),
                    "candidate_present": candidate_present,
                    "supporting_marker_present": bool(supporting_here),
                    "candidate_and_supporting_marker": candidate_and_supporting,
                }
            )

        unexplained_supporting = [
            g for g in supporting_genes
            if g.upper() not in seen_supporting
        ]

        unexplained_observed = [
            g for g in observed_markers
            if g.upper() not in seen_observed
        ]

        summary = {
            "cluster_id": cluster_id,
            "candidate": candidate,
            "eligible_passage_count": result.get("eligible_passage_count"),
            "top_passage_count": len(top),
            "independent_top_paper_count": len(unique_papers),

            # Retrieval similarity only — NOT confidence.
            "mean_similarity": result.get("mean_similarity"),
            "max_similarity": result.get("max_similarity"),
            "median_similarity": result.get("median_similarity"),

            "candidate_mention_passages": candidate_mention_passages,
            "observed_marker_passages": marker_mention_passages,
            "supporting_marker_passages": supporting_marker_passages,
            "candidate_and_supporting_marker_passages":
                candidate_supporting_marker_passages,

            "context_matched_passages": context_passages,
            "tissue_and_disease_matched_passages":
                tissue_disease_passages,

            "candidate_supporting_gene_count": len(supporting_genes),
            "distinct_supporting_markers_retrieved":
                len(seen_supporting),
            "distinct_observed_markers_retrieved":
                len(seen_observed),

            "supporting_genes": "; ".join(supporting_genes),
            "retrieved_supporting_genes":
                "; ".join(sorted(seen_supporting)),
            "unexplained_supporting_genes":
                "; ".join(unexplained_supporting),
            "unexplained_observed_markers":
                "; ".join(unexplained_observed),
        }

        summary_rows.append(summary)

        audit_results.append(
            {
                **summary,
                "supporting_genes_list": supporting_genes,
                "observed_markers_list": observed_markers,
                "retrieved_supporting_genes_list":
                    sorted(seen_supporting),
                "retrieved_observed_markers_list":
                    sorted(seen_observed),
                "unexplained_supporting_genes_list":
                    unexplained_supporting,
                "unexplained_observed_markers_list":
                    unexplained_observed,
            }
        )

    summary_csv = outdir / f"cluster_{cluster_id}_pubmedbert_evidence_summary.csv"
    passages_csv = outdir / f"cluster_{cluster_id}_pubmedbert_evidence_passages.csv"
    audit_json = outdir / f"cluster_{cluster_id}_pubmedbert_evidence_audit.json"

    if summary_rows:
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(summary_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(summary_rows)

    if passage_rows:
        with passages_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(passage_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(passage_rows)

    audit = {
        "schema_name": "pubmedbert_evidence_audit",
        "schema_version": "1.0",
        "cluster_id": cluster_id,
        "source_pubmedbert_json": str(src),
        "important_note":
            "PubMedBERT similarity is semantic retrieval relevance, not biological confidence or annotation probability.",
        "candidate_count": len(audit_results),
        "results": audit_results,
    }

    audit_json.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("PubMedBERT evidence audit completed successfully.")
    print("Cluster:", cluster_id)
    print("Candidates:", len(summary_rows))
    print()

    for row in summary_rows:
        print(
            f"{row['candidate']}: "
            f"papers={row['independent_top_paper_count']}, "
            f"candidate_mentions={row['candidate_mention_passages']}, "
            f"supporting_marker_passages={row['supporting_marker_passages']}, "
            f"candidate+supporting_marker="
            f"{row['candidate_and_supporting_marker_passages']}, "
            f"distinct_supporting_markers="
            f"{row['distinct_supporting_markers_retrieved']}/"
            f"{row['candidate_supporting_gene_count']}"
        )

    print()
    print("Summary CSV:", summary_csv)
    print("Passage CSV:", passages_csv)
    print("Audit JSON:", audit_json)


if __name__ == "__main__":
    main()
