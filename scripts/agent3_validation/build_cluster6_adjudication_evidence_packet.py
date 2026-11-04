from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def clean(value: Any) -> str:
    return " ".join(
        str(value or "").split()
    )


def load_json(
    path: Path,
) -> dict:
    return json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )


def normalize_path(
    value: Any,
) -> str:
    return clean(value).replace(
        "\\",
        "/",
    ).casefold()


def index_results(
    rows: list[dict],
) -> dict[str, dict]:

    output = {}

    for row in rows:
        candidate = clean(
            row.get(
                "candidate",
                "",
            )
        )

        if not candidate:
            raise ValueError(
                "Candidate name missing."
            )

        key = candidate.casefold()

        if key in output:
            raise ValueError(
                f"Duplicate candidate: "
                f"{candidate}"
            )

        output[key] = row

    return output


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Construct the frozen Cluster 6 "
            "candidate-adjudication evidence packet "
            "without generating a combined winner "
            "score or calibrated annotation decision."
        )
    )

    parser.add_argument(
        "--pubmedbert-json",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--evidence-audit-json",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-directory",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    pubmedbert = load_json(
        args.pubmedbert_json
    )

    audit = load_json(
        args.evidence_audit_json
    )

    #
    # --------------------------------------------------
    # Cluster consistency
    # --------------------------------------------------
    #

    pub_cluster = clean(
        pubmedbert.get(
            "cluster_id",
            "",
        )
    )

    audit_cluster = clean(
        audit.get(
            "cluster_id",
            "",
        )
    )

    if (
        not pub_cluster
        or not audit_cluster
        or pub_cluster != audit_cluster
    ):
        raise ValueError(
            "Cluster mismatch: "
            f"PubMedBERT={pub_cluster}, "
            f"audit={audit_cluster}"
        )

    cluster_id = pub_cluster

    #
    # --------------------------------------------------
    # Verify that the evidence audit was created from
    # the exact PubMedBERT artifact supplied here.
    # --------------------------------------------------
    #

    recorded_source = normalize_path(
        audit.get(
            "source_pubmedbert_json",
            "",
        )
    )

    supplied_source = normalize_path(
        args.pubmedbert_json
    )

    if (
        recorded_source
        and recorded_source
        != supplied_source
    ):
        raise ValueError(
            "Evidence-audit PubMedBERT source "
            "does not match supplied artifact.\n"
            f"Recorded: {recorded_source}\n"
            f"Supplied: {supplied_source}"
        )

    pub_rows = pubmedbert.get(
        "results",
        [],
    )

    audit_rows = audit.get(
        "results",
        [],
    )

    if not isinstance(
        pub_rows,
        list,
    ):
        raise ValueError(
            "PubMedBERT results must be a list."
        )

    if not isinstance(
        audit_rows,
        list,
    ):
        raise ValueError(
            "Evidence-audit results must be a list."
        )

    pub_index = index_results(
        pub_rows
    )

    audit_index = index_results(
        audit_rows
    )

    if set(pub_index) != set(
        audit_index
    ):
        raise ValueError(
            "Candidate sets differ between "
            "PubMedBERT and evidence audit.\n"
            f"PubMedBERT: "
            f"{sorted(pub_index)}\n"
            f"Audit: "
            f"{sorted(audit_index)}"
        )

    #
    # Preserve PubMedBERT candidate order.
    #
    candidates = []

    for pub_item in pub_rows:

        candidate = clean(
            pub_item.get(
                "candidate",
                "",
            )
        )

        audit_item = audit_index[
            candidate.casefold()
        ]

        #
        # ----------------------------------------------
        # Similarity consistency check
        # ----------------------------------------------
        #

        for field in (
            "mean_similarity",
            "max_similarity",
            "median_similarity",
        ):

            pub_value = pub_item.get(
                field
            )

            audit_value = audit_item.get(
                field
            )

            if (
                pub_value is None
                or audit_value is None
            ):
                raise ValueError(
                    f"{candidate}: missing "
                    f"{field}"
                )

            if abs(
                float(pub_value)
                - float(audit_value)
            ) > 1e-9:
                raise ValueError(
                    f"{candidate}: {field} "
                    "differs between artifacts."
                )

        supporting_gene_count = int(
            audit_item.get(
                "candidate_supporting_gene_count",
                0,
            )
            or 0
        )

        retrieved_supporting_count = int(
            audit_item.get(
                "distinct_supporting_markers_retrieved",
                0,
            )
            or 0
        )

        if supporting_gene_count > 0:
            supporting_gene_retrieval_fraction = (
                retrieved_supporting_count
                / supporting_gene_count
            )
        else:
            supporting_gene_retrieval_fraction = None

        record = {

            #
            # ------------------------------------------
            # Candidate identity
            # ------------------------------------------
            #

            "cluster_id":
                cluster_id,

            "candidate":
                candidate,

            "candidate_source":
                "Agent2",

            "candidate_already_upstream":
                True,

            #
            # ------------------------------------------
            # Candidate evidence pool
            # ------------------------------------------
            #

            "eligible_passage_count":
                int(
                    audit_item.get(
                        "eligible_passage_count",
                        0,
                    )
                    or 0
                ),

            "top_passage_count":
                int(
                    audit_item.get(
                        "top_passage_count",
                        0,
                    )
                    or 0
                ),

            "independent_top_paper_count":
                int(
                    audit_item.get(
                        "independent_top_paper_count",
                        0,
                    )
                    or 0
                ),

            #
            # ------------------------------------------
            # Literature grounding
            # ------------------------------------------
            #

            "candidate_mention_passages":
                int(
                    audit_item.get(
                        "candidate_mention_passages",
                        0,
                    )
                    or 0
                ),

            "observed_marker_passages":
                int(
                    audit_item.get(
                        "observed_marker_passages",
                        0,
                    )
                    or 0
                ),

            "supporting_marker_passages":
                int(
                    audit_item.get(
                        "supporting_marker_passages",
                        0,
                    )
                    or 0
                ),

            "candidate_and_supporting_marker_passages":
                int(
                    audit_item.get(
                        "candidate_and_supporting_marker_passages",
                        0,
                    )
                    or 0
                ),

            "context_matched_passages":
                int(
                    audit_item.get(
                        "context_matched_passages",
                        0,
                    )
                    or 0
                ),

            "tissue_and_disease_matched_passages":
                int(
                    audit_item.get(
                        "tissue_and_disease_matched_passages",
                        0,
                    )
                    or 0
                ),

            #
            # ------------------------------------------
            # Agent 2 supporting genes
            # ------------------------------------------
            #

            "candidate_supporting_gene_count":
                supporting_gene_count,

            "distinct_supporting_markers_retrieved":
                retrieved_supporting_count,

            "supporting_gene_retrieval_fraction":
                supporting_gene_retrieval_fraction,

            "supporting_genes":
                clean(
                    audit_item.get(
                        "supporting_genes",
                        "",
                    )
                ),

            "retrieved_supporting_genes":
                clean(
                    audit_item.get(
                        "retrieved_supporting_genes",
                        "",
                    )
                ),

            "unexplained_supporting_genes":
                clean(
                    audit_item.get(
                        "unexplained_supporting_genes",
                        "",
                    )
                ),

            "distinct_observed_markers_retrieved":
                int(
                    audit_item.get(
                        "distinct_observed_markers_retrieved",
                        0,
                    )
                    or 0
                ),

            "unexplained_observed_markers":
                clean(
                    audit_item.get(
                        "unexplained_observed_markers",
                        "",
                    )
                ),

            #
            # ------------------------------------------
            # PubMedBERT semantic relevance
            #
            # IMPORTANT:
            # These are NOT probabilities/confidence.
            # ------------------------------------------
            #

            "pubmedbert_model":
                clean(
                    pubmedbert.get(
                        "model_id",
                        "",
                    )
                ),

            "pubmedbert_mean_similarity":
                float(
                    pub_item[
                        "mean_similarity"
                    ]
                ),

            "pubmedbert_max_similarity":
                float(
                    pub_item[
                        "max_similarity"
                    ]
                ),

            "pubmedbert_median_similarity":
                float(
                    pub_item[
                        "median_similarity"
                    ]
                ),

            "insufficient_candidate_evidence":
                bool(
                    pub_item.get(
                        "insufficient_candidate_evidence",
                        False,
                    )
                ),

            #
            # ------------------------------------------
            # Decision status
            # ------------------------------------------
            #

            "evidence_profile":
                "CANDIDATE_ADJUDICATION_EVIDENCE",

            "automatic_acceptance":
                "NOT_CALIBRATED",

            "final_annotation":
                None,

            #
            # ------------------------------------------
            # Full PubMedBERT provenance retained
            # in JSON.
            # ------------------------------------------
            #

            "pubmedbert_candidate_query":
                clean(
                    pub_item.get(
                        "candidate_query",
                        "",
                    )
                ),

            "pubmedbert_top_passages":
                pub_item.get(
                    "top_passages",
                    [],
                ),
        }

        candidates.append(
            record
        )

    #
    # --------------------------------------------------
    # Output
    # --------------------------------------------------
    #

    output_directory = (
        args.output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        output_directory
        / (
            f"cluster_{cluster_id}_"
            "adjudication_evidence_packet.json"
        )
    )

    csv_path = (
        output_directory
        / (
            f"cluster_{cluster_id}_"
            "adjudication_evidence_packet.csv"
        )
    )

    output = {

        "schema_name":
            "Agent3CandidateAdjudicationEvidencePacket",

        "schema_version":
            "1.0",

        "cluster_id":
            cluster_id,

        "candidate_count":
            len(candidates),

        "model_id":
            clean(
                pubmedbert.get(
                    "model_id",
                    "",
                )
            ),

        "corpus_json":
            clean(
                pubmedbert.get(
                    "corpus_json",
                    "",
                )
            ),

        "corpus_passage_count":
            int(
                pubmedbert.get(
                    "passage_count",
                    0,
                )
                or 0
            ),

        "top_marker_count":
            int(
                pubmedbert.get(
                    "top_marker_count",
                    0,
                )
                or 0
            ),

        "top_k_passages":
            int(
                pubmedbert.get(
                    "top_k_passages",
                    0,
                )
                or 0
            ),

        "passage_selection_method":
            clean(
                pubmedbert.get(
                    "passage_selection_method",
                    "",
                )
            ),

        "candidate_eligibility_rule":
            clean(
                pubmedbert.get(
                    "candidate_eligibility_rule",
                    "",
                )
            ),

        "expert_label_used":
            False,

        "reference_label_used":
            False,

        "adjudication_status":
            "UNCALIBRATED_COMPETING_EVIDENCE",

        "automatic_acceptance":
            "NOT_CALIBRATED",

        "final_annotation":
            None,

        "important_note": (
            "Candidate evidence dimensions are "
            "reported separately. No weighted "
            "candidate score is computed. "
            "PubMedBERT cosine similarity represents "
            "semantic relevance and must not be "
            "interpreted as annotation probability "
            "or confidence."
        ),

        "source_artifacts": {

            "pubmedbert":
                str(
                    args.pubmedbert_json
                ),

            "evidence_audit":
                str(
                    args.evidence_audit_json
                ),
        },

        "candidates":
            candidates,
    }

    json_path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    #
    # CSV contains scalar summary fields only.
    #
    csv_rows = []

    for record in candidates:

        row = {
            key: value
            for key, value
            in record.items()
            if key not in {
                "pubmedbert_top_passages",
            }
        }

        csv_rows.append(
            row
        )

    if csv_rows:

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:

            writer = csv.DictWriter(
                handle,
                fieldnames=list(
                    csv_rows[0].keys()
                ),
            )

            writer.writeheader()

            writer.writerows(
                csv_rows
            )

    #
    # --------------------------------------------------
    # Terminal audit
    # --------------------------------------------------
    #

    print(
        "Agent 3 Cluster 6 adjudication "
        "evidence packet completed successfully."
    )

    print(
        "Cluster:",
        cluster_id,
    )

    print(
        "Candidates:",
        len(candidates),
    )

    print(
        "Reference label used:",
        False,
    )

    print(
        "Automatic acceptance:",
        "NOT_CALIBRATED",
    )

    print()

    print(
        "CANDIDATE EVIDENCE PROFILES"
    )

    print("=" * 120)

    for record in candidates:

        print()

        print(
            record[
                "candidate"
            ]
        )

        print(
            "  semantic mean:",
            round(
                record[
                    "pubmedbert_mean_similarity"
                ],
                6,
            ),
        )

        print(
            "  independent top papers:",
            record[
                "independent_top_paper_count"
            ],
        )

        print(
            "  candidate mentions:",
            record[
                "candidate_mention_passages"
            ],
        )

        print(
            "  supporting-marker passages:",
            record[
                "supporting_marker_passages"
            ],
        )

        print(
            "  candidate+supporting-marker:",
            record[
                "candidate_and_supporting_marker_passages"
            ],
        )

        print(
            "  supporting genes retrieved:",
            (
                f"{record['distinct_supporting_markers_retrieved']}"
                f"/"
                f"{record['candidate_supporting_gene_count']}"
            ),
        )

        print(
            "  context matched:",
            record[
                "context_matched_passages"
            ],
        )

        print(
            "  tissue+disease matched:",
            record[
                "tissue_and_disease_matched_passages"
            ],
        )

    print()

    print(
        "Cluster adjudication status:",
        "UNCALIBRATED_COMPETING_EVIDENCE",
    )

    print()

    print(
        "JSON:",
        json_path,
    )

    print(
        "CSV:",
        csv_path,
    )


if __name__ == "__main__":
    main()
