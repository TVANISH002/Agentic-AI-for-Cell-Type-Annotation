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


def load_json(path: Path) -> dict:
    return json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )


def unique_index(
    rows: list[dict],
    key_name: str,
) -> dict[str, dict]:

    result = {}

    for row in rows:
        value = clean(
            row.get(key_name, "")
        )

        if not value:
            continue

        key = value.casefold()

        if key in result:
            raise ValueError(
                f"Duplicate {key_name}: {value}"
            )

        result[key] = row

    return result


def bool_count(
    value: Any,
) -> bool:
    try:
        return int(value or 0) > 0
    except (
        TypeError,
        ValueError,
    ):
        return False


def classify_evidence(
    *,
    novel: bool,
    candidate_mentions: int,
    marker_papers: int,
    target_candidate_papers: int,
    target_marker_papers: int,
) -> tuple[str, str]:

    marker_grounded = (
        marker_papers > 0
    )

    target_supported = (
        target_candidate_papers > 0
    )

    target_marker_grounded = (
        target_marker_papers > 0
    )

    if target_marker_grounded:
        evidence_state = (
            "TARGET_CONTEXT_MARKER_GROUNDED"
        )

    elif (
        target_supported
        and marker_grounded
    ):
        evidence_state = (
            "TARGET_CONTEXT_SUPPORTED_"
            "WITH_CROSS_CONTEXT_MARKER_GROUNDING"
        )

    elif marker_grounded:
        evidence_state = (
            "CROSS_CONTEXT_MARKER_GROUNDED"
        )

    elif target_supported:
        evidence_state = (
            "TARGET_CONTEXT_ONLY"
        )

    elif candidate_mentions > 0:
        evidence_state = (
            "LITERATURE_MENTION_ONLY"
        )

    else:
        evidence_state = (
            "INSUFFICIENT_RESCUE_EVIDENCE"
        )

    if not novel:
        prefix = "UPSTREAM_CANDIDATE"

    else:
        prefix = "RECOVERED_CANDIDATE"

    if evidence_state == (
        "TARGET_CONTEXT_MARKER_GROUNDED"
    ):
        reporting_state = (
            prefix
            + "_WITH_TARGET_CONTEXT_"
            + "MARKER_GROUNDING"
        )

    elif evidence_state == (
        "TARGET_CONTEXT_SUPPORTED_"
        "WITH_CROSS_CONTEXT_MARKER_GROUNDING"
    ):
        reporting_state = (
            prefix
            + "_WITH_PARTIAL_CONTEXT_GROUNDING"
        )

    elif evidence_state == (
        "CROSS_CONTEXT_MARKER_GROUNDED"
    ):
        reporting_state = (
            prefix
            + "_WITH_MARKER_GROUNDING_ONLY"
        )

    elif evidence_state == (
        "TARGET_CONTEXT_ONLY"
    ):
        reporting_state = (
            prefix
            + "_WITH_TARGET_CONTEXT_ONLY"
        )

    elif evidence_state == (
        "LITERATURE_MENTION_ONLY"
    ):
        reporting_state = (
            prefix
            + "_WITH_LITERATURE_MENTION_ONLY"
        )

    else:
        reporting_state = (
            "INSUFFICIENT_RESCUE_EVIDENCE"
        )

    return (
        evidence_state,
        reporting_state,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Combine Agent 3 rescue candidate "
            "discovery, provenance, grounding, "
            "and PubMedBERT evidence without "
            "creating a winner score."
        )
    )

    parser.add_argument(
        "--discovery-json",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--provenance-json",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--grounding-json",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--pubmedbert-json",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-directory",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    discovery = load_json(
        args.discovery_json
    )

    provenance = load_json(
        args.provenance_json
    )

    grounding = load_json(
        args.grounding_json
    )

    pubmedbert = load_json(
        args.pubmedbert_json
    )

    # Cluster consistency
    
    cluster_ids = {
        clean(
            artifact.get(
                "cluster_id",
                "",
            )
        )
        for artifact in (
            discovery,
            provenance,
            grounding,
            pubmedbert,
        )
        if clean(
            artifact.get(
                "cluster_id",
                "",
            )
        )
    }

    if len(cluster_ids) != 1:
        raise ValueError(
            "Cluster mismatch across "
            f"source artifacts: {cluster_ids}"
        )

    cluster_id = next(
        iter(cluster_ids)
    )

    
    # Leakage guards where fields exist.
   
    for name, artifact in (
        (
            "discovery",
            discovery,
        ),
        (
            "grounding",
            grounding,
        ),
        (
            "pubmedbert",
            pubmedbert,
        ),
    ):
        if artifact.get(
            "expert_label_used",
            False,
        ):
            raise ValueError(
                f"{name}: expert label "
                "usage detected."
            )

        if artifact.get(
            "reference_label_used",
            False,
        ):
            raise ValueError(
                f"{name}: reference label "
                "usage detected."
            )

    discovery_rows = discovery.get(
        "results",
        [],
    )

    provenance_rows = provenance.get(
        "results",
        [],
    )

    grounding_rows = grounding.get(
        "results",
        [],
    )

    pubmedbert_rows = pubmedbert.get(
        "results",
        [],
    )

    if not (
        isinstance(
            discovery_rows,
            list,
        )
        and isinstance(
            provenance_rows,
            list,
        )
        and isinstance(
            grounding_rows,
            list,
        )
        and isinstance(
            pubmedbert_rows,
            list,
        )
    ):
        raise ValueError(
            "One or more results fields "
            "are not lists."
        )

    
    # Preserve the Cell Ontology discovery ordering. No new ranking is introduced.
    
    provenance_by_ontology = (
        unique_index(
            provenance_rows,
            "ontology_id",
        )
    )

    grounding_by_name = (
        unique_index(
            grounding_rows,
            "candidate",
        )
    )

    pubmedbert_by_name = (
        unique_index(
            pubmedbert_rows,
            "candidate",
        )
    )

    consolidated = []

    missing_records = []

    for discovery_rank, item in enumerate(
        discovery_rows,
        start=1,
    ):
        candidate = clean(
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

        if not candidate:
            raise ValueError(
                "Discovery candidate "
                "without cell_type."
            )

        provenance_item = (
            provenance_by_ontology.get(
                ontology_id.casefold()
            )
            if ontology_id
            else None
        )

        grounding_item = (
            grounding_by_name.get(
                candidate.casefold()
            )
        )

        pubmedbert_item = (
            pubmedbert_by_name.get(
                candidate.casefold()
            )
        )

        if (
            provenance_item is None
            or grounding_item is None
            or pubmedbert_item is None
        ):
            missing_records.append(
                {
                    "candidate":
                        candidate,

                    "ontology_id":
                        ontology_id,

                    "missing_provenance":
                        provenance_item
                        is None,

                    "missing_grounding":
                        grounding_item
                        is None,

                    "missing_pubmedbert":
                        pubmedbert_item
                        is None,
                }
            )

            continue

        
        # Cross-check ontology where available.
        
        pubmedbert_ontology = clean(
            pubmedbert_item.get(
                "ontology_id",
                "",
            )
        )

        if (
            ontology_id
            and pubmedbert_ontology
            and ontology_id.casefold()
            != pubmedbert_ontology.casefold()
        ):
            raise ValueError(
                "Ontology mismatch for "
                f"{candidate}: "
                f"{ontology_id} vs "
                f"{pubmedbert_ontology}"
            )

        novel = bool(
            item.get(
                "is_novel_to_agent2",
                False,
            )
        )

        candidate_mentions = int(
            grounding_item.get(
                "candidate_mention_papers",
                0,
            )
            or 0
        )

        marker_papers = int(
            grounding_item.get(
                "candidate_marker_papers",
                0,
            )
            or 0
        )

        target_candidate_papers = int(
            grounding_item.get(
                "target_context_candidate_papers",
                0,
            )
            or 0
        )

        target_marker_papers = int(
            grounding_item.get(
                "target_context_candidate_marker_papers",
                0,
            )
            or 0
        )

        (
            evidence_state,
            reporting_state,
        ) = classify_evidence(
            novel=novel,
            candidate_mentions=(
                candidate_mentions
            ),
            marker_papers=(
                marker_papers
            ),
            target_candidate_papers=(
                target_candidate_papers
            ),
            target_marker_papers=(
                target_marker_papers
            ),
        )

        record = {
           
            # Identity
            
            "cluster_id":
                cluster_id,

            "discovery_rank":
                discovery_rank,

            "candidate":
                candidate,

            "ontology_id":
                ontology_id,

            "is_upstream_candidate":
                bool(
                    item.get(
                        "is_upstream_candidate",
                        False,
                    )
                ),

            "is_novel_to_agent2":
                novel,

            
            # Marker-rescue discovery
            
            "discovery_independent_papers":
                int(
                    item.get(
                        "independent_paper_count",
                        0,
                    )
                    or 0
                ),

            "discovery_canonical_label_papers":
                int(
                    item.get(
                        "canonical_label_paper_count",
                        0,
                    )
                    or 0
                ),

            "discovery_title_mention_papers":
                int(
                    item.get(
                        "title_mention_paper_count",
                        0,
                    )
                    or 0
                ),

            "discovery_rescue_query_count":
                int(
                    item.get(
                        "rescue_query_count",
                        0,
                    )
                    or 0
                ),

            "discovery_marker_linked_papers":
                int(
                    item.get(
                        "marker_linked_paper_count",
                        0,
                    )
                    or 0
                ),

            "discovery_linked_markers":
                clean(
                    item.get(
                        "linked_markers",
                        "",
                    )
                ),

          
            # Rescue provenance
            
            "provenance_total_papers":
                int(
                    provenance_item.get(
                        "total_papers",
                        0,
                    )
                    or 0
                ),

            "provenance_target_context_papers":
                int(
                    provenance_item.get(
                        "target_context_papers",
                        0,
                    )
                    or 0
                ),

            "provenance_title_mention_papers":
                int(
                    provenance_item.get(
                        "title_mention_papers",
                        0,
                    )
                    or 0
                ),

            "provenance_pairwise_marker_papers":
                int(
                    provenance_item.get(
                        "pairwise_marker_papers",
                        0,
                    )
                    or 0
                ),

            "provenance_distinct_pairwise_queries":
                int(
                    provenance_item.get(
                        "distinct_pairwise_queries",
                        0,
                    )
                    or 0
                ),

            "provenance_pairwise_query_ids":
                "; ".join(
                    provenance_item.get(
                        "pairwise_query_ids",
                        [],
                    )
                ),

            "provenance_same_sentence_marker_papers":
                int(
                    provenance_item.get(
                        "same_sentence_marker_papers",
                        0,
                    )
                    or 0
                ),

            "provenance_same_sentence_markers":
                clean(
                    provenance_item.get(
                        "same_sentence_markers",
                        "",
                    )
                ),

            # Confirmation grounding
            
            "confirmation_query_id":
                clean(
                    grounding_item.get(
                        "query_id",
                        "",
                    )
                ),

            "confirmation_returned_papers":
                int(
                    grounding_item.get(
                        "returned_papers",
                        0,
                    )
                    or 0
                ),

            "confirmation_candidate_mention_papers":
                candidate_mentions,

            "confirmation_candidate_marker_papers":
                marker_papers,

            "same_sentence_link_papers":
                int(
                    grounding_item.get(
                        "same_sentence_link_papers",
                        0,
                    )
                    or 0
                ),

            "same_sentence_linked_genes":
                clean(
                    grounding_item.get(
                        "same_sentence_linked_genes",
                        "",
                    )
                ),

            "adjacent_sentence_link_papers":
                int(
                    grounding_item.get(
                        "adjacent_sentence_link_papers",
                        0,
                    )
                    or 0
                ),

            "adjacent_sentence_linked_genes":
                clean(
                    grounding_item.get(
                        "adjacent_sentence_linked_genes",
                        "",
                    )
                ),

            "within_three_sentence_link_papers":
                int(
                    grounding_item.get(
                        "within_three_sentence_link_papers",
                        0,
                    )
                    or 0
                ),

            "within_three_sentence_linked_genes":
                clean(
                    grounding_item.get(
                        "within_three_sentence_linked_genes",
                        "",
                    )
                ),

            "target_context_candidate_papers":
                target_candidate_papers,

            "target_context_candidate_marker_papers":
                target_marker_papers,

        
            
            "ontology_grounded":
                bool(
                    ontology_id
                ),

            "marker_grounded":
                marker_papers > 0,

            "direct_same_sentence_marker_grounding":
                bool_count(
                    grounding_item.get(
                        "same_sentence_link_papers",
                        0,
                    )
                ),

            "target_context_supported":
                target_candidate_papers
                > 0,

            "target_context_marker_grounded":
                target_marker_papers
                > 0,

            
            # PubMedBERT
            # Semantic relevance only.
           
            "pubmedbert_model":
                clean(
                    pubmedbert.get(
                        "model_id",
                        "",
                    )
                ),

            "pubmedbert_retrieved_papers":
                int(
                    pubmedbert_item.get(
                        "retrieved_paper_count",
                        0,
                    )
                    or 0
                ),

            "pubmedbert_eligible_passages":
                int(
                    pubmedbert_item.get(
                        "eligible_passage_count",
                        0,
                    )
                    or 0
                ),

            "pubmedbert_eligible_unique_papers":
                int(
                    pubmedbert_item.get(
                        "eligible_unique_paper_count",
                        0,
                    )
                    or 0
                ),

            "pubmedbert_selected_top_k":
                int(
                    pubmedbert_item.get(
                        "selected_top_k",
                        0,
                    )
                    or 0
                ),

            "pubmedbert_mean_similarity":
                pubmedbert_item.get(
                    "mean_topk_similarity"
                ),

            "pubmedbert_max_similarity":
                pubmedbert_item.get(
                    "max_topk_similarity"
                ),

            "pubmedbert_median_similarity":
                pubmedbert_item.get(
                    "median_topk_similarity"
                ),

            "semantic_evidence_available":
                not bool(
                    pubmedbert_item.get(
                        "insufficient_candidate_evidence",
                        True,
                    )
                ),

          
            # Descriptive state only.
           
            "evidence_state":
                evidence_state,

            "reporting_state":
                reporting_state,

            "automatic_acceptance":
                "NOT_CALIBRATED",

            "final_annotation":
                None,

            
            # Full nested provenance retained in JSON.
          
            "discovery_paper_provenance":
                item.get(
                    "paper_provenance",
                    [],
                ),

            "confirmation_paper_evidence":
                grounding_item.get(
                    "paper_evidence",
                    [],
                ),

            "pubmedbert_top_passages":
                pubmedbert_item.get(
                    "top_passages",
                    [],
                ),
        }

        consolidated.append(
            record
        )

    if missing_records:
        raise ValueError(
            "Missing candidate records across "
            "artifacts:\n"
            + json.dumps(
                missing_records,
                indent=2,
            )
        )

    if len(consolidated) != len(
        discovery_rows
    ):
        raise ValueError(
            "Consolidated candidate count "
            "does not equal discovery count."
        )

   
    # Output
   
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
            "rescue_evidence_packet.json"
        )
    )

    csv_path = (
        output_directory
        / (
            f"cluster_{cluster_id}_"
            "rescue_evidence_packet.csv"
        )
    )

    output = {
        "schema_name":
            "Agent3RescueEvidencePacket",

        "schema_version":
            "1.0",

        "cluster_id":
            cluster_id,

        "expert_label_used":
            False,

        "reference_label_used":
            False,

        "candidate_count":
            len(consolidated),

        "candidate_ordering":
            (
                "Original ontology rescue "
                "discovery order preserved. "
                "No new combined ranking "
                "or weighted score applied."
            ),

        "state_rule":
            (
                "Evidence states are descriptive "
                "and depend only on whether "
                "candidate, marker, and target-context "
                "evidence is present. They are not "
                "calibrated annotation decisions."
            ),

        "automatic_acceptance":
            "NOT_CALIBRATED",

        "important_note":
            (
                "PubMedBERT cosine similarities "
                "represent semantic retrieval relevance, "
                "not probability or annotation confidence."
            ),

        "source_artifacts": {
            "candidate_discovery":
                str(
                    args.discovery_json
                ),

            "rescue_provenance":
                str(
                    args.provenance_json
                ),

            "confirmation_grounding":
                str(
                    args.grounding_json
                ),

            "pubmedbert":
                str(
                    args.pubmedbert_json
                ),
        },

        "candidates":
            consolidated,
    }

    json_path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    
    # CSV contains scalar summary fields only.
    
    nested_keys = {
        "discovery_paper_provenance",
        "confirmation_paper_evidence",
        "pubmedbert_top_passages",
    }

    csv_rows = [
        {
            key: value
            for key, value
            in record.items()
            if key not in nested_keys
        }
        for record in consolidated
    ]

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

   
    # Terminal audit
    
    print(
        "Agent 3 rescue evidence packet "
        "completed successfully."
    )

    print(
        "Cluster:",
        cluster_id,
    )

    print(
        "Candidates:",
        len(consolidated),
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
        "JSON:",
        json_path,
    )

    print(
        "CSV:",
        csv_path,
    )


if __name__ == "__main__":
    main()
