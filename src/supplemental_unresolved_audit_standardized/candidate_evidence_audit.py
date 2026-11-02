
from __future__ import annotations

import ast
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Mapping


SCHEMA_NAME = "StandardizedSupplementalCandidateEvidenceAudit"
SCHEMA_VERSION = "1.1"

EXPECTED_PACKET_STATUS = (
    "STANDARDIZED_EVIDENCE_PACKET_BUILT_"
    "NOT_YET_ADJUDICATED"
)

# BASIC UTILITIES


def clean(value: Any) -> str:
    """
    Convert a value into a clean single-space string.
    """
    return " ".join(str(value or "").split())


def gene_key(value: Any) -> str:
    """
    Case-normalized exact gene/probe identifier.

    IMPORTANT:
    Composite probes remain intact.

    Examples:
        HLA-DQB1/2 -> HLA-DQB1/2
        FCGR3A/B   -> FCGR3A/B
        TPSAB1/B2  -> TPSAB1/B2

    We do NOT split '/'.
    """
    return clean(value).upper()


def parse_bool(value: Any) -> bool:
    """
    Convert common boolean representations to bool.
    """

    if isinstance(value, bool):
        return value

    return clean(value).lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def parse_float(
    value: Any,
) -> float | None:
    """
    Safely convert a value to float.
    """

    text = clean(value)

    if not text:
        return None

    try:
        return float(text)

    except (TypeError, ValueError):
        return None


# SUPPORTING-GENE PARSER

def parse_gene_list(
    value: Any,
) -> list[str]:
    """
    Parse an upstream supporting-gene field without changing
    biological identifiers.

    Accepted formats include:

        ["CCL5", "CD2", "ETS1"]

        CCL5 | CD2 | ETS1

        CCL5; CD2; ETS1

        CCL5, CD2, ETS1

    IMPORTANT:
    '/' is NOT treated as a delimiter because composite probes
    such as:

        HLA-DQB1/2
        FCGR3A/B
        TPSAB1/B2
        CXCL1/2/3

    must remain unchanged.

    No genes are:
        - invented
        - expanded
        - remapped
        - manually added
    """

    if value is None:
        return []

    # Already a Python list/tuple.
    if isinstance(
        value,
        (list, tuple),
    ):

        raw_items = list(
            value
        )

    else:

        text = str(
            value
        ).strip()

        if not text:
            return []

        raw_items: list[Any] | None = None

        
        # Attempt valid JSON list first.
        

        try:

            parsed = json.loads(
                text
            )

            if isinstance(
                parsed,
                list,
            ):

                raw_items = parsed

        except (
            json.JSONDecodeError,
            TypeError,
        ):
            pass

       
        # Some CSVs can contain Python-style list strings:
        # ['CCL5', 'CD2', 'ETS1']
       

        if raw_items is None:

            try:

                parsed_literal = ast.literal_eval(
                    text
                )

                if isinstance(
                    parsed_literal,
                    (list, tuple),
                ):

                    raw_items = list(
                        parsed_literal
                    )

            except (
                ValueError,
                SyntaxError,
            ):
                pass

        

        if raw_items is None:

            normalized = (
                text
                .replace("|", ";")
                .replace(",", ";")
            )

            raw_items = normalized.split(
                ";"
            )

    # Clean + de-duplicate while preserving original order.
   

    result: list[str] = []

    seen: set[str] = set()

    for item in raw_items:

        gene = clean(
            item
        )

        if not gene:
            continue

        key = gene_key(
            gene
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        result.append(
            gene
        )

    return result


# GENERIC OPTIONAL EXPRESSION-EVIDENCE INTERPRETATION



def expression_found_from_summary(
    summary: Mapping[str, Any],
) -> bool:
    """
    Determine whether an optional expression-reference record
    reports positive expression.

    This deliberately avoids requiring a breast-specific field.

    Preference order:

        1. generic `expression_found`
        2. positive_record_count > 0
        3. source_status == expression_found
        4. source_status ending in `_expression_found`

    Historical HPA records may contain fields such as:

        breast_expression_found

    but the standardized supplemental logic should not depend on
    a particular tissue name.

    IMPORTANT:
    Expression evidence is complementary evidence only.
    It does NOT generate candidate votes here.
    """

    
    # Preferred generic standardized field.
   

    if "expression_found" in summary:

        value = summary.get(
            "expression_found"
        )

        if isinstance(
            value,
            bool,
        ):
            return value

        text = clean(
            value
        ).lower()

        if text:

            return text in {
                "true",
                "1",
                "yes",
                "y",
            }

    # -------------------------------------------------------------
    # Generic positive-record evidence.
    # -------------------------------------------------------------

    positive_count = parse_float(
        summary.get(
            "positive_record_count"
        )
    )

    if positive_count is not None:

        return (
            positive_count > 0
        )

   
    # Generic status field.
    # Examples:
    # breast_expression_found
    # lung_expression_found
    # expression_found
   

    source_status = clean(
        summary.get(
            "source_status"
        )
    ).lower()

    if (
        source_status == "expression_found"
        or source_status.endswith(
            "_expression_found"
        )
    ):

        return True

    return False


# LOAD STANDARDIZED PACKET



def load_packet(
    path: str | Path,
) -> dict[str, Any]:
    """
    Load the standardized supplemental evidence packet.
    """

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            "Evidence packet does not exist: "
            f"{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        packet = json.load(
            handle
        )

    if not isinstance(
        packet,
        dict,
    ):

        raise ValueError(
            "Standardized evidence packet must "
            "be a JSON object."
        )

    return packet


# SAFETY VALIDATION



def validate_packet(
    packet: Mapping[str, Any],
) -> None:
    """
    Confirm that this packet is eligible for a supplemental
    candidate-evidence audit.

    This audit is allowed only after frozen core Agent 3 returns
    UNRESOLVED.
    """

    status = clean(
        packet.get(
            "status"
        )
    )

    if status != EXPECTED_PACKET_STATUS:

        raise RuntimeError(
            "Unexpected evidence packet status. "
            f"Expected={EXPECTED_PACKET_STATUS}; "
            f"received={status}"
        )

    
    # Core Agent 3 must be UNRESOLVED.
    

    core_agent3 = packet.get(
        "core_agent3"
    )

    if not isinstance(
        core_agent3,
        Mapping,
    ):

        raise ValueError(
            "Packet is missing core_agent3."
        )

    decision = clean(
        core_agent3.get(
            "decision_state"
        )
    ).upper()

    if decision != "UNRESOLVED":

        raise RuntimeError(
            "Candidate evidence audit is permitted only "
            "for a frozen Agent 3 UNRESOLVED case. "
            f"Received: {decision}"
        )

    
    # Safety metadata.
    

    safety = packet.get(
        "safety"
    )

    if not isinstance(
        safety,
        Mapping,
    ):

        raise ValueError(
            "Packet is missing safety metadata."
        )

    required_false = (
        "core_agent3_output_changed",
        "reference_label_used",
        "expert_label_used",
        "manual_cell_type_label_injected",
        "manual_gene_selection_used",
    )

    for field in required_false:

        if safety.get(
            field
        ) is not False:

            raise RuntimeError(
                "Safety condition failed: "
                f"{field}"
            )

    # Evidence packet itself must already have passed hash validation.


    validation = packet.get(
        "validation"
    )

    if not isinstance(
        validation,
        Mapping,
    ):

        raise ValueError(
            "Packet is missing validation metadata."
        )

    if validation.get(
        "frozen_hashes_verified"
    ) is not True:

        raise RuntimeError(
            "Frozen evidence hashes were not verified."
        )

    if validation.get(
        "biological_adjudication_performed"
    ) is not False:

        raise RuntimeError(
            "Input packet has already been adjudicated."
        )



# MARKER INDEX



def build_marker_rank_index(
    packet: Mapping[str, Any],
) -> tuple[
    dict[str, int],
    dict[str, str],
]:
    """
    Build:

        gene_key -> rank
        gene_key -> original display name

    from the exact frozen cluster marker set.
    """

    marker_set = packet.get(
        "marker_set"
    )

    if not isinstance(
        marker_set,
        Mapping,
    ):

        raise ValueError(
            "Packet is missing marker_set."
        )

    markers = marker_set.get(
        "markers"
    )

    if not isinstance(
        markers,
        list,
    ):

        raise ValueError(
            "marker_set.markers must be an array."
        )

    rank_index: dict[
        str,
        int
    ] = {}

    display_index: dict[
        str,
        str
    ] = {}

    for row in markers:

        if not isinstance(
            row,
            Mapping,
        ):
            continue

        gene = clean(
            row.get(
                "gene"
            )
        )

        rank_text = clean(
            row.get(
                "rank_within_cluster"
            )
        )

        if not gene:
            continue

        try:

            rank = int(
                float(
                    rank_text
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            raise ValueError(
                "Invalid marker rank for "
                f"{gene}: {rank_text}"
            )

        key = gene_key(
            gene
        )

        if key in rank_index:

            raise ValueError(
                "Duplicate marker identifier in "
                "standardized packet: "
                f"{gene}"
            )

        rank_index[
            key
        ] = rank

        display_index[
            key
        ] = gene

    if not rank_index:

        raise ValueError(
            "No ranked markers found in packet."
        )

    return (
        rank_index,
        display_index,
    )



# GENERIC GENE-OBJECT INDEX



def build_gene_object_index(
    objects: Any,
) -> dict[
    str,
    dict[str, Any],
]:
    """
    Build exact case-normalized gene -> evidence-object index.
    """

    if not isinstance(
        objects,
        list,
    ):

        return {}

    result: dict[
        str,
        dict[str, Any]
    ] = {}

    for obj in objects:

        if not isinstance(
            obj,
            Mapping,
        ):
            continue

        gene = clean(
            obj.get(
                "gene"
            )
        )

        if not gene:
            continue

        key = gene_key(
            gene
        )

        if key not in result:

            result[
                key
            ] = dict(
                obj
            )

    return result


def first_available_list(
    mapping: Mapping[str, Any],
    keys: tuple[str, ...],
) -> list[Any]:
    """
    Return the first list-valued field found among several compatible
    schema field names.

    This is schema compatibility logic, not biological logic.
    """

    for key in keys:

        value = mapping.get(
            key
        )

        if isinstance(
            value,
            list,
        ):

            return value

    return []



# CANDIDATE-LEVEL EVIDENCE AUDIT


def audit_candidate(
    *,
    candidate: Mapping[str, Any],
    marker_rank_index: Mapping[str, int],
    marker_display_index: Mapping[str, str],
    agent1_index: Mapping[
        str,
        Mapping[str, Any],
    ],
    hpa_index: Mapping[
        str,
        Mapping[str, Any],
    ],
) -> dict[str, Any]:
    """
    Audit one existing frozen Agent 2 candidate.

    IMPORTANT:
    This function does NOT:
        - create a new candidate
        - remove an Agent 2 candidate
        - assign a final cell type
        - use expert/reference labels
        - make RESOLVED/AMBIGUOUS/UNRESOLVED decisions
    """

    cell_type = clean(
        candidate.get(
            "cell_type"
        )
    )

    if not cell_type:

        raise ValueError(
            "Agent 2 candidate is missing cell_type."
        )

   
    # Parse upstream supporting genes.
   

    supporting_genes = parse_gene_list(
        candidate.get(
            "supporting_genes"
        )
    )

    both_source_genes = parse_gene_list(
        candidate.get(
            "both_source_supporting_genes"
        )
    )


    # Match candidate-supporting genes against exact frozen marker set.


    supporting_marker_hits: list[
        dict[str, Any]
    ] = []

    for gene in supporting_genes:

        key = gene_key(
            gene
        )

        rank = marker_rank_index.get(
            key
        )

        if rank is None:
            continue

        supporting_marker_hits.append(
            {
                "gene": (
                    marker_display_index.get(
                        key,
                        gene,
                    )
                ),
                "rank": rank,
            }
        )

    supporting_marker_hits.sort(
        key=lambda item: item[
            "rank"
        ]
    )

    marker_ranks = [
        int(
            item[
                "rank"
            ]
        )
        for item in supporting_marker_hits
    ]

 
    # Unique supporting gene keys.


    supporting_keys = {
        gene_key(
            gene
        )
        for gene in supporting_genes
    }

  
    # Agent 1 coverage.
    # This asks only whether those supporting markers are represented in the frozen Agent 1 evidence.
    

    agent1_matched_genes: list[
        str
    ] = []

    for key in supporting_keys:

        if (
            key in agent1_index
            and key in marker_rank_index
        ):

            agent1_matched_genes.append(
                marker_display_index.get(
                    key,
                    key,
                )
            )

    agent1_matched_genes.sort(
        key=lambda gene: marker_rank_index[
            gene_key(
                gene
            )
        ]
    )

   
    # Optional HPA/expression-reference coverage.
   

    hpa_matched_genes: list[
        str
    ] = []

    hpa_expression_found_genes: list[
        str
    ] = []

    for key in supporting_keys:

        if (
            key not in hpa_index
            or key not in marker_rank_index
        ):
            continue

        gene = marker_display_index.get(
            key,
            key,
        )

        hpa_matched_genes.append(
            gene
        )

        summary = hpa_index[
            key
        ]

        if expression_found_from_summary(
            summary
        ):

            hpa_expression_found_genes.append(
                gene
            )

    hpa_matched_genes.sort(
        key=lambda gene: marker_rank_index[
            gene_key(
                gene
            )
        ]
    )

    hpa_expression_found_genes.sort(
        key=lambda gene: marker_rank_index[
            gene_key(
                gene
            )
        ]
    )

    
    # Recomputed rank summaries.
   

    best_marker_rank = (
        min(
            marker_ranks
        )
        if marker_ranks
        else None
    )

    median_marker_rank = (
        statistics.median(
            marker_ranks
        )
        if marker_ranks
        else None
    )

    mean_marker_rank = (
        statistics.mean(
            marker_ranks
        )
        if marker_ranks
        else None
    )


    # Build auditable candidate record.


    audit = {
        "candidate_rank": parse_float(
            candidate.get(
                "candidate_rank"
            )
        ),

        "cell_type": cell_type,

        "selected_for_downstream": parse_bool(
            candidate.get(
                "selected_for_downstream"
            )
        ),

        "selected_by_sudden_drop": parse_bool(
            candidate.get(
                "selected_by_sudden_drop"
            )
        ),

        "selected_by_min_support_threshold": parse_bool(
            candidate.get(
                "selected_by_min_support_threshold"
            )
        ),

        "reported_unique_supporting_gene_count": (
            parse_float(
                candidate.get(
                    "unique_supporting_gene_count"
                )
            )
        ),

        "reported_both_source_supporting_gene_count": (
            parse_float(
                candidate.get(
                    "both_source_supporting_gene_count"
                )
            )
        ),

        "reported_specificity_weighted_support": (
            parse_float(
                candidate.get(
                    "specificity_weighted_support"
                )
            )
        ),

        "reported_mean_supporting_gene_rank": (
            parse_float(
                candidate.get(
                    "mean_supporting_gene_rank"
                )
            )
        ),

        "reported_mean_supporting_log_fold_change": (
            parse_float(
                candidate.get(
                    "mean_supporting_log_fold_change"
                )
            )
        ),

        "reported_evidence_count": (
            parse_float(
                candidate.get(
                    "evidence_count"
                )
            )
        ),

        "evidence_sources": clean(
            candidate.get(
                "evidence_sources"
            )
        ),

        "tissue_contexts": clean(
            candidate.get(
                "tissue_contexts"
            )
        ),

        "disease_contexts": clean(
            candidate.get(
                "disease_contexts"
            )
        ),

        "cell_ontology_ids": clean(
            candidate.get(
                "cell_ontology_ids"
            )
        ),

       
        # Supporting-gene evidence.

        "supporting_genes": (
            supporting_genes
        ),

        "supporting_gene_count_parsed": len(
            supporting_genes
        ),

        "both_source_supporting_genes": (
            both_source_genes
        ),

        "both_source_gene_count_parsed": len(
            both_source_genes
        ),

        # Marker-set evidence.
    

        "supporting_marker_hits": (
            supporting_marker_hits
        ),

        "supporting_marker_hit_count": len(
            supporting_marker_hits
        ),

        "best_supporting_marker_rank": (
            best_marker_rank
        ),

        "median_supporting_marker_rank": (
            median_marker_rank
        ),

        "mean_supporting_marker_rank_recomputed": (
            mean_marker_rank
        ),

    
        # Agent 1 coverage.
      

        "agent1_matched_supporting_genes": (
            agent1_matched_genes
        ),

        "agent1_matched_supporting_gene_count": len(
            agent1_matched_genes
        ),

        
        # Optional HPA/expression reference.
        # Explicitly NOT a candidate vote.
        

        "hpa_coverage": {
            "candidate_vote_generated": False,

            "matched_supporting_genes": (
                hpa_matched_genes
            ),

            "matched_supporting_gene_count": len(
                hpa_matched_genes
            ),

            "expression_found_supporting_genes": (
                hpa_expression_found_genes
            ),

            "expression_found_supporting_gene_count": len(
                hpa_expression_found_genes
            ),
        },
    }

    return audit



# FULL STANDARDIZED CANDIDATE AUDIT


def build_candidate_evidence_audit(
    *,
    evidence_packet_json: str | Path,
) -> dict[str, Any]:
    """
    Build a dataset-agnostic candidate-level evidence audit.

    No biological adjudication is performed here.
    """

    packet_path = Path(
        evidence_packet_json
    )

    packet = load_packet(
        packet_path
    )

    validate_packet(
        packet
    )

    (
        marker_rank_index,
        marker_display_index,
    ) = build_marker_rank_index(
        packet
    )

    # Retrieve packet sections.


    agent1 = packet.get(
        "agent1",
        {},
    )

    agent2 = packet.get(
        "agent2",
        {},
    )

    hpa = packet.get(
        "hpa",
        {},
    )

    if not isinstance(
        agent1,
        Mapping,
    ):
        agent1 = {}

    if not isinstance(
        agent2,
        Mapping,
    ):
        agent2 = {}

    if not isinstance(
        hpa,
        Mapping,
    ):
        hpa = {}

    # Agent 1 schema compatibility.

    # The standardized packet may call the retained list:
    # marker_evidence
    # matching_evidence
    # gene_evidence
  

    agent1_objects = first_available_list(
        agent1,
        (
            "marker_evidence",
            "matching_evidence",
            "gene_evidence",
        ),
    )

    agent1_index = build_gene_object_index(
        agent1_objects
    )

    
    # Optional HPA/expression-reference summaries.
  

    hpa_objects = first_available_list(
        hpa,
        (
            "gene_summaries",
            "marker_evidence",
            "matching_evidence",
        ),
    )

    hpa_index = build_gene_object_index(
        hpa_objects
    )

   
    # Existing frozen Agent 2 candidates.
 

    candidates = agent2.get(
        "ranked_candidates"
    )

    if not isinstance(
        candidates,
        list,
    ):

        raise ValueError(
            "Packet does not contain Agent 2 "
            "ranked_candidates."
        )

    if not candidates:

        raise ValueError(
            "Agent 2 ranked candidate list is empty."
        )

    # Audit all existing candidates.

    candidate_audits = [
        audit_candidate(
            candidate=candidate,
            marker_rank_index=marker_rank_index,
            marker_display_index=marker_display_index,
            agent1_index=agent1_index,
            hpa_index=hpa_index,
        )
        for candidate in candidates
        if isinstance(
            candidate,
            Mapping,
        )
    ]

    candidate_audits.sort(
        key=lambda row: (
            row.get(
                "candidate_rank"
            )
            if row.get(
                "candidate_rank"
            ) is not None
            else float(
                "inf"
            )
        )
    )

    selected_candidates = [
        row
        for row in candidate_audits
        if row.get(
            "selected_for_downstream"
        ) is True
    ]

    # Final audit object.
   

    audit = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,

        "dataset_id": packet.get(
            "dataset_id"
        ),

        "cluster_id": packet.get(
            "cluster_id"
        ),

        "species": packet.get(
            "species"
        ),

        "tissue": packet.get(
            "tissue"
        ),

        "platform": packet.get(
            "platform"
        ),

        "disease_context": packet.get(
            "disease_context"
        ),

        "core_agent3_decision": (
            "UNRESOLVED"
        ),

       
        # Policy declaration.

        "audit_policy": {
            "candidate_source": (
                "Existing frozen Agent 2 ranked "
                "candidate table only"
            ),

            "candidate_selection_changed": False,

            "manual_candidate_added": False,

            "manual_gene_added": False,

            "reference_label_used": False,

            "expert_label_used": False,

            "hpa_candidate_vote_generated": False,

            "hpa_role": (
                "Complementary expression/context "
                "coverage only"
            ),

            "gene_matching_policy": (
                "Case-normalized exact identifier matching; "
                "composite probes preserved"
            ),

            "accepted_supporting_gene_delimiters": [
                "|",
                ";",
                ",",
            ],

            "composite_probe_slash_preserved": True,

            "cell_type_specific_hardcoding_used": False,

            "cluster_specific_hardcoding_used": False,

            "dataset_specific_hardcoding_used": False,

            "decision_threshold_applied": False,
        },

       
        # Summary.


        "summary": {
            "marker_count": len(
                marker_rank_index
            ),

            "candidate_count": len(
                candidate_audits
            ),

            "selected_downstream_candidate_count": len(
                selected_candidates
            ),

            "agent1_indexed_gene_count": len(
                agent1_index
            ),

            "optional_expression_indexed_gene_count": len(
                hpa_index
            ),
        },

        # Existing Agent 2 selected candidates.

        "selected_downstream_candidates": [
            row.get(
                "cell_type"
            )
            for row in selected_candidates
        ],

        # Full candidate evidence.

        "candidate_evidence": (
            candidate_audits
        ),

        
        # NO biological adjudication here.

        "decision": {
            "supplemental_decision": None,

            "supplemental_possible_labels": [],

            "decision_reason": (
                "Candidate-level evidence audit completed. "
                "No supplemental adjudication policy has "
                "yet been applied."
            ),

            "core_agent3_output_changed": False,
        },

        # Provenance.

        "provenance": {
            "standardized_evidence_packet": str(
                packet_path
            ),
        },

        "status": (
            "CANDIDATE_EVIDENCE_AUDIT_BUILT_"
            "NOT_YET_ADJUDICATED"
        ),
    }

    return audit


# JSON WRITER


def write_candidate_evidence_audit(
    audit: Mapping[str, Any],
    output_json: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """
    Write candidate audit JSON.

    Existing files are protected by default.
    """

    output_path = Path(
        output_json
    )

    if (
        output_path.exists()
        and not overwrite
    ):

        raise FileExistsError(
            "Refusing to overwrite existing candidate "
            f"evidence audit: {output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            dict(
                audit
            ),
            handle,
            indent=2,
            ensure_ascii=False,
        )

        handle.write(
            "\n"
        )

    return output_path


# CSV WRITER


def write_candidate_evidence_csv(
    audit: Mapping[str, Any],
    output_csv: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """
    Write reviewer-friendly candidate evidence summary CSV.

    This remains an evidence table only.
    """

    output_path = Path(
        output_csv
    )

    if (
        output_path.exists()
        and not overwrite
    ):

        raise FileExistsError(
            "Refusing to overwrite existing candidate "
            f"evidence CSV: {output_path}"
        )

    rows = audit.get(
        "candidate_evidence"
    )

    if not isinstance(
        rows,
        list,
    ):

        raise ValueError(
            "Audit candidate_evidence must "
            "be an array."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    flat_rows: list[
        dict[str, Any]
    ] = []

    for row in rows:

        if not isinstance(
            row,
            Mapping,
        ):
            continue

        hpa_coverage = row.get(
            "hpa_coverage",
            {},
        )

        if not isinstance(
            hpa_coverage,
            Mapping,
        ):
            hpa_coverage = {}

        supporting_hits = row.get(
            "supporting_marker_hits",
            [],
        )

        if not isinstance(
            supporting_hits,
            list,
        ):
            supporting_hits = []

        supporting_hit_genes = [
            clean(
                item.get(
                    "gene"
                )
            )
            for item in supporting_hits
            if isinstance(
                item,
                Mapping,
            )
            and clean(
                item.get(
                    "gene"
                )
            )
        ]

        flat_rows.append(
            {
                "candidate_rank": row.get(
                    "candidate_rank"
                ),

                "cell_type": row.get(
                    "cell_type"
                ),

                "selected_for_downstream": row.get(
                    "selected_for_downstream"
                ),

                "selected_by_sudden_drop": row.get(
                    "selected_by_sudden_drop"
                ),

                "selected_by_min_support_threshold": row.get(
                    "selected_by_min_support_threshold"
                ),

                "unique_supporting_gene_count": row.get(
                    "reported_unique_supporting_gene_count"
                ),

                "both_source_supporting_gene_count": row.get(
                    "reported_both_source_supporting_gene_count"
                ),

                "specificity_weighted_support": row.get(
                    "reported_specificity_weighted_support"
                ),

                "reported_mean_supporting_gene_rank": row.get(
                    "reported_mean_supporting_gene_rank"
                ),

                "reported_mean_supporting_log_fold_change": row.get(
                    "reported_mean_supporting_log_fold_change"
                ),

                "reported_evidence_count": row.get(
                    "reported_evidence_count"
                ),

                "supporting_gene_count_parsed": row.get(
                    "supporting_gene_count_parsed"
                ),

                "supporting_marker_hit_count": row.get(
                    "supporting_marker_hit_count"
                ),

                "best_supporting_marker_rank": row.get(
                    "best_supporting_marker_rank"
                ),

                "median_supporting_marker_rank": row.get(
                    "median_supporting_marker_rank"
                ),

                "mean_supporting_marker_rank_recomputed": row.get(
                    "mean_supporting_marker_rank_recomputed"
                ),

                "agent1_matched_supporting_gene_count": row.get(
                    "agent1_matched_supporting_gene_count"
                ),

                "hpa_matched_supporting_gene_count": (
                    hpa_coverage.get(
                        "matched_supporting_gene_count"
                    )
                ),

                "hpa_expression_found_supporting_gene_count": (
                    hpa_coverage.get(
                        "expression_found_supporting_gene_count"
                    )
                ),

                "evidence_sources": row.get(
                    "evidence_sources"
                ),

                "tissue_contexts": row.get(
                    "tissue_contexts"
                ),

                "disease_contexts": row.get(
                    "disease_contexts"
                ),

                "cell_ontology_ids": row.get(
                    "cell_ontology_ids"
                ),

                "supporting_genes": " | ".join(
                    row.get(
                        "supporting_genes",
                        [],
                    )
                ),

                "supporting_marker_hit_genes": " | ".join(
                    supporting_hit_genes
                ),

                "agent1_matched_supporting_genes": " | ".join(
                    row.get(
                        "agent1_matched_supporting_genes",
                        [],
                    )
                ),

                "hpa_matched_supporting_genes": " | ".join(
                    hpa_coverage.get(
                        "matched_supporting_genes",
                        [],
                    )
                ),

                "hpa_expression_found_supporting_genes": " | ".join(
                    hpa_coverage.get(
                        "expression_found_supporting_genes",
                        [],
                    )
                ),

                "both_source_supporting_genes": " | ".join(
                    row.get(
                        "both_source_supporting_genes",
                        [],
                    )
                ),
            }
        )

    if not flat_rows:

        raise ValueError(
            "No candidate evidence rows to write."
        )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                flat_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            flat_rows
        )

    return output_path