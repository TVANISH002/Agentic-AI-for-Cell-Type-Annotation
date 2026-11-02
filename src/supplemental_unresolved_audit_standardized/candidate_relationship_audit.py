
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from literature_retrieval_agent.agent3_decision_summary import (
    CellOntologyIndex,
)


SCHEMA_NAME = (
    "StandardizedSupplementalCandidateRelationshipAudit"
)

SCHEMA_VERSION = "1.1"

EXPECTED_CANDIDATE_AUDIT_STATUS = (
    "CANDIDATE_EVIDENCE_AUDIT_BUILT_NOT_YET_ADJUDICATED"
)


# BASIC UTILITIES

def clean(value: Any) -> str:
    return " ".join(
        str(value or "").split()
    )


def normalize_cl_id(
    value: Any,
) -> str | None:

    text = clean(value)

    if not text:
        return None

    if text.startswith("CL:"):
        return text.replace(
            ":",
            "_",
            1,
        )

    prefix = (
        "http://purl.obolibrary.org/obo/"
    )

    if text.startswith(prefix + "CL_"):
        return text.rsplit(
            "/",
            1,
        )[-1]

    if re.fullmatch(
        r"CL_\d+",
        text,
    ):
        return text

    return None


def display_cl_id(
    value: str | None,
) -> str | None:

    if not value:
        return None

    if value.startswith("CL_"):
        return value.replace(
            "_",
            ":",
            1,
        )

    return value


def load_json(
    path: str | Path,
) -> dict[str, Any]:

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"JSON file does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        payload = json.load(
            handle
        )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            f"Expected JSON object: {path}"
        )

    return payload


# ONTOLOGY-ID PARSER

def parse_ontology_ids(
    value: Any,
) -> list[str]:
    """
    Parse frozen Agent 2 ontology identifiers.

    No ontology identifier is invented here.
    """

    if value is None:
        return []

    raw_values: list[Any]

    if isinstance(
        value,
        list,
    ):
        raw_values = value

    else:
        text = clean(value)

        if not text:
            return []

        # JSON-list compatibility.
        if (
            text.startswith("[")
            and text.endswith("]")
        ):
            try:
                parsed = json.loads(
                    text
                )

                if isinstance(
                    parsed,
                    list,
                ):
                    raw_values = parsed
                else:
                    raw_values = [
                        text
                    ]

            except json.JSONDecodeError:
                raw_values = re.split(
                    r"[|;,]",
                    text,
                )

        else:
            raw_values = re.split(
                r"[|;,]",
                text,
            )

    normalized: list[str] = []

    seen: set[str] = set()

    for raw in raw_values:

        cl_id = normalize_cl_id(
            raw
        )

        if (
            cl_id
            and cl_id not in seen
        ):
            seen.add(
                cl_id
            )

            normalized.append(
                cl_id
            )

    return normalized


# EXACT SYNONYM INDEX


def build_exact_synonym_index(
    ontology_json: str | Path,
) -> dict[str, list[str]]:
    """
    Build an EXACT Cell Ontology synonym lookup.

    Only hasExactSynonym entries are indexed.

    Broad, narrow and related synonyms are intentionally not promoted to
    exact mappings because doing so could change biological meaning.
    """

    payload = load_json(
        ontology_json
    )

    graphs = payload.get(
        "graphs",
        [],
    )

    if not isinstance(
        graphs,
        list,
    ) or not graphs:

        raise ValueError(
            "Cell Ontology JSON contains no graph."
        )

    graph = graphs[0]

    if not isinstance(
        graph,
        Mapping,
    ):
        raise ValueError(
            "Invalid Cell Ontology graph."
        )

    nodes = graph.get(
        "nodes",
        [],
    )

    index: dict[
        str,
        list[str],
    ] = {}

    if not isinstance(
        nodes,
        list,
    ):
        return index

    for node in nodes:

        if not isinstance(
            node,
            Mapping,
        ):
            continue

        cl_id = normalize_cl_id(
            node.get("id")
        )

        if not cl_id:
            continue

        meta = node.get(
            "meta",
            {},
        )

        if not isinstance(
            meta,
            Mapping,
        ):
            continue

        synonyms = meta.get(
            "synonyms",
            [],
        )

        if not isinstance(
            synonyms,
            list,
        ):
            continue

        for synonym in synonyms:

            if not isinstance(
                synonym,
                Mapping,
            ):
                continue

            predicate = clean(
                synonym.get(
                    "pred"
                )
            )

            if not predicate.endswith(
                "hasExactSynonym"
            ):
                continue

            synonym_value = clean(
                synonym.get(
                    "val"
                )
            )

            if not synonym_value:
                continue

            key = synonym_value.casefold()

            index.setdefault(
                key,
                [],
            )

            if cl_id not in index[key]:
                index[key].append(
                    cl_id
                )

    return index


# INPUT VALIDATION


def validate_candidate_audit(
    audit: Mapping[str, Any],
) -> None:

    status = clean(
        audit.get(
            "status"
        )
    )

    if status != EXPECTED_CANDIDATE_AUDIT_STATUS:

        raise RuntimeError(
            "Unexpected candidate evidence audit status. "
            f"Received: {status}"
        )

    if clean(
        audit.get(
            "core_agent3_decision"
        )
    ).upper() != "UNRESOLVED":

        raise RuntimeError(
            "Relationship audit is allowed only for a "
            "frozen core Agent 3 UNRESOLVED case."
        )

    decision = audit.get(
        "decision",
        {},
    )

    if not isinstance(
        decision,
        Mapping,
    ):
        raise ValueError(
            "Candidate audit decision section missing."
        )

    if decision.get(
        "supplemental_decision"
    ) is not None:

        raise RuntimeError(
            "Candidate audit already contains a "
            "supplemental decision."
        )

    if decision.get(
        "core_agent3_output_changed"
    ) is not False:

        raise RuntimeError(
            "Core Agent 3 output must remain unchanged."
        )

    policy = audit.get(
        "audit_policy",
        {},
    )

    if not isinstance(
        policy,
        Mapping,
    ):
        raise ValueError(
            "Candidate audit policy section missing."
        )

    safety_fields = (
        "candidate_selection_changed",
        "manual_candidate_added",
        "manual_gene_added",
        "reference_label_used",
        "expert_label_used",
        "hpa_candidate_vote_generated",
        "cell_type_specific_hardcoding_used",
        "cluster_specific_hardcoding_used",
        "dataset_specific_hardcoding_used",
        "decision_threshold_applied",
    )

    for field in safety_fields:

        if (
            field in policy
            and policy.get(
                field
            ) is not False
        ):
            raise RuntimeError(
                "Candidate audit safety condition failed: "
                f"{field}"
            )


# SELECTED CANDIDATES


def extract_selected_candidates(
    audit: Mapping[str, Any],
) -> list[dict[str, Any]]:

    rows = audit.get(
        "candidate_evidence"
    )

    if not isinstance(
        rows,
        list,
    ):
        raise ValueError(
            "candidate_evidence must be an array."
        )

    selected: list[
        dict[str, Any]
    ] = []

    seen: set[str] = set()

    for row in rows:

        if not isinstance(
            row,
            Mapping,
        ):
            continue

        if row.get(
            "selected_for_downstream"
        ) is not True:
            continue

        label = clean(
            row.get(
                "cell_type"
            )
        )

        if not label:
            raise ValueError(
                "Selected candidate missing cell_type."
            )

        key = label.casefold()

        if key in seen:
            raise RuntimeError(
                "Duplicate selected candidate label: "
                f"{label}"
            )

        seen.add(
            key
        )

        selected.append(
            dict(row)
        )

    if not selected:
        raise RuntimeError(
            "No downstream-selected candidates found."
        )

    selected.sort(
        key=lambda row: (
            row.get(
                "candidate_rank"
            )
            if isinstance(
                row.get(
                    "candidate_rank"
                ),
                (int, float),
            )
            else float("inf"),
            clean(
                row.get(
                    "cell_type"
                )
            ),
        )
    )

    return selected


# ONTOLOGY RESOLUTION


def resolve_candidate_ontology(
    *,
    candidate: Mapping[str, Any],
    ontology: CellOntologyIndex,
    exact_synonym_index: Mapping[
        str,
        list[str],
    ],
) -> dict[str, Any]:

    label = clean(
        candidate.get(
            "cell_type"
        )
    )

    supplied_ids = parse_ontology_ids(
        candidate.get(
            "cell_ontology_ids"
        )
    )

    # 1. Frozen Agent 2 ontology ID.

    valid_supplied_ids = [
        cl_id
        for cl_id in supplied_ids
        if cl_id
        in ontology.labels_by_id
    ]

    valid_supplied_ids = list(
        dict.fromkeys(
            valid_supplied_ids
        )
    )

    if len(
        valid_supplied_ids
    ) == 1:

        cl_id = valid_supplied_ids[
            0
        ]

        return {
            "cell_type": label,

            "supplied_ontology_ids": [
                display_cl_id(
                    value
                )
                for value in supplied_ids
            ],

            "resolved_ontology_id": (
                display_cl_id(
                    cl_id
                )
            ),

            "resolved_ontology_label": (
                ontology.labels_by_id.get(
                    cl_id
                )
            ),

            "ontology_resolution_source": (
                "FROZEN_AGENT2_ONTOLOGY_ID"
            ),

            "ontology_mapping_resolved": (
                True
            ),

            "ontology_mapping_warning": (
                None
            ),
        }

    if len(
        valid_supplied_ids
    ) > 1:

        return {
            "cell_type": label,

            "supplied_ontology_ids": [
                display_cl_id(
                    value
                )
                for value in supplied_ids
            ],

            "resolved_ontology_id": None,

            "resolved_ontology_label": None,

            "ontology_resolution_source": (
                "UNRESOLVED_MULTIPLE_VALID_AGENT2_IDS"
            ),

            "ontology_mapping_resolved": (
                False
            ),

            "ontology_mapping_warning": (
                "Multiple valid frozen Agent 2 ontology "
                "IDs were supplied; no ID was selected."
            ),
        }

    # 2. Exact Cell Ontology primary label.

    exact_label_id = ontology.resolve_id(
        label=label,
        ontology_id=None,
    )

    if (
        exact_label_id
        and exact_label_id
        in ontology.labels_by_id
    ):

        return {
            "cell_type": label,

            "supplied_ontology_ids": [
                display_cl_id(
                    value
                )
                for value in supplied_ids
            ],

            "resolved_ontology_id": (
                display_cl_id(
                    exact_label_id
                )
            ),

            "resolved_ontology_label": (
                ontology.labels_by_id.get(
                    exact_label_id
                )
            ),

            "ontology_resolution_source": (
                "EXACT_CELL_ONTOLOGY_LABEL"
            ),

            "ontology_mapping_resolved": (
                True
            ),

            "ontology_mapping_warning": (
                None
            ),
        }

    # 3. Exact Cell Ontology synonym.
    # No fuzzy matching.
    # No manual terminology substitution.

    synonym_matches = list(
        exact_synonym_index.get(
            label.casefold(),
            [],
        )
    )

    synonym_matches = list(
        dict.fromkeys(
            synonym_matches
        )
    )

    if len(
        synonym_matches
    ) == 1:

        cl_id = synonym_matches[
            0
        ]

        return {
            "cell_type": label,

            "supplied_ontology_ids": [
                display_cl_id(
                    value
                )
                for value in supplied_ids
            ],

            "resolved_ontology_id": (
                display_cl_id(
                    cl_id
                )
            ),

            "resolved_ontology_label": (
                ontology.labels_by_id.get(
                    cl_id
                )
            ),

            "ontology_resolution_source": (
                "EXACT_CELL_ONTOLOGY_SYNONYM"
            ),

            "ontology_mapping_resolved": (
                True
            ),

            "ontology_mapping_warning": (
                None
            ),
        }

    if len(
        synonym_matches
    ) > 1:

        return {
            "cell_type": label,

            "supplied_ontology_ids": [
                display_cl_id(
                    value
                )
                for value in supplied_ids
            ],

            "resolved_ontology_id": None,

            "resolved_ontology_label": None,

            "ontology_resolution_source": (
                "UNRESOLVED_AMBIGUOUS_EXACT_SYNONYM"
            ),

            "ontology_mapping_resolved": (
                False
            ),

            "ontology_mapping_warning": (
                "The exact synonym maps to multiple "
                "Cell Ontology identifiers; no ID was "
                "selected."
            ),
        }

    # 4. Preserve unmapped candidate.

    return {
        "cell_type": label,

        "supplied_ontology_ids": [
            display_cl_id(
                value
            )
            for value in supplied_ids
        ],

        "resolved_ontology_id": None,

        "resolved_ontology_label": None,

        "ontology_resolution_source": (
            "UNRESOLVED_NO_EXACT_ONTOLOGY_MAPPING"
        ),

        "ontology_mapping_resolved": (
            False
        ),

        "ontology_mapping_warning": (
            "No frozen Agent 2 ontology ID, exact Cell "
            "Ontology primary label, or exact Cell "
            "Ontology synonym was available. Candidate "
            "was preserved without inventing a mapping."
        ),
    }


# PAIRWISE RELATIONSHIPS


def compare_candidate_pair(
    *,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    ontology: CellOntologyIndex,
) -> dict[str, Any]:

    left_label = clean(
        left.get(
            "cell_type"
        )
    )

    right_label = clean(
        right.get(
            "cell_type"
        )
    )

    left_id = normalize_cl_id(
        left.get(
            "resolved_ontology_id"
        )
    )

    right_id = normalize_cl_id(
        right.get(
            "resolved_ontology_id"
        )
    )

    if (
        not left_id
        or not right_id
    ):

        return {
            "left_label": (
                left_label
            ),

            "right_label": (
                right_label
            ),

            "relationship_evaluable": (
                False
            ),

            "closely_related": (
                None
            ),

            "relationship_evidence": {
                "reason": (
                    "ontology_mapping_incomplete"
                ),

                "common_ancestor_id": (
                    None
                ),

                "common_ancestor_label": (
                    None
                ),

                "ontology_distance": (
                    None
                ),
            },
        }

    related, evidence = (
        ontology.are_closely_related(
            {
                "label": left_label,

                "_resolved_ontology_id": (
                    left_id
                ),
            },
            {
                "label": right_label,

                "_resolved_ontology_id": (
                    right_id
                ),
            },
        )
    )

    if not isinstance(
        evidence,
        dict,
    ):
        evidence = {}

    return {
        "left_label": (
            left_label
        ),

        "right_label": (
            right_label
        ),

        "relationship_evaluable": (
            True
        ),

        "closely_related": bool(
            related
        ),

        "relationship_evidence": (
            evidence
        ),
    }


# RELATIONSHIP COMPONENTS

def build_relationship_components(
    *,
    resolved_candidates: list[
        Mapping[str, Any]
    ],
    unresolved_candidates: list[
        Mapping[str, Any]
    ],
    pairwise_checks: list[
        Mapping[str, Any]
    ],
) -> list[list[str]]:
    """
    Build connected ontology relationship components.

    Resolved candidates are connected using closely-related edges.

    Every ontology-unmapped candidate is retained as its own singleton
    component. This does NOT assert that the candidate is biologically
    distinct; it simply prevents evidence from being discarded.
    """

    resolved_labels = [
        clean(
            row.get(
                "cell_type"
            )
        )
        for row in resolved_candidates
    ]

    graph: dict[
        str,
        set[str],
    ] = {
        label: set()
        for label in resolved_labels
    }

    for row in pairwise_checks:

        if row.get(
            "relationship_evaluable"
        ) is not True:
            continue

        if row.get(
            "closely_related"
        ) is not True:
            continue

        left = clean(
            row.get(
                "left_label"
            )
        )

        right = clean(
            row.get(
                "right_label"
            )
        )

        if (
            left in graph
            and right in graph
        ):

            graph[left].add(
                right
            )

            graph[right].add(
                left
            )

    components: list[
        list[str]
    ] = []

    visited: set[str] = set()

    for label in resolved_labels:

        if label in visited:
            continue

        stack = [
            label
        ]

        component: list[
            str
        ] = []

        while stack:

            current = stack.pop()

            if current in visited:
                continue

            visited.add(
                current
            )

            component.append(
                current
            )

            neighbours = sorted(
                graph.get(
                    current,
                    set(),
                ),
                reverse=True,
            )

            stack.extend(
                neighbours
            )

        components.append(
            component
        )

    
    # Preserve ontology-unmapped candidates.
    #
    # These singleton components mean "relationship unknown",
    # NOT "biologically independent".

    for row in unresolved_candidates:

        label = clean(
            row.get(
                "cell_type"
            )
        )

        if label:
            components.append(
                [
                    label
                ]
            )

    return components


# STRUCTURE CLASSIFICATION


def classify_relationship_structure(
    *,
    selected_count: int,
    unresolved_count: int,
    total_pair_count: int,
    evaluable_pair_count: int,
    related_pair_count: int,
    component_count: int,
) -> str:

    if unresolved_count > 0:
        return (
            "ONTOLOGY_MAPPING_INCOMPLETE"
        )

    if selected_count == 1:
        return (
            "SINGLE_SELECTED_CANDIDATE"
        )

    if (
        total_pair_count > 0
        and evaluable_pair_count == 0
    ):
        return (
            "NO_EVALUABLE_RELATIONSHIPS"
        )

    if (
        total_pair_count > 0
        and related_pair_count
        == total_pair_count
    ):
        return (
            "ALL_SELECTED_CANDIDATES_PAIRWISE_RELATED"
        )

    if component_count == 1:
        return (
            "ONE_CONNECTED_RELATIONSHIP_COMPONENT_"
            "WITH_NONRELATED_PAIRS"
        )

    return (
        "MULTIPLE_RELATIONSHIP_COMPONENTS"
    )


# MAIN BUILDER


def build_candidate_relationship_audit(
    *,
    candidate_audit_json: str | Path,
    cell_ontology_json: str | Path,
) -> dict[str, Any]:

    candidate_path = Path(
        candidate_audit_json
    )

    ontology_path = Path(
        cell_ontology_json
    )

    candidate_audit = load_json(
        candidate_path
    )

    validate_candidate_audit(
        candidate_audit
    )

    if not ontology_path.exists():
        raise FileNotFoundError(
            "Cell Ontology JSON does not exist: "
            f"{ontology_path}"
        )

    ontology = (
        CellOntologyIndex.from_json(
            ontology_path
        )
    )

    if ontology is None:
        raise RuntimeError(
            "Could not load Cell Ontology."
        )

    exact_synonym_index = (
        build_exact_synonym_index(
            ontology_path
        )
    )

    selected_candidates = (
        extract_selected_candidates(
            candidate_audit
        )
    )

    resolution_rows = [
        resolve_candidate_ontology(
            candidate=row,
            ontology=ontology,
            exact_synonym_index=(
                exact_synonym_index
            ),
        )
        for row in selected_candidates
    ]

    resolved_candidates = [
        row
        for row in resolution_rows
        if row.get(
            "ontology_mapping_resolved"
        ) is True
    ]

    unresolved_candidates = [
        row
        for row in resolution_rows
        if row.get(
            "ontology_mapping_resolved"
        ) is not True
    ]

    # Every pair is inspected.

    pairwise_checks: list[
        dict[str, Any]
    ] = []

    for left_index in range(
        len(
            resolution_rows
        )
    ):

        for right_index in range(
            left_index + 1,
            len(
                resolution_rows
            ),
        ):

            pairwise_checks.append(
                compare_candidate_pair(
                    left=resolution_rows[
                        left_index
                    ],
                    right=resolution_rows[
                        right_index
                    ],
                    ontology=ontology,
                )
            )

    total_pair_count = len(
        pairwise_checks
    )

    evaluable_pair_count = sum(
        1
        for row in pairwise_checks
        if row.get(
            "relationship_evaluable"
        ) is True
    )

    related_pair_count = sum(
        1
        for row in pairwise_checks
        if (
            row.get(
                "relationship_evaluable"
            ) is True
            and row.get(
                "closely_related"
            ) is True
        )
    )

    nonrelated_pair_count = sum(
        1
        for row in pairwise_checks
        if (
            row.get(
                "relationship_evaluable"
            ) is True
            and row.get(
                "closely_related"
            ) is False
        )
    )

    components = (
        build_relationship_components(
            resolved_candidates=(
                resolved_candidates
            ),
            unresolved_candidates=(
                unresolved_candidates
            ),
            pairwise_checks=(
                pairwise_checks
            ),
        )
    )

    relationship_structure = (
        classify_relationship_structure(
            selected_count=len(
                resolution_rows
            ),
            unresolved_count=len(
                unresolved_candidates
            ),
            total_pair_count=(
                total_pair_count
            ),
            evaluable_pair_count=(
                evaluable_pair_count
            ),
            related_pair_count=(
                related_pair_count
            ),
            component_count=len(
                components
            ),
        )
    )

    audit = {
        "schema_name": (
            SCHEMA_NAME
        ),

        "schema_version": (
            SCHEMA_VERSION
        ),

        "dataset_id": (
            candidate_audit.get(
                "dataset_id"
            )
        ),

        "cluster_id": (
            candidate_audit.get(
                "cluster_id"
            )
        ),

        "species": (
            candidate_audit.get(
                "species"
            )
        ),

        "tissue": (
            candidate_audit.get(
                "tissue"
            )
        ),

        "platform": (
            candidate_audit.get(
                "platform"
            )
        ),

        "disease_context": (
            candidate_audit.get(
                "disease_context"
            )
        ),

        "core_agent3_decision": (
            "UNRESOLVED"
        ),

        "ontology_resource": {
            "path": str(
                ontology_path
            ),

            "exact_primary_label_matching": (
                True
            ),

            "exact_synonym_matching": (
                True
            ),

            "broad_synonym_matching": (
                False
            ),

            "narrow_synonym_matching": (
                False
            ),

            "fuzzy_matching": (
                False
            ),

            "manual_ontology_id_assignment": (
                False
            ),
        },

        "relationship_policy": {
            "uses_existing_agent3_cell_ontology_index": (
                True
            ),

            "modifies_agent3": (
                False
            ),

            "evaluates_every_candidate_pair": (
                True
            ),

            "ontology_unmapped_candidates_preserved": (
                True
            ),

            "ontology_unmapped_candidates_assumed_unrelated": (
                False
            ),

            "reference_label_used": (
                False
            ),

            "expert_label_used": (
                False
            ),

            "manual_candidate_added": (
                False
            ),

            "manual_ontology_mapping_added": (
                False
            ),

            "dataset_specific_hardcoding_used": (
                False
            ),

            "cluster_specific_hardcoding_used": (
                False
            ),

            "cell_type_specific_hardcoding_used": (
                False
            ),

            "decision_threshold_applied": (
                False
            ),

            "supplemental_biological_decision_made": (
                False
            ),
        },

        "candidate_ontology_resolution": (
            resolution_rows
        ),

        "pairwise_relationship_checks": (
            pairwise_checks
        ),

        "relationship_components": (
            components
        ),

        "summary": {
            "selected_candidate_count": len(
                resolution_rows
            ),

            "ontology_resolved_candidate_count": len(
                resolved_candidates
            ),

            "ontology_unresolved_candidate_count": len(
                unresolved_candidates
            ),

            "total_candidate_pairs": (
                total_pair_count
            ),

            "evaluable_candidate_pairs": (
                evaluable_pair_count
            ),

            "closely_related_pair_count": (
                related_pair_count
            ),

            "nonrelated_pair_count": (
                nonrelated_pair_count
            ),

            "relationship_component_count": len(
                components
            ),

            "mapped_relationship_component_count": (
                len(
                    components
                )
                - len(
                    unresolved_candidates
                )
            ),

            "unmapped_singleton_component_count": len(
                unresolved_candidates
            ),

            "relationship_structure": (
                relationship_structure
            ),
        },

        "decision": {
            "supplemental_decision": (
                None
            ),

            "supplemental_possible_labels": (
                []
            ),

            "decision_reason": (
                "Candidate ontology relationships were "
                "audited descriptively. No supplemental "
                "biological adjudication was performed."
            ),

            "core_agent3_output_changed": (
                False
            ),
        },

        "status": (
            "CANDIDATE_RELATIONSHIP_AUDIT_BUILT_"
            "NOT_YET_ADJUDICATED"
        ),
    }

    return audit


# WRITER


def write_candidate_relationship_audit(
    audit: Mapping[str, Any],
    output_json: str | Path,
    *,
    overwrite: bool = False,
) -> Path:

    output_path = Path(
        output_json
    )

    if (
        output_path.exists()
        and not overwrite
    ):
        raise FileExistsError(
            "Refusing to overwrite existing candidate "
            f"relationship audit: {output_path}"
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
            dict(audit),
            handle,
            indent=2,
            ensure_ascii=False,
        )

        handle.write(
            "\n"
        )

    return output_path

