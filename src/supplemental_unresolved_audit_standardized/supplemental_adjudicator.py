
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_NAME = "StandardizedSupplementalAdjudication"
SCHEMA_VERSION = "1.1"

EXPECTED_COMPONENT_STATUS = (
    "COMPONENT_EVIDENCE_AUDIT_BUILT_NOT_YET_ADJUDICATED"
)

RESOLVED = "RESOLVED"
AMBIGUOUS = "AMBIGUOUS"
UNRESOLVED = "UNRESOLVED"

CLOSELY_RELATED_STATES = (
    "CLOSELY_RELATED_STATES"
)

POSSIBLE_MIXED_CLUSTER = (
    "POSSIBLE_MIXED_CLUSTER"
)


# FROZEN MINIMUM SUPPORT POLICY
# This is inherited from the already-used Agent 2 minimum-support
# eligibility rule. It is NOT tuned using Breast Cluster 3.

MIN_UNIQUE_SUPPORTING_GENES = 2
MIN_RANKED_SUPPORTING_MARKERS = 2


# UTILITIES

def clean(value: Any) -> str:
    return " ".join(
        str(value or "").split()
    )


def to_int(value: Any) -> int:
    try:
        return int(
            float(value)
        )

    except (
        TypeError,
        ValueError,
    ):
        return 0


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def load_json(
    path: str | Path,
) -> dict[str, Any]:

    path = Path(
        path
    )

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


# VALIDATION


def validate_component_audit(
    audit: Mapping[str, Any],
) -> None:

    status = clean(
        audit.get(
            "status"
        )
    )

    if status != EXPECTED_COMPONENT_STATUS:

        raise RuntimeError(
            "Unexpected component audit status. "
            f"Expected={EXPECTED_COMPONENT_STATUS}; "
            f"received={status}"
        )

    if clean(
        audit.get(
            "core_agent3_decision"
        )
    ).upper() != "UNRESOLVED":

        raise RuntimeError(
            "Supplemental adjudication is allowed only "
            "after frozen core Agent 3 UNRESOLVED."
        )

    policy = audit.get(
        "component_policy",
        {},
    )

    if not isinstance(
        policy,
        Mapping,
    ):
        raise ValueError(
            "Component policy missing."
        )

    required_false = (
        "shared_gene_double_counting_allowed",
        "candidate_specificity_scores_summed",
        "agent1_counts_as_additional_vote",
        "hpa_counts_as_additional_vote",
        "manual_candidate_added",
        "manual_gene_added",
        "manual_component_added",
        "manual_ontology_mapping_added",
        "reference_label_used",
        "expert_label_used",
        "cluster_specific_hardcoding_used",
        "dataset_specific_hardcoding_used",
        "cell_type_specific_hardcoding_used",
        "decision_threshold_applied",
        "supplemental_biological_decision_made",
    )

    for field in required_false:

        if (
            field in policy
            and policy.get(
                field
            ) is not False
        ):

            raise RuntimeError(
                "Component audit safety condition failed: "
                f"{field}"
            )

    previous_decision = audit.get(
        "decision",
        {},
    )

    if not isinstance(
        previous_decision,
        Mapping,
    ):
        raise ValueError(
            "Component decision section missing."
        )

    if previous_decision.get(
        "supplemental_decision"
    ) is not None:

        raise RuntimeError(
            "Component audit already contains a "
            "supplemental decision."
        )

    if previous_decision.get(
        "core_agent3_output_changed"
    ) is not False:

        raise RuntimeError(
            "Core Agent 3 output must remain unchanged."
        )


# COMPONENT EVIDENCE QUALIFICATION


def evaluate_component(
    component: Mapping[str, Any],
) -> dict[str, Any]:

    labels = component.get(
        "member_candidate_labels",
        [],
    )

    if not isinstance(
        labels,
        list,
    ):
        labels = []

    labels = [
        clean(label)
        for label in labels
        if clean(label)
    ]

    unique_gene_count = to_int(
        component.get(
            "unique_supporting_gene_count"
        )
    )

    ranked_marker_count = to_int(
        component.get(
            "unique_ranked_supporting_marker_count"
        )
    )

    gene_requirement_met = (
        unique_gene_count
        >= MIN_UNIQUE_SUPPORTING_GENES
    )

    ranked_requirement_met = (
        ranked_marker_count
        >= MIN_RANKED_SUPPORTING_MARKERS
    )

    qualifies = (
        gene_requirement_met
        and ranked_requirement_met
    )

    ontology_complete = component.get(
        "ontology_mapping_complete"
    )

    # Compatibility safeguard:
    # If the field exists, it must be a Boolean.
    if ontology_complete not in (
        True,
        False,
    ):
        raise RuntimeError(
            "Component is missing explicit "
            "ontology_mapping_complete status."
        )

    relationship = component.get(
        "relationship_summary",
        {},
    )

    if not isinstance(
        relationship,
        Mapping,
    ):
        relationship = {}

    reasons: list[str] = []

    if gene_requirement_met:
        reasons.append(
            "minimum_unique_supporting_gene_requirement_met"
        )
    else:
        reasons.append(
            "minimum_unique_supporting_gene_requirement_not_met"
        )

    if ranked_requirement_met:
        reasons.append(
            "minimum_ranked_marker_requirement_met"
        )
    else:
        reasons.append(
            "minimum_ranked_marker_requirement_not_met"
        )

    if ontology_complete:
        reasons.append(
            "ontology_relationship_available"
        )
    else:
        reasons.append(
            "ontology_relationship_incomplete"
        )

    return {
        "component_id": (
            component.get(
                "component_id"
            )
        ),

        "component_type": (
            component.get(
                "component_type"
            )
        ),

        "ontology_component_status": (
            component.get(
                "ontology_component_status"
            )
        ),

        "ontology_mapping_complete": (
            ontology_complete
        ),

        "ontology_relationship_known": (
            component.get(
                "ontology_relationship_known"
            )
        ),

        "member_candidate_count": len(
            labels
        ),

        "member_candidate_labels": (
            labels
        ),

        "unique_supporting_gene_count": (
            unique_gene_count
        ),

        "unique_ranked_supporting_marker_count": (
            ranked_marker_count
        ),

        "best_supporting_marker_rank": (
            component.get(
                "best_supporting_marker_rank"
            )
        ),

        "median_supporting_marker_rank": (
            component.get(
                "median_supporting_marker_rank"
            )
        ),

        "all_member_pairs_related": (
            relationship.get(
                "all_member_pairs_related"
            )
        ),

        "qualifies_minimum_evidence": (
            qualifies
        ),

        "qualification_reasons": (
            reasons
        ),
    }


# LABEL HELPERS


def collect_labels(
    components: list[
        Mapping[str, Any]
    ],
) -> list[str]:

    labels: list[str] = []

    seen: set[str] = set()

    for component in components:

        values = component.get(
            "member_candidate_labels",
            [],
        )

        if not isinstance(
            values,
            list,
        ):
            continue

        for value in values:

            label = clean(
                value
            )

            if not label:
                continue

            key = label.casefold()

            if key in seen:
                continue

            seen.add(
                key
            )

            labels.append(
                label
            )

    return labels

# COMPLETE-ONTOLOGY ADJUDICATION



def adjudicate_complete_mapping(
    qualifying_components: list[
        Mapping[str, Any]
    ],
) -> tuple[
    str,
    str | None,
    list[str],
    str,
]:

    if len(
        qualifying_components
    ) == 0:

        return (
            UNRESOLVED,
            None,
            [],
            (
                "No Cell Ontology relationship component "
                "met the standardized minimum evidence "
                "requirement."
            ),
        )

    if len(
        qualifying_components
    ) > 1:

        return (
            AMBIGUOUS,
            POSSIBLE_MIXED_CLUSTER,
            collect_labels(
                qualifying_components
            ),
            (
                "Multiple distinct Cell Ontology "
                "relationship components independently "
                "met the standardized minimum evidence "
                "requirement. Competing lineage evidence "
                "was therefore preserved."
            ),
        )

    component = qualifying_components[
        0
    ]

    labels = collect_labels(
        [
            component
        ]
    )

    member_count = to_int(
        component.get(
            "member_candidate_count"
        )
    )

    all_related = component.get(
        "all_member_pairs_related"
    )

    if (
        member_count == 1
        and len(labels) == 1
    ):

        return (
            RESOLVED,
            None,
            labels,
            (
                "Exactly one fully ontology-mapped "
                "component met the standardized minimum "
                "evidence requirement and contained one "
                "candidate."
            ),
        )

    if (
        member_count > 1
        and all_related is True
    ):

        return (
            AMBIGUOUS,
            CLOSELY_RELATED_STATES,
            labels,
            (
                "Exactly one fully ontology-mapped "
                "component met the evidence requirement, "
                "but it contained multiple closely "
                "related candidate states."
            ),
        )

    return (
        AMBIGUOUS,
        POSSIBLE_MIXED_CLUSTER,
        labels,
        (
            "Exactly one ontology component met the "
            "minimum evidence requirement, but its "
            "candidate members were not all pairwise "
            "closely related."
        ),
    )

# INCOMPLETE-ONTOLOGY ADJUDICATION



def adjudicate_incomplete_mapping(
    *,
    qualifying_mapped_components: list[
        Mapping[str, Any]
    ],
    qualifying_unmapped_components: list[
        Mapping[str, Any]
    ],
) -> tuple[
    str,
    str | None,
    list[str],
    str,
]:
    """
    Conservative policy for incomplete ontology mappings.

    Critical rule:
        INCOMPLETE ONTOLOGY MAPPING CAN NEVER PRODUCE RESOLVED.

    Unmapped singleton components are not interpreted as independent
    biological lineages because their ontology relationships are unknown.
    """

    mapped_count = len(
        qualifying_mapped_components
    )

    unmapped_count = len(
        qualifying_unmapped_components
    )

  
    # At least two ontology-mapped components independently satisfy the evidence rule.
    # Distinct mapped components are sufficient to establish
    # competing biological evidence even though other candidates remain unmapped.
   
    if mapped_count >= 2:

        mapped_labels = collect_labels(
            qualifying_mapped_components
        )

        return (
            AMBIGUOUS,
            POSSIBLE_MIXED_CLUSTER,
            mapped_labels,
            (
                "Ontology mapping was incomplete, so "
                "RESOLVED was prohibited. However, at "
                "least two distinct fully ontology-mapped "
                "components independently met the "
                "standardized minimum evidence requirement. "
                "This is sufficient to preserve competing "
                "mapped lineage evidence as AMBIGUOUS. "
                "Ontology-unmapped candidates were retained "
                "separately and did not create additional "
                "independent lineage votes."
            ),
        )

    # Zero or one qualifying mapped component.
    # We cannot establish that an unmapped candidate belongs to the
    # mapped component or represents another distinct component. Therefore we abstain.
  

    if mapped_count == 1:

        return (
            UNRESOLVED,
            None,
            [],
            (
                "Ontology mapping was incomplete and only "
                "one fully ontology-mapped component met "
                "the standardized minimum evidence "
                "requirement. Because relationships between "
                "that component and the ontology-unmapped "
                "candidates are unknown, the supplemental "
                "audit conservatively abstained."
            ),
        )

    if unmapped_count > 0:

        return (
            UNRESOLVED,
            None,
            [],
            (
                "Ontology mapping was incomplete and no "
                "fully ontology-mapped component provided "
                "sufficient independent evidence for a "
                "safe adjudication. Evidence-supported "
                "unmapped candidates were retained, but "
                "their relationships cannot be assumed."
            ),
        )

    return (
        UNRESOLVED,
        None,
        [],
        (
            "Ontology mapping was incomplete and no fully "
            "ontology-mapped component met the standardized "
            "minimum evidence requirement."
        ),
    )


# MAIN BUILDER



def build_supplemental_adjudication(
    *,
    component_audit_json: str | Path,
) -> dict[str, Any]:

    component_path = Path(
        component_audit_json
    )

    component_audit = load_json(
        component_path
    )

    validate_component_audit(
        component_audit
    )

    components = component_audit.get(
        "components"
    )

    if not isinstance(
        components,
        list,
    ):
        raise ValueError(
            "Component audit components must be an array."
        )

    evaluated_components = [
        evaluate_component(
            component
        )
        for component in components
        if isinstance(
            component,
            Mapping,
        )
    ]

    qualifying_components = [
        component
        for component in evaluated_components
        if component.get(
            "qualifies_minimum_evidence"
        ) is True
    ]

    qualifying_mapped_components = [
        component
        for component in qualifying_components
        if component.get(
            "ontology_mapping_complete"
        ) is True
    ]

    qualifying_unmapped_components = [
        component
        for component in qualifying_components
        if component.get(
            "ontology_mapping_complete"
        ) is False
    ]

    nonqualifying_components = [
        component
        for component in evaluated_components
        if component.get(
            "qualifies_minimum_evidence"
        ) is not True
    ]

    ontology_mapping_complete = (
        component_audit.get(
            "ontology_mapping_complete"
        )
    )

    if ontology_mapping_complete not in (
        True,
        False,
    ):
        raise RuntimeError(
            "Component audit missing explicit global "
            "ontology_mapping_complete status."
        )

    if ontology_mapping_complete:

        (
            decision_state,
            ambiguity_subtype,
            possible_labels,
            decision_reason,
        ) = adjudicate_complete_mapping(
            qualifying_mapped_components
        )

    else:

        (
            decision_state,
            ambiguity_subtype,
            possible_labels,
            decision_reason,
        ) = adjudicate_incomplete_mapping(
            qualifying_mapped_components=(
                qualifying_mapped_components
            ),
            qualifying_unmapped_components=(
                qualifying_unmapped_components
            ),
        )

        # Hard safety invariant.
        if decision_state == RESOLVED:

            raise RuntimeError(
                "Safety invariant violated: an "
                "ontology-incomplete case cannot be "
                "RESOLVED."
            )

    unmapped_candidate_labels = (
        collect_labels(
            [
                component
                for component in evaluated_components
                if component.get(
                    "ontology_mapping_complete"
                ) is False
            ]
        )
    )

    qualifying_unmapped_labels = (
        collect_labels(
            qualifying_unmapped_components
        )
    )

    result = {
        "schema_name": (
            SCHEMA_NAME
        ),

        "schema_version": (
            SCHEMA_VERSION
        ),

        "dataset_id": (
            component_audit.get(
                "dataset_id"
            )
        ),

        "cluster_id": (
            component_audit.get(
                "cluster_id"
            )
        ),

        "species": (
            component_audit.get(
                "species"
            )
        ),

        "tissue": (
            component_audit.get(
                "tissue"
            )
        ),

        "platform": (
            component_audit.get(
                "platform"
            )
        ),

        "disease_context": (
            component_audit.get(
                "disease_context"
            )
        ),

        "core_agent3_decision": (
            "UNRESOLVED"
        ),

        "core_agent3_output_changed": (
            False
        ),

        "ontology_mapping_complete": (
            ontology_mapping_complete
        ),

        "relationship_structure": (
            component_audit.get(
                "relationship_structure"
            )
        ),

        "adjudication_policy": {
            "policy_version": (
                "1.1"
            ),

            "policy_type": (
                "NON_CALIBRATED_EVIDENCE_STATE"
            ),

            "minimum_unique_supporting_genes": (
                MIN_UNIQUE_SUPPORTING_GENES
            ),

            "minimum_ranked_supporting_markers": (
                MIN_RANKED_SUPPORTING_MARKERS
            ),

            "minimum_support_threshold_source": (
                "Inherited from frozen Agent 2 "
                "minimum-support eligibility rule"
            ),

            "ontology_incomplete_can_resolve": (
                False
            ),

            "unmapped_candidate_creates_independent_lineage_vote": (
                False
            ),

            "multiple_qualifying_mapped_components_can_establish_ambiguity": (
                True
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

            "manual_gene_added": (
                False
            ),

            "manual_component_added": (
                False
            ),

            "manual_ontology_mapping_added": (
                False
            ),

            "posthoc_breast_threshold_tuning_used": (
                False
            ),

            "posthoc_cosmx_threshold_tuning_used": (
                False
            ),

            "candidate_specificity_scores_summed": (
                False
            ),

            "shared_gene_double_counting_allowed": (
                False
            ),

            "agent1_counts_as_additional_vote": (
                False
            ),

            "hpa_counts_as_additional_vote": (
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

            "automatic_acceptance": (
                "NOT_CALIBRATED"
            ),
        },

        "component_evidence_evaluation": {
            "total_component_count": len(
                evaluated_components
            ),

            "qualifying_component_count": len(
                qualifying_components
            ),

            "qualifying_mapped_component_count": len(
                qualifying_mapped_components
            ),

            "qualifying_unmapped_component_count": len(
                qualifying_unmapped_components
            ),

            "nonqualifying_component_count": len(
                nonqualifying_components
            ),

            "components": (
                evaluated_components
            ),
        },

        "ontology_incomplete_evidence": {
            "ontology_mapping_complete": (
                ontology_mapping_complete
            ),

            "unmapped_candidate_count": len(
                unmapped_candidate_labels
            ),

            "unmapped_candidate_labels": (
                unmapped_candidate_labels
            ),

            "qualifying_unmapped_candidate_labels": (
                qualifying_unmapped_labels
            ),

            "unmapped_candidates_used_as_independent_lineage_votes": (
                False
            ),
        },

        "supplemental_decision_state": (
            decision_state
        ),

        "supplemental_ambiguity_subtype": (
            ambiguity_subtype
        ),

        "supplemental_possible_label_count": len(
            possible_labels
        ),

        "supplemental_possible_labels": (
            possible_labels
        ),

        "supplemental_decision_reason": (
            decision_reason
        ),

        "supplemental_confidence": (
            None
        ),

        "final_annotation": (
            None
        ),

        "decision_policy_statement": (
            "Human-triggered standardized supplemental "
            "evidence-state adjudication following frozen "
            "core Agent 3 abstention. Ontology-unmapped "
            "candidates are preserved but are not assumed "
            "to represent independent biological lineages. "
            "Incomplete ontology mapping can never produce "
            "a RESOLVED supplemental state."
        ),

        "provenance": {
            "component_evidence_audit": str(
                component_path
            ),

            "component_evidence_audit_sha256": (
                sha256_file(
                    component_path
                )
            ),
        },

        "status": (
            "SUPPLEMENTAL_ADJUDICATION_COMPLETED_"
            "CORE_AGENT3_UNCHANGED"
        ),
    }

    return result


# WRITER


def write_supplemental_adjudication(
    result: Mapping[str, Any],
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
            "Refusing to overwrite existing supplemental "
            f"adjudication: {output_path}"
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
            dict(result),
            handle,
            indent=2,
            ensure_ascii=False,
        )

        handle.write(
            "\n"
        )

    return output_path


