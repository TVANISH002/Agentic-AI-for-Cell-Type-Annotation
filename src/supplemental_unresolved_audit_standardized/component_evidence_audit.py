
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Mapping


SCHEMA_NAME = "StandardizedSupplementalComponentEvidenceAudit"
SCHEMA_VERSION = "1.1"

EXPECTED_CANDIDATE_STATUS = (
    "CANDIDATE_EVIDENCE_AUDIT_BUILT_NOT_YET_ADJUDICATED"
)

EXPECTED_RELATIONSHIP_STATUS = (
    "CANDIDATE_RELATIONSHIP_AUDIT_BUILT_NOT_YET_ADJUDICATED"
)


def clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def gene_key(value: Any) -> str:
    return clean(value).upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"JSON file does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected JSON object: {path}"
        )

    return payload


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None

    return float(statistics.mean(values))


def safe_median(values: list[float]) -> float | None:
    if not values:
        return None

    return float(statistics.median(values))



# VALIDATION


def validate_inputs(
    *,
    candidate_audit: Mapping[str, Any],
    relationship_audit: Mapping[str, Any],
) -> None:

    if clean(
        candidate_audit.get("status")
    ) != EXPECTED_CANDIDATE_STATUS:

        raise RuntimeError(
            "Unexpected candidate audit status."
        )

    if clean(
        relationship_audit.get("status")
    ) != EXPECTED_RELATIONSHIP_STATUS:

        raise RuntimeError(
            "Unexpected relationship audit status."
        )

    identity_fields = (
        "dataset_id",
        "cluster_id",
        "species",
        "tissue",
        "platform",
        "disease_context",
        "core_agent3_decision",
    )

    mismatches: list[str] = []

    for field in identity_fields:

        if clean(
            candidate_audit.get(field)
        ) != clean(
            relationship_audit.get(field)
        ):
            mismatches.append(field)

    if mismatches:
        raise RuntimeError(
            "Candidate/relationship audit mismatch: "
            + ", ".join(mismatches)
        )

    if clean(
        candidate_audit.get(
            "core_agent3_decision"
        )
    ).upper() != "UNRESOLVED":

        raise RuntimeError(
            "Component audit is allowed only after "
            "core Agent 3 UNRESOLVED."
        )

    for source_name, source in (
        (
            "candidate",
            candidate_audit,
        ),
        (
            "relationship",
            relationship_audit,
        ),
    ):

        decision = source.get(
            "decision",
            {},
        )

        if not isinstance(
            decision,
            Mapping,
        ):
            raise ValueError(
                f"{source_name} decision section missing."
            )

        if decision.get(
            "supplemental_decision"
        ) is not None:
            raise RuntimeError(
                f"{source_name} audit already contains "
                "a supplemental decision."
            )

        if decision.get(
            "core_agent3_output_changed"
        ) is not False:
            raise RuntimeError(
                "Core Agent 3 output must remain unchanged."
            )

    # IMPORTANT v1.1 CHANGE:
    # Ontology-incomplete cases are now allowed to continue.
    # Unmapped candidates must already have been preserved by the
    # relationship audit. We do NOT guess mappings here.

    relationship_policy = (
        relationship_audit.get(
            "relationship_policy",
            {},
        )
    )

    summary = relationship_audit.get(
        "summary",
        {},
    )

    unresolved_count = int(
        summary.get(
            "ontology_unresolved_candidate_count",
            0,
        )
        or 0
    )

    if unresolved_count > 0:

        if relationship_policy.get(
            "ontology_unmapped_candidates_preserved"
        ) is not True:

            raise RuntimeError(
                "Ontology mappings are incomplete, but "
                "unmapped candidates were not explicitly "
                "preserved."
            )

        if relationship_policy.get(
            "ontology_unmapped_candidates_assumed_unrelated"
        ) is not False:

            raise RuntimeError(
                "Unmapped candidates must not be assumed "
                "biologically unrelated."
            )


# SELECTED CANDIDATES


def extract_selected_candidates(
    candidate_audit: Mapping[str, Any],
) -> list[dict[str, Any]]:

    rows = candidate_audit.get(
        "candidate_evidence"
    )

    if not isinstance(rows, list):
        raise ValueError(
            "candidate_evidence must be an array."
        )

    selected: list[dict[str, Any]] = []

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
            row.get("cell_type")
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

        seen.add(key)

        selected.append(
            dict(row)
        )

    if not selected:
        raise RuntimeError(
            "No selected candidates."
        )

    selected.sort(
        key=lambda row: (
            row.get("candidate_rank")
            if isinstance(
                row.get("candidate_rank"),
                (int, float),
            )
            else float("inf")
        )
    )

    return selected


# ONTOLOGY RESOLUTION LOOKUP



def build_resolution_lookup(
    relationship_audit: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:

    rows = relationship_audit.get(
        "candidate_ontology_resolution",
        [],
    )

    if not isinstance(rows, list):
        raise ValueError(
            "candidate_ontology_resolution must be an array."
        )

    lookup: dict[
        str,
        dict[str, Any],
    ] = {}

    for row in rows:

        if not isinstance(
            row,
            Mapping,
        ):
            continue

        label = clean(
            row.get("cell_type")
        )

        if not label:
            continue

        lookup[
            label.casefold()
        ] = dict(row)

    return lookup


# COMPONENT VALIDATION



def validate_components(
    *,
    selected_candidates: list[dict[str, Any]],
    components: Any,
) -> list[list[str]]:

    if not isinstance(
        components,
        list,
    ):
        raise ValueError(
            "relationship_components must be an array."
        )

    selected_labels = [
        clean(
            row.get("cell_type")
        )
        for row in selected_candidates
    ]

    selected_keys = {
        label.casefold()
        for label in selected_labels
    }

    observed_keys: list[str] = []

    normalized: list[
        list[str]
    ] = []

    for component in components:

        if not isinstance(
            component,
            list,
        ):
            raise ValueError(
                "Each relationship component must be an array."
            )

        labels = [
            clean(value)
            for value in component
            if clean(value)
        ]

        if not labels:
            raise ValueError(
                "Empty relationship component."
            )

        observed_keys.extend(
            label.casefold()
            for label in labels
        )

        normalized.append(
            labels
        )

    if len(
        observed_keys
    ) != len(
        set(observed_keys)
    ):
        raise RuntimeError(
            "Candidate appears in multiple components."
        )

    if set(
        observed_keys
    ) != selected_keys:

        raise RuntimeError(
            "Relationship components do not exactly "
            "cover all selected candidates."
        )

    return normalized

# RELATIONSHIP SUMMARY

def component_relationship_summary(
    *,
    member_labels: list[str],
    pairwise_checks: Any,
) -> dict[str, Any]:

    expected_pair_count = (
        len(member_labels)
        * (
            len(member_labels) - 1
        )
        // 2
    )

    if expected_pair_count == 0:

        return {
            "member_pair_count": 0,
            "evaluable_member_pair_count": 0,
            "closely_related_member_pair_count": 0,
            "nonrelated_member_pair_count": 0,
            "all_member_pairs_related": None,
        }

    member_keys = {
        label.casefold()
        for label in member_labels
    }

    relevant: list[
        Mapping[str, Any]
    ] = []

    if isinstance(
        pairwise_checks,
        list,
    ):

        for row in pairwise_checks:

            if not isinstance(
                row,
                Mapping,
            ):
                continue

            left = clean(
                row.get("left_label")
            ).casefold()

            right = clean(
                row.get("right_label")
            ).casefold()

            if (
                left in member_keys
                and right in member_keys
            ):
                relevant.append(row)

    evaluable = [
        row
        for row in relevant
        if row.get(
            "relationship_evaluable"
        ) is True
    ]

    related = [
        row
        for row in evaluable
        if row.get(
            "closely_related"
        ) is True
    ]

    nonrelated = [
        row
        for row in evaluable
        if row.get(
            "closely_related"
        ) is False
    ]

    if (
        len(evaluable)
        == expected_pair_count
    ):
        all_related = (
            len(related)
            == expected_pair_count
        )
    else:
        all_related = None

    return {
        "member_pair_count": (
            expected_pair_count
        ),

        "evaluable_member_pair_count": len(
            evaluable
        ),

        "closely_related_member_pair_count": len(
            related
        ),

        "nonrelated_member_pair_count": len(
            nonrelated
        ),

        "all_member_pairs_related": (
            all_related
        ),
    }


# COMPONENT AGGREGATION


def aggregate_component(
    *,
    component_number: int,
    member_labels: list[str],
    candidate_by_label: Mapping[
        str,
        Mapping[str, Any],
    ],
    resolution_lookup: Mapping[
        str,
        Mapping[str, Any],
    ],
    pairwise_checks: Any,
) -> dict[str, Any]:

    members = [
        dict(
            candidate_by_label[
                label.casefold()
            ]
        )
        for label in member_labels
    ]

    resolution_rows = [
        resolution_lookup.get(
            label.casefold(),
            {},
        )
        for label in member_labels
    ]

    mapped_count = sum(
        1
        for row in resolution_rows
        if row.get(
            "ontology_mapping_resolved"
        ) is True
    )

    unmapped_count = (
        len(member_labels)
        - mapped_count
    )

    if unmapped_count == 0:

        ontology_component_status = (
            "MAPPED_COMPONENT"
        )

    elif (
        len(member_labels) == 1
        and mapped_count == 0
    ):

        ontology_component_status = (
            "UNMAPPED_SINGLETON_COMPONENT"
        )

    else:

        ontology_component_status = (
            "PARTIALLY_MAPPED_COMPONENT"
        )

    # Gene deduplication

    genes: dict[
        str,
        dict[str, Any],
    ] = {}

    total_memberships = 0

    both_source_keys: set[str] = set()
    agent1_keys: set[str] = set()
    hpa_matched_keys: set[str] = set()
    hpa_expression_keys: set[str] = set()

    specificity_values: list[
        float
    ] = []

    for member in members:

        label = clean(
            member.get("cell_type")
        )

        specificity = member.get(
            "reported_specificity_weighted_support"
        )

        if isinstance(
            specificity,
            (int, float),
        ):
            specificity_values.append(
                float(specificity)
            )

        supporting_genes = member.get(
            "supporting_genes",
            [],
        )

        if not isinstance(
            supporting_genes,
            list,
        ):
            supporting_genes = []

        total_memberships += len(
            supporting_genes
        )

        hit_rank_by_gene: dict[
            str,
            int
        ] = {}

        hits = member.get(
            "supporting_marker_hits",
            [],
        )

        if isinstance(
            hits,
            list,
        ):

            for hit in hits:

                if not isinstance(
                    hit,
                    Mapping,
                ):
                    continue

                gene = clean(
                    hit.get("gene")
                )

                rank = hit.get(
                    "rank"
                )

                if (
                    gene
                    and isinstance(
                        rank,
                        (int, float),
                    )
                ):
                    hit_rank_by_gene[
                        gene_key(gene)
                    ] = int(rank)

        for gene in supporting_genes:

            gene = clean(gene)

            if not gene:
                continue

            key = gene_key(gene)

            marker_rank = (
                hit_rank_by_gene.get(
                    key
                )
            )

            if key not in genes:

                genes[key] = {
                    "gene": gene,
                    "marker_rank": (
                        marker_rank
                    ),
                    "supporting_candidate_labels": [],
                    "supporting_candidate_count": 0,
                }

            if label not in genes[
                key
            ][
                "supporting_candidate_labels"
            ]:

                genes[
                    key
                ][
                    "supporting_candidate_labels"
                ].append(label)

        for gene in member.get(
            "both_source_supporting_genes",
            [],
        ) or []:

            if clean(gene):
                both_source_keys.add(
                    gene_key(gene)
                )

        for gene in member.get(
            "agent1_matched_supporting_genes",
            [],
        ) or []:

            if clean(gene):
                agent1_keys.add(
                    gene_key(gene)
                )

        hpa = member.get(
            "hpa_coverage",
            {},
        )

        if isinstance(
            hpa,
            Mapping,
        ):

            for gene in hpa.get(
                "matched_supporting_genes",
                [],
            ) or []:

                if clean(gene):
                    hpa_matched_keys.add(
                        gene_key(gene)
                    )

            for gene in hpa.get(
                "expression_found_supporting_genes",
                [],
            ) or []:

                if clean(gene):
                    hpa_expression_keys.add(
                        gene_key(gene)
                    )

    for record in genes.values():

        record[
            "supporting_candidate_count"
        ] = len(
            record[
                "supporting_candidate_labels"
            ]
        )

    gene_records = list(
        genes.values()
    )

    gene_records.sort(
        key=lambda row: (
            row.get("marker_rank")
            if row.get(
                "marker_rank"
            ) is not None
            else float("inf"),
            clean(
                row.get("gene")
            ),
        )
    )

    ranked = [
        row
        for row in gene_records
        if row.get(
            "marker_rank"
        ) is not None
    ]

    ranks = [
        float(
            row["marker_rank"]
        )
        for row in ranked
    ]

    relationship_summary = (
        component_relationship_summary(
            member_labels=member_labels,
            pairwise_checks=pairwise_checks,
        )
    )

    if (
        ontology_component_status
        == "UNMAPPED_SINGLETON_COMPONENT"
    ):

        component_type = (
            "UNMAPPED_SINGLETON_COMPONENT"
        )

    elif len(
        member_labels
    ) == 1:

        component_type = (
            "SINGLE_CANDIDATE_COMPONENT"
        )

    elif relationship_summary.get(
        "all_member_pairs_related"
    ) is True:

        component_type = (
            "RELATED_CANDIDATE_COMPONENT"
        )

    else:

        component_type = (
            "CONNECTED_CANDIDATE_COMPONENT"
        )

    return {
        "component_id": (
            f"COMPONENT_{component_number:02d}"
        ),

        "component_type": (
            component_type
        ),

        "ontology_component_status": (
            ontology_component_status
        ),

        "ontology_mapping_complete": (
            unmapped_count == 0
        ),

        "ontology_mapped_member_count": (
            mapped_count
        ),

        "ontology_unmapped_member_count": (
            unmapped_count
        ),

        "ontology_relationship_known": (
            unmapped_count == 0
        ),

        "member_candidate_count": len(
            member_labels
        ),

        "member_candidate_labels": (
            member_labels
        ),

        "member_candidate_ranks": [
            row.get(
                "candidate_rank"
            )
            for row in members
        ],

        "relationship_summary": (
            relationship_summary
        ),

        "candidate_specificity_weighted_support_values": (
            specificity_values
        ),

        "max_candidate_specificity_weighted_support": (
            max(
                specificity_values
            )
            if specificity_values
            else None
        ),

        "mean_candidate_specificity_weighted_support": (
            safe_mean(
                specificity_values
            )
        ),

        "total_candidate_gene_memberships_before_deduplication": (
            total_memberships
        ),

        "unique_supporting_gene_count": len(
            gene_records
        ),

        "repeated_gene_membership_count_removed": max(
            0,
            total_memberships
            - len(
                gene_records
            ),
        ),

        "unique_supporting_genes": [
            row.get("gene")
            for row in gene_records
        ],

        "unique_supporting_gene_records": (
            gene_records
        ),

        "unique_ranked_supporting_marker_count": len(
            ranked
        ),

        "best_supporting_marker_rank": (
            min(ranks)
            if ranks
            else None
        ),

        "median_supporting_marker_rank": (
            safe_median(
                ranks
            )
        ),

        "mean_supporting_marker_rank": (
            safe_mean(
                ranks
            )
        ),

        "both_source_unique_supporting_gene_count": len(
            both_source_keys
        ),

        "agent1_matched_unique_supporting_gene_count": len(
            agent1_keys
        ),

        "hpa_matched_unique_supporting_gene_count": len(
            hpa_matched_keys
        ),

        "hpa_expression_found_unique_supporting_gene_count": len(
            hpa_expression_keys
        ),

        "coverage_policy": {
            "agent1_counts_as_additional_vote": False,
            "hpa_counts_as_additional_vote": False,
        },
    }


# MAIN BUILDER


def build_component_evidence_audit(
    *,
    candidate_audit_json: str | Path,
    relationship_audit_json: str | Path,
) -> dict[str, Any]:

    candidate_path = Path(
        candidate_audit_json
    )

    relationship_path = Path(
        relationship_audit_json
    )

    candidate_audit = load_json(
        candidate_path
    )

    relationship_audit = load_json(
        relationship_path
    )

    validate_inputs(
        candidate_audit=candidate_audit,
        relationship_audit=relationship_audit,
    )

    selected = extract_selected_candidates(
        candidate_audit
    )

    components = validate_components(
        selected_candidates=selected,
        components=relationship_audit.get(
            "relationship_components"
        ),
    )

    candidate_by_label = {
        clean(
            row.get("cell_type")
        ).casefold(): row
        for row in selected
    }

    resolution_lookup = (
        build_resolution_lookup(
            relationship_audit
        )
    )

    pairwise_checks = (
        relationship_audit.get(
            "pairwise_relationship_checks",
            [],
        )
    )

    component_records: list[
        dict[str, Any]
    ] = []

    for index, labels in enumerate(
        components,
        start=1,
    ):

        component_records.append(
            aggregate_component(
                component_number=index,
                member_labels=labels,
                candidate_by_label=(
                    candidate_by_label
                ),
                resolution_lookup=(
                    resolution_lookup
                ),
                pairwise_checks=(
                    pairwise_checks
                ),
            )
        )

    relationship_summary = (
        relationship_audit.get(
            "summary",
            {},
        )
    )

    ontology_unresolved_count = int(
        relationship_summary.get(
            "ontology_unresolved_candidate_count",
            0,
        )
        or 0
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

        "ontology_mapping_complete": (
            ontology_unresolved_count == 0
        ),

        "relationship_structure": (
            relationship_summary.get(
                "relationship_structure"
            )
        ),

        "component_policy": {
            "candidate_source": (
                "Frozen Agent 2 downstream-selected candidates only"
            ),

            "component_source": (
                "Standardized Cell Ontology relationship audit"
            ),

            "ontology_unmapped_candidates_preserved": (
                True
            ),

            "ontology_unmapped_candidates_assumed_unrelated": (
                False
            ),

            "unmapped_singleton_means_relationship_unknown": (
                True
            ),

            "gene_deduplication": (
                "Case-normalized exact gene/probe identifier"
            ),

            "shared_gene_double_counting_allowed": (
                False
            ),

            "candidate_specificity_scores_summed": (
                False
            ),

            "agent1_counts_as_additional_vote": (
                False
            ),

            "hpa_counts_as_additional_vote": (
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

            "reference_label_used": (
                False
            ),

            "expert_label_used": (
                False
            ),

            "cluster_specific_hardcoding_used": (
                False
            ),

            "dataset_specific_hardcoding_used": (
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

        "summary": {
            "selected_candidate_count": len(
                selected
            ),

            "relationship_component_count": len(
                component_records
            ),

            "mapped_component_count": sum(
                1
                for row in component_records
                if row.get(
                    "ontology_mapping_complete"
                ) is True
            ),

            "unmapped_singleton_component_count": sum(
                1
                for row in component_records
                if row.get(
                    "ontology_component_status"
                )
                == "UNMAPPED_SINGLETON_COMPONENT"
            ),

            "partially_mapped_component_count": sum(
                1
                for row in component_records
                if row.get(
                    "ontology_component_status"
                )
                == "PARTIALLY_MAPPED_COMPONENT"
            ),

            "ontology_unresolved_candidate_count": (
                ontology_unresolved_count
            ),
        },

        "components": (
            component_records
        ),

        "decision": {
            "supplemental_decision": (
                None
            ),

            "supplemental_possible_labels": (
                []
            ),

            "decision_reason": (
                "Component evidence was aggregated while "
                "preserving ontology-unmapped candidates. "
                "No supplemental biological adjudication "
                "policy has yet been applied."
            ),

            "core_agent3_output_changed": (
                False
            ),
        },

        "provenance": {
            "candidate_evidence_audit": str(
                candidate_path
            ),

            "candidate_evidence_audit_sha256": (
                sha256_file(
                    candidate_path
                )
            ),

            "candidate_relationship_audit": str(
                relationship_path
            ),

            "candidate_relationship_audit_sha256": (
                sha256_file(
                    relationship_path
                )
            ),
        },

        "status": (
            "COMPONENT_EVIDENCE_AUDIT_BUILT_"
            "NOT_YET_ADJUDICATED"
        ),
    }

    return audit


# WRITERS


def write_component_evidence_audit(
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
            "Refusing to overwrite existing component "
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
            dict(audit),
            handle,
            indent=2,
            ensure_ascii=False,
        )

        handle.write("\n")

    return output_path


def write_component_evidence_csv(
    audit: Mapping[str, Any],
    output_csv: str | Path,
    *,
    overwrite: bool = False,
) -> Path:

    output_path = Path(
        output_csv
    )

    if (
        output_path.exists()
        and not overwrite
    ):
        raise FileExistsError(
            "Refusing to overwrite existing component CSV: "
            f"{output_path}"
        )

    rows: list[
        dict[str, Any]
    ] = []

    for component in audit.get(
        "components",
        [],
    ):

        if not isinstance(
            component,
            Mapping,
        ):
            continue

        relationship = component.get(
            "relationship_summary",
            {},
        )

        rows.append(
            {
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
                    component.get(
                        "ontology_mapping_complete"
                    )
                ),

                "member_candidates": " | ".join(
                    component.get(
                        "member_candidate_labels",
                        [],
                    )
                ),

                "unique_supporting_gene_count": (
                    component.get(
                        "unique_supporting_gene_count"
                    )
                ),

                "unique_ranked_supporting_marker_count": (
                    component.get(
                        "unique_ranked_supporting_marker_count"
                    )
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

                "unique_supporting_genes": " | ".join(
                    component.get(
                        "unique_supporting_genes",
                        [],
                    )
                ),
            }
        )

    if not rows:
        raise ValueError(
            "No component rows to write."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    return output_path