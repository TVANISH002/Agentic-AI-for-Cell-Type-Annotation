from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import deque
from pathlib import Path

import pandas as pd


# Convert the mapping-candidate audit into ontology-aware EVALUATION metadata.


REQUIRED_MAPPING_COLUMNS = {
    "row_id",
    "dataset",
    "cluster_id",
    "label_role",
    "label_raw",
    "existing_agent2_ids_validated",
    "existing_agent2_id_validation",
    "exact_label_matches",
    "exact_synonym_matches",
    "lexical_mapping_status",
}

REQUIRED_POLICY_KEYS = {
    "policy_version",
    "targets",
}

CL_PATTERN = re.compile(
    r"CL[:_](\d+)",
    flags=re.IGNORECASE,
)

DISEASE_OR_COMPARTMENT_PATTERN = re.compile(
    r"\b(tumou?r|cancer|carcinoma|neoplasm|malignan\w*)\b",
    flags=re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip().casefold()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_cl_id(value: object) -> str:
    if value is None:
        return ""

    match = CL_PATTERN.search(
        str(value)
    )

    if not match:
        return ""

    return f"CL:{match.group(1)}"


def extract_ids(value: object) -> list[str]:
    if pd.isna(value):
        return []

    ids: list[str] = []

    for match in CL_PATTERN.finditer(
        str(value)
    ):
        curie = f"CL:{match.group(1)}"

        if curie not in ids:
            ids.append(curie)

    return ids


def parse_ontology(
    ontology_path: Path,
) -> tuple[
    dict[str, str],
    dict[str, list[str]],
    dict[str, set[str]],
]:

    with ontology_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)

    graphs = payload.get(
        "graphs",
        []
    )

    if not graphs:
        raise ValueError(
            "Ontology JSON contains no graphs."
        )

    id_to_label: dict[str, str] = {}
    label_index: dict[str, list[str]] = {}
    parents: dict[str, set[str]] = {}

    for graph in graphs:
        for node in graph.get(
            "nodes",
            [],
        ):
            curie = normalize_cl_id(
                node.get("id")
            )

            if not curie:
                continue

            label = str(
                node.get("lbl") or ""
            ).strip()

            id_to_label[curie] = label

            normalized = normalize_text(
                label
            )

            if normalized:
                label_index.setdefault(
                    normalized,
                    [],
                ).append(curie)

            parents.setdefault(
                curie,
                set(),
            )

        for edge in graph.get(
            "edges",
            [],
        ):
            pred = str(
                edge.get("pred") or ""
            ).casefold()

            if pred not in {
                "is_a",
                "rdfs:subclassof",
                "http://www.w3.org/2000/01/rdf-schema#subclassof",
            }:
                continue

            child = normalize_cl_id(
                edge.get("sub")
            )

            parent = normalize_cl_id(
                edge.get("obj")
            )

            if child and parent:
                parents.setdefault(
                    child,
                    set(),
                ).add(parent)

    if not id_to_label:
        raise ValueError(
            "No Cell Ontology nodes were parsed."
        )

    return (
        id_to_label,
        label_index,
        parents,
    )


def grammatical_variants(
    raw_label: str,
) -> list[str]:
    """
    Generate conservative lexical variants only.

    No biological substitutions are made.
    """

    base = normalize_text(
        raw_label
    )

    variants: list[str] = []

    def add(value: str) -> None:
        value = normalize_text(value)

        if value and value not in variants:
            variants.append(value)

    add(base)

    # Plural -> singular.
    if base.endswith(" cells"):
        add(
            base[:-1]
        )

    elif base.endswith("s"):
        add(
            base[:-1]
        )

    # Missing explicit "cell" suffix.
    if (
        base
        and not base.endswith("cell")
        and not base.endswith("cells")
    ):
        add(
            f"{base} cell"
        )

    # Also try suffix after singularization.
    for current in list(variants):
        if (
            current
            and not current.endswith("cell")
            and not current.endswith("cells")
        ):
            add(
                f"{current} cell"
            )

    return variants


def resolve_mapping(
    row: pd.Series,
    *,
    id_to_label: dict[str, str],
    label_index: dict[str, list[str]],
) -> dict[str, object]:

    raw_label = str(
        row["label_raw"]
    )

    role = str(
        row["label_role"]
    )

    exact_ids = extract_ids(
        row["exact_label_matches"]
    )

    # -------------------------------------------------------------
    # 1. UNIQUE EXACT CANONICAL LABEL
    # -------------------------------------------------------------
    if len(exact_ids) == 1:
        curie = exact_ids[0]

        return {
            "accepted_ontology_id": curie,
            "accepted_ontology_label": id_to_label.get(
                curie,
                "",
            ),
            "mapping_method": "EXACT_CANONICAL_LABEL",
            "mapping_status": "RESOLVED",
            "mapping_evidence": (
                "Unique exact canonical label in pinned Cell Ontology."
            ),
        }

    # -------------------------------------------------------------
    # 2. AGENT 2 EXISTING ONTOLOGY ID
    # Only accept when exactly one validated CL ID exists.
    # -------------------------------------------------------------
    if role == "agent2_prediction":
        validated_ids = extract_ids(
            row["existing_agent2_ids_validated"]
        )

        if len(validated_ids) == 1:
            curie = validated_ids[0]

            return {
                "accepted_ontology_id": curie,
                "accepted_ontology_label": id_to_label.get(
                    curie,
                    "",
                ),
                "mapping_method": "UNIQUE_EXISTING_AGENT2_CL_ID",
                "mapping_status": "RESOLVED",
                "mapping_evidence": (
                    "Exactly one existing Agent 2 Cell Ontology ID was "
                    "validated against the pinned ontology."
                ),
            }

        if len(validated_ids) > 1:
            return {
                "accepted_ontology_id": "",
                "accepted_ontology_label": "",
                "mapping_method": "AMBIGUOUS_EXISTING_AGENT2_CL_IDS",
                "mapping_status": "UNRESOLVED",
                "mapping_evidence": (
                    "Multiple valid Agent 2 Cell Ontology IDs were present; "
                    "no ontology term was selected automatically."
                ),
            }

    # -------------------------------------------------------------
    # 3. REFERENCE GRAMMATICAL CANONICALIZATION
    # -------------------------------------------------------------
    if role == "primary_reference":
        if DISEASE_OR_COMPARTMENT_PATTERN.search(
            raw_label
        ):
            return {
                "accepted_ontology_id": "",
                "accepted_ontology_label": "",
                "mapping_method": "DISEASE_OR_COMPARTMENT_TERM",
                "mapping_status": "UNRESOLVED",
                "mapping_evidence": (
                    "Reference appears to describe tumor/disease/compartment "
                    "rather than an unambiguous Cell Ontology cell type."
                ),
            }

        candidate_ids: list[str] = []
        matched_variants: list[str] = []

        for variant in grammatical_variants(
            raw_label
        ):
            ids = list(
                dict.fromkeys(
                    label_index.get(
                        variant,
                        [],
                    )
                )
            )

            if len(ids) == 1:
                candidate_ids.append(
                    ids[0]
                )
                matched_variants.append(
                    variant
                )

        unique_ids = list(
            dict.fromkeys(
                candidate_ids
            )
        )

        if len(unique_ids) == 1:
            curie = unique_ids[0]

            return {
                "accepted_ontology_id": curie,
                "accepted_ontology_label": id_to_label.get(
                    curie,
                    "",
                ),
                "mapping_method": "SAFE_GRAMMATICAL_CANONICALIZATION",
                "mapping_status": "RESOLVED",
                "mapping_evidence": (
                    "Reference label resolved by conservative lexical "
                    "normalization only; matched canonical variant(s): "
                    + " | ".join(
                        matched_variants
                    )
                ),
            }

        if len(unique_ids) > 1:
            return {
                "accepted_ontology_id": "",
                "accepted_ontology_label": "",
                "mapping_method": "AMBIGUOUS_GRAMMATICAL_CANONICALIZATION",
                "mapping_status": "UNRESOLVED",
                "mapping_evidence": (
                    "Conservative grammatical normalization produced "
                    "multiple canonical Cell Ontology IDs."
                ),
            }

    # -------------------------------------------------------------
    # 4. DO NOT AUTO-ACCEPT SYNONYM-ONLY OR FUZZY MATCHES
    # -------------------------------------------------------------
    return {
        "accepted_ontology_id": "",
        "accepted_ontology_label": "",
        "mapping_method": "NO_SAFE_AUTOMATIC_MAPPING",
        "mapping_status": "UNRESOLVED",
        "mapping_evidence": (
            "No unique exact/canonical mapping was available. "
            "Synonym-only and fuzzy suggestions were not auto-accepted."
        ),
    }


def shortest_ancestor_distance(
    child: str,
    target_ancestor: str,
    parents: dict[str, set[str]],
) -> int | None:

    if child == target_ancestor:
        return 0

    queue: deque[
        tuple[str, int]
    ] = deque(
        [(child, 0)]
    )

    visited = {
        child
    }

    while queue:
        current, distance = queue.popleft()

        for parent in parents.get(
            current,
            set(),
        ):
            if parent == target_ancestor:
                return distance + 1

            if parent not in visited:
                visited.add(parent)
                queue.append(
                    (
                        parent,
                        distance + 1,
                    )
                )

    return None


def classify_relation(
    prediction_id: str,
    reference_id: str,
    parents: dict[str, set[str]],
) -> tuple[str, int | None]:

    if (
        not prediction_id
        or not reference_id
    ):
        return (
            "unmapped",
            None,
        )

    if prediction_id == reference_id:
        return (
            "exact",
            0,
        )

    child_distance = shortest_ancestor_distance(
        prediction_id,
        reference_id,
        parents,
    )

    if child_distance is not None:
        return (
            "predicted_child_of_reference",
            child_distance,
        )

    parent_distance = shortest_ancestor_distance(
        reference_id,
        prediction_id,
        parents,
    )

    if parent_distance is not None:
        return (
            "predicted_parent_of_reference",
            parent_distance,
        )

    # "Sibling" is kept deliberately strict:
    # at least one DIRECT asserted is-a parent must be shared.
    prediction_parents = parents.get(
        prediction_id,
        set(),
    )

    reference_parents = parents.get(
        reference_id,
        set(),
    )

    shared_direct_parents = (
        prediction_parents
        & reference_parents
    )

    if shared_direct_parents:
        return (
            "sibling",
            2,
        )

    return (
        "no_match",
        None,
    )


def target_from_policy(
    relation: str,
    target_policy: dict[str, object],
) -> object:

    correct = set(
        target_policy.get(
            "correct_relations",
            [],
        )
    )

    incorrect = set(
        target_policy.get(
            "incorrect_relations",
            [],
        )
    )

    excluded = set(
        target_policy.get(
            "excluded_relations",
            [],
        )
    )

    if relation in correct:
        return 1

    if relation in incorrect:
        return 0

    if relation in excluded:
        return pd.NA

    return pd.NA


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve conservative ontology mappings and derive "
            "ontology-aware evaluation targets."
        )
    )

    parser.add_argument(
        "--mapping-candidates",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--ontology-json",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--ontology-manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--correctness-policy",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/analysis/router_ontology_relations_v1"
        ),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mapping = pd.read_csv(
        args.mapping_candidates
    )

    missing = REQUIRED_MAPPING_COLUMNS - set(
        mapping.columns
    )

    if missing:
        raise ValueError(
            "Mapping candidate table missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    if mapping.duplicated(
        subset=[
            "row_id",
            "label_role",
        ]
    ).any():
        raise ValueError(
            "Duplicate row_id/label_role mapping rows."
        )

    manifest = json.loads(
        args.ontology_manifest.read_text(
            encoding="utf-8"
        )
    )

    policy = json.loads(
        args.correctness_policy.read_text(
            encoding="utf-8"
        )
    )

    missing_policy_keys = REQUIRED_POLICY_KEYS - set(
        policy
    )

    if missing_policy_keys:
        raise ValueError(
            "Correctness policy missing keys: "
            + ", ".join(
                sorted(
                    missing_policy_keys
                )
            )
        )

    expected_sha = str(
        manifest.get(
            "verified_sha256",
            "",
        )
    ).strip().casefold()

    actual_sha = sha256_file(
        args.ontology_json
    ).casefold()

    if (
        not expected_sha
        or actual_sha != expected_sha
    ):
        raise ValueError(
            "Pinned ontology checksum does not match manifest."
        )

    (
        id_to_label,
        label_index,
        parents,
    ) = parse_ontology(
        args.ontology_json
    )

    resolved_rows = []

    for _, row in mapping.iterrows():
        resolution = resolve_mapping(
            row,
            id_to_label=id_to_label,
            label_index=label_index,
        )

        resolved_rows.append(
            {
                "row_id": int(
                    row["row_id"]
                ),
                "dataset": row["dataset"],
                "cluster_id": int(
                    row["cluster_id"]
                ),
                "label_role": row["label_role"],
                "label_raw": row["label_raw"],
                **resolution,
            }
        )

    resolved = pd.DataFrame(
        resolved_rows
    )

    prediction = resolved.loc[
        resolved["label_role"]
        == "agent2_prediction"
    ].copy()

    reference = resolved.loc[
        resolved["label_role"]
        == "primary_reference"
    ].copy()

    prediction = prediction.rename(
        columns={
            "label_raw": "prediction_label_raw",
            "accepted_ontology_id": "prediction_ontology_id",
            "accepted_ontology_label": "prediction_ontology_label",
            "mapping_method": "prediction_mapping_method",
            "mapping_status": "prediction_mapping_status",
            "mapping_evidence": "prediction_mapping_evidence",
        }
    )

    reference = reference.rename(
        columns={
            "label_raw": "reference_label_raw",
            "accepted_ontology_id": "reference_ontology_id",
            "accepted_ontology_label": "reference_ontology_label",
            "mapping_method": "reference_mapping_method",
            "mapping_status": "reference_mapping_status",
            "mapping_evidence": "reference_mapping_evidence",
        }
    )

    pair = prediction[
        [
            "row_id",
            "dataset",
            "cluster_id",
            "prediction_label_raw",
            "prediction_ontology_id",
            "prediction_ontology_label",
            "prediction_mapping_method",
            "prediction_mapping_status",
            "prediction_mapping_evidence",
        ]
    ].merge(
        reference[
            [
                "row_id",
                "reference_label_raw",
                "reference_ontology_id",
                "reference_ontology_label",
                "reference_mapping_method",
                "reference_mapping_status",
                "reference_mapping_evidence",
            ]
        ],
        on="row_id",
        how="inner",
        validate="one_to_one",
    )

    relations = []

    for _, row in pair.iterrows():
        relation, distance = classify_relation(
            str(
                row["prediction_ontology_id"]
                or ""
            ),
            str(
                row["reference_ontology_id"]
                or ""
            ),
            parents,
        )

        relations.append(
            (
                relation,
                distance,
            )
        )

    pair["ontology_relation"] = [
        relation
        for relation, _ in relations
    ]

    pair["ontology_distance"] = [
        distance
        for _, distance in relations
    ]

    targets = policy["targets"]

    pair["strict_exact_target"] = pair[
        "ontology_relation"
    ].map(
        lambda relation: target_from_policy(
            relation,
            targets["strict_exact"],
        )
    )

    pair[
        "primary_router_sufficiency_target"
    ] = pair[
        "ontology_relation"
    ].map(
        lambda relation: target_from_policy(
            relation,
            targets[
                "primary_router_sufficiency"
            ],
        )
    )

    pair[
        "permissive_hierarchy_target"
    ] = pair[
        "ontology_relation"
    ].map(
        lambda relation: target_from_policy(
            relation,
            targets[
                "permissive_hierarchy"
            ],
        )
    )

    pair["target_eligible"] = (
        pair[
            "primary_router_sufficiency_target"
        ]
        .notna()
        .astype(int)
    )

    pair["target_role"] = (
        "DEVELOPMENT_EVALUATION_ONLY"
    )

    pair["router_inference_input"] = False

    mapping_file = (
        args.output_dir
        / "resolved_ontology_mappings_v1.csv"
    )

    relation_file = (
        args.output_dir
        / "router_ontology_relations_and_targets_v1.csv"
    )

    audit_file = (
        args.output_dir
        / "router_ontology_relations_audit_v1.json"
    )

    resolved.to_csv(
        mapping_file,
        index=False,
    )

    pair.to_csv(
        relation_file,
        index=False,
    )

    relation_counts = (
        pair["ontology_relation"]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    audit = {
        "ontology_release_version": manifest.get(
            "release_version"
        ),
        "ontology_release_tag": manifest.get(
            "release_tag"
        ),
        "ontology_sha256": actual_sha,
        "ontology_checksum_against_manifest": "PASS",
        "correctness_policy_version": policy.get(
            "policy_version"
        ),
        "cluster_pairs_processed": int(
            len(pair)
        ),
        "resolved_prediction_mappings": int(
            (
                pair[
                    "prediction_mapping_status"
                ]
                == "RESOLVED"
            ).sum()
        ),
        "resolved_reference_mappings": int(
            (
                pair[
                    "reference_mapping_status"
                ]
                == "RESOLVED"
            ).sum()
        ),
        "primary_target_eligible_rows": int(
            pair["target_eligible"].sum()
        ),
        "primary_target_excluded_rows": int(
            (
                pair["target_eligible"]
                == 0
            ).sum()
        ),
        "ontology_relation_counts": {
            str(key): int(value)
            for key, value in relation_counts.items()
        },
        "synonym_only_matches_auto_accepted": False,
        "fuzzy_matches_auto_accepted": False,
        "ambiguous_multiple_ids_auto_resolved": False,
        "router_model_trained": False,
        "router_threshold_selected": False,
        "routing_decisions_created": False,
        "existing_agent_outputs_modified": False,
    }

    audit_file.write_text(
        json.dumps(
            audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Resolved ontology mappings: {mapping_file}"
    )

    print(
        f"Ontology relations and targets: {relation_file}"
    )

    print(
        f"Ontology relation audit: {audit_file}"
    )

    print(
        f"Cluster pairs processed: {len(pair)}"
    )

    print(
        f"Primary target eligible rows: {int(pair['target_eligible'].sum())}"
    )

    print(
        "Synonym-only/fuzzy matches were NOT auto-accepted."
    )

    print(
        "Ambiguous multiple ontology IDs were NOT auto-resolved."
    )

    print(
        "Evaluation targets were created only for ontology-resolvable "
        "reference/prediction pairs."
    )

    print(
        "No Router model, threshold, or routing decision was created."
    )

    print(
        "No existing Agent output or annotation was changed."
    )


if __name__ == "__main__":
    main()
