from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd



# Generate auditable mapping CANDIDATES for:
#   - Agent 2 top-ranked cell-type labels
#   - independent reference labels

# MAPPING PRIORITY
# 1. Validate an existing Agent 2 Cell Ontology ID when one is available.
# 2. Search normalized exact ontology labels.
# 3. Search normalized exact ontology synonyms.
# 4. If still unresolved, provide top lexical SUGGESTIONS for human review.

# Lexical suggestions are never treated as accepted ontology mappings.



REQUIRED_REVIEW_COLUMNS = {
    "row_id",
    "dataset",
    "cluster_id",
    "agent2_top1_candidate",
    "agent2_top1_cell_ontology_ids",
    "reference_label_raw",
    "reference_type",
}


CL_PATTERN = re.compile(
    r"(?:CL[:_])(\d+)",
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

    text = unicodedata.normalize(
        "NFKC",
        str(value),
    ).casefold()

    text = text.replace(
        "_",
        " ",
    ).replace(
        "-",
        " ",
    )

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def normalize_cl_id(value: object) -> str:
    if value is None:
        return ""

    text = str(value)

    match = CL_PATTERN.search(
        text
    )

    if not match:
        return ""

    return f"CL:{match.group(1)}"


def extract_candidate_cl_ids(value: object) -> list[str]:
    if pd.isna(value):
        return []

    found = []

    for match in CL_PATTERN.finditer(
        str(value)
    ):
        curie = f"CL:{match.group(1)}"

        if curie not in found:
            found.append(
                curie
            )

    return found


def parse_synonyms(meta: object) -> list[str]:
    if not isinstance(meta, dict):
        return []

    raw = meta.get(
        "synonyms",
        [],
    )

    synonyms: list[str] = []

    if not isinstance(raw, list):
        return synonyms

    for item in raw:
        if isinstance(item, dict):
            value = item.get(
                "val"
            )

            if value:
                synonyms.append(
                    str(value)
                )

        elif isinstance(item, str):
            synonyms.append(
                item
            )

    return synonyms


def load_ontology(
    ontology_path: Path,
) -> tuple[
    dict[str, str],
    dict[str, list[str]],
    dict[str, list[str]],
    dict[str, list[str]],
]:

    with ontology_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(
            handle
        )

    graphs = payload.get(
        "graphs",
        []
    )

    if not graphs:
        raise ValueError(
            "Ontology JSON does not contain a 'graphs' collection."
        )

    id_to_label: dict[str, str] = {}
    id_to_synonyms: dict[str, list[str]] = {}

    label_index: dict[
        str,
        list[str],
    ] = {}

    synonym_index: dict[
        str,
        list[str],
    ] = {}

    for graph in graphs:
        nodes = graph.get(
            "nodes",
            []
        )

        for node in nodes:
            raw_id = node.get(
                "id",
                "",
            )

            cl_id = normalize_cl_id(
                raw_id
            )

            if not cl_id:
                continue

            label = str(
                node.get(
                    "lbl",
                    "",
                )
                or ""
            ).strip()

            synonyms = parse_synonyms(
                node.get(
                    "meta",
                    {},
                )
            )

            id_to_label[
                cl_id
            ] = label

            id_to_synonyms[
                cl_id
            ] = synonyms

            normalized_label = normalize_text(
                label
            )

            if normalized_label:
                label_index.setdefault(
                    normalized_label,
                    [],
                ).append(
                    cl_id
                )

            for synonym in synonyms:
                normalized_synonym = normalize_text(
                    synonym
                )

                if normalized_synonym:
                    synonym_index.setdefault(
                        normalized_synonym,
                        [],
                    ).append(
                        cl_id
                    )

    if not id_to_label:
        raise ValueError(
            "No CL nodes were extracted from ontology JSON."
        )

    return (
        id_to_label,
        id_to_synonyms,
        label_index,
        synonym_index,
    )


def format_ids_with_labels(
    ids: list[str],
    id_to_label: dict[str, str],
) -> str:
    parts = []

    for curie in ids:
        label = id_to_label.get(
            curie,
            "",
        )

        if label:
            parts.append(
                f"{curie}::{label}"
            )
        else:
            parts.append(
                curie
            )

    return " | ".join(
        parts
    )


def build_search_entries(
    id_to_label: dict[str, str],
    id_to_synonyms: dict[str, list[str]],
) -> dict[str, tuple[str, str, str]]:
    # normalized term ->
    #   (CL ID, canonical label, matched surface form)
    #
    # If the same lexical form belongs to multiple ontology IDs, keep every
    # candidate in a list outside this helper by using unique composite keys.

    entries: dict[
        str,
        tuple[str, str, str],
    ] = {}

    counter = 0

    for curie, label in id_to_label.items():
        norm = normalize_text(
            label
        )

        if norm:
            key = f"{norm}||{counter}"
            entries[
                key
            ] = (
                curie,
                label,
                label,
            )
            counter += 1

        for synonym in id_to_synonyms.get(
            curie,
            [],
        ):
            norm_syn = normalize_text(
                synonym
            )

            if norm_syn:
                key = f"{norm_syn}||{counter}"
                entries[
                    key
                ] = (
                    curie,
                    label,
                    synonym,
                )
                counter += 1

    return entries


def lexical_suggestions(
    raw_label: str,
    search_entries: dict[
        str,
        tuple[str, str, str],
    ],
    top_n: int = 5,
) -> list[
    tuple[float, str, str, str]
]:
    query = normalize_text(
        raw_label
    )

    if not query:
        return []

    scored: list[
        tuple[
            float,
            str,
            str,
            str,
        ]
    ] = []

    for composite_key, (
        curie,
        canonical_label,
        surface_form,
    ) in search_entries.items():

        candidate_norm = composite_key.split(
            "||",
            1,
        )[0]

        score = difflib.SequenceMatcher(
            None,
            query,
            candidate_norm,
        ).ratio()

        scored.append(
            (
                score,
                curie,
                canonical_label,
                surface_form,
            )
        )

    scored.sort(
        key=lambda x: (
            -x[0],
            x[1],
            x[2],
        )
    )

    unique: list[
        tuple[
            float,
            str,
            str,
            str,
        ]
    ] = []

    seen: set[str] = set()

    for item in scored:
        curie = item[1]

        if curie in seen:
            continue

        seen.add(
            curie
        )

        unique.append(
            item
        )

        if len(unique) >= top_n:
            break

    return unique


def analyze_label(
    *,
    row_id: int,
    dataset: str,
    cluster_id: int,
    label_role: str,
    raw_label: str,
    existing_ontology_ids: object,
    id_to_label: dict[str, str],
    label_index: dict[str, list[str]],
    synonym_index: dict[str, list[str]],
    search_entries: dict[
        str,
        tuple[str, str, str],
    ],
) -> dict[str, object]:

    normalized = normalize_text(
        raw_label
    )

    exact_label_ids = list(
        dict.fromkeys(
            label_index.get(
                normalized,
                [],
            )
        )
    )

    exact_synonym_ids = list(
        dict.fromkeys(
            synonym_index.get(
                normalized,
                [],
            )
        )
    )

    existing_ids = (
        extract_candidate_cl_ids(
            existing_ontology_ids
        )
        if label_role == "agent2_prediction"
        else []
    )

    valid_existing_ids = [
        curie
        for curie in existing_ids
        if curie in id_to_label
    ]

    invalid_existing_ids = [
        curie
        for curie in existing_ids
        if curie not in id_to_label
    ]

    if existing_ids:
        existing_validation = (
            "PASS"
            if not invalid_existing_ids
            else "PARTIAL_OR_FAIL"
        )
    else:
        existing_validation = "ABSENT"

    if exact_label_ids:
        lexical_status = (
            "EXACT_LABEL_MATCH"
            if len(exact_label_ids) == 1
            else "AMBIGUOUS_EXACT_LABEL_MATCH"
        )

    elif exact_synonym_ids:
        lexical_status = (
            "EXACT_SYNONYM_MATCH"
            if len(exact_synonym_ids) == 1
            else "AMBIGUOUS_EXACT_SYNONYM_MATCH"
        )

    else:
        lexical_status = (
            "NO_EXACT_MATCH_MANUAL_REVIEW"
        )

    suggestions = []

    if not exact_label_ids and not exact_synonym_ids:
        suggestions = lexical_suggestions(
            raw_label,
            search_entries,
            top_n=5,
        )

    suggestion_text = " | ".join(
        (
            f"{curie}::{label}"
            f"::surface={surface}"
            f"::similarity={score:.4f}"
        )
        for (
            score,
            curie,
            label,
            surface,
        ) in suggestions
    )

    return {
        "row_id": row_id,
        "dataset": dataset,
        "cluster_id": cluster_id,
        "label_role": label_role,
        "label_raw": raw_label,
        "label_normalized": normalized,
        "existing_agent2_ontology_ids_raw": (
            str(existing_ontology_ids)
            if label_role == "agent2_prediction"
            and not pd.isna(existing_ontology_ids)
            else ""
        ),
        "existing_agent2_ids_validated": (
            format_ids_with_labels(
                valid_existing_ids,
                id_to_label,
            )
        ),
        "existing_agent2_ids_invalid": (
            " | ".join(
                invalid_existing_ids
            )
        ),
        "existing_agent2_id_validation": (
            existing_validation
        ),
        "exact_label_matches": (
            format_ids_with_labels(
                exact_label_ids,
                id_to_label,
            )
        ),
        "exact_synonym_matches": (
            format_ids_with_labels(
                exact_synonym_ids,
                id_to_label,
            )
        ),
        "lexical_mapping_status": lexical_status,
        "top5_lexical_suggestions_review_only": suggestion_text,
        "accepted_ontology_id": "",
        "accepted_ontology_label": "",
        "mapping_decision": "NOT_ASSIGNED",
        "mapping_reviewer": "",
        "mapping_evidence": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate review-only Cell Ontology mapping candidates "
            "for Router reference harmonization."
        )
    )

    parser.add_argument(
        "--review-template",
        type=Path,
        required=True,
        help=(
            "router_harmonization_review_template_v1.csv"
        ),
    )

    parser.add_argument(
        "--ontology-json",
        type=Path,
        required=True,
        help="Pinned cl-base.json",
    )

    parser.add_argument(
        "--ontology-manifest",
        type=Path,
        required=True,
        help="cell_ontology_manifest.json",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/analysis/router_ontology_mapping_v1"
        ),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    review = pd.read_csv(
        args.review_template
    )

    missing = REQUIRED_REVIEW_COLUMNS - set(
        review.columns
    )

    if missing:
        raise ValueError(
            "Review template missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    if review["row_id"].duplicated().any():
        raise ValueError(
            "Duplicate row_id values in review template."
        )

    manifest = json.loads(
        args.ontology_manifest.read_text(
            encoding="utf-8"
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

    if not expected_sha:
        raise ValueError(
            "Ontology manifest does not contain verified_sha256."
        )

    if actual_sha != expected_sha:
        raise ValueError(
            "Ontology checksum does not match manifest.\n"
            f"Manifest: {expected_sha}\n"
            f"Actual:   {actual_sha}"
        )

    (
        id_to_label,
        id_to_synonyms,
        label_index,
        synonym_index,
    ) = load_ontology(
        args.ontology_json
    )

    search_entries = build_search_entries(
        id_to_label,
        id_to_synonyms,
    )

    rows: list[
        dict[str, object]
    ] = []

    for _, record in review.iterrows():
        common = {
            "row_id": int(
                record["row_id"]
            ),
            "dataset": str(
                record["dataset"]
            ),
            "cluster_id": int(
                record["cluster_id"]
            ),
        }

        rows.append(
            analyze_label(
                **common,
                label_role="agent2_prediction",
                raw_label=str(
                    record[
                        "agent2_top1_candidate"
                    ]
                ),
                existing_ontology_ids=record[
                    "agent2_top1_cell_ontology_ids"
                ],
                id_to_label=id_to_label,
                label_index=label_index,
                synonym_index=synonym_index,
                search_entries=search_entries,
            )
        )

        rows.append(
            analyze_label(
                **common,
                label_role="primary_reference",
                raw_label=str(
                    record[
                        "reference_label_raw"
                    ]
                ),
                existing_ontology_ids="",
                id_to_label=id_to_label,
                label_index=label_index,
                synonym_index=synonym_index,
                search_entries=search_entries,
            )
        )

    output = pd.DataFrame(
        rows
    )

    mapping_file = (
        args.output_dir
        / "router_ontology_mapping_candidates_v1.csv"
    )

    audit_file = (
        args.output_dir
        / "router_ontology_mapping_audit_v1.json"
    )

    mapping_file.write_text(
        output.to_csv(
            index=False
        ),
        encoding="utf-8",
    )

    audit = {
        "review_rows": int(
            len(review)
        ),
        "label_rows_generated": int(
            len(output)
        ),
        "ontology_release_version": manifest.get(
            "release_version"
        ),
        "ontology_release_tag": manifest.get(
            "release_tag"
        ),
        "ontology_sha256": actual_sha,
        "ontology_checksum_against_manifest": "PASS",
        "cell_ontology_nodes_indexed": int(
            len(id_to_label)
        ),
        "exact_label_match_rows": int(
            (
                output[
                    "lexical_mapping_status"
                ]
                == "EXACT_LABEL_MATCH"
            ).sum()
        ),
        "exact_synonym_match_rows": int(
            (
                output[
                    "lexical_mapping_status"
                ]
                == "EXACT_SYNONYM_MATCH"
            ).sum()
        ),
        "rows_requiring_manual_mapping_review": int(
            output[
                "lexical_mapping_status"
            ]
            .str.contains(
                "MANUAL_REVIEW",
                na=False,
            )
            .sum()
        ),
        "lexical_suggestions_auto_accepted": False,
        "ontology_relations_assigned": False,
        "correctness_targets_created": False,
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
        f"Ontology mapping candidates: {mapping_file}"
    )

    print(
        f"Ontology mapping audit: {audit_file}"
    )

    print(
        f"Label rows generated: {len(output)}"
    )

    print(
        f"Cell Ontology nodes indexed: {len(id_to_label)}"
    )

    print(
        "Ontology checksum against manifest: PASS"
    )

    print(
        "Lexical suggestions were NOT auto-accepted."
    )

    print(
        "No ontology relation, correctness target, Router model, "
        "threshold, or routing decision was created."
    )

    print(
        "No existing Agent output or annotation was changed."
    )


if __name__ == "__main__":
    main()
