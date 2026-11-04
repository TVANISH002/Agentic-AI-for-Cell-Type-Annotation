from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import deque
from pathlib import Path

import pandas as pd

# Audit the ontology relation labels created in Step 14 before they are ever
# used for Router training.


REQUIRED_COLUMNS = {
    "row_id",
    "dataset",
    "cluster_id",
    "prediction_label_raw",
    "prediction_ontology_id",
    "reference_label_raw",
    "reference_ontology_id",
    "ontology_relation",
    "strict_exact_target",
    "primary_router_sufficiency_target",
    "permissive_hierarchy_target",
    "target_eligible",
}


CL_PATTERN = re.compile(
    r"CL[:_](\d+)",
    flags=re.IGNORECASE,
)


def normalize_cl_id(value: object) -> str:
    if value is None or pd.isna(value):
        return ""

    match = CL_PATTERN.search(
        str(value)
    )

    if not match:
        return ""

    return f"CL:{match.group(1)}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def parse_ontology(
    ontology_path: Path,
) -> tuple[
    dict[str, str],
    dict[str, set[str]],
]:

    with ontology_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)

    graphs = payload.get(
        "graphs",
        [],
    )

    if not graphs:
        raise ValueError(
            "Ontology JSON contains no graphs."
        )

    labels: dict[str, str] = {}
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

            labels[curie] = str(
                node.get("lbl") or ""
            ).strip()

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

    if not labels:
        raise ValueError(
            "No Cell Ontology nodes parsed."
        )

    return labels, parents


def ancestor_distances(
    node: str,
    parents: dict[str, set[str]],
) -> dict[str, int]:

    if not node:
        return {}

    distances = {
        node: 0
    }

    queue: deque[
        tuple[str, int]
    ] = deque(
        [(node, 0)]
    )

    while queue:
        current, distance = queue.popleft()

        for parent in parents.get(
            current,
            set(),
        ):
            proposed = distance + 1

            if (
                parent not in distances
                or proposed < distances[parent]
            ):
                distances[parent] = proposed

                queue.append(
                    (
                        parent,
                        proposed,
                    )
                )

    return distances


def shortest_upward_path(
    child: str,
    ancestor: str,
    parents: dict[str, set[str]],
) -> list[str] | None:

    if (
        not child
        or not ancestor
    ):
        return None

    if child == ancestor:
        return [child]

    queue: deque[
        tuple[
            str,
            list[str],
        ]
    ] = deque(
        [
            (
                child,
                [child],
            )
        ]
    )

    visited = {
        child
    }

    while queue:
        current, path = queue.popleft()

        for parent in parents.get(
            current,
            set(),
        ):
            new_path = path + [
                parent
            ]

            if parent == ancestor:
                return new_path

            if parent not in visited:
                visited.add(parent)

                queue.append(
                    (
                        parent,
                        new_path,
                    )
                )

    return None


def format_path(
    path: list[str] | None,
    labels: dict[str, str],
) -> str:
    if not path:
        return ""

    return " -> ".join(
        f"{curie}::{labels.get(curie, '')}"
        for curie in path
    )


def nearest_shared_ancestors(
    prediction_id: str,
    reference_id: str,
    parents: dict[str, set[str]],
    labels: dict[str, str],
    top_n: int = 5,
) -> list[dict[str, object]]:

    pred_dist = ancestor_distances(
        prediction_id,
        parents,
    )

    ref_dist = ancestor_distances(
        reference_id,
        parents,
    )

    shared = set(
        pred_dist
    ) & set(
        ref_dist
    )

    # The prediction/reference nodes themselves have already been handled by
    # exact/ancestor checks. Remove them from shared-ancestor candidates.
    shared.discard(
        prediction_id
    )

    shared.discard(
        reference_id
    )

    candidates = []

    for ancestor in shared:
        d_pred = int(
            pred_dist[ancestor]
        )

        d_ref = int(
            ref_dist[ancestor]
        )

        candidates.append(
            {
                "ancestor_id": ancestor,
                "ancestor_label": labels.get(
                    ancestor,
                    "",
                ),
                "prediction_distance": d_pred,
                "reference_distance": d_ref,
                "distance_sum": (
                    d_pred + d_ref
                ),
                "max_branch_distance": max(
                    d_pred,
                    d_ref,
                ),
            }
        )

    # No arbitrary biological cutoff is used.
    # We only rank shared ancestors by graph proximity.
    candidates.sort(
        key=lambda item: (
            item[
                "distance_sum"
            ],
            item[
                "max_branch_distance"
            ],
            item[
                "ancestor_id"
            ],
        )
    )

    return candidates[
        :top_n
    ]


def format_shared(
    shared: list[
        dict[str, object]
    ],
) -> str:

    return " | ".join(
        (
            f"{item['ancestor_id']}::{item['ancestor_label']}"
            f"::pred_d={item['prediction_distance']}"
            f"::ref_d={item['reference_distance']}"
            f"::sum={item['distance_sum']}"
        )
        for item in shared
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit ontology relation paths and nearest shared ancestors."
        )
    )

    parser.add_argument(
        "--relations",
        type=Path,
        required=True,
        help=(
            "router_ontology_relations_and_targets_v1.csv"
        ),
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
        "--output-dir",
        type=Path,
        default=Path(
            "data/analysis/router_ontology_path_audit_v1"
        ),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    relations = pd.read_csv(
        args.relations
    )

    missing = REQUIRED_COLUMNS - set(
        relations.columns
    )

    if missing:
        raise ValueError(
            "Relations table missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
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

    if (
        not expected_sha
        or actual_sha != expected_sha
    ):
        raise ValueError(
            "Ontology checksum does not match manifest."
        )

    labels, parents = parse_ontology(
        args.ontology_json
    )

    output_rows = []

    for _, row in relations.iterrows():
        prediction_id = normalize_cl_id(
            row[
                "prediction_ontology_id"
            ]
        )

        reference_id = normalize_cl_id(
            row[
                "reference_ontology_id"
            ]
        )

        relation = str(
            row[
                "ontology_relation"
            ]
        ).strip()

        path = None

        if relation == "predicted_child_of_reference":
            path = shortest_upward_path(
                prediction_id,
                reference_id,
                parents,
            )

        elif relation == "predicted_parent_of_reference":
            # Traverse from the more-specific reference upward to prediction.
            path = shortest_upward_path(
                reference_id,
                prediction_id,
                parents,
            )

        elif relation == "exact":
            path = (
                [prediction_id]
                if prediction_id
                else None
            )

        shared = []

        if (
            prediction_id
            and reference_id
            and relation
            in {
                "sibling",
                "no_match",
            }
        ):
            shared = nearest_shared_ancestors(
                prediction_id,
                reference_id,
                parents,
                labels,
                top_n=5,
            )

        nearest_shared = (
            shared[0]
            if shared
            else None
        )

        if relation == "unmapped":
            review_flag = (
                "MAPPING_REVIEW_REQUIRED"
            )

        elif relation == "no_match":
            review_flag = (
                "SHARED_ANCESTRY_REVIEW_REQUIRED"
            )

        elif relation == "sibling":
            review_flag = (
                "SIBLING_RELATION_REVIEW"
            )

        else:
            review_flag = (
                "ONTOLOGY_PATH_AUDIT_AVAILABLE"
            )

        output_rows.append(
            {
                "row_id": int(
                    row["row_id"]
                ),
                "dataset": row["dataset"],
                "cluster_id": int(
                    row["cluster_id"]
                ),
                "prediction_label_raw": row[
                    "prediction_label_raw"
                ],
                "prediction_ontology_id": prediction_id,
                "prediction_ontology_label": labels.get(
                    prediction_id,
                    "",
                ),
                "reference_label_raw": row[
                    "reference_label_raw"
                ],
                "reference_ontology_id": reference_id,
                "reference_ontology_label": labels.get(
                    reference_id,
                    "",
                ),
                "step14_relation": relation,
                "step14_ontology_distance": row[
                    "ontology_distance"
                ],
                "shortest_relation_path": format_path(
                    path,
                    labels,
                ),
                "nearest_shared_ancestor_id": (
                    nearest_shared[
                        "ancestor_id"
                    ]
                    if nearest_shared
                    else ""
                ),
                "nearest_shared_ancestor_label": (
                    nearest_shared[
                        "ancestor_label"
                    ]
                    if nearest_shared
                    else ""
                ),
                "prediction_to_shared_distance": (
                    nearest_shared[
                        "prediction_distance"
                    ]
                    if nearest_shared
                    else pd.NA
                ),
                "reference_to_shared_distance": (
                    nearest_shared[
                        "reference_distance"
                    ]
                    if nearest_shared
                    else pd.NA
                ),
                "top5_shared_ancestors": format_shared(
                    shared
                ),
                "relation_review_flag": review_flag,
                "strict_exact_target_step14": row[
                    "strict_exact_target"
                ],
                "primary_router_target_step14": row[
                    "primary_router_sufficiency_target"
                ],
                "permissive_target_step14": row[
                    "permissive_hierarchy_target"
                ],
                "targets_modified_by_step15": False,
            }
        )

    output = pd.DataFrame(
        output_rows
    )

    audit_file = (
        args.output_dir
        / "ontology_path_audit_v1.csv"
    )

    summary_file = (
        args.output_dir
        / "ontology_path_audit_summary_v1.json"
    )

    output.to_csv(
        audit_file,
        index=False,
    )

    no_match_rows = output.loc[
        output[
            "step14_relation"
        ] == "no_match"
    ]

    summary = {
        "rows_processed": int(
            len(output)
        ),
        "ontology_sha256": actual_sha,
        "ontology_checksum_against_manifest": "PASS",
        "step14_targets_modified": False,
        "no_match_rows": int(
            len(no_match_rows)
        ),
        "no_match_rows_with_shared_ancestor": int(
            no_match_rows[
                "nearest_shared_ancestor_id"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        ),
        "unmapped_rows": int(
            (
                output[
                    "step14_relation"
                ]
                == "unmapped"
            ).sum()
        ),
        "router_model_trained": False,
        "router_threshold_selected": False,
        "routing_decisions_created": False,
        "existing_agent_outputs_modified": False,
        "important_interpretation": (
            "A Step-14 'no_match' must not be interpreted as biological "
            "unrelatedness. This audit reports shared ancestry without "
            "changing the frozen Step-14 target."
        ),
    }

    summary_file.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Ontology path audit: {audit_file}"
    )

    print(
        f"Ontology path audit summary: {summary_file}"
    )

    print(
        f"Rows processed: {len(output)}"
    )

    print(
        f"Step-14 no_match rows: {len(no_match_rows)}"
    )

    print(
        "Step-14 targets were NOT modified."
    )

    print(
        "No Router model, threshold, or routing decision was created."
    )

    print(
        "No existing Agent output or annotation was changed."
    )


if __name__ == "__main__":
    main()
