from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any



# This script reads an existing Agent 2 ranked-candidate CSV and creates:
#
#   1. agent2_evidence_features.csv
#      -> only information available from Agent 2
#      -> safe for Router development
#   2. agent2_evaluation_view.csv
#      -> optional joined view with expert/LLM annotations
#      -> evaluation only
#      -> must NEVER be fed into Agent 1, Agent 2, Router, or Agent 3
#


CLUSTER_ALIASES = (
    "cluster_id",
    "cluster",
    "spatial_cluster",
    "group",
)

CANDIDATE_ALIASES = (
    "candidate_cell_type",
    "candidate",
    "cell_type",
    "normalized_cell_type",
    "annotation",
    "candidate_label",
)

RANK_ALIASES = (
    "candidate_rank",
    "rank",
    "cell_type_rank",
)

SUPPORT_COUNT_ALIASES = (
    "unique_supporting_gene_count",
    "supporting_gene_count",
    "support_count",
    "unique_gene_count",
    "n_supporting_genes",
)

SUPPORTING_GENES_ALIASES = (
    "supporting_genes",
    "support_genes",
    "marker_genes",
    "supporting_marker_genes",
)

REVIEW_FLAG_ALIASES = (
    "review_flag",
    "requires_review",
    "needs_review",
    "manual_review",
)

WARNING_ALIASES = (
    "warning_codes",
    "warnings",
    "review_reasons",
    "flags",
)

SOURCE_ALIASES = (
    "evidence_sources",
    "sources",
    "source_summary",
)

CROSS_DATABASE_ALIASES = (
    "both_source",
    "both_source_supporting_gene_count",
    "cross_database_support",
    "both_sources_count",
)


def normalize_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]

    if not fieldnames:
        raise ValueError(f"No header row found in {path}")

    return rows, fieldnames


def choose_column(
    fieldnames: list[str],
    aliases: tuple[str, ...],
    *,
    required: bool,
    semantic_name: str,
) -> str | None:
    normalized = {
        normalize_column(name): name
        for name in fieldnames
    }

    for alias in aliases:
        if alias in normalized:
            return normalized[alias]

    if required:
        raise ValueError(
            f"Could not find a column for {semantic_name!r}. "
            f"Available columns: {fieldnames}"
        )

    return None


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def to_float(value: Any) -> float | None:
    text = clean_text(value)

    if not text:
        return None

    try:
        result = float(text)
    except ValueError:
        return None

    if not math.isfinite(result):
        return None

    return result


def to_int(value: Any) -> int | None:
    number = to_float(value)

    if number is None:
        return None

    return int(number)


def parse_bool(value: Any) -> bool | None:
    text = clean_text(value).casefold()

    if text in {"true", "1", "yes", "y"}:
        return True

    if text in {"false", "0", "no", "n"}:
        return False

    return None


def natural_cluster_key(value: str) -> tuple[int, str]:
    text = clean_text(value)

    try:
        return (0, f"{int(float(text)):010d}")
    except ValueError:
        return (1, text)


def derive_ranked_rows(
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:

    cluster_col = choose_column(
        fieldnames,
        CLUSTER_ALIASES,
        required=True,
        semantic_name="cluster identifier",
    )

    candidate_col = choose_column(
        fieldnames,
        CANDIDATE_ALIASES,
        required=True,
        semantic_name="candidate cell type",
    )

    rank_col = choose_column(
        fieldnames,
        RANK_ALIASES,
        required=False,
        semantic_name="candidate rank",
    )

    support_col = choose_column(
        fieldnames,
        SUPPORT_COUNT_ALIASES,
        required=True,
        semantic_name="supporting-gene count",
    )

    genes_col = choose_column(
        fieldnames,
        SUPPORTING_GENES_ALIASES,
        required=False,
        semantic_name="supporting genes",
    )

    review_col = choose_column(
        fieldnames,
        REVIEW_FLAG_ALIASES,
        required=False,
        semantic_name="review flag",
    )

    warning_col = choose_column(
        fieldnames,
        WARNING_ALIASES,
        required=False,
        semantic_name="warning codes",
    )

    source_col = choose_column(
        fieldnames,
        SOURCE_ALIASES,
        required=False,
        semantic_name="evidence sources",
    )

    cross_db_col = choose_column(
        fieldnames,
        CROSS_DATABASE_ALIASES,
        required=False,
        semantic_name="cross-database support",
    )

    normalized_rows: list[dict[str, Any]] = []

    for original_index, row in enumerate(rows, start=1):
        cluster_id = clean_text(row.get(cluster_col))
        candidate = clean_text(row.get(candidate_col))
        support_count = to_int(row.get(support_col))

        if not cluster_id:
            continue

        if not candidate:
            continue

        if support_count is None:
            continue

        candidate_rank = (
            to_int(row.get(rank_col))
            if rank_col is not None
            else None
        )

        normalized_rows.append(
            {
                "cluster_id": cluster_id,
                "candidate_cell_type": candidate,
                "candidate_rank": candidate_rank,
                "support_count": support_count,
                "supporting_genes": (
                    clean_text(row.get(genes_col))
                    if genes_col is not None
                    else ""
                ),
                "review_flag": (
                    parse_bool(row.get(review_col))
                    if review_col is not None
                    else None
                ),
                "warning_codes": (
                    clean_text(row.get(warning_col))
                    if warning_col is not None
                    else ""
                ),
                "evidence_sources": (
                    clean_text(row.get(source_col))
                    if source_col is not None
                    else ""
                ),
                "cross_database_support": (
                    to_int(row.get(cross_db_col))
                    if cross_db_col is not None
                    else None
                ),
                "_original_index": original_index,
            }
        )

    # If an explicit candidate rank exists, use it.
    # Otherwise rank deterministically by descending support count.
    by_cluster: dict[str, list[dict[str, Any]]] = {}

    for row in normalized_rows:
        by_cluster.setdefault(
            row["cluster_id"],
            [],
        ).append(row)

    ranked_rows: list[dict[str, Any]] = []

    for cluster_id, cluster_rows in by_cluster.items():
        all_have_rank = all(
            row["candidate_rank"] is not None
            for row in cluster_rows
        )

        if all_have_rank:
            cluster_rows.sort(
                key=lambda row: (
                    int(row["candidate_rank"]),
                    -int(row["support_count"]),
                    str(row["candidate_cell_type"]).casefold(),
                )
            )
        else:
            cluster_rows.sort(
                key=lambda row: (
                    -int(row["support_count"]),
                    str(row["candidate_cell_type"]).casefold(),
                    int(row["_original_index"]),
                )
            )

            for rank, row in enumerate(
                cluster_rows,
                start=1,
            ):
                row["candidate_rank"] = rank

        ranked_rows.extend(cluster_rows)

    audit = {
        "detected_columns": {
            "cluster_id": cluster_col,
            "candidate_cell_type": candidate_col,
            "candidate_rank": rank_col,
            "support_count": support_col,
            "supporting_genes": genes_col,
            "review_flag": review_col,
            "warning_codes": warning_col,
            "evidence_sources": source_col,
            "cross_database_support": cross_db_col,
        },
        "input_row_count": len(rows),
        "usable_candidate_row_count": len(normalized_rows),
        "cluster_count": len(by_cluster),
    }

    return ranked_rows, audit


def build_cluster_features(
    dataset: str,
    ranked_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    by_cluster: dict[str, list[dict[str, Any]]] = {}

    for row in ranked_rows:
        by_cluster.setdefault(
            row["cluster_id"],
            [],
        ).append(row)

    output: list[dict[str, Any]] = []

    for cluster_id in sorted(
        by_cluster,
        key=natural_cluster_key,
    ):
        candidates = sorted(
            by_cluster[cluster_id],
            key=lambda row: int(row["candidate_rank"]),
        )

        top1 = candidates[0] if len(candidates) >= 1 else None
        top2 = candidates[1] if len(candidates) >= 2 else None
        top3 = candidates[2] if len(candidates) >= 3 else None

        s1 = int(top1["support_count"]) if top1 else 0
        s2 = int(top2["support_count"]) if top2 else 0
        s3 = int(top3["support_count"]) if top3 else 0

        margin_1_2 = (
            s1 - s2
            if top2 is not None
            else None
        )

        ratio_1_2 = (
            round(s1 / s2, 4)
            if top2 is not None and s2 > 0
            else None
        )

        relative_margin_1_2 = (
            round((s1 - s2) / s1, 4)
            if top2 is not None and s1 > 0
            else None
        )

        cluster_review_flags = [
            row["review_flag"]
            for row in candidates
            if row["review_flag"] is not None
        ]

        any_review_flag = (
            any(cluster_review_flags)
            if cluster_review_flags
            else None
        )

        warnings = sorted(
            {
                clean_text(row["warning_codes"])
                for row in candidates
                if clean_text(row["warning_codes"])
            }
        )

        output.append(
            {
                "dataset": dataset,
                "cluster_id": cluster_id,

                "candidate_count": len(candidates),

                "top1_candidate": (
                    top1["candidate_cell_type"]
                    if top1 else ""
                ),
                "top1_support_count": s1,
                "top1_supporting_genes": (
                    top1["supporting_genes"]
                    if top1 else ""
                ),
                "top1_evidence_sources": (
                    top1["evidence_sources"]
                    if top1 else ""
                ),
                "top1_cross_database_support": (
                    top1["cross_database_support"]
                    if top1 else None
                ),

                "top2_candidate": (
                    top2["candidate_cell_type"]
                    if top2 else ""
                ),
                "top2_support_count": (
                    s2 if top2 else None
                ),
                "top2_supporting_genes": (
                    top2["supporting_genes"]
                    if top2 else ""
                ),

                "top3_candidate": (
                    top3["candidate_cell_type"]
                    if top3 else ""
                ),
                "top3_support_count": (
                    s3 if top3 else None
                ),
                "top3_supporting_genes": (
                    top3["supporting_genes"]
                    if top3 else ""
                ),

                "support_margin_top1_top2": margin_1_2,
                "support_ratio_top1_top2": ratio_1_2,
                "relative_margin_top1_top2": relative_margin_1_2,

                "any_agent2_review_flag": any_review_flag,
                "agent2_warning_codes": " | ".join(warnings),
            }
        )

    return output


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        raise ValueError(
            f"No rows available to write to {path}"
        )

    fieldnames = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def read_reference_registry(
    path: Path,
) -> dict[tuple[str, str], dict[str, str]]:

    rows, _ = read_csv(path)

    registry: dict[
        tuple[str, str],
        dict[str, str],
    ] = {}

    for row in rows:
        dataset = clean_text(
            row.get("dataset")
        ).casefold()

        cluster_id = clean_text(
            row.get("cluster_id")
        )

        registry[
            (dataset, cluster_id)
        ] = row

    return registry


def attach_evaluation_metadata(
    feature_rows: list[dict[str, Any]],
    registry: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, Any]]:

    output: list[dict[str, Any]] = []

    for row in feature_rows:
        key = (
            str(row["dataset"]).casefold(),
            str(row["cluster_id"]),
        )

        reference = registry.get(
            key,
            {},
        )

        evaluation_row = {
            **row,

            
            # EVALUATION-ONLY FIELDS BELOW
            # Never feed these columns back into Router or Agent 3.
            "biology_expert_label": clean_text(
                reference.get(
                    "biology_expert_label"
                )
            ),
            "llm_label": clean_text(
                reference.get(
                    "llm_label"
                )
            ),
            "primary_reference_label": clean_text(
                reference.get(
                    "primary_reference_label"
                )
            ),
            "primary_reference_type": clean_text(
                reference.get(
                    "primary_reference_type"
                )
            ),
            "use_for_primary_accuracy": clean_text(
                reference.get(
                    "use_for_primary_accuracy"
                )
            ),
        }

        output.append(
            evaluation_row
        )

    return output


def process_dataset(
    *,
    dataset: str,
    candidate_file: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:

    rows, fieldnames = read_csv(
        candidate_file
    )

    ranked_rows, audit = derive_ranked_rows(
        rows,
        fieldnames,
    )

    features = build_cluster_features(
        dataset,
        ranked_rows,
    )

    audit.update(
        {
            "dataset": dataset,
            "candidate_file": str(
                candidate_file
            ),
            "feature_cluster_count": len(
                features
            ),
        }
    )

    return features, audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build cluster-level evidence features from existing Agent 2 "
            "ranked-candidate outputs without modifying Agent 2."
        )
    )

    parser.add_argument(
        "--breast-candidates",
        type=Path,
        required=True,
        help=(
            "Breast Agent 2 ranked-candidate CSV."
        ),
    )

    parser.add_argument(
        "--lung-candidates",
        type=Path,
        required=True,
        help=(
            "Lung Agent 2 ranked-candidate CSV."
        ),
    )

    parser.add_argument(
        "--reference-registry",
        type=Path,
        default=None,
        help=(
            "Optional reference_registry_breast_lung_v2.csv. "
            "Used only to make a separate evaluation view."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "agent2_feature_outputs"
        ),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    breast_features, breast_audit = process_dataset(
        dataset="breast",
        candidate_file=args.breast_candidates,
    )

    lung_features, lung_audit = process_dataset(
        dataset="lung",
        candidate_file=args.lung_candidates,
    )

    all_features = (
        breast_features
        + lung_features
    )

    feature_file = (
        args.output_dir
        / "agent2_evidence_features.csv"
    )

    write_csv(
        feature_file,
        all_features,
    )

    audit = {
        "purpose": (
            "Read-only extraction of Router-safe Agent 2 evidence features."
        ),
        "important_rule": (
            "Expert/LLM annotations are not used to construct evidence features."
        ),
        "datasets": {
            "breast": breast_audit,
            "lung": lung_audit,
        },
    }

    audit_file = (
        args.output_dir
        / "agent2_feature_audit.json"
    )

    with audit_file.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            audit,
            handle,
            indent=2,
        )

    print(
        f"Router-safe evidence features: {feature_file}"
    )

    print(
        f"Feature audit: {audit_file}"
    )

    if args.reference_registry is not None:
        registry = read_reference_registry(
            args.reference_registry
        )

        evaluation_rows = (
            attach_evaluation_metadata(
                all_features,
                registry,
            )
        )

        evaluation_file = (
            args.output_dir
            / "agent2_evaluation_view.csv"
        )

        write_csv(
            evaluation_file,
            evaluation_rows,
        )

        print(
            "Evaluation-only view: "
            f"{evaluation_file}"
        )

        print(
            "WARNING: Do NOT use evaluation-only columns "
            "as Router or Agent 3 input."
        )


if __name__ == "__main__":
    main()
