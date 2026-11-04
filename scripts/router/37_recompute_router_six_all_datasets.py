
"""
Recalculate the six ORIGINAL, UNSTANDARDIZED Router features
for all 29 clusters:

    Breast Xenium: 8 clusters
    Lung Xenium:   12 clusters
    Breast CosMX:  9 clusters

All six features are recalculated from original Agent 2 evidence.
No previously saved feature values are used in the computation.

After all 29 clusters pass validation, this script automatically
creates a new output directory and writes the final CSV files.

Previously validated pipeline files are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


FEATURES = [
    "direct_evidence_coverage",
    "top1_support_fraction",
    "top1_top2_support_gap",
    "cross_database_support_fraction",
    "first_direct_supported_marker_rank_norm",
    "rank_weighted_unexplained_marker_score",
]

DATASETS = [
    (
        "Breast Xenium",
        "Xenium",
        8,
        "data/pipeline_runs/breast_xenium_marker_only_clean/agent2_output",
    ),
    (
        "Lung Xenium",
        "Xenium",
        12,
        "data/pipeline_runs/lung_validated_clean/agent2_output",
    ),
    (
        "Breast CosMX",
        "CosMX",
        9,
        None,
    ),
]

HISTORICAL_TABLE = (
    "data/analysis/router_29_cluster_consolidation_v1/"
    "router_original_six_29_clusters.csv"
)

COEFFICIENTS_FILE = (
    "data/analysis/router_l2_logistic_feasibility_v1/"
    "router_l2_coefficients.csv"
)

OUTPUT_DIRECTORY = (
    "data/analysis/router_six_features_all_datasets_v3"
)


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Required input file is missing: {path}"
        )

    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header: {path}")

        return list(reader)


def check_columns(rows, required, path):
    if not rows:
        raise ValueError(f"Empty CSV: {path}")

    missing = set(required) - set(rows[0])

    if missing:
        raise ValueError(
            f"Missing columns in {path}: {sorted(missing)}"
        )


def file_sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            digest.update(block)

    return digest.hexdigest()


def as_integer(value, description):
    number = float(value)

    if not math.isfinite(number) or number != int(number):
        raise ValueError(
            f"{description}: expected integer, got {value}"
        )

    return int(number)


def is_supported(value):
    text = str(value).strip().lower()

    if text in {"true", "yes", "1"}:
        return True

    if text in {"false", "no", "0"}:
        return False

    raise ValueError(
        f"Invalid direct_marker_supported value: {value}"
    )


def index_by_cluster(rows, description):
    indexed = {}

    for row in rows:
        cluster = as_integer(
            row["cluster_id"],
            f"{description}.cluster_id",
        )

        if cluster in indexed:
            raise ValueError(
                f"Duplicate cluster {cluster}: {description}"
            )

        indexed[cluster] = row

    return indexed


def save_csv(path, columns, rows):
    with path.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columns,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def compute_six_features(
    top1,
    top2,
    summary,
    gene_evidence,
    dataset,
    cluster_id,
):
    label = f"{dataset} cluster {cluster_id}"

    total_genes = as_integer(
        summary["total_input_genes"],
        f"{label}: total genes",
    )

    if total_genes <= 0:
        raise ValueError(f"{label}: no input marker genes")

    if len(gene_evidence) != total_genes:
        raise ValueError(
            f"{label}: expected {total_genes} gene rows, "
            f"found {len(gene_evidence)}"
        )

    direct_count = as_integer(
        summary["direct_marker_supported_genes"],
        f"{label}: supported genes",
    )

    follow_up_count = as_integer(
        summary["follow_up_gene_count"],
        f"{label}: follow-up genes",
    )

    if direct_count + follow_up_count != total_genes:
        raise ValueError(
            f"{label}: summary gene counts are inconsistent"
        )

    # Build gene-rank -> direct-support mapping.
    ranked_support = {}

    for gene in gene_evidence:
        gene_cluster = as_integer(
            gene["cluster_id"],
            f"{label}: gene cluster",
        )

        if gene_cluster != cluster_id:
            raise ValueError(
                f"{label}: inconsistent gene cluster ID"
            )

        rank = as_integer(
            gene["rank_within_cluster"],
            f"{label}: marker rank",
        )

        if rank in ranked_support:
            raise ValueError(
                f"{label}: duplicate marker rank {rank}"
            )

        ranked_support[rank] = is_supported(
            gene["direct_marker_supported"]
        )

    expected_ranks = set(range(1, total_genes + 1))

    if set(ranked_support) != expected_ranks:
        raise ValueError(
            f"{label}: marker ranks are incomplete"
        )

    observed_direct = sum(ranked_support.values())

    if observed_direct != direct_count:
        raise ValueError(
            f"{label}: direct-support counts do not match"
        )

    supported_ranks = [
        rank
        for rank, supported in ranked_support.items()
        if supported
    ]

    if not supported_ranks:
        raise ValueError(
            f"{label}: first supported-marker rank undefined"
        )

    first_supported_rank = min(supported_ranks)

    top1_count = as_integer(
        top1["unique_supporting_gene_count"],
        f"{label}: Top-1 support",
    )

    top2_count = as_integer(
        top2["unique_supporting_gene_count"],
        f"{label}: Top-2 support",
    )

    if (
        top1_count <= 0
        or top2_count < 0
        or top2_count > top1_count
    ):
        raise ValueError(
            f"{label}: invalid Top-1/Top-2 support counts"
        )

    cross_source_column = next(
        (
            column
            for column in [
                "both_source_supporting_gene_count",
                "both_database_supporting_gene_count",
            ]
            if column in top1
        ),
        None,
    )

    if cross_source_column is None:
        raise ValueError(
            f"{label}: cross-database support column missing"
        )

    both_sources_count = as_integer(
        top1[cross_source_column],
        f"{label}: cross-database support",
    )

    if not 0 <= both_sources_count <= top1_count:
        raise ValueError(
            f"{label}: invalid cross-database support"
        )

    # Rank-weighted unexplained evidence.
    unexplained_weight = sum(
        total_genes - rank + 1
        for rank in range(1, total_genes + 1)
        if not ranked_support[rank]
    )

    total_rank_weight = (
        total_genes * (total_genes + 1) / 2
    )

    # Original unstandardized features.
    return {
        "direct_evidence_coverage": round(
            observed_direct / total_genes, 4
        ),
        "top1_support_fraction": round(
            top1_count / total_genes, 4
        ),
        "top1_top2_support_gap": round(
            (top1_count - top2_count) / total_genes, 4
        ),
        "cross_database_support_fraction": round(
            both_sources_count / top1_count, 4
        ),
        "first_direct_supported_marker_rank_norm": round(
            first_supported_rank / total_genes, 4
        ),
        "rank_weighted_unexplained_marker_score": round(
            unexplained_weight / total_rank_weight, 4
        ),
    }



def process_dataset(
    dataset,
    platform,
    expected_clusters,
    agent2_dir,
    input_hashes,
):
    candidate_file = (
        agent2_dir / "all_clusters_threshold_candidates.csv"
    )

    summary_file = (
        agent2_dir / "all_clusters_summary.csv"
    )

    candidates = read_csv(candidate_file)
    summaries = read_csv(summary_file)

    check_columns(
        candidates,
        [
            "cluster_id",
            "candidate_rank",
            "cell_type",
            "unique_supporting_gene_count",
        ],
        candidate_file,
    )

    check_columns(
        summaries,
        [
            "cluster_id",
            "total_input_genes",
            "direct_marker_supported_genes",
            "follow_up_gene_count",
        ],
        summary_file,
    )

    input_hashes[str(candidate_file)] = file_sha256(
        candidate_file
    )

    input_hashes[str(summary_file)] = file_sha256(
        summary_file
    )

    indexed_summaries = index_by_cluster(
        summaries, dataset
    )

    expected_ids = set(range(expected_clusters))

    if set(indexed_summaries) != expected_ids:
        raise ValueError(
            f"{dataset}: expected clusters "
            f"{sorted(expected_ids)}, "
            f"found {sorted(indexed_summaries)}"
        )

    ranked_candidates = {}

    for candidate in candidates:
        cid = as_integer(
            candidate["cluster_id"],
            f"{dataset}: candidate cluster",
        )

        rank = as_integer(
            candidate["candidate_rank"],
            f"{dataset}: candidate rank",
        )

        ranked_candidates.setdefault(cid, {})

        if rank in ranked_candidates[cid]:
            raise ValueError(
                f"{dataset} cluster {cid}: "
                f"duplicate candidate rank {rank}"
            )

        ranked_candidates[cid][rank] = candidate

    if set(ranked_candidates) != expected_ids:
        raise ValueError(
            f"{dataset}: candidate clusters incomplete"
        )

    results = []

    for cid in range(expected_clusters):
        ranked = ranked_candidates[cid]

        if 1 not in ranked or 2 not in ranked:
            raise ValueError(
                f"{dataset} cluster {cid}: "
                f"missing Top-1 or Top-2 candidate"
            )

        top1 = ranked[1]
        top2 = ranked[2]

        annotation = top1["cell_type"].strip()

        if not annotation:
            raise ValueError(
                f"{dataset} cluster {cid}: "
                f"empty Top-1 annotation"
            )

        master_file = (
            agent2_dir
            / "clusters"
            / f"cluster_{cid}"
            / f"cluster_{cid}_master_gene_evidence.csv"
        )

        gene_evidence = read_csv(master_file)

        check_columns(
            gene_evidence,
            [
                "cluster_id",
                "rank_within_cluster",
                "direct_marker_supported",
            ],
            master_file,
        )

        input_hashes[str(master_file)] = file_sha256(
            master_file
        )

        features = compute_six_features(
            top1,
            top2,
            indexed_summaries[cid],
            gene_evidence,
            dataset,
            cid,
        )

        results.append({
            "dataset": dataset,
            "platform": platform,
            "cluster_id": cid,
            "agent2_top1_annotation": annotation,
            **features,
        })

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
    )

    parser.add_argument(
        "--cosmx-agent2-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--historical-table",
        type=Path,
        default=Path(HISTORICAL_TABLE),
    )

    parser.add_argument(
        "--coefficients",
        type=Path,
        default=Path(COEFFICIENTS_FILE),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(OUTPUT_DIRECTORY),
    )

    args = parser.parse_args()

    root = args.project_root.resolve()

    def locate(path):
        path = Path(path)
        return path if path.is_absolute() else root / path

    output_dir = locate(args.output_dir)

    # Never overwrite an earlier analysis.
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. "
            "Choose a new directory with --output-dir."
        )

    historical_file = locate(args.historical_table)
    coefficient_file = locate(args.coefficients)

    historical = read_csv(historical_file)

    check_columns(
        historical,
        [
            "dataset",
            "platform",
            "cluster_id",
            "agent2_top1_annotation",
            "reference_cell_type_annotation",
            "correctness_label_0_1",
            "router_v4_decision",
            *FEATURES,
        ],
        historical_file,
    )

    previous_rows = {}

    for row in historical:
        key = (
            row["dataset"],
            as_integer(row["cluster_id"], "historical ID"),
        )

        if key in previous_rows:
            raise ValueError(
                f"Duplicate historical cluster: {key}"
            )

        previous_rows[key] = row

    coefficient_rows = read_csv(coefficient_file)

    check_columns(
        coefficient_rows,
        ["term", "coefficient"],
        coefficient_file,
    )

    coefficients = {
        row["term"]: float(row["coefficient"])
        for row in coefficient_rows
    }

    if set(coefficients) != {"intercept", *FEATURES}:
        raise ValueError(
            "Expected intercept and exactly six "
            "original logistic-regression coefficients"
        )

    if not all(
        math.isfinite(v) for v in coefficients.values()
    ):
        raise ValueError(
            "Non-finite logistic-regression coefficient"
        )

    input_hashes = {
        str(historical_file): file_sha256(historical_file),
        str(coefficient_file): file_sha256(coefficient_file),
    }

    calculated_rows = []
    evidence_sources = {}

    # Recalculate all datasets, including CosMX.
    for (
        dataset,
        platform,
        expected_count,
        default_directory,
    ) in DATASETS:

        source_dir = locate(
            args.cosmx_agent2_dir
            if default_directory is None
            else Path(default_directory)
        )

        print(
            f"\nRECOMPUTING: {dataset}\n"
            f"Source: {source_dir}",
            flush=True,
        )

        batch = process_dataset(
            dataset,
            platform,
            expected_count,
            source_dir,
            input_hashes,
        )

        calculated_rows.extend(batch)

        evidence_sources[dataset] = str(source_dir)

        print(
            f"PASS: {len(batch)} clusters, "
            f"{len(batch) * 6} features computed",
            flush=True,
        )

    # Confirm all 29 clusters.
    expected_keys = {
        (name, cid)
        for name, _, count, _ in DATASETS
        for cid in range(count)
    }

    actual_keys = {
        (row["dataset"], row["cluster_id"])
        for row in calculated_rows
    }

    if (
        len(calculated_rows) != 29
        or actual_keys != expected_keys
    ):
        raise ValueError(
            "All 29 clusters were not successfully recomputed"
        )

    if set(previous_rows) != expected_keys:
        raise ValueError(
            "Historical comparison table must have "
            "the same 29 cluster identifiers"
        )

    final_rows = []
    feature_changes = []
    annotation_changes = []
    v4_rerun_needed = []

    for current in calculated_rows:
        key = (
            current["dataset"],
            current["cluster_id"],
        )

        previous = previous_rows[key]

        same_annotation = (
            current["agent2_top1_annotation"].casefold()
            ==
            previous[
                "agent2_top1_annotation"
            ].strip().casefold()
        )

        if not same_annotation:
            annotation_changes.append({
                "dataset": key[0],
                "cluster_id": key[1],
                "previous_top1":
                    previous["agent2_top1_annotation"],
                "recomputed_top1":
                    current["agent2_top1_annotation"],
            })

        changed_features = False

        for feature in FEATURES:
            old = float(previous[feature])
            new = float(current[feature])

            if abs(old - new) > 0.00005001:
                changed_features = True

                feature_changes.append({
                    "dataset": key[0],
                    "cluster_id": key[1],
                    "feature": feature,
                    "previous_value": old,
                    "recomputed_value": new,
                    "difference": round(new - old, 8),
                })

        # Apply the existing frozen logistic model to
        # the newly recalculated six feature values.
        linear_score = coefficients["intercept"]

        for feature in FEATURES:
            linear_score += (
                coefficients[feature] * current[feature]
            )

        probability = 1 / (1 + math.exp(-linear_score))

        # Previous V4 decisions are not newly rerun.
        # Do not retain them if evidence has changed.
        safe_to_retain_v4 = (
            same_annotation and not changed_features
        )

        if not safe_to_retain_v4:
            v4_rerun_needed.append({
                "dataset": key[0],
                "cluster_id": key[1],
                "reason": (
                    "TOP1_CHANGED"
                    if not same_annotation
                    else "FEATURE_VALUES_CHANGED"
                ),
            })

        final_rows.append({
            "dataset": current["dataset"],
            "platform": current["platform"],
            "cluster_id": current["cluster_id"],
            "agent2_top1_annotation":
                current["agent2_top1_annotation"],
            "reference_cell_type_annotation":
                previous["reference_cell_type_annotation"],
            "correctness_label_0_1": (
                previous["correctness_label_0_1"]
                if same_annotation
                else ""
            ),
            **{
                feature: current[feature]
                for feature in FEATURES
            },
            "logistic_predicted_correctness_probability":
                round(probability, 10),
            "router_v4_decision": (
                previous["router_v4_decision"]
                if safe_to_retain_v4
                else ""
            ),
        })

    # CosMX must not receive fabricated reference labels.
    for row in final_rows:
        if row["dataset"] == "Breast CosMX":
            if (
                str(
                    row["reference_cell_type_annotation"]
                ).strip()
                or
                str(
                    row["correctness_label_0_1"]
                ).strip()
            ):
                raise ValueError(
                    "CosMX reference labels must remain "
                    "unassigned until verified"
                )

    # ------------------------------------------------------
    # 6. AUTOMATICALLY CREATE OUTPUT DIRECTORY AND CSV FILES
    # ------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    feature_columns = [
        "dataset",
        "platform",
        "cluster_id",
        "agent2_top1_annotation",
        *FEATURES,
    ]

    comparison_columns = [
        "dataset",
        "platform",
        "cluster_id",
        "agent2_top1_annotation",
        "reference_cell_type_annotation",
        "correctness_label_0_1",
        *FEATURES,
        "logistic_predicted_correctness_probability",
        "router_v4_decision",
    ]

    save_csv(
        output_dir
        / "router_original_six_29_clusters_recomputed.csv",
        feature_columns,
        final_rows,
    )

    save_csv(
        output_dir
        / "router_comparison_29_clusters.csv",
        comparison_columns,
        final_rows,
    )

    save_csv(
        output_dir
        / "router_feature_changes_vs_previous.csv",
        [
            "dataset",
            "cluster_id",
            "feature",
            "previous_value",
            "recomputed_value",
            "difference",
        ],
        feature_changes,
    )

    save_csv(
        output_dir
        / "router_annotation_changes.csv",
        [
            "dataset",
            "cluster_id",
            "previous_top1",
            "recomputed_top1",
        ],
        annotation_changes,
    )

    save_csv(
        output_dir
        / "router_v4_decisions_needing_rerun.csv",
        ["dataset", "cluster_id", "reason"],
        v4_rerun_needed,
    )

    audit = {
        "created_at_utc":
            datetime.now(timezone.utc).isoformat(),
        "total_clusters": len(final_rows),
        "recomputed_feature_values": len(final_rows) * 6,
        "dataset_counts": dict(
            Counter(
                row["dataset"]
                for row in final_rows
            )
        ),
        "features": FEATURES,
        "feature_scale": "original_unstandardized",
        "evidence_sources": evidence_sources,
        "feature_changes": len(feature_changes),
        "annotation_changes": len(annotation_changes),
        "v4_decisions_needing_rerun":
            len(v4_rerun_needed),
        "historical_feature_values_used_for_calculation":
            False,
        "logistic_model": "existing_frozen_coefficients",
        "v4_decisions_newly_recomputed": False,
        "reference_labels": (
            "historical provisional labels; "
            "CosMX unassigned"
        ),
        "input_sha256": input_hashes,
    }

    audit_file = (
        output_dir / "router_recalculation_audit.json"
    )

    audit_file.write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("STATUS: SUCCESS - ALL 29 CLUSTERS RECALCULATED")
    print("Dataset counts:", audit["dataset_counts"])
    print("Feature values computed: 174 / 174")
    print("Feature differences:", len(feature_changes))
    print("Agent 2 annotation changes:", len(annotation_changes))
    print("V4 decisions needing rerun:", len(v4_rerun_needed))
    print("Original pipeline files modified: NONE")
    print("=" * 60)
    print("\nFINAL CSV:")
    print(output_dir / "router_comparison_29_clusters.csv")
    print("\nAUDIT JSON:")
    print(audit_file)


if __name__ == "__main__":
    main()
