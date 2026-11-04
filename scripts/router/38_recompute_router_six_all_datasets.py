
"""
Recompute the six original, UNSTANDARDIZED Router features.

Datasets:
    Breast Xenium: 8 clusters
    Lung Xenium: 12 clusters
    Breast CosMX: 9 clusters

Total: 29 clusters, 174 feature values.

All features are calculated from Agent 2 evidence.
No historical feature values are used in recalculation.

The script:
- Validates each dataset.
- Recomputes all six features.
- Compares results with the previous consolidated table.
- Applies the previously fitted logistic-regression coefficients.
- Preserves historical reference annotations where applicable.
- Keeps unverified CosMX reference annotations blank.
- Creates new CSV and JSON outputs automatically.
- Never overwrites validated pipeline results.
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

# CONFIGURATION

FEATURES = (
    "direct_evidence_coverage",
    "top1_support_fraction",
    "top1_top2_support_gap",
    "cross_database_support_fraction",
    "first_direct_supported_marker_rank_norm",
    "rank_weighted_unexplained_marker_score",
)

DATASETS = (
    (
        "Breast Xenium",
        "Xenium",
        8,
        "data/pipeline_runs/"
        "breast_xenium_marker_only_clean/agent2_output",
    ),
    (
        "Lung Xenium",
        "Xenium",
        12,
        "data/pipeline_runs/"
        "lung_validated_clean/agent2_output",
    ),
    (
        "Breast CosMX",
        "CosMX",
        9,
        None,
    ),
)

HISTORICAL_DEFAULT = (
    "data/analysis/router_29_cluster_consolidation_v1/"
    "router_original_six_29_clusters.csv"
)

COEFFICIENTS_DEFAULT = (
    "data/analysis/router_l2_logistic_feasibility_v1/"
    "router_l2_coefficients.csv"
)

OUTPUT_DEFAULT = (
    "data/analysis/router_six_features_all_datasets_v4"
)


# FILE UTILITIES

def read_csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Required input file missing: {path}"
        )

    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        reader = csv.DictReader(stream)

        if not reader.fieldnames:
            raise ValueError(
                f"CSV has no header: {path}"
            )

        return list(reader)


def require(rows, columns, path):
    if not rows:
        raise ValueError(
            f"CSV is empty: {path}"
        )

    missing = set(columns) - rows[0].keys()

    if missing:
        raise ValueError(
            f"Missing columns {sorted(missing)} in {path}"
        )


def sha256(path: Path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024), b""
        ):
            digest.update(block)

    return digest.hexdigest()


def as_int(value, name):
    number = float(value)

    if (
        not math.isfinite(number)
        or number != int(number)
    ):
        raise ValueError(
            f"{name} must be an integer: {value}"
        )

    return int(number)


def supported(value):
    text = str(value).strip().casefold()

    if text in {"true", "yes", "1"}:
        return True

    if text in {"false", "no", "0"}:
        return False

    raise ValueError(
        f"Invalid direct_marker_supported: {value}"
    )


def index_on_cluster(rows, description):
    result = {}

    for row in rows:
        cluster_id = as_int(
            row["cluster_id"],
            f"{description}.cluster_id",
        )

        if cluster_id in result:
            raise ValueError(
                f"Duplicate cluster {cluster_id}: {description}"
            )

        result[cluster_id] = row

    return result


def write_csv(path, columns, rows):
    with path.open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=columns,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


# RECALCULATE SIX ROUTER FEATURES FOR ONE CLUSTER


def compute_one(
    top1,
    top2,
    summary,
    master,
    dataset,
    cluster_id,
):
    label = f"{dataset} cluster {cluster_id}"

    total_genes = as_int(
        summary["total_input_genes"],
        f"{label}: total_input_genes",
    )

    if total_genes <= 0:
        raise ValueError(
            f"{label}: no marker genes"
        )

    if len(master) != total_genes:
        raise ValueError(
            f"{label}: master evidence has {len(master)} "
            f"rows; expected {total_genes}"
        )

    direct_count = as_int(
        summary["direct_marker_supported_genes"],
        f"{label}: direct_count",
    )

    follow_up_count = as_int(
        summary["follow_up_gene_count"],
        f"{label}: follow_up_count",
    )


    if not 0 <= direct_count <= total_genes:
        raise ValueError(
            f"{label}: invalid direct-support count "
            f"{direct_count}"
        )

    if not 0 <= follow_up_count <= total_genes:
        raise ValueError(
            f"{label}: invalid follow-up count "
            f"{follow_up_count}"
        )


    ranked_support = {}

    for gene in master:
        gene_cluster = as_int(
            gene["cluster_id"],
            f"{label}: gene cluster",
        )

        if gene_cluster != cluster_id:
            raise ValueError(
                f"{label}: inconsistent gene cluster"
            )

        rank = as_int(
            gene["rank_within_cluster"],
            f"{label}: gene rank",
        )

        if rank in ranked_support:
            raise ValueError(
                f"{label}: duplicate marker rank {rank}"
            )

        ranked_support[rank] = supported(
            gene["direct_marker_supported"]
        )

    expected_ranks = set(
        range(1, total_genes + 1)
    )

    if set(ranked_support) != expected_ranks:
        raise ValueError(
            f"{label}: marker ranks must be complete "
            "and consecutive"
        )

    observed_direct_count = sum(
        ranked_support.values()
    )

    if observed_direct_count != direct_count:
        raise ValueError(
            f"{label}: direct-support count differs "
            "between summary and master evidence"
        )

    supported_ranks = [
        rank
        for rank in range(1, total_genes + 1)
        if ranked_support[rank]
    ]

    if not supported_ranks:
        raise ValueError(
            f"{label}: no directly supported gene; "
            "first-rank feature is undefined"
        )

    first_direct_rank = min(supported_ranks)


    top1_support = as_int(
        top1["unique_supporting_gene_count"],
        f"{label}: Top-1 support",
    )

    top2_support = as_int(
        top2["unique_supporting_gene_count"],
        f"{label}: Top-2 support",
    )

    if (
        top1_support <= 0
        or top1_support > total_genes
        or top2_support < 0
        or top2_support > top1_support
    ):
        raise ValueError(
            f"{label}: invalid Top-1/Top-2 "
            "support counts"
        )

    cross_database_column = next(
        (
            column
            for column in (
                "both_source_supporting_gene_count",
                "both_database_supporting_gene_count",
            )
            if column in top1
        ),
        None,
    )

    if cross_database_column is None:
        raise ValueError(
            f"{label}: cross-database support "
            "column missing"
        )

    cross_database_count = as_int(
        top1[cross_database_column],
        f"{label}: cross-database support",
    )

    if not (
        0 <= cross_database_count <= top1_support
    ):
        raise ValueError(
            f"{label}: invalid cross-database "
            "support count"
        )

 

    unexplained_weight = sum(
        total_genes - rank + 1
        for rank, is_direct in ranked_support.items()
        if not is_direct
    )

    total_rank_weight = (
        total_genes * (total_genes + 1) / 2
    )


    return {
        "direct_evidence_coverage": round(
            observed_direct_count / total_genes, 4
        ),

        "top1_support_fraction": round(
            top1_support / total_genes, 4
        ),

        "top1_top2_support_gap": round(
            (top1_support - top2_support) / total_genes,
            4,
        ),

        "cross_database_support_fraction": round(
            cross_database_count / top1_support, 4
        ),

        "first_direct_supported_marker_rank_norm": round(
            first_direct_rank / total_genes, 4
        ),

        "rank_weighted_unexplained_marker_score": round(
            unexplained_weight / total_rank_weight, 4
        ),
    }



def calculate_dataset(
    name,
    platform,
    expected_count,
    agent2_dir,
    input_hashes,
):
    candidate_path = (
        agent2_dir / "all_clusters_threshold_candidates.csv"
    )

    summary_path = (
        agent2_dir / "all_clusters_summary.csv"
    )

    candidates = read_csv(candidate_path)
    summaries = read_csv(summary_path)

    require(
        candidates,
        [
            "cluster_id",
            "candidate_rank",
            "cell_type",
            "unique_supporting_gene_count",
        ],
        candidate_path,
    )

    require(
        summaries,
        [
            "cluster_id",
            "total_input_genes",
            "direct_marker_supported_genes",
            "follow_up_gene_count",
        ],
        summary_path,
    )

    input_hashes[str(candidate_path)] = sha256(
        candidate_path
    )

    input_hashes[str(summary_path)] = sha256(
        summary_path
    )

    indexed_summaries = index_on_cluster(
        summaries,
        f"{name} summaries",
    )

    expected_ids = set(
        range(expected_count)
    )

    if set(indexed_summaries) != expected_ids:
        raise ValueError(
            f"{name}: expected clusters "
            f"{sorted(expected_ids)}, found "
            f"{sorted(indexed_summaries)}"
        )

    ranked_candidates = {}

    for row in candidates:
        cluster_id = as_int(
            row["cluster_id"],
            f"{name}: candidate cluster",
        )

        rank = as_int(
            row["candidate_rank"],
            f"{name}: candidate rank",
        )

        ranked_candidates.setdefault(
            cluster_id, {}
        )

        if rank in ranked_candidates[cluster_id]:
            raise ValueError(
                f"{name}: duplicate candidate rank "
                f"{rank} for cluster {cluster_id}"
            )

        ranked_candidates[cluster_id][rank] = row

    if set(ranked_candidates) != expected_ids:
        raise ValueError(
            f"{name}: candidate clusters incomplete"
        )

    results = []

    for cluster_id in range(expected_count):
        ranked = ranked_candidates[cluster_id]

        if not {1, 2}.issubset(ranked):
            raise ValueError(
                f"{name} cluster {cluster_id}: "
                "missing Top-1 or Top-2 candidate"
            )

        top1 = ranked[1]
        top2 = ranked[2]

        annotation = top1["cell_type"].strip()

        if not annotation:
            raise ValueError(
                f"{name} cluster {cluster_id}: "
                "empty Agent 2 annotation"
            )

        master_path = (
            agent2_dir
            / "clusters"
            / f"cluster_{cluster_id}"
            / f"cluster_{cluster_id}_master_gene_evidence.csv"
        )

        master = read_csv(master_path)

        require(
            master,
            [
                "cluster_id",
                "rank_within_cluster",
                "direct_marker_supported",
            ],
            master_path,
        )

        input_hashes[str(master_path)] = sha256(
            master_path
        )

        features = compute_one(
            top1,
            top2,
            indexed_summaries[cluster_id],
            master,
            name,
            cluster_id,
        )

        results.append({
            "dataset": name,
            "platform": platform,
            "cluster_id": cluster_id,
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
        default=Path(HISTORICAL_DEFAULT),
    )

    parser.add_argument(
        "--coefficients",
        type=Path,
        default=Path(COEFFICIENTS_DEFAULT),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(OUTPUT_DEFAULT),
    )

    args = parser.parse_args()

    root = args.project_root.resolve()

    def locate(path):
        path = Path(path)

        return (
            path
            if path.is_absolute()
            else root / path
        )

    output_dir = locate(args.output_dir)

    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: "
            f"{output_dir}. No overwrite allowed."
        )

    historical_path = locate(
        args.historical_table
    )

    coefficients_path = locate(
        args.coefficients
    )

    historical_rows = read_csv(
        historical_path
    )

    require(
        historical_rows,
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
        historical_path,
    )

    previous_rows = {}

    for row in historical_rows:
        key = (
            row["dataset"],
            as_int(
                row["cluster_id"],
                "historical cluster_id",
            ),
        )

        if key in previous_rows:
            raise ValueError(
                f"Duplicate historical cluster: {key}"
            )

        previous_rows[key] = row

    coefficient_rows = read_csv(
        coefficients_path
    )

    require(
        coefficient_rows,
        ["term", "coefficient"],
        coefficients_path,
    )

    coefficients = {
        row["term"]: float(row["coefficient"])
        for row in coefficient_rows
    }

    expected_terms = {
        "intercept",
        *FEATURES,
    }

    if set(coefficients) != expected_terms:
        raise ValueError(
            "Expected intercept and exactly "
            "six logistic-regression coefficients"
        )

    if not all(
        math.isfinite(value)
        for value in coefficients.values()
    ):
        raise ValueError(
            "Logistic coefficients must be finite"
        )

    input_hashes = {
        str(historical_path): sha256(historical_path),
        str(coefficients_path): sha256(coefficients_path),
    }

    results = []
    evidence_sources = {}


    for (
        name,
        platform,
        expected_count,
        default_directory,
    ) in DATASETS:

        source_dir = locate(
            args.cosmx_agent2_dir
            if default_directory is None
            else default_directory
        )

        print(
            f"\nRECOMPUTING: {name}\n"
            f"Source: {source_dir}",
            flush=True,
        )

        batch = calculate_dataset(
            name,
            platform,
            expected_count,
            source_dir,
            input_hashes,
        )

        results.extend(batch)

        evidence_sources[name] = str(source_dir)

        print(
            f"PASS: {len(batch)} clusters, "
            f"{len(batch) * 6} features computed",
            flush=True,
        )



    expected_keys = {
        (name, cluster_id)
        for name, _, count, _ in DATASETS
        for cluster_id in range(count)
    }

    actual_keys = {
        (row["dataset"], row["cluster_id"])
        for row in results
    }

    if (
        len(results) != 29
        or actual_keys != expected_keys
    ):
        raise ValueError(
            "Expected exactly 29 recomputed clusters"
        )

    if set(previous_rows) != expected_keys:
        raise ValueError(
            "Historical table must cover "
            "the same 29 cluster IDs"
        )

 

    final_rows = []
    feature_changes = []
    annotation_changes = []
    v4_rerun_needed = []

    for current in results:
        key = (
            current["dataset"],
            current["cluster_id"],
        )

        previous = previous_rows[key]

        same_annotation = (
            current["agent2_top1_annotation"].casefold()
            ==
            previous["agent2_top1_annotation"]
            .strip()
            .casefold()
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

        features_changed = False

        for feature in FEATURES:
            old_value = float(
                previous[feature]
            )

            new_value = float(
                current[feature]
            )

            if abs(
                old_value - new_value
            ) > 0.00005001:

                features_changed = True

                feature_changes.append({
                    "dataset": key[0],
                    "cluster_id": key[1],
                    "feature": feature,
                    "previous_value": old_value,
                    "recomputed_value": new_value,
                    "difference": round(
                        new_value - old_value, 8
                    ),
                })

        # Apply frozen logistic-regression coefficients
        # to the newly recalculated features.

        linear_score = coefficients["intercept"]

        for feature in FEATURES:
            linear_score += (
                coefficients[feature]
                * current[feature]
            )

        probability = 1.0 / (
            1.0 + math.exp(-linear_score)
        )

   

        retain_previous_v4 = (
            same_annotation
            and not features_changed
        )

        if not retain_previous_v4:
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
                if retain_previous_v4
                else ""
            ),
        })

  

    for row in final_rows:
        if row["dataset"] != "Breast CosMX":
            continue

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
                "CosMX reference/correctness labels "
                "must remain unassigned until verified"
            )


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

    write_csv(
        output_dir
        / "router_original_six_29_clusters_recomputed.csv",
        feature_columns,
        final_rows,
    )

    write_csv(
        output_dir
        / "router_comparison_29_clusters.csv",
        comparison_columns,
        final_rows,
    )

    write_csv(
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

    write_csv(
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

    write_csv(
        output_dir
        / "router_v4_decisions_needing_rerun.csv",
        [
            "dataset",
            "cluster_id",
            "reason",
        ],
        v4_rerun_needed,
    )


    audit = {
        "created_at_utc":
            datetime.now(timezone.utc).isoformat(),

        "total_clusters": len(final_rows),

        "recomputed_feature_values":
            len(final_rows) * len(FEATURES),

        "dataset_counts": dict(
            Counter(
                row["dataset"]
                for row in final_rows
            )
        ),

        "original_unstandardized_features":
            list(FEATURES),

        "evidence_sources": evidence_sources,

        "feature_changes": len(feature_changes),

        "annotation_changes": len(annotation_changes),

        "v4_decisions_needing_rerun":
            len(v4_rerun_needed),

        "historical_features_used_in_calculation":
            False,

        "logistic_model":
            "previously fitted original L2 coefficients",

        "v4_decisions_newly_computed":
            False,

        "v4_decision_source":
            str(historical_path),

        "reference_labels":
            "provisional historical Breast/Lung labels; "
            "CosMX labels unavailable",

        "input_sha256": input_hashes,
    }

    audit_path = (
        output_dir
        / "router_recalculation_audit.json"
    )

    audit_path.write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)

    print(
        "STATUS: SUCCESS - ALL 29 CLUSTERS RECALCULATED"
    )

    print(
        "Dataset counts:",
        audit["dataset_counts"],
    )

    print(
        "Original feature values recomputed: 174 / 174"
    )

    print(
        "Feature differences:",
        len(feature_changes),
    )

    print(
        "Agent 2 annotation changes:",
        len(annotation_changes),
    )

    print(
        "V4 decisions needing rerun:",
        len(v4_rerun_needed),
    )

    print(
        "Existing validated pipeline files modified: NONE"
    )

    print("=" * 60)

    print("\nFINAL COMPARISON CSV:")
    print(
        output_dir
        / "router_comparison_29_clusters.csv"
    )

    print("\nAUDIT JSON:")
    print(audit_path)


if __name__ == "__main__":
    main()
