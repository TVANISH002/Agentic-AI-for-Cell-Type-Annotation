"""Offline Agent 2 mean-input-gene-rank comparison.

READ-ONLY INPUTS:
- Existing Breast Xenium Agent 2 candidate and gene-evidence CSVs
- Existing Router V4 configuration and decisions

OUTPUTS:
- Full candidate comparison
- Eight-cluster comparison
- Router shadow comparison
- Detailed Router audit
- Experiment manifest

This script does not edit Agent 1, Agent 2, Router, or Agent 3.
Reference and expert annotations are not used during ranking.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys

from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        return list(csv.DictReader(stream))


def write_csv(
    path: Path,
    rows: list[dict],
    columns: list[str],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=columns,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def checksum(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def numeric(value, label: str) -> float:
    try:
        result = float(str(value))
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Invalid {label}: {value!r}"
        ) from exc

    if not math.isfinite(result):
        raise ValueError(
            f"Non-finite {label}: {value!r}"
        )

    return result


def unique(
    rows: list[dict[str, str]],
    key: str,
) -> dict[str, dict[str, str]]:
    result = {}

    for row in rows:
        value = str(row[key]).strip()

        if not value or value in result:
            raise ValueError(
                f"Missing or duplicated {key}: {value!r}"
            )

        result[value] = row

    return result


def compare_sort_key(row: dict) -> tuple:
    """Lower mean input rank is better.

    Ties:
    1. More supporting genes
    2. Earlier original candidate rank
    3. Cell-type name for deterministic ordering
    """
    return (
        numeric(
            row["mean_supporting_gene_rank"],
            "mean gene rank",
        ),
        -numeric(
            row["unique_supporting_gene_count"],
            "support count",
        ),
        int(row["candidate_rank"]),
        row["cell_type"].casefold(),
    )


def verify_means(
    cluster: str,
    candidates: list[dict],
    genes: list[dict],
) -> None:
    """Verify stored means against original input gene ranks."""

    ranks = unique(genes, "gene")

    for row in candidates:
        supporting = [
            gene.strip()
            for gene in row["supporting_genes"].split("|")
            if gene.strip()
        ]

        if (
            not supporting
            or len(supporting) != len(set(supporting))
        ):
            raise ValueError(
                f"Cluster {cluster}, "
                f"{row['cell_type']}: invalid supporting genes"
            )

        if len(supporting) != int(
            row["unique_supporting_gene_count"]
        ):
            raise ValueError(
                f"Cluster {cluster}, "
                f"{row['cell_type']}: count mismatch"
            )

        for gene in supporting:
            if (
                gene not in ranks
                or str(
                    ranks[gene]["direct_marker_supported"]
                ).lower() != "true"
            ):
                raise ValueError(
                    f"Cluster {cluster}, "
                    f"{row['cell_type']}: "
                    f"missing direct marker {gene}"
                )

        real_mean = sum(
            numeric(
                ranks[gene]["rank_within_cluster"],
                "gene rank",
            )
            for gene in supporting
        ) / len(supporting)

        stored_mean = numeric(
            row["mean_supporting_gene_rank"],
            "candidate mean rank",
        )

        if abs(real_mean - stored_mean) > 1e-6:
            raise ValueError(
                f"Cluster {cluster}, "
                f"{row['cell_type']}: "
                f"mean-rank mismatch "
                f"{real_mean} vs {stored_mean}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare existing Agent 2 rankings with "
            "mean input-gene-rank rankings and preview "
            "Router V4 decisions without modifying them."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--agent2-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--router-config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--router-decisions",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    project = args.project_root.resolve(strict=True)
    agent2 = args.agent2_root.resolve(strict=True)
    policy = args.router_config.resolve(strict=True)
    decisions = args.router_decisions.resolve(strict=True)
    output = args.output_dir.resolve()

    # Never overwrite prior experiment results.
    if output.exists():
        parser.error(
            "Output directory already exists. "
            "Choose a new directory."
        )

    # Never write into the production repository.
    if output == project or project in output.parents:
        parser.error(
            "Output directory must be outside the project."
        )

    for protected in (
        project,
        agent2,
        policy,
        decisions,
    ):
        if (
            output == protected
            or protected in output.parents
        ):
            parser.error(
                "Output directory overlaps protected inputs."
            )

    # Import the EXISTING Router implementation.
    # No changes are made to Router code or thresholds.
    sys.path.insert(
        0,
        str(project / "src"),
    )

    from annotation_router.runners.run_scored_router import (
        route_one_cluster,
    )

    summary_path = (
        agent2 / "all_clusters_summary.csv"
    )

    threshold_path = (
        agent2 / "all_clusters_threshold_candidates.csv"
    )

    summary = unique(
        read_csv(summary_path),
        "cluster_id",
    )

    historical_router = unique(
        read_csv(decisions),
        "cluster_id",
    )

    threshold_rows = read_csv(threshold_path)

    groups = defaultdict(list)

    for row in threshold_rows:
        groups[row["cluster_id"]].append(row)

    config = json.loads(
        policy.read_text(encoding="utf-8-sig")
    )

    # This experiment is explicitly the 8-cluster
    # Breast Xenium development dataset.
    expected_clusters = {
        str(i) for i in range(8)
    }

    if set(summary) != expected_clusters:
        raise ValueError(
            "Expected exactly Breast clusters 0 through 7."
        )

    if (
        set(historical_router) != set(summary)
        or set(groups) != set(summary)
    ):
        raise ValueError(
            "Historical Router, threshold candidates, "
            "and Agent 2 cluster IDs differ."
        )

    source_files = [
        summary_path,
        threshold_path,
        policy,
        decisions,
    ]

    for i in range(8):
        folder = (
            agent2 / "clusters" / f"cluster_{i}"
        )

        for suffix in (
            "ranked_candidate_cell_types",
            "master_gene_evidence",
        ):
            source_files.append(
                folder / f"cluster_{i}_{suffix}.csv"
            )

    missing = [
        str(path)
        for path in source_files
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing input files: " + "; ".join(missing)
        )

    before = {
        str(path): checksum(path)
        for path in source_files
    }

    full_comparison = []
    cluster_comparison = []
    router_preview = []
    router_breakdown = []

    # Calculate and validate everything before writing outputs.
    for cluster_id in sorted(summary, key=int):

        folder = (
            agent2
            / "clusters"
            / f"cluster_{cluster_id}"
        )

        candidate_file = (
            folder
            / f"cluster_{cluster_id}_ranked_candidate_cell_types.csv"
        )

        gene_file = (
            folder
            / f"cluster_{cluster_id}_master_gene_evidence.csv"
        )

        candidates = read_csv(candidate_file)
        genes = read_csv(gene_file)

        if not candidates:
            raise ValueError(
                f"No candidates for cluster {cluster_id}"
            )

        original = sorted(
            candidates,
            key=lambda row: int(row["candidate_rank"]),
        )

        candidate_ranks = [
            int(row["candidate_rank"])
            for row in original
        ]

        if candidate_ranks != list(
            range(1, len(original) + 1)
        ):
            raise ValueError(
                f"Nonconsecutive original candidate ranks "
                f"for cluster {cluster_id}"
            )

        # Verify candidate mean ranks using direct supporting genes.
        verify_means(
            cluster_id,
            original,
            genes,
        )

        # Reproduce the existing Router decision first.
        original_threshold = sorted(
            groups[cluster_id],
            key=lambda row: int(row["candidate_rank"]),
        )

        baseline = route_one_cluster(
            summary[cluster_id],
            original_threshold,
            config,
        )

        historical = historical_router[cluster_id]

        fields_to_verify = (
            "router_decision",
            "leading_candidate",
            "top_supporting_gene_count",
            "second_supporting_gene_count",
            "support_gap",
            "router_confidence_score",
        )

        number_fields = (
            "router_confidence_score",
            "top_supporting_gene_count",
            "second_supporting_gene_count",
            "support_gap",
        )

        for field in fields_to_verify:

            actual = baseline[field]
            expected = historical[field]

            if str(actual) == str(expected):
                continue

            if field in number_fields:
                if abs(
                    numeric(actual, field)
                    - numeric(expected, field)
                ) < 1e-9:
                    continue

            raise ValueError(
                f"Historical Router validation FAILED "
                f"cluster {cluster_id}, field {field}: "
                f"{actual!r} != {expected!r}"
            )

        # Approach A:
        # Rank ALL candidates by mean gene rank.
        literal_sorted = sorted(
            original,
            key=compare_sort_key,
        )

        # Approach B:
        # Rank only candidates already passing the existing
        # Agent 2 supporting-gene eligibility threshold.
        eligible_labels = {
            row["cell_type"]
            for row in original_threshold
        }

        if len(eligible_labels) != len(original_threshold):
            raise ValueError(
                "Duplicate threshold candidate labels."
            )

        by_label = unique(
            original,
            "cell_type",
        )

        if not eligible_labels.issubset(by_label):
            raise ValueError(
                f"Cluster {cluster_id}: "
                f"threshold candidate missing from original list"
            )

        eligible = sorted(
            (
                by_label[label]
                for label in eligible_labels
            ),
            key=compare_sort_key,
        )

        min_support = int(
            original_threshold[0]["minimum_supporting_genes"]
        )

        for row in eligible:
            if int(
                row["unique_supporting_gene_count"]
            ) < min_support:
                raise ValueError(
                    "Existing threshold eligibility is inconsistent."
                )

        literal_positions = {
            row["cell_type"]: index
            for index, row in enumerate(
                literal_sorted,
                start=1,
            )
        }

        eligible_positions = {
            row["cell_type"]: index
            for index, row in enumerate(
                eligible,
                start=1,
            )
        }

        # Construct a SHADOW Router input.
        # Only candidate order and leading label change.
        # Existing Router configuration, Agent 2 review
        # flags, summary counts, and thresholds stay fixed.
        shadow_summary = dict(
            summary[cluster_id]
        )

        shadow_summary["leading_candidate"] = (
            eligible[0]["cell_type"]
        )

        shadow_threshold = []

        for index, candidate in enumerate(
            eligible,
            start=1,
        ):
            row = dict(original_threshold[0])
            row.update(candidate)
            row["candidate_rank"] = str(index)
            shadow_threshold.append(row)

        shadow = route_one_cluster(
            shadow_summary,
            shadow_threshold,
            config,
        )

        decision_changed = (
            baseline["router_decision"]
            != shadow["router_decision"]
        )

        # Existing support-count selector expects
        # descending support counts.
        nonmonotonic = any(
            int(a["unique_supporting_gene_count"])
            < int(b["unique_supporting_gene_count"])
            for a, b in zip(
                eligible,
                eligible[1:],
            )
        )

        for row in original:

            label = row["cell_type"]

            full_comparison.append({
                "cluster_id": cluster_id,
                "cell_type": label,
                "original_rank": int(
                    row["candidate_rank"]
                ),
                "all_candidate_mean_rank_position": (
                    literal_positions[label]
                ),
                "existing_min_support_eligible": (
                    label in eligible_labels
                ),
                "eligible_mean_rank_position": (
                    eligible_positions.get(label, "")
                ),
                "supporting_gene_count": int(
                    row["unique_supporting_gene_count"]
                ),
                "mean_input_gene_rank": numeric(
                    row["mean_supporting_gene_rank"],
                    "mean gene rank",
                ),
                "supporting_genes": row["supporting_genes"],
            })

        cluster_comparison.append({
            "cluster_id": cluster_id,
            "original_top_candidate": (
                original[0]["cell_type"]
            ),
            "original_top_support_count": int(
                original[0]["unique_supporting_gene_count"]
            ),
            "literal_mean_rank_top_candidate": (
                literal_sorted[0]["cell_type"]
            ),
            "literal_top_support_count": int(
                literal_sorted[0]["unique_supporting_gene_count"]
            ),
            "eligible_mean_rank_top_candidate": (
                eligible[0]["cell_type"]
            ),
            "eligible_top_support_count": int(
                eligible[0]["unique_supporting_gene_count"]
            ),
            "eligible_top_mean_gene_rank": numeric(
                eligible[0]["mean_supporting_gene_rank"],
                "mean rank",
            ),
            "candidate_count_all": len(original),
            "candidate_count_existing_threshold": len(eligible),
            "literal_top_is_single_gene": (
                int(
                    literal_sorted[0]["unique_supporting_gene_count"]
                ) == 1
            ),
            "eligible_top_changed": (
                original[0]["cell_type"]
                != eligible[0]["cell_type"]
            ),
        })

        router_preview.append({
            "cluster_id": cluster_id,
            "baseline_top_candidate": (
                baseline["leading_candidate"]
            ),
            "shadow_eligible_top_candidate": (
                shadow["leading_candidate"]
            ),
            "baseline_top_gene_count": (
                baseline["top_supporting_gene_count"]
            ),
            "baseline_second_gene_count": (
                baseline["second_supporting_gene_count"]
            ),
            "shadow_top_gene_count": (
                shadow["top_supporting_gene_count"]
            ),
            "shadow_second_gene_count": (
                shadow["second_supporting_gene_count"]
            ),
            "baseline_support_gap": (
                baseline["support_gap"]
            ),
            "shadow_support_gap": (
                shadow["support_gap"]
            ),
            "baseline_router_score": (
                baseline["router_confidence_score"]
            ),
            "shadow_router_score": (
                shadow["router_confidence_score"]
            ),
            "baseline_router_decision": (
                baseline["router_decision"]
            ),
            "shadow_router_decision": (
                shadow["router_decision"]
            ),
            "decision_changed": decision_changed,
            "nonmonotonic_support_counts_in_shadow": (
                nonmonotonic
            ),
            "agent2_review_flags_held_fixed": True,
            "router_thresholds_unchanged": True,
        })

        router_breakdown.append({
            "cluster_id": cluster_id,
            "baseline": baseline,
            "shadow_eligible_rank": shadow,
        })

    # Verify input files remained untouched.
    after = {
        str(path): checksum(path)
        for path in source_files
    }

    if before != after:
        raise ValueError(
            "Protected input files changed during the experiment."
        )

    # Outputs are written only after all checks pass.
    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        output / "candidate_rank_comparison.csv",
        full_comparison,
        list(full_comparison[0]),
    )

    write_csv(
        output / "eight_cluster_rank_summary.csv",
        cluster_comparison,
        list(cluster_comparison[0]),
    )

    write_csv(
        output / "router_v4_shadow_comparison.csv",
        router_preview,
        list(router_preview[0]),
    )

    write_json(
        output / "router_v4_detailed_audit.json",
        router_breakdown,
    )

    write_json(
        output / "experiment_manifest.json",
        {
            "experiment": (
                "Agent 2 mean input-gene-rank comparison"
            ),
            "method": (
                "Reuse stored mean_supporting_gene_rank "
                "from unchanged historical Agent 2 evidence."
            ),
            "literal_mean_rank_rule": (
                "Sort all candidates by ascending mean gene rank; "
                "ties by descending support, then original rank."
            ),
            "router_shadow_rule": (
                "Reorder existing eligible candidates; hold "
                "original Agent 2 review flags and Router "
                "V4 configuration unchanged."
            ),
            "limitations": [
                "Offline shadow comparison, not deployed.",
                "Existing support-drop selection is not recalculated.",
                "Shadow candidate support counts may be nonmonotonic.",
                "Router support ratios and gaps may change meaning.",
                "Expert/reference labels are excluded from inference.",
            ],
            "input_sha256": before,
            "historical_router_reproduced_for_all_eight_clusters": True,
            "original_inputs_unchanged": True,
            "changed_eligible_top_candidates": sum(
                row["eligible_top_changed"]
                for row in cluster_comparison
            ),
            "changed_shadow_router_decisions": sum(
                row["decision_changed"]
                for row in router_preview
            ),
        },
    )

    print(
        "\nHISTORICAL ROUTER REPRODUCTION: 8/8 PASS"
    )
    print(
        "PROTECTED INPUT HASHES: UNCHANGED"
    )

    print(
        "\nCLUSTER | ORIGINAL TOP -> NEW ELIGIBLE TOP "
        "| ROUTER ORIGINAL -> SHADOW"
    )

    for cluster, router in zip(
        cluster_comparison,
        router_preview,
    ):
        print(
            f"{cluster['cluster_id']} | "
            f"{cluster['original_top_candidate']} -> "
            f"{cluster['eligible_mean_rank_top_candidate']} | "
            f"{router['baseline_router_decision']} -> "
            f"{router['shadow_router_decision']}"
        )

    print(
        "\nChanged eligible top candidates:",
        sum(
            row["eligible_top_changed"]
            for row in cluster_comparison
        ),
        "/ 8",
    )

    print(
        "Changed Router shadow decisions:",
        sum(
            row["decision_changed"]
            for row in router_preview
        ),
        "/ 8",
    )

    print(
        "\nRankings comparison saved:",
        output,
    )

    print(
        "NO SCIENTIFIC CODE OR ROUTER CONFIGURATION MODIFIED"
    )


if __name__ == "__main__":
    main()