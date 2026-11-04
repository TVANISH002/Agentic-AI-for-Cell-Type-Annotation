
"""Verify CosMX V4 settings against the saved 29-cluster export.

Read-only. Does not modify V4, Agent 2, or any previous results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

BASELINE = {
    "min_top_support_to_accept": 6,
    "close_candidate_ratio": 1.25,
    "clear_candidate_ratio": 1.50,
    "high_candidate_count": 10,
    "high_candidate_min_support": 6,
    "broad_label_candidate_threshold": 3,
    "critical_reason_count_to_route": 2,
    "accept_score_threshold": 0.75,
    "monitor_score_threshold": 0.55,
    "broad_label_terms": [
        "Cancer cell",
        "Epithelial cell",
        "Immune cell",
        "Lymphocyte",
        "Myeloid cell",
        "Stromal cell",
    ],
}

FLAGS = [
    "signal_weak_top_marker_support",
    "signal_close_competing_candidate_support",
    "signal_moderate_candidate_competition",
    "signal_large_candidate_set",
    "signal_broad_or_nonspecific_leading_label",
    "signal_agent2_review_flag",
]

WEIGHTS = [0.25, 0.25, 0.10, 0.15, 0.15, 0.05]

EXTRA = "critical_large_candidate_set_with_limited_top_support"


def read_json(path):
    with path.open(encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def differences(config):
    return {
        key: {
            "expected": expected,
            "found": config.get(key, "MISSING"),
        }
        for key, expected in BASELINE.items()
        if config.get(key) != expected
    }


def find_cosmx_configs(root):
    candidates = []

    locations = [
        root / "configs" / "router",
        root / "data" / "pipeline_runs"
        / "cosmx_brcr_validated_clean",
    ]

    for folder in locations:
        if not folder.is_dir():
            continue

        for path in folder.rglob("*.json"):
            try:
                config = read_json(path)
            except (OSError, ValueError):
                continue

            if not all(k in config for k in BASELINE):
                continue

            identity = " ".join(
                str(config.get(k, ""))
                for k in (
                    "dataset_name",
                    "agent2_summary_csv",
                    "output_dir",
                )
            ).lower()

            if "cosmx" in str(path).lower() or "cosmx" in identity:
                candidates.append(path)

    return sorted(set(candidates))


def verify_cosmx(rows, config):
    expected_ids = set(range(9))
    actual_ids = {int(row["cluster_id"]) for row in rows}

    if len(rows) != 9 or actual_ids != expected_ids:
        raise ValueError("Expected CosMX clusters 0 through 8")

    broad_labels = {
        label.casefold()
        for label in config["broad_label_terms"]
    }

    for row in rows:
        cid = int(row["cluster_id"])

        signals = [
            int(row[name])
            for name in FLAGS
        ]

        critical_extra = int(row[EXTRA])

        top = int(row["v4_top_supporting_gene_count"])
        second = int(row["v4_second_supporting_gene_count"])
        candidate_count = int(row["v4_candidate_count"])

        ratio = top / second if second else None

        expected_signals = [
            int(top < config["min_top_support_to_accept"]),

            int(
                ratio is not None
                and ratio <= config["close_candidate_ratio"]
            ),

            int(
                ratio is not None
                and config["close_candidate_ratio"]
                < ratio
                < config["clear_candidate_ratio"]
            ),

            int(
                candidate_count >= config["high_candidate_count"]
            ),

            int(
                row["agent2_top1_annotation"]
                .strip()
                .casefold()
                in broad_labels
                and candidate_count
                >= config["broad_label_candidate_threshold"]
            ),

            signals[5],
        ]

        if signals != expected_signals:
            raise ValueError(
                f"CosMX cluster {cid}: uncertainty indicator mismatch"
            )

        expected_extra = int(
            signals[3] == 1
            and top < config["high_candidate_min_support"]
        )

        if critical_extra != expected_extra:
            raise ValueError(
                f"CosMX cluster {cid}: critical condition mismatch"
            )

        critical_count = (
            signals[0] + signals[1] + critical_extra
        )

        if critical_count != int(
            row["v4_critical_uncertainty_count"]
        ):
            raise ValueError(
                f"CosMX cluster {cid}: critical count mismatch"
            )

        penalty = sum(
            flag * weight
            for flag, weight in zip(signals, WEIGHTS)
        )

        score = round(
            max(0.0, 1.0 - penalty),
            3,
        )

        if not math.isclose(
            score,
            float(row["v4_score"]),
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"CosMX cluster {cid}: V4 score mismatch"
            )

        if critical_count >= config["critical_reason_count_to_route"]:
            decision = "ROUTE"

        elif (
            score >= config["accept_score_threshold"]
            and critical_count == 0
        ):
            decision = "ACCEPT"

        elif score >= config["monitor_score_threshold"]:
            decision = "MONITOR"

        else:
            decision = "ROUTE"

        if decision != row["v4_final_routing_decision"]:
            raise ValueError(
                f"CosMX cluster {cid}: routing decision mismatch"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
    )

    parser.add_argument(
        "--cosmx-config",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    root = args.root.resolve()

    csv_path = (
        root
        / "data"
        / "analysis"
        / "router_v4_uncertainty_export_v1"
        / "router_v4_uncertainty_indicators_29_clusters.csv"
    )

    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    with csv_path.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        all_rows = list(csv.DictReader(handle))

    if len(all_rows) != 29:
        raise ValueError(
            f"Expected 29 clusters; found {len(all_rows)}"
        )

    cosmx_rows = [
        row
        for row in all_rows
        if row["dataset"] == "Breast CosMX"
    ]

    if len(cosmx_rows) != 9:
        raise ValueError("Expected nine CosMX clusters")

    # Check Breast/Lung V4 configurations.
    for dataset in ("breast", "lung"):
        config_path = (
            root
            / "configs"
            / "router"
            / f"{dataset}_router_v4.json"
        )

        config = read_json(config_path)

        mismatch = differences(config)

        if mismatch:
            raise ValueError(
                f"{dataset} V4 settings differ: {mismatch}"
            )

        print(
            f"PASS: {dataset.title()} V4 configuration"
        )

    # Find a saved CosMX-specific configuration.
    if args.cosmx_config:
        path = args.cosmx_config
        if not path.is_absolute():
            path = root / path

        if not path.is_file():
            raise FileNotFoundError(path)

        candidates = [path]

    else:
        candidates = find_cosmx_configs(root)

    if len(candidates) > 1:
        print("Multiple CosMX configurations found:")

        for path in candidates:
            print(path)

        raise SystemExit(
            "Specify the correct file with --cosmx-config."
        )

    if candidates:
        path = candidates[0]
        config = read_json(path)

        mismatch = differences(config)

        if mismatch:
            raise ValueError(
                f"CosMX V4 settings differ: {mismatch}"
            )

        expected_output = (
            "cosmx_brcr_validated_clean/router_output_v4"
        )

        output_path = (
            str(config.get("output_dir", ""))
            .replace("\\", "/")
            .lower()
        )

        dataset_name = str(
            config.get("dataset_name", "")
        ).lower()

        if (
            expected_output not in output_path
            or "cosmx" not in dataset_name
        ):
            raise ValueError(
                "The configuration does not identify "
                "the validated CosMX V4 run."
            )

        verify_cosmx(cosmx_rows, config)

        print("PASS: CosMX V4 configuration values")
        print("PASS: All nine CosMX indicator records")
        print("Configuration:", path)
        print("STATUS: CONFIG_FILE_AND_OUTPUTS_CONSISTENT")

    else:
        # Check numerical consistency without claiming
        # the original configuration file was recovered.
        verify_cosmx(cosmx_rows, BASELINE)

        print(
            "PASS: All nine CosMX records match "
            "the common V4 defaults"
        )

        print(
            "WARNING: Original CosMX run configuration "
            "was not found"
        )

        print(
            "STATUS: OUTPUTS_VERIFIED_CONFIG_PROVENANCE_PENDING"
        )

    print("Existing files modified: NONE")


if __name__ == "__main__":
    main()
