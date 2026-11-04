
"""Export frozen V4 pre-penalty uncertainty indicators for 29 clusters.

Run at the VS Code project root. Reads data only; does not train trees.
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

EVIDENCE = [
    "direct_evidence_coverage",
    "top1_support_fraction",
    "top1_top2_support_gap",
    "cross_database_support_fraction",
    "first_direct_supported_marker_rank_norm",
    "rank_weighted_unexplained_marker_score",
]

UNCERTAINTY = [
    "signal_weak_top_marker_support",
    "signal_close_competing_candidate_support",
    "signal_moderate_candidate_competition",
    "signal_large_candidate_set",
    "signal_broad_or_nonspecific_leading_label",
    "signal_agent2_review_flag",
]

WEIGHTS = dict(
    zip(UNCERTAINTY, [0.25, 0.25, 0.10, 0.15, 0.15, 0.05])
)

EXTRA = "critical_large_candidate_set_with_limited_top_support"

DATASETS = {
    "Breast Xenium": (8, "breast_xenium_marker_only_clean"),
    "Lung Xenium": (12, "lung_validated_clean"),
    "Breast CosMX": (9, "cosmx_brcr_validated_clean"),
}

DECISION_MAP = {
    "accept_agent2_annotation": "ACCEPT",
    "monitor_uncertainty": "MONITOR",
    "route_to_agent3": "ROUTE",
}

DEFAULTS = {
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


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}")

    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")

        return list(reader)


def require(rows, fields, name):
    if not rows:
        raise ValueError(f"No rows in {name}")

    missing = set(fields) - rows[0].keys()

    if missing:
        raise ValueError(
            f"Missing columns in {name}: {sorted(missing)}"
        )


def write_csv(path, cols, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=cols,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()


def as_int(value, name):
    try:
        x = float(value)

        if not math.isfinite(x) or not x.is_integer():
            raise ValueError()

        return int(x)

    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid integer {name}: {value!r}"
        ) from None


def as_float(value, name):
    try:
        x = float(value)

        if not math.isfinite(x):
            raise ValueError()

        return x

    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid number {name}: {value!r}"
        ) from None


def as_bit(value, name):
    value = str(value).strip().lower()

    if value in ("true", "yes", "1"):
        return 1

    if value in ("false", "no", "0"):
        return 0

    raise ValueError(
        f"Invalid Boolean {name}: {value!r}"
    )


def resolve(root, value):
    path = Path(value)

    return path if path.is_absolute() else root / path


def find_cosmx_v4(root, explicit):
    if explicit:
        return resolve(root, explicit)

    base = root / "data/pipeline_runs"

    paths = [
        base
        / "cosmx_brcr_validated_clean"
        / "router_output_v4"
        / "router_decisions.csv",

        base
        / "cosmx_brcr_validated_clean"
        / "router_output"
        / "router_decisions.csv",

        base
        / "cosmx_brcr_orchestration_validation"
        / "router_output_v4"
        / "router_decisions.csv",

        base
        / "cosmx_brcr_orchestration_validation"
        / "router_output"
        / "router_decisions.csv",
    ]

    e2e = base / "cosmx_brcr_end_to_end_runs"

    if e2e.exists():
        paths += sorted(
            e2e.glob("*/router_output_v4/router_decisions.csv")
        )
        paths += sorted(
            e2e.glob("*/router_output/router_decisions.csv")
        )

    existing = [p for p in paths if p.is_file()]

    if len(existing) == 1:
        return existing[0]

    raise FileNotFoundError(
        "Expected exactly one CosMX V4 "
        f"router_decisions.csv, found {len(existing)}. "
        "Use --cosmx-v4-csv with the correct "
        "frozen V4 file.\n"
        + "\n".join(map(str, existing))
    )


def get_config(root, dataset, explicit_cosmx_config):
    if dataset == "Breast CosMX":
        if not explicit_cosmx_config:
            return (
                DEFAULTS.copy(),
                "V4 defaults (CosMX config not supplied)",
                None,
            )

        path = resolve(root, explicit_cosmx_config)

    else:
        prefix = (
            "breast"
            if dataset == "Breast Xenium"
            else "lung"
        )

        path = (
            root
            / f"configs/router/{prefix}_router_v4.json"
        )

    return (
        {
            **DEFAULTS,
            **json.loads(
                path.read_text(encoding="utf-8-sig")
            ),
        },
        str(path),
        sha256(path),
    )


def validate(row, cfg, dataset, cid):
    label = f"{dataset} cluster {cid}"

    top = as_int(
        row["top_supporting_gene_count"],
        label + ".top",
    )

    second = as_int(
        row["second_supporting_gene_count"],
        label + ".second",
    )

    count = as_int(
        row["candidate_count"],
        label + ".candidates",
    )

    if top < 0 or second < 0 or count < 1:
        raise ValueError(
            f"Invalid evidence values: {label}"
        )

    ratio = top / second if second > 0 else None

    flags = {
        x: as_bit(row[x], label + "." + x)
        for x in UNCERTAINTY
    }

    broad = {
        s.lower() for s in cfg["broad_label_terms"]
    }

    predicted = [
        top < cfg["min_top_support_to_accept"],

        ratio is not None
        and ratio <= cfg["close_candidate_ratio"],

        ratio is not None
        and cfg["close_candidate_ratio"]
        < ratio
        < cfg["clear_candidate_ratio"],

        count >= cfg["high_candidate_count"],

        row["leading_candidate"].strip().lower() in broad
        and count >= cfg["broad_label_candidate_threshold"],

        bool(
            as_bit(
                row["human_review_required"],
                label + ".review",
            )
        ),
    ]

    for key, expected in zip(UNCERTAINTY, predicted):
        if flags[key] != int(expected):
            raise ValueError(
                f"{label}: saved {key} disagrees "
                "with config. Check you selected "
                "the original V4 config and output."
            )

    extra = int(
        flags[UNCERTAINTY[3]] == 1
        and top < cfg["high_candidate_min_support"]
    )

    critical_expected = []

    if flags[UNCERTAINTY[0]]:
        critical_expected.append(
            "weak_top_marker_support"
        )

    if flags[UNCERTAINTY[1]]:
        critical_expected.append(
            "close_competing_candidate_support"
        )

    if extra:
        critical_expected.append(
            "large_candidate_set_with_limited_top_support"
        )

    critical_actual = [
        s.strip()
        for s in row["critical_reasons"].split("|")
        if s.strip()
    ]

    if (
        len(critical_actual) != len(set(critical_actual))
        or set(critical_actual) != set(critical_expected)
    ):
        raise ValueError(
            f"{label}: critical reason mismatch: "
            f"{critical_actual}"
        )

    total_penalty = round(
        sum(
            flags[x] * WEIGHTS[x]
            for x in UNCERTAINTY
        ),
        3,
    )

    score = as_float(
        row["router_confidence_score"],
        label + ".score",
    )

    if abs(
        as_float(
            row["total_penalty"],
            label + ".penalty",
        )
        - total_penalty
    ) > 1e-9:
        raise ValueError(
            f"{label}: penalty mismatch"
        )

    if abs(
        score
        - round(
            max(0.0, 1.0 - total_penalty),
            3,
        )
    ) > 1e-9:
        raise ValueError(
            f"{label}: V4 score mismatch"
        )

    critical_count = len(critical_actual)

    expected_decision = (
        "ROUTE"
        if critical_count >= cfg["critical_reason_count_to_route"]
        else "ACCEPT"
        if (
            score >= cfg["accept_score_threshold"]
            and critical_count == 0
        )
        else "MONITOR"
        if score >= cfg["monitor_score_threshold"]
        else "ROUTE"
    )

    if (
        DECISION_MAP.get(
            row["router_decision"].strip()
        )
        != expected_decision
    ):
        raise ValueError(
            f"{label}: V4 decision mismatch"
        )

    return (
        flags,
        extra,
        critical_count,
        score,
        top,
        second,
        ratio,
        count,
    )


def definitions(cfg, dataset, source):
    terms = ", ".join(cfg["broad_label_terms"])

    rules = [
        (
            "Top-1 supporting gene count "
            f"< {cfg['min_top_support_to_accept']}"
        ),

        (
            "Top-2 support > 0 and Top-1/Top-2 "
            f"support ratio <= {cfg['close_candidate_ratio']}"
        ),

        (
            "Top-2 support > 0 and "
            f"{cfg['close_candidate_ratio']} "
            f"< ratio < {cfg['clear_candidate_ratio']}"
        ),

        (
            "Number of candidate cell types "
            f">= {cfg['high_candidate_count']}"
        ),

        (
            f"Top-1 in broad labels [{terms}] "
            "and candidate count >= "
            f"{cfg['broad_label_candidate_threshold']}"
        ),

        "Agent 2 human_review_required is True",

        (
            "Large candidate set = 1 and "
            "Top-1 supporting genes "
            f"< {cfg['high_candidate_min_support']}"
        ),
    ]

    names = UNCERTAINTY + [EXTRA]

    return [
        {
            "dataset": dataset,
            "indicator": key,
            "definition_of_value_1": rule,
            "value_0": "Condition absent",
            "value_1": "Condition present",
            "role": (
                "critical-only, no additional penalty"
                if key == EXTRA
                else "penalty trigger; also critical"
                if key in UNCERTAINTY[:2]
                else "penalty trigger"
            ),
            "configuration_source": source,
        }
        for key, rule in zip(names, rules)
    ]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--project-root",
        default=".",
    )

    parser.add_argument(
        "--master-csv",
        default=(
            "data/analysis/"
            "router_six_features_all_datasets_v4/"
            "router_comparison_29_clusters.csv"
        ),
    )

    parser.add_argument(
        "--cosmx-v4-csv",
        default=None,
    )

    parser.add_argument(
        "--cosmx-config",
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "data/analysis/"
            "router_v4_uncertainty_export_v1"
        ),
    )

    args = parser.parse_args()

    root = Path(args.project_root).resolve()

    output_dir = resolve(
        root, args.output_dir
    )

    if output_dir.exists():
        raise FileExistsError(
            "Safety stop; directory already exists: "
            f"{output_dir}"
        )

    master_path = resolve(
        root, args.master_csv
    )

    master_rows = read_csv(
        master_path
    )

    require(
        master_rows,
        [
            "dataset",
            "platform",
            "cluster_id",
            "agent2_top1_annotation",
            "reference_cell_type_annotation",
            "correctness_label_0_1",
            "router_v4_decision",
            *EVIDENCE,
        ],
        master_path,
    )

    master = {}

    for row in master_rows:
        key = (
            row["dataset"],
            as_int(
                row["cluster_id"],
                "master.cluster_id",
            ),
        )

        if key in master:
            raise ValueError(
                f"Duplicate master cluster: {key}"
            )

        master[key] = row

    expected = {
        (dataset, cid)
        for dataset, (n, _) in DATASETS.items()
        for cid in range(n)
    }

    if set(master) != expected:
        raise ValueError(
            "Master must contain exactly 29 clusters "
            "across all three datasets"
        )

    hashes = {
        str(master_path): sha256(master_path)
    }

    output_rows = []
    definition_rows = []

    for dataset, (n, run) in DATASETS.items():
        if dataset == "Breast CosMX":
            v4_path = find_cosmx_v4(
                root, args.cosmx_v4_csv
            )

        else:
            v4_path = (
                root
                / "data/pipeline_runs"
                / run
                / "router_output_v4"
                / "router_decisions.csv"
            )

        cfg, cfg_source, cfg_hash = get_config(
            root,
            dataset,
            args.cosmx_config,
        )

        if cfg_hash:
            hashes[cfg_source] = cfg_hash

        v4_rows = read_csv(v4_path)

        require(
            v4_rows,
            [
                "cluster_id",
                "leading_candidate",
                "top_supporting_gene_count",
                "second_supporting_gene_count",
                "candidate_count",
                "human_review_required",
                "critical_reasons",
                "total_penalty",
                "router_confidence_score",
                "router_decision",
                *UNCERTAINTY,
            ],
            v4_path,
        )

        if len(v4_rows) != n:
            raise ValueError(
                f"{dataset}: expected {n} V4 rows, "
                f"found {len(v4_rows)}"
            )

        hashes[str(v4_path)] = sha256(
            v4_path
        )

        seen = set()

        for v4 in v4_rows:
            cid = as_int(
                v4["cluster_id"],
                dataset + ".cluster_id",
            )

            if cid not in range(n) or cid in seen:
                raise ValueError(
                    "Unexpected/duplicate cluster "
                    f"in {dataset}: {cid}"
                )

            seen.add(cid)

            m = master[(dataset, cid)]

            if (
                v4["leading_candidate"].strip().casefold()
                != m["agent2_top1_annotation"]
                .strip().casefold()
            ):
                raise ValueError(
                    "Agent2 label mismatch: "
                    f"{dataset} cluster {cid}"
                )

            if (
                DECISION_MAP.get(
                    v4["router_decision"].strip()
                )
                != m["router_v4_decision"]
                .strip().upper()
            ):
                raise ValueError(
                    "Stored decision mismatch: "
                    f"{dataset} cluster {cid}"
                )

            (
                flags,
                extra,
                critical_count,
                score,
                top,
                second,
                ratio,
                cand_count,
            ) = validate(
                v4,
                cfg,
                dataset,
                cid,
            )

            if (
                dataset == "Breast CosMX"
                and m["correctness_label_0_1"].strip()
            ):
                raise ValueError(
                    "CosMX correctness is unverified; "
                    "expected blank"
                )

            output_rows.append({
                "dataset": dataset,
                "platform": m["platform"],
                "cluster_id": cid,

                "agent2_top1_annotation":
                    m["agent2_top1_annotation"],

                "reference_cell_type_annotation":
                    m["reference_cell_type_annotation"],

                "correctness_label_0_1":
                    m["correctness_label_0_1"],

                **{
                    f: as_float(
                        m[f],
                        f"{dataset}.{cid}.{f}"
                    )
                    for f in EVIDENCE
                },

                **flags,

                EXTRA: extra,

                "v4_top_supporting_gene_count":
                    top,

                "v4_second_supporting_gene_count":
                    second,

                "v4_support_ratio":
                    ""
                    if ratio is None
                    else round(ratio, 6),

                "v4_candidate_count":
                    cand_count,

                "v4_score":
                    score,

                "v4_critical_uncertainty_count":
                    critical_count,

                "v4_critical_reasons":
                    v4["critical_reasons"],

                "v4_final_routing_decision":
                    DECISION_MAP[
                        v4["router_decision"].strip()
                    ],
            })

        definition_rows.extend(
            definitions(
                cfg,
                dataset,
                cfg_source,
            )
        )

        print(
            f"PASS: {dataset}: {n} clusters",
            flush=True,
        )

    if len(output_rows) != 29:
        raise ValueError(
            "Expected 29 rows"
        )

    metadata = [
        "dataset",
        "platform",
        "cluster_id",
        "agent2_top1_annotation",
        "reference_cell_type_annotation",
        "correctness_label_0_1",
    ]

    verification = [
        "v4_top_supporting_gene_count",
        "v4_second_supporting_gene_count",
        "v4_support_ratio",
        "v4_candidate_count",
        "v4_score",
        "v4_critical_uncertainty_count",
        "v4_critical_reasons",
        "v4_final_routing_decision",
    ]

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    outputs = {
        "router_v4_uncertainty_indicators_29_clusters.csv":
            metadata + UNCERTAINTY + [EXTRA] + verification,

        "router_tree_inputs_evidence_29.csv":
            metadata + EVIDENCE,

        "router_tree_inputs_uncertainty_29.csv":
            metadata + UNCERTAINTY,

        "router_tree_inputs_combined_29.csv":
            metadata + EVIDENCE + UNCERTAINTY,

        "router_full_v4_feature_audit_29.csv":
            metadata
            + EVIDENCE
            + UNCERTAINTY
            + [EXTRA]
            + verification,
    }

    for filename, cols in outputs.items():
        write_csv(
            output_dir / filename,
            cols,
            output_rows,
        )

    write_csv(
        output_dir / "router_v4_indicator_definitions.csv",
        list(definition_rows[0]),
        definition_rows,
    )

    audit = {
        "created_utc":
            datetime.now(timezone.utc).isoformat(),

        "clusters":
            len(output_rows),

        "dataset_counts":
            dict(
                Counter(
                    r["dataset"]
                    for r in output_rows
                )
            ),

        "evidence_feature_columns":
            EVIDENCE,

        "v4_uncertainty_feature_columns":
            UNCERTAINTY,

        "separate_critical_only_indicator":
            EXTRA,

        "tree_inputs_exclude_v4_score_critical_count_and_final_decision":
            True,

        "cosmx_configuration":
            (
                "provided"
                if args.cosmx_config
                else
                "V4 code defaults only; "
                "confirm original CosMX config"
            ),

        "input_sha256":
            hashes,

        "retrained_or_modified_router":
            False,
    }

    (
        output_dir
        / "router_v4_uncertainty_export_audit.json"
    ).write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    print(
        "STATUS: SUCCESS - ALL 29 "
        "V4 UNCERTAINTY ROWS EXPORTED"
    )

    print(
        "Output directory:",
        output_dir,
    )

    print(
        "No existing files changed; "
        "no decision trees trained"
    )


if __name__ == "__main__":
    main()
