from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


 #INDEPENDENT ROUTER V4 RECONSTRUCTION AUDIT



PENALTIES = {
    "weak_top_marker_support": 0.25,
    "close_competing_candidate_support": 0.25,
    "moderate_candidate_competition": 0.10,
    "large_candidate_set": 0.15,
    "broad_or_nonspecific_leading_label": 0.15,
    "agent2_review_flag": 0.05,
}

DECISION_FIELDS = [
    "cluster_id",
    "leading_candidate",
    "top_supporting_gene_count",
    "second_supporting_gene_count",
    "support_gap",
    "support_ratio",
    "candidate_count",
    "human_review_required",
    "agent2_flag_codes",
    "signal_weak_top_marker_support",
    "signal_close_competing_candidate_support",
    "signal_moderate_candidate_competition",
    "signal_large_candidate_set",
    "signal_broad_or_nonspecific_leading_label",
    "signal_agent2_review_flag",
    "penalty_weak_top_marker_support",
    "penalty_close_competing_candidate_support",
    "penalty_moderate_candidate_competition",
    "penalty_large_candidate_set",
    "penalty_broad_or_nonspecific_leading_label",
    "penalty_agent2_review_flag",
    "total_penalty",
    "score_calculation",
    "router_confidence_score",
    "router_confidence_level",
    "router_decision",
    "human_in_loop_default_action",
    "router_reasons",
    "critical_reasons",
    "selected_candidate_cell_types",
    "supporting_markers",
]

INTEGER_FIELDS = {
    "top_supporting_gene_count",
    "second_supporting_gene_count",
    "support_gap",
    "candidate_count",
}

FLOAT_FIELDS = {
    "support_ratio",
    "penalty_weak_top_marker_support",
    "penalty_close_competing_candidate_support",
    "penalty_moderate_candidate_competition",
    "penalty_large_candidate_set",
    "penalty_broad_or_nonspecific_leading_label",
    "penalty_agent2_review_flag",
    "total_penalty",
    "router_confidence_score",
}

BOOLEAN_FIELDS = {
    "human_review_required",
    "signal_weak_top_marker_support",
    "signal_close_competing_candidate_support",
    "signal_moderate_candidate_competition",
    "signal_large_candidate_set",
    "signal_broad_or_nonspecific_leading_label",
    "signal_agent2_review_flag",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(
            csv.DictReader(
                handle
            )
        )


def as_int(
    value: Any,
    default: int = 0,
) -> int:
    try:
        if value is None or str(value).strip() == "":
            return default

        return int(
            float(
                value
            )
        )
    except Exception:
        return default


def as_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default

        return float(
            value
        )
    except Exception:
        return default


def as_bool(
    value: Any,
) -> bool:
    return (
        str(
            value
        )
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
            "y",
        }
    )


def clean_text(
    value: Any,
) -> str:
    if value is None:
        return ""

    return str(
        value
    ).strip()


def split_cell_types(
    value: Any,
) -> list[str]:
    text = clean_text(
        value
    )

    if not text:
        return []

    return [
        item.strip()
        for item in (
            text
            .replace(
                "|",
                ";",
            )
            .split(
                ";"
            )
        )
        if item.strip()
    ]


def is_broad_label(
    label: str,
    broad_terms: list[str],
) -> bool:
    target = (
        clean_text(
            label
        ).lower()
    )

    return target in {
        clean_text(
            item
        ).lower()
        for item in broad_terms
    }


def load_candidates(
    path: Path,
) -> dict[str, list[dict[str, str]]]:

    grouped: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(
        list
    )

    for row in read_csv(
        path
    ):
        cluster_id = clean_text(
            row.get(
                "cluster_id"
            )
        )

        if cluster_id:
            grouped[
                cluster_id
            ].append(
                row
            )

    for cluster_id in grouped:
        grouped[
            cluster_id
        ].sort(
            key=lambda row: as_int(
                row.get(
                    "candidate_rank"
                ),
                999999,
            )
        )

    return grouped


def reconstruct_one_cluster(
    summary_row: dict[str, str],
    candidates: list[
        dict[str, str]
    ],
    config: dict[str, Any],
) -> dict[str, Any]:

    cluster_id = clean_text(
        summary_row.get(
            "cluster_id"
        )
    )

    leading_candidate = clean_text(
        summary_row.get(
            "leading_candidate"
        )
    )

    selected = split_cell_types(
        summary_row.get(
            "selected_candidate_cell_types",
            "",
        )
    )

    candidate_count = max(
        as_int(
            summary_row.get(
                "selected_candidate_count"
            ),
            len(
                selected
            ),
        ),
        as_int(
            summary_row.get(
                "threshold_candidate_count"
            ),
            len(
                candidates
            ),
        ),
        len(
            candidates
        ),
        1,
    )

    if candidates:
        top_support = as_int(
            candidates[
                0
            ].get(
                "unique_supporting_gene_count"
            ),
            0,
        )

        second_support = (
            as_int(
                candidates[
                    1
                ].get(
                    "unique_supporting_gene_count"
                ),
                0,
            )
            if len(
                candidates
            )
            > 1
            else 0
        )

    else:
        top_support = as_int(
            summary_row.get(
                "supporting_gene_count"
            ),
            as_int(
                summary_row.get(
                    "direct_marker_supported_genes"
                ),
                0,
            ),
        )

        second_support = 0

    human_review_required = as_bool(
        summary_row.get(
            "human_review_required"
        )
    )

    flag_codes = (
        summary_row.get(
            "review_flag_codes"
        )
        or summary_row.get(
            "warning_flag_codes"
        )
        or ""
    )

    min_top_support = as_int(
        config.get(
            "min_top_support_to_accept"
        ),
        6,
    )

    close_ratio = as_float(
        config.get(
            "close_candidate_ratio"
        ),
        1.25,
    )

    clear_ratio = as_float(
        config.get(
            "clear_candidate_ratio"
        ),
        1.50,
    )

    high_candidate_count = as_int(
        config.get(
            "high_candidate_count"
        ),
        10,
    )

    high_candidate_min_support = as_int(
        config.get(
            "high_candidate_min_support"
        ),
        min_top_support,
    )

    broad_threshold = as_int(
        config.get(
            "broad_label_candidate_threshold"
        ),
        3,
    )

    broad_terms = list(
        config.get(
            "broad_label_terms",
            [],
        )
    )

    accept_score = as_float(
        config.get(
            "accept_score_threshold"
        ),
        0.75,
    )

    monitor_score = as_float(
        config.get(
            "monitor_score_threshold"
        ),
        0.55,
    )

    critical_count_to_route = as_int(
        config.get(
            "critical_reason_count_to_route"
        ),
        2,
    )

    support_gap = (
        top_support
        - second_support
    )

    support_ratio: float | str = (
        round(
            top_support
            / second_support,
            3,
        )
        if second_support > 0
        else ""
    )

    score = 1.0

    reasons: list[
        str
    ] = []

    critical_reasons: list[
        str
    ] = []

    signal_values = {
        key: False
        for key in PENALTIES
    }

    applied_penalties = {
        key: 0.0
        for key in PENALTIES
    }

    if (
        top_support
        < min_top_support
    ):
        signal_values[
            "weak_top_marker_support"
        ] = True

        applied_penalties[
            "weak_top_marker_support"
        ] = PENALTIES[
            "weak_top_marker_support"
        ]

        score -= applied_penalties[
            "weak_top_marker_support"
        ]

        reasons.append(
            "weak_top_marker_support"
        )

        critical_reasons.append(
            "weak_top_marker_support"
        )

    if second_support > 0:
        ratio = (
            top_support
            / second_support
        )

        if ratio <= close_ratio:
            signal_values[
                "close_competing_candidate_support"
            ] = True

            applied_penalties[
                "close_competing_candidate_support"
            ] = PENALTIES[
                "close_competing_candidate_support"
            ]

            score -= applied_penalties[
                "close_competing_candidate_support"
            ]

            reasons.append(
                "close_competing_candidate_support"
            )

            critical_reasons.append(
                "close_competing_candidate_support"
            )

        elif ratio < clear_ratio:
            signal_values[
                "moderate_candidate_competition"
            ] = True

            applied_penalties[
                "moderate_candidate_competition"
            ] = PENALTIES[
                "moderate_candidate_competition"
            ]

            score -= applied_penalties[
                "moderate_candidate_competition"
            ]

            reasons.append(
                "moderate_candidate_competition"
            )

    if (
        candidate_count
        >= high_candidate_count
    ):
        signal_values[
            "large_candidate_set"
        ] = True

        applied_penalties[
            "large_candidate_set"
        ] = PENALTIES[
            "large_candidate_set"
        ]

        score -= applied_penalties[
            "large_candidate_set"
        ]

        reasons.append(
            "large_candidate_set"
        )

        if (
            top_support
            < high_candidate_min_support
        ):
            critical_reasons.append(
                "large_candidate_set_with_limited_top_support"
            )

    if (
        is_broad_label(
            leading_candidate,
            broad_terms,
        )
        and candidate_count
        >= broad_threshold
    ):
        signal_values[
            "broad_or_nonspecific_leading_label"
        ] = True

        applied_penalties[
            "broad_or_nonspecific_leading_label"
        ] = PENALTIES[
            "broad_or_nonspecific_leading_label"
        ]

        score -= applied_penalties[
            "broad_or_nonspecific_leading_label"
        ]

        reasons.append(
            "broad_or_nonspecific_leading_label"
        )

    if human_review_required:
        signal_values[
            "agent2_review_flag"
        ] = True

        applied_penalties[
            "agent2_review_flag"
        ] = PENALTIES[
            "agent2_review_flag"
        ]

        score -= applied_penalties[
            "agent2_review_flag"
        ]

        reasons.append(
            "agent2_review_flag"
        )

    total_penalty = round(
        sum(
            applied_penalties.values()
        ),
        3,
    )

    score = round(
        max(
            score,
            0.0,
        ),
        3,
    )

    if (
        len(
            critical_reasons
        )
        >= critical_count_to_route
    ):
        decision = (
            "route_to_agent3"
        )

    elif (
        score
        >= accept_score
        and not critical_reasons
    ):
        decision = (
            "accept_agent2_annotation"
        )

    elif (
        score
        >= monitor_score
    ):
        decision = (
            "monitor_uncertainty"
        )

    else:
        decision = (
            "route_to_agent3"
        )

    confidence_level = (
        "high"
        if score >= accept_score
        else (
            "moderate"
            if score >= monitor_score
            else "low"
        )
    )

    if (
        decision
        == "accept_agent2_annotation"
    ):
        default_action = "accept"

    elif (
        decision
        == "monitor_uncertainty"
    ):
        default_action = (
            "expert_review_or_route_to_agent3_if_no_expert"
        )

    else:
        default_action = (
            "route_to_agent3"
        )

    return {
        "cluster_id": cluster_id,
        "leading_candidate": leading_candidate,
        "top_supporting_gene_count": top_support,
        "second_supporting_gene_count": second_support,
        "support_gap": support_gap,
        "support_ratio": support_ratio,
        "candidate_count": candidate_count,
        "human_review_required": str(
            human_review_required
        ),
        "agent2_flag_codes": flag_codes,
        "signal_weak_top_marker_support": str(
            signal_values[
                "weak_top_marker_support"
            ]
        ),
        "signal_close_competing_candidate_support": str(
            signal_values[
                "close_competing_candidate_support"
            ]
        ),
        "signal_moderate_candidate_competition": str(
            signal_values[
                "moderate_candidate_competition"
            ]
        ),
        "signal_large_candidate_set": str(
            signal_values[
                "large_candidate_set"
            ]
        ),
        "signal_broad_or_nonspecific_leading_label": str(
            signal_values[
                "broad_or_nonspecific_leading_label"
            ]
        ),
        "signal_agent2_review_flag": str(
            signal_values[
                "agent2_review_flag"
            ]
        ),
        "penalty_weak_top_marker_support": applied_penalties[
            "weak_top_marker_support"
        ],
        "penalty_close_competing_candidate_support": applied_penalties[
            "close_competing_candidate_support"
        ],
        "penalty_moderate_candidate_competition": applied_penalties[
            "moderate_candidate_competition"
        ],
        "penalty_large_candidate_set": applied_penalties[
            "large_candidate_set"
        ],
        "penalty_broad_or_nonspecific_leading_label": applied_penalties[
            "broad_or_nonspecific_leading_label"
        ],
        "penalty_agent2_review_flag": applied_penalties[
            "agent2_review_flag"
        ],
        "total_penalty": total_penalty,
        "score_calculation": (
            f"max(0, 1.0 - {total_penalty:.3f})"
        ),
        "router_confidence_score": score,
        "router_confidence_level": confidence_level,
        "router_decision": decision,
        "human_in_loop_default_action": default_action,
        "router_reasons": (
            " | ".join(
                reasons
            )
            if reasons
            else "clear_agent2_evidence"
        ),
        "critical_reasons": (
            " | ".join(
                critical_reasons
            )
        ),
        "selected_candidate_cell_types": summary_row.get(
            "selected_candidate_cell_types",
            "",
        ),
        "supporting_markers": summary_row.get(
            "supporting_markers",
            "",
        ),
    }


def normalize_for_compare(
    field: str,
    value: Any,
) -> Any:

    if field in INTEGER_FIELDS:
        return as_int(
            value,
            0,
        )

    if field in FLOAT_FIELDS:
        text = clean_text(
            value
        )

        if not text:
            return None

        return float(
            text
        )

    if field in BOOLEAN_FIELDS:
        return as_bool(
            value
        )

    return clean_text(
        value
    )


def values_equal(
    field: str,
    stored: Any,
    reconstructed: Any,
) -> bool:

    left = normalize_for_compare(
        field,
        stored,
    )

    right = normalize_for_compare(
        field,
        reconstructed,
    )

    if field in FLOAT_FIELDS:
        if (
            left is None
            and right is None
        ):
            return True

        if (
            left is None
            or right is None
        ):
            return False

        return math.isclose(
            float(
                left
            ),
            float(
                right
            ),
            rel_tol=1e-9,
            abs_tol=1e-9,
        )

    return left == right


def cluster_ids_from_rows(
    rows: list[
        dict[str, Any]
    ],
) -> list[str]:

    return [
        clean_text(
            row.get(
                "cluster_id"
            )
        )
        for row in rows
    ]


def resolve_path(
    root: Path,
    path_text: str,
) -> Path:

    path = Path(
        path_text
    )

    if not path.is_absolute():
        path = (
            root
            / path
        )

    return path


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(
            "."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/analysis/pipeline_validation/"
            "router_v4_independent_reconstruction"
        ),
    )

    args = parser.parse_args()

    with args.manifest.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        manifest = list(
            csv.DictReader(
                handle
            )
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_reports: list[
        dict[str, Any]
    ] = []

    mismatch_rows: list[
        dict[str, Any]
    ] = []

    reconstructed_rows: list[
        dict[str, Any]
    ] = []

    for item in manifest:
        dataset = clean_text(
            item.get(
                "dataset"
            )
        )

        config_path = resolve_path(
            args.project_root,
            clean_text(
                item.get(
                    "router_config"
                )
            ),
        )

        if not config_path.exists():
            raise FileNotFoundError(
                f"{dataset}: config not found: {config_path}"
            )

        config = json.loads(
            config_path.read_text(
                encoding="utf-8-sig"
            )
        )

        summary_path = resolve_path(
            args.project_root,
            config[
                "agent2_summary_csv"
            ],
        )

        candidates_path = resolve_path(
            args.project_root,
            config[
                "agent2_candidates_csv"
            ],
        )

        router_output_dir = resolve_path(
            args.project_root,
            config[
                "output_dir"
            ],
        )

        stored_decision_path = (
            router_output_dir
            / "router_decisions.csv"
        )

        stored_summary_path = (
            router_output_dir
            / "router_summary.json"
        )

        stored_agent3_path = (
            router_output_dir
            / "agent3_queue.csv"
        )

        stored_monitor_path = (
            router_output_dir
            / "monitor_queue.csv"
        )

        stored_no_expert_path = (
            router_output_dir
            / "agent3_queue_if_no_expert.csv"
        )

        required_files = [
            summary_path,
            candidates_path,
            stored_decision_path,
            stored_summary_path,
            stored_agent3_path,
            stored_monitor_path,
            stored_no_expert_path,
        ]

        for path in required_files:
            if not path.exists():
                raise FileNotFoundError(
                    f"{dataset}: required file not found: {path}"
                )

        summary_rows = read_csv(
            summary_path
        )

        candidates_by_cluster = load_candidates(
            candidates_path
        )

        reconstructed: list[
            dict[str, Any]
        ] = []

        for summary_row in summary_rows:
            cluster_id = clean_text(
                summary_row.get(
                    "cluster_id"
                )
            )

            decision = reconstruct_one_cluster(
                summary_row,
                candidates_by_cluster.get(
                    cluster_id,
                    [],
                ),
                config,
            )

            reconstructed.append(
                decision
            )

            reconstructed_rows.append(
                {
                    "dataset": dataset,
                    **decision,
                }
            )

        stored_decisions = read_csv(
            stored_decision_path
        )

        stored_by_cluster = {
            clean_text(
                row.get(
                    "cluster_id"
                )
            ): row
            for row in stored_decisions
        }

        recon_by_cluster = {
            clean_text(
                row.get(
                    "cluster_id"
                )
            ): row
            for row in reconstructed
        }

        stored_ids = set(
            stored_by_cluster
        )

        recon_ids = set(
            recon_by_cluster
        )

        missing_ids = sorted(
            stored_ids
            - recon_ids
        )

        extra_ids = sorted(
            recon_ids
            - stored_ids
        )

        for cluster_id in missing_ids:
            mismatch_rows.append(
                {
                    "dataset": dataset,
                    "cluster_id": cluster_id,
                    "field": "cluster_presence",
                    "stored": "present",
                    "reconstructed": "missing",
                }
            )

        for cluster_id in extra_ids:
            mismatch_rows.append(
                {
                    "dataset": dataset,
                    "cluster_id": cluster_id,
                    "field": "cluster_presence",
                    "stored": "missing",
                    "reconstructed": "present",
                }
            )

        field_mismatch_count = 0
        exact_cluster_count = 0

        for cluster_id in sorted(
            stored_ids
            & recon_ids,
            key=lambda value: (
                as_int(
                    value,
                    999999,
                ),
                value,
            ),
        ):
            stored_row = stored_by_cluster[
                cluster_id
            ]

            recon_row = recon_by_cluster[
                cluster_id
            ]

            cluster_mismatch = False

            for field in DECISION_FIELDS:
                if not values_equal(
                    field,
                    stored_row.get(
                        field,
                        "",
                    ),
                    recon_row.get(
                        field,
                        "",
                    ),
                ):
                    cluster_mismatch = True
                    field_mismatch_count += 1

                    mismatch_rows.append(
                        {
                            "dataset": dataset,
                            "cluster_id": cluster_id,
                            "field": field,
                            "stored": stored_row.get(
                                field,
                                "",
                            ),
                            "reconstructed": recon_row.get(
                                field,
                                "",
                            ),
                        }
                    )

            if not cluster_mismatch:
                exact_cluster_count += 1

        accepted = [
            row
            for row in reconstructed
            if row[
                "router_decision"
            ]
            == "accept_agent2_annotation"
        ]

        monitored = [
            row
            for row in reconstructed
            if row[
                "router_decision"
            ]
            == "monitor_uncertainty"
        ]

        routed = [
            row
            for row in reconstructed
            if row[
                "router_decision"
            ]
            == "route_to_agent3"
        ]

        no_expert = (
            routed
            + monitored
        )

        stored_agent3 = read_csv(
            stored_agent3_path
        )

        stored_monitor = read_csv(
            stored_monitor_path
        )

        stored_no_expert = read_csv(
            stored_no_expert_path
        )

        queue_checks = {
            "agent3_queue_ids_match": (
                cluster_ids_from_rows(
                    stored_agent3
                )
                == cluster_ids_from_rows(
                    routed
                )
            ),
            "monitor_queue_ids_match": (
                cluster_ids_from_rows(
                    stored_monitor
                )
                == cluster_ids_from_rows(
                    monitored
                )
            ),
            "no_expert_queue_ids_match": (
                cluster_ids_from_rows(
                    stored_no_expert
                )
                == cluster_ids_from_rows(
                    no_expert
                )
            ),
        }

        stored_router_summary = json.loads(
            stored_summary_path.read_text(
                encoding="utf-8-sig"
            )
        )

        summary_checks = {
            "total_clusters_match": (
                as_int(
                    stored_router_summary.get(
                        "total_clusters"
                    )
                )
                == len(
                    reconstructed
                )
            ),
            "accepted_clusters_match": (
                as_int(
                    stored_router_summary.get(
                        "accepted_clusters"
                    )
                )
                == len(
                    accepted
                )
            ),
            "monitored_clusters_match": (
                as_int(
                    stored_router_summary.get(
                        "monitored_clusters"
                    )
                )
                == len(
                    monitored
                )
            ),
            "routed_clusters_match": (
                as_int(
                    stored_router_summary.get(
                        "routed_to_agent3_clusters"
                    )
                )
                == len(
                    routed
                )
            ),
            "no_expert_clusters_match": (
                as_int(
                    stored_router_summary.get(
                        "agent3_if_no_expert_clusters"
                    )
                )
                == len(
                    no_expert
                )
            ),
            "accepted_ids_match": (
                [
                    clean_text(
                        value
                    )
                    for value in stored_router_summary.get(
                        "accepted_cluster_ids",
                        [],
                    )
                ]
                == cluster_ids_from_rows(
                    accepted
                )
            ),
            "monitored_ids_match": (
                [
                    clean_text(
                        value
                    )
                    for value in stored_router_summary.get(
                        "monitored_cluster_ids",
                        [],
                    )
                ]
                == cluster_ids_from_rows(
                    monitored
                )
            ),
            "routed_ids_match": (
                [
                    clean_text(
                        value
                    )
                    for value in stored_router_summary.get(
                        "routed_cluster_ids",
                        [],
                    )
                ]
                == cluster_ids_from_rows(
                    routed
                )
            ),
            "no_expert_ids_match": (
                [
                    clean_text(
                        value
                    )
                    for value in stored_router_summary.get(
                        "agent3_if_no_expert_cluster_ids",
                        [],
                    )
                ]
                == cluster_ids_from_rows(
                    no_expert
                )
            ),
        }

        decision_pass = (
            not missing_ids
            and not extra_ids
            and field_mismatch_count == 0
        )

        queues_pass = all(
            queue_checks.values()
        )

        summary_pass = all(
            summary_checks.values()
        )

        dataset_pass = (
            decision_pass
            and queues_pass
            and summary_pass
        )

        dataset_reports.append(
            {
                "dataset": dataset,
                "config_path": str(
                    config_path
                ),
                "cluster_count": len(
                    reconstructed
                ),
                "stored_decision_rows": len(
                    stored_decisions
                ),
                "exact_cluster_decisions_all_fields": exact_cluster_count,
                "field_mismatch_count": field_mismatch_count,
                "missing_clusters": missing_ids,
                "extra_clusters": extra_ids,
                "accepted_clusters": len(
                    accepted
                ),
                "monitored_clusters": len(
                    monitored
                ),
                "routed_to_agent3_clusters": len(
                    routed
                ),
                "agent3_if_no_expert_clusters": len(
                    no_expert
                ),
                "queue_checks": queue_checks,
                "summary_checks": summary_checks,
                "status": (
                    "PASS"
                    if dataset_pass
                    else "FAIL"
                ),
            }
        )

    overall_pass = all(
        report[
            "status"
        ]
        == "PASS"
        for report in dataset_reports
    )

    audit = {
        "audit_role": (
            "READ_ONLY_INDEPENDENT_ROUTER_V4_RECONSTRUCTION"
        ),
        "router_implementation_imported": False,
        "expert_or_reference_labels_used": False,
        "router_v4_interpretation": (
            "heuristic baseline; computational reproducibility "
            "does not imply calibrated confidence or optimal thresholds"
        ),
        "datasets": dataset_reports,
        "overall_status": (
            "PASS"
            if overall_pass
            else "FAIL"
        ),
        "existing_outputs_modified": False,
    }

    report_path = (
        args.output_dir
        / "router_v4_independent_reconstruction_audit_v1.json"
    )

    mismatch_path = (
        args.output_dir
        / "router_v4_independent_reconstruction_mismatches_v1.csv"
    )

    reconstruction_path = (
        args.output_dir
        / "router_v4_reconstructed_decisions_v1.csv"
    )

    report_path.write_text(
        json.dumps(
            audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    with mismatch_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "cluster_id",
                "field",
                "stored",
                "reconstructed",
            ],
        )

        writer.writeheader()
        writer.writerows(
            mismatch_rows
        )

    reconstruction_fields = [
        "dataset",
        *DECISION_FIELDS,
    ]

    with reconstruction_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=reconstruction_fields,
        )

        writer.writeheader()
        writer.writerows(
            reconstructed_rows
        )

    print(
        f"Router V4 independent reconstruction audit: {report_path}"
    )

    print(
        f"Mismatch table: {mismatch_path}"
    )

    print(
        f"Reconstructed decisions: {reconstruction_path}"
    )

    print()

    for report in dataset_reports:
        print(
            report[
                "dataset"
            ].upper()
            + ":"
        )

        print(
            f"  Clusters: "
            f"{report['cluster_count']}"
        )

        print(
            f"  Stored decision rows: "
            f"{report['stored_decision_rows']}"
        )

        print(
            "  Exact cluster decisions across all fields: "
            f"{report['exact_cluster_decisions_all_fields']}/"
            f"{report['cluster_count']}"
        )

        print(
            f"  Field mismatches: "
            f"{report['field_mismatch_count']}"
        )

        print(
            f"  Accepted: "
            f"{report['accepted_clusters']}"
        )

        print(
            f"  Monitored: "
            f"{report['monitored_clusters']}"
        )

        print(
            f"  Routed to Agent 3: "
            f"{report['routed_to_agent3_clusters']}"
        )

        print(
            f"  Agent 3 if no expert: "
            f"{report['agent3_if_no_expert_clusters']}"
        )

        print(
            "  Queue membership checks: "
            + (
                "PASS"
                if all(
                    report[
                        "queue_checks"
                    ].values()
                )
                else "FAIL"
            )
        )

        print(
            "  Summary consistency checks: "
            + (
                "PASS"
                if all(
                    report[
                        "summary_checks"
                    ].values()
                )
                else "FAIL"
            )
        )

        print(
            f"  STATUS: "
            f"{report['status']}"
        )

        print()

    print(
        "OVERALL ROUTER V4 INDEPENDENT RECONSTRUCTION STATUS: "
        f"{audit['overall_status']}"
    )

    print()

    print(
        "Router implementation code was NOT imported by this validator."
    )

    print(
        "Expert/reference labels were NOT used."
    )

    print(
        "No Agent 1, Agent 2, Agent 3, Router, orchestrator, "
        "or annotation output was changed."
    )

    print(
        "V4 remains a heuristic baseline; this audit tests "
        "reproducibility, not calibration."
    )


if __name__ == "__main__":
    main()
