
"""Compute six original, unstandardized Router features for MERSCOPE.

Inputs:
    Frozen Agent 2 cluster evidence
    Frozen Agent 2 candidate summaries
    Frozen Router V4 decisions
    Frozen Breast Xenium logistic regression coefficients

Outputs:
    Eight-cluster Router feature table
    Reproducibility audit JSON

Does not:
    Modify Agent 1, Agent 2, Router, or Agent 3
    Modify CosMX, Xenium, or Lung datasets
    Train or refit any model
    Use reference labels during inference
"""

from __future__ import annotations

import csv
import hashlib
import json
import math

from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATASET = "merscope_brcr_validated_clean"

BASE = ROOT / "data" / "pipeline_runs" / DATASET

AGENT2 = BASE / "agent2_output"

SUMMARY = AGENT2 / "all_clusters_summary.csv"

CANDIDATES = (
    AGENT2 / "all_clusters_threshold_candidates.csv"
)

ROUTER = (
    BASE / "router_output_v4" / "router_decisions.csv"
)

# Frozen Breast Xenium logistic regression model.
COEFFICIENTS = (
    ROOT
    / "data"
    / "analysis"
    / "router_l2_logistic_feasibility_v1"
    / "router_l2_coefficients.csv"
)

# MERSCOPE-specific outputs only.
OUTPUT = (
    BASE / "router_original_six_feature_table_v1"
)

TABLE = (
    OUTPUT / "merscope_router_original_six_8_clusters.csv"
)

AUDIT = (
    OUTPUT / "merscope_router_original_six_audit.json"
)


# ============================================================
# 2. ORIGINAL SIX UNSTANDARDIZED FEATURES
# ============================================================

FEATURES = (
    "direct_evidence_coverage",
    "top1_support_fraction",
    "top1_top2_support_gap",
    "cross_database_support_fraction",
    "first_direct_supported_marker_rank_norm",
    "rank_weighted_unexplained_marker_score",
)

COLUMNS = (
    "dataset",
    "platform",
    "cluster_id",
    "agent2_top1_annotation",
    "reference_cell_type_annotation",
    "correctness_label_0_1",
    *FEATURES,
    "logistic_predicted_correctness_probability",
    "router_v4_decision",
)

DECISION_MAP = {
    "accept_agent2_annotation": "ACCEPT",
    "monitor_uncertainty": "MONITOR",
    "route_to_agent3": "ROUTE",
}


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def read_csv(path: Path) -> list[dict[str, str]]:

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        reader = csv.DictReader(handle)

        if not reader.fieldnames:
            raise ValueError(
                f"CSV has no header: {path}"
            )

        rows = list(reader)

    if not rows:
        raise ValueError(
            f"CSV is empty: {path}"
        )

    return rows


def require(row, columns, path):

    missing = set(columns) - set(row)

    if missing:
        raise ValueError(
            f"Missing columns in {path}: "
            f"{sorted(missing)}"
        )


def index_one(rows, name):

    indexed = {}

    for row in rows:

        cid = int(row["cluster_id"])

        if cid in indexed:
            raise ValueError(
                f"Duplicate cluster {cid} in {name}"
            )

        indexed[cid] = row

    return indexed


def sha256(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as file:

        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def parse_bool(value: str) -> bool:

    text = str(value).strip().lower()

    if text in ("true", "1", "yes"):
        return True

    if text in ("false", "0", "no"):
        return False

    raise ValueError(
        f"Invalid direct-support value: {value!r}"
    )


# ============================================================
# 4. LOAD FROZEN LOGISTIC REGRESSION MODEL
# ============================================================

def read_model():

    rows = read_csv(COEFFICIENTS)

    require(
        rows[0],
        ("term", "coefficient"),
        COEFFICIENTS,
    )

    model = {}

    for row in rows:

        key = row["term"].strip()

        if key in model:
            raise ValueError(
                f"Duplicate coefficient: {key}"
            )

        model[key] = float(row["coefficient"])

    expected = {"intercept", *FEATURES}

    if set(model) != expected:
        raise ValueError(
            "Frozen model features do not match "
            "the original six Router features."
        )

    return model


# ============================================================
# 5. COMPUTE FEATURES FOR ALL EIGHT CLUSTERS
# ============================================================

def compute():

    # Never overwrite an existing results directory.
    if OUTPUT.exists():
        raise FileExistsError(
            "Output directory already exists. "
            "Refusing overwrite:\n"
            f"{OUTPUT}"
        )

    summaries = index_one(
        read_csv(SUMMARY),
        "Agent 2 summary",
    )

    decisions = index_one(
        read_csv(ROUTER),
        "Router V4 decisions",
    )

    raw_candidates = read_csv(CANDIDATES)

    require(
        raw_candidates[0],
        (
            "cluster_id",
            "candidate_rank",
            "cell_type",
            "unique_supporting_gene_count",
            "both_source_supporting_gene_count",
        ),
        CANDIDATES,
    )

    candidates = {}

    for row in raw_candidates:

        cid = int(row["cluster_id"])

        candidates.setdefault(cid, []).append(row)

    for group in candidates.values():

        group.sort(
            key=lambda row: int(row["candidate_rank"])
        )

        ranks = [
            int(row["candidate_rank"])
            for row in group
        ]

        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError(
                f"Invalid candidate ranking: {ranks}"
            )

    expected_clusters = set(range(8))

    if (
        set(summaries) != expected_clusters
        or set(decisions) != expected_clusters
        or set(candidates) != expected_clusters
    ):
        raise ValueError(
            "Expected exactly MERSCOPE clusters 0-7 "
            "in all upstream tables."
        )

    model = read_model()

    source_paths = [
        SUMMARY,
        CANDIDATES,
        ROUTER,
        COEFFICIENTS,
    ]

    results = []
    counts = {}

    for cid in range(8):

        # ----------------------------------------------------
        # A. LOAD FROZEN GENE EVIDENCE
        # ----------------------------------------------------

        path = (
            AGENT2
            / "clusters"
            / f"cluster_{cid}"
            / f"cluster_{cid}_master_gene_evidence.csv"
        )

        master = read_csv(path)

        source_paths.append(path)

        require(
            master[0],
            (
                "cluster_id",
                "gene",
                "rank_within_cluster",
                "direct_marker_supported",
            ),
            path,
        )

        if any(
            int(row["cluster_id"]) != cid
            for row in master
        ):
            raise ValueError(
                f"Cluster identity mismatch for C{cid}"
            )

        master.sort(
            key=lambda row: int(
                row["rank_within_cluster"]
            )
        )

        n = len(master)

        ranks = [
            int(row["rank_within_cluster"])
            for row in master
        ]

        if ranks != list(range(1, n + 1)):
            raise ValueError(
                f"Nonconsecutive marker ranks in C{cid}"
            )

        direct = [
            parse_bool(row["direct_marker_supported"])
            for row in master
        ]

        # ----------------------------------------------------
        # B. VALIDATE AGENT 2 ACCOUNTING
        # ----------------------------------------------------

        summary = summaries[cid]

        require(
            summary,
            (
                "total_input_genes",
                "direct_marker_supported_genes",
                "follow_up_gene_count",
                "leading_candidate",
            ),
            SUMMARY,
        )

        if int(summary["total_input_genes"]) != n:
            raise ValueError(
                f"Marker count mismatch in C{cid}"
            )

        direct_count = sum(direct)

        if (
            int(summary["direct_marker_supported_genes"])
            != direct_count
        ):
            raise ValueError(
                f"Direct-support count mismatch in C{cid}"
            )

        # ----------------------------------------------------
        # CORRECTED VALIDATION
        #
        # The follow-up queue is a separate review policy.
        # Its count need not equal:
        #
        # total markers - directly supported markers
        #
        # Especially when HPA evidence is enabled.
        #
        # Record both counts separately instead of
        # forcing them to be equal.
        # ----------------------------------------------------

        follow_up_count = int(
            summary["follow_up_gene_count"]
        )

        non_direct_count = n - direct_count

        if not (0 <= follow_up_count <= n):
            raise ValueError(
                f"Invalid follow-up count in C{cid}: "
                f"follow_up={follow_up_count}, total={n}"
            )

        print(
            f"C{cid} accounting: "
            f"total={n}, "
            f"direct={direct_count}, "
            f"non_direct={non_direct_count}, "
            f"follow_up={follow_up_count}"
        )

        # ----------------------------------------------------
        # C. GET AGENT 2 TOP-1 AND TOP-2 CANDIDATES
        # ----------------------------------------------------

        pool = candidates[cid]

        if len(pool) < 2:
            raise ValueError(
                f"Missing Top-2 candidate for C{cid}"
            )

        top1, top2 = pool[0], pool[1]

        label = top1["cell_type"].strip()

        if (
            label.casefold()
            != summary["leading_candidate"].strip().casefold()
        ):
            raise ValueError(
                f"Agent 2 leading candidate mismatch in C{cid}"
            )

        if (
            label.casefold()
            != decisions[cid]["leading_candidate"].strip().casefold()
        ):
            raise ValueError(
                f"Router / Agent 2 label mismatch in C{cid}"
            )

        s1 = int(
            top1["unique_supporting_gene_count"]
        )

        s2 = int(
            top2["unique_supporting_gene_count"]
        )

        both = int(
            top1["both_source_supporting_gene_count"]
        )

        if not (
            0 <= both <= s1 <= n
            and 0 <= s2 <= n
        ):
            raise ValueError(
                f"Invalid supporting gene counts in C{cid}"
            )

        if s1 < s2:
            raise ValueError(
                f"Top-2 support exceeds Top-1 in C{cid}; "
                "inspect the candidate ranking."
            )

        # ----------------------------------------------------
        # D. FIRST DIRECTLY SUPPORTED MARKER RANK
        # ----------------------------------------------------

        first_rank = next(
            (
                rank
                for rank, flag in zip(ranks, direct)
                if flag
            ),
            None,
        )

        if first_rank is None:
            raise ValueError(
                f"No directly supported marker in C{cid}. "
                "The original feature is undefined."
            )

        # ----------------------------------------------------
        # E. RANK WEIGHTS
        # ----------------------------------------------------

        # Rank 1 receives weight N.
        # Rank N receives weight 1.

        weights = [
            n - rank + 1
            for rank in ranks
        ]

        unsupported_weight = sum(
            weight
            for weight, flag in zip(weights, direct)
            if not flag
        )

        total_weight = n * (n + 1) // 2

        # ----------------------------------------------------
        # F. ORIGINAL SIX FEATURES
        # ----------------------------------------------------

        features = {

            "direct_evidence_coverage": round(
                direct_count / n,
                4,
            ),

            "top1_support_fraction": round(
                s1 / n,
                4,
            ),

            "top1_top2_support_gap": round(
                (s1 - s2) / n,
                4,
            ),

            "cross_database_support_fraction": (
                round(both / s1, 4)
                if s1
                else None
            ),

            "first_direct_supported_marker_rank_norm": round(
                first_rank / n,
                4,
            ),

            "rank_weighted_unexplained_marker_score": round(
                unsupported_weight / total_weight,
                4,
            ),
        }

        if any(
            value is None
            for value in features.values()
        ):
            raise ValueError(
                f"Missing Router feature in C{cid}. "
                "Artificial imputation is not allowed."
            )

        # ----------------------------------------------------
        # G. FROZEN LOGISTIC REGRESSION SCORE
        # ----------------------------------------------------

        # No model training or standardization.
        # Transfer the existing Breast Xenium model.

        logit = model["intercept"] + sum(
            model[key] * features[key]
            for key in FEATURES
        )

        logistic_score = 1.0 / (
            1.0 + math.exp(-logit)
        )

        # This is an uncalibrated model score.
        # It is not biological annotation confidence.

        # ----------------------------------------------------
        # H. FROZEN ROUTER V4 DECISION
        # ----------------------------------------------------

        route = decisions[cid][
            "router_decision"
        ].strip()

        if route not in DECISION_MAP:
            raise ValueError(
                f"Unexpected Router decision in C{cid}: "
                f"{route}"
            )

        # ----------------------------------------------------
        # I. BUILD MANUSCRIPT TABLE ROW
        # ----------------------------------------------------

        results.append({

            "dataset": "Breast MERSCOPE",

            "platform": "MERSCOPE",

            "cluster_id": cid,

            "agent2_top1_annotation": label,

            # Independent reference labels not supplied.
            "reference_cell_type_annotation": "",

            # Do not infer correctness from Agent 3.
            "correctness_label_0_1": "",

            **features,

            "logistic_predicted_correctness_probability":
                f"{logistic_score:.10f}",

            "router_v4_decision":
                DECISION_MAP[route],
        })

        # ----------------------------------------------------
        # J. REPRODUCIBILITY ACCOUNTING
        # ----------------------------------------------------

        counts[str(cid)] = {

            "input_marker_count": n,

            "top1_direct_supporting_genes": s1,

            "top2_direct_supporting_genes": s2,

            "both_database_top1_supporting_genes": both,

            "direct_supported_marker_count": direct_count,

            "first_direct_supported_rank": first_rank,

            "follow_up_gene_count": follow_up_count,

            "no_direct_support_count": non_direct_count,

            "follow_up_equals_no_direct_support_count": (
                follow_up_count == non_direct_count
            ),
        }

    # ========================================================
    # 6. VERIFY FROZEN INPUTS
    # ========================================================

    hashes_before = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in source_paths
    }

    # ========================================================
    # 7. WRITE ONLY NEW MERSCOPE OUTPUTS
    # ========================================================

    OUTPUT.mkdir(
        parents=True,
        exist_ok=False,
    )

    with TABLE.open(
        "x",
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=COLUMNS,
        )

        writer.writeheader()
        writer.writerows(results)

    hashes_after = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in source_paths
    }

    if hashes_before != hashes_after:
        raise RuntimeError(
            "Protected source hashes changed unexpectedly."
        )

    # ========================================================
    # 8. SAVE REPRODUCIBILITY AUDIT
    # ========================================================

    metadata = {

        "dataset": DATASET,

        "feature_contract":
            "original_six_unstandardized_4_decimal",

        "clusters": list(range(8)),

        "feature_columns": list(FEATURES),

        "source_sha256": hashes_after,

        "input_accounting": counts,

        "logistic_model":
            "frozen_l2_breast_xenium_train_8_C1_liblinear",

        "logistic_is_calibrated": False,

        "reference_labels_used": False,

        "original_router_decisions_modified": False,

        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    with AUDIT.open(
        "x",
        encoding="utf-8",
    ) as handle:

        json.dump(
            metadata,
            handle,
            indent=2,
        )

        handle.write("\n")

    return results


# ============================================================
# 9. PRINT EIGHT-CLUSTER RESULTS
# ============================================================

def main():

    rows = compute()

    print("\n" + "=" * 75)
    print("MERSCOPE — ORIGINAL SIX ROUTER FEATURES")
    print("=" * 75)

    for row in rows:

        values = " | ".join(
            f"{row[name]:.4f}"
            for name in FEATURES
        )

        print(
            f"C{row['cluster_id']}: "
            f"{row['agent2_top1_annotation']} | "
            f"{values} | "
            f"LR={row['logistic_predicted_correctness_probability']} | "
            f"Router={row['router_v4_decision']}"
        )

    print()
    print("Clusters evaluated : 8")
    print("Feature values      : 48")
    print(f"Output CSV          : {TABLE}")
    print(f"Audit JSON          : {AUDIT}")
    print("Reference labels    : NOT USED")
    print("Frozen input files  : UNCHANGED")
    print("=" * 75)


if __name__ == "__main__":
    main()
