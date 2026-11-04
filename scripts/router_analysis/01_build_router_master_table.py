
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path



FEATURES = (
    "direct_evidence_coverage",
    "top1_support_fraction",
    "top1_top2_support_gap",
    "cross_database_support_fraction",
    "first_direct_supported_marker_rank_norm",
    "rank_weighted_unexplained_marker_score",
)

ROOT = Path.cwd()

OUTPUT_DIR = (
    ROOT
    / "data/analysis/router_29_cluster_consolidation_v1"
)


# ORIGINAL SOURCE FILES — READ ONLY
# ============================================================

SOURCES = {
    "historical_features": (
        "data/analysis/router_feature_table_for_review_20261008/"
        "router_feature_table_20_clusters.csv"
    ),
    "historical_logistic": (
        "data/analysis/router_l2_logistic_feasibility_v1/"
        "router_l2_cluster_predictions.csv"
    ),
    "historical_coefficients": (
        "data/analysis/router_l2_logistic_feasibility_v1/"
        "router_l2_coefficients.csv"
    ),
    "historical_references": (
        "data/analysis/router_reference_harmonization_v1/"
        "router_reference_harmonization_worklist_v1.csv"
    ),
    "breast_router": (
        "data/pipeline_runs/breast_xenium_marker_only_clean/"
        "router_output_v4/router_decisions.csv"
    ),
    "lung_router": (
        "data/pipeline_runs/lung_validated_clean/"
        "router_output_v4/router_decisions.csv"
    ),
    "cosmx_features": (
        "data/analysis/router_feature_matrix_cosmx_v1/"
        "cosmx_original_six_router_features.csv"
    ),
    "cosmx_router": (
        "data/pipeline_runs/cosmx_brcr_orchestration_validation/"
        "router_output_v4/router_decisions.csv"
    ),
}


# ============================================================
# OUTPUT COLUMNS
# ============================================================

OUTPUT_COLUMNS = (
    "dataset",
    "platform",
    "cluster_id",
    "agent2_top1_annotation",
    "reference_cell_type_annotation",
    "reference_label_raw",
    "correctness_label_0_1",
    "correctness_status",
    "correctness_note",
    *FEATURES,
    "logistic_predicted_correctness_probability",
    "logistic_decision_at_0_5",
    "logistic_score_status",
    "router_v4_decision",
    "router_v4_decision_raw",
)



# READ-ONLY UTILITIES
# ============================================================

def read_csv(path: Path) -> list[dict[str, str]]:

    if not path.is_file():
        raise FileNotFoundError(
            f"Required input does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.DictReader(handle)

        if not reader.fieldnames:
            raise ValueError(
                f"Missing CSV header: {path}"
            )

        return list(reader)


def require_columns(rows, required, name):

    if not rows:
        raise ValueError(f"No rows: {name}")

    missing = set(required) - set(rows[0])

    if missing:
        raise ValueError(
            f"{name} missing columns: {sorted(missing)}"
        )


def index_rows(rows, fields, name):

    result = {}

    for row in rows:

        key = tuple(
            str(row[field]).strip()
            for field in fields
        )

        if key in result:
            raise ValueError(
                f"Duplicate key in {name}: {key}"
            )

        result[key] = row

    return result


def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as handle:

        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# FROZEN LOGISTIC REGRESSION
# NO TRAINING OR FEATURE CHANGES
# ============================================================

def logistic_probability(row, coefficients):

    z = coefficients["intercept"]

    for name in FEATURES:
        z += (
            coefficients[name]
            * float(row[name])
        )

    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))

    exp_z = math.exp(z)

    return exp_z / (1.0 + exp_z)


def router_decision(raw):

    mapping = {
        "accept_agent2_annotation": "ACCEPT",
        "monitor_uncertainty": "MONITOR",
        "route_to_agent3": "ROUTE",
    }

    if raw not in mapping:
        raise ValueError(
            f"Unknown Router V4 decision: {raw}"
        )

    return mapping[raw]


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # SAFETY: DO NOT OVERWRITE AN EXISTING OUTPUT
    # --------------------------------------------------------

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            "SAFETY STOP: Output directory already exists.\n"
            f"{OUTPUT_DIR}\n"
            "No existing results were overwritten."
        )

    paths = {
        name: ROOT / relative_path
        for name, relative_path in SOURCES.items()
    }

    missing_files = [
        str(path)
        for path in paths.values()
        if not path.is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing required input files:\n"
            + "\n".join(missing_files)
        )

    # Record source hashes before consolidation.
    original_hashes = {
        name: sha256_file(path)
        for name, path in paths.items()
    }

    data = {
        name: read_csv(path)
        for name, path in paths.items()
    }

    # --------------------------------------------------------
    # VALIDATE ORIGINAL FEATURE TABLES
    # --------------------------------------------------------

    require_columns(
        data["historical_features"],
        ("dataset", "cluster_id", "top1_candidate", *FEATURES),
        "Breast/Lung feature table",
    )

    require_columns(
        data["historical_logistic"],
        (
            "dataset",
            "cluster_id",
            "top1_candidate",
            "reference_label",
            "agent2_correct",
            "predicted_correctness_score",
            *FEATURES,
        ),
        "Historical logistic results",
    )

    require_columns(
        data["cosmx_features"],
        (
            "dataset",
            "platform",
            "cluster_id",
            "top1_candidate",
            *FEATURES,
        ),
        "CosMX features",
    )

    require_columns(
        data["historical_references"],
        (
            "dataset",
            "cluster_id",
            "reference_label_raw",
        ),
        "Historical reference worklist",
    )

    if len(data["historical_features"]) != 20:
        raise ValueError("Expected 20 Breast/Lung clusters")

    if len(data["historical_logistic"]) != 20:
        raise ValueError("Expected 20 historical logistic rows")

    if len(data["cosmx_features"]) != 9:
        raise ValueError("Expected nine CosMX clusters")

    historical = index_rows(
        data["historical_features"],
        ("dataset", "cluster_id"),
        "Historical features",
    )

    logistic_rows = index_rows(
        data["historical_logistic"],
        ("dataset", "cluster_id"),
        "Historical logistic",
    )

    references = index_rows(
        data["historical_references"],
        ("dataset", "cluster_id"),
        "Historical references",
    )

    # --------------------------------------------------------
    # LOAD FROZEN LOGISTIC COEFFICIENTS
    # --------------------------------------------------------

    coefficients = {
        row["term"]: float(row["coefficient"])
        for row in data["historical_coefficients"]
    }

    if set(coefficients) != {"intercept", *FEATURES}:
        raise ValueError(
            "Frozen model must contain the original "
            "six features and one intercept."
        )

    # --------------------------------------------------------
    # LOAD ORIGINAL ROUTER V4 DECISIONS
    # --------------------------------------------------------

    router = {}

    router_sources = (
        ("breast", "breast_router", 8),
        ("lung", "lung_router", 12),
        ("breast_cosmx", "cosmx_router", 9),
    )

    for dataset, source_name, expected in router_sources:

        rows = data[source_name]

        require_columns(
            rows,
            (
                "cluster_id",
                "leading_candidate",
                "router_decision",
            ),
            source_name,
        )

        if len(rows) != expected:
            raise ValueError(
                f"{source_name}: expected {expected} rows"
            )

        indexed = index_rows(
            rows,
            ("cluster_id",),
            source_name,
        )

        for (cluster_id,), row in indexed.items():
            router[(dataset, cluster_id)] = row

    consolidated = []

    # ========================================================
    # BREAST XENIUM AND LUNG XENIUM
    # ========================================================

    for (dataset, cluster_id), row in historical.items():

        if dataset not in ("breast", "lung"):
            raise ValueError(
                f"Unexpected historical dataset: {dataset}"
            )

        key = (dataset, cluster_id)

        if key not in logistic_rows:
            raise ValueError(
                f"Missing logistic row: {key}"
            )

        saved = logistic_rows[key]
        v4 = router[key]

        if (
            row["top1_candidate"].casefold()
            != saved["top1_candidate"].casefold()
        ):
            raise ValueError(
                f"Agent 2 Top-1 mismatch: {key}"
            )

        if (
            row["top1_candidate"].casefold()
            != v4["leading_candidate"].casefold()
        ):
            raise ValueError(
                f"Router Top-1 mismatch: {key}"
            )

        # Confirm same six historical feature values.
        for feature in FEATURES:

            x = float(row[feature])
            y = float(saved[feature])

            if not math.isclose(
                x,
                y,
                abs_tol=1e-10,
                rel_tol=0,
            ):
                raise ValueError(
                    f"Feature mismatch: {key}, {feature}"
                )

        if saved["agent2_correct"] not in ("0", "1"):
            raise ValueError(
                f"Invalid historical correctness: {key}"
            )

        # Recalculate score using frozen coefficients.
        probability = logistic_probability(
            row,
            coefficients,
        )

        historical_probability = float(
            saved["predicted_correctness_score"]
        )

        if not math.isclose(
            probability,
            historical_probability,
            abs_tol=1e-12,
            rel_tol=0,
        ):
            raise ValueError(
                f"Frozen logistic prediction mismatch: {key}"
            )

        raw_reference = references.get(
            key,
            {},
        ).get("reference_label_raw", "")

        consolidated.append({
            "dataset": (
                "Breast Xenium"
                if dataset == "breast"
                else "Lung Xenium"
            ),
            "platform": "Xenium",
            "cluster_id": cluster_id,
            "agent2_top1_annotation":
                row["top1_candidate"],
            "reference_cell_type_annotation":
                saved["reference_label"],
            "reference_label_raw": raw_reference,
            "correctness_label_0_1":
                int(saved["agent2_correct"]),
            "correctness_status":
                "HISTORICAL_PROVISIONAL_MANUAL_LABEL",
            "correctness_note":
                saved.get("ontology_aware_relation_note", ""),
            **{
                feature: float(row[feature])
                for feature in FEATURES
            },
            "logistic_predicted_correctness_probability":
                probability,
            "logistic_decision_at_0_5":
                int(probability >= 0.5),
            "logistic_score_status":
                "HISTORICAL_BREAST_TRAIN_OR_LUNG_TEST",
            "router_v4_decision":
                router_decision(v4["router_decision"]),
            "router_v4_decision_raw":
                v4["router_decision"],
        })

    # ========================================================
    # COSMX BREAST — NINE CLUSTERS
    # ========================================================

    cosmx_rows = index_rows(
        data["cosmx_features"],
        ("cluster_id",),
        "CosMX features",
    )

    if set(k[0] for k in cosmx_rows) != set(
        str(i) for i in range(9)
    ):
        raise ValueError(
            "CosMX must contain exactly clusters 0–8"
        )

    for (cluster_id,), row in cosmx_rows.items():

        key = ("breast_cosmx", cluster_id)

        if key not in router:
            raise ValueError(
                f"Missing CosMX Router row: {cluster_id}"
            )

        v4 = router[key]

        if (
            row["top1_candidate"].casefold()
            != v4["leading_candidate"].casefold()
        ):
            raise ValueError(
                f"CosMX Top-1 mismatch: {cluster_id}"
            )

        for feature in FEATURES:
            value = float(row[feature])

            if not math.isfinite(value):
                raise ValueError(
                    f"Invalid feature: {cluster_id}, {feature}"
                )

        # Frozen historical model applied to CosMX.
        # Exploratory score only, NOT calibrated confidence.
        probability = logistic_probability(
            row,
            coefficients,
        )

        consolidated.append({
            "dataset": "Breast CosMX",
            "platform": "CosMX",
            "cluster_id": cluster_id,
            "agent2_top1_annotation":
                row["top1_candidate"],

            # CosMX reference annotations are not verified.
            "reference_cell_type_annotation": "",
            "reference_label_raw": "",
            "correctness_label_0_1": "",
            "correctness_status":
                "PENDING_VERIFIED_REFERENCE",
            "correctness_note":
                "CosMX reference and correctness not yet verified",

            **{
                feature: float(row[feature])
                for feature in FEATURES
            },

            "logistic_predicted_correctness_probability":
                probability,
            "logistic_decision_at_0_5":
                int(probability >= 0.5),
            "logistic_score_status":
                "FROZEN_MODEL_TRANSFER_EXPLORATORY",
            "router_v4_decision":
                router_decision(v4["router_decision"]),
            "router_v4_decision_raw":
                v4["router_decision"],
        })

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    dataset_order = {
        "Breast Xenium": 0,
        "Lung Xenium": 1,
        "Breast CosMX": 2,
    }

    consolidated.sort(
        key=lambda row: (
            dataset_order[row["dataset"]],
            int(row["cluster_id"]),
        )
    )

    if len(consolidated) != 29:
        raise ValueError(
            f"Expected 29 rows, got {len(consolidated)}"
        )

    # Verify source files still have original hashes.
    for name, path in paths.items():

        if sha256_file(path) != original_hashes[name]:
            raise RuntimeError(
                f"Source file changed during consolidation: {path}"
            )

    # ========================================================
    # WRITE ONLY INTO NEW OUTPUT DIRECTORY
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    master_csv = (
        OUTPUT_DIR / "router_original_six_29_clusters.csv"
    )

    with master_csv.open(
        "x",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_COLUMNS,
        )

        writer.writeheader()
        writer.writerows(consolidated)

    # --------------------------------------------------------
    # COSMX REFERENCE REVIEW TEMPLATE
    # --------------------------------------------------------

    reference_template = (
        OUTPUT_DIR / "cosmx_reference_review_template.csv"
    )

    with reference_template.open(
        "x",
        encoding="utf-8",
        newline="",
    ) as handle:

        fields = (
            "cluster_id",
            "agent2_top1_annotation",
            "reference_cell_type_annotation",
            "correctness_label_0_1",
            "reviewer_note",
        )

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in consolidated:

            if row["dataset"] == "Breast CosMX":
                writer.writerow({
                    "cluster_id": row["cluster_id"],
                    "agent2_top1_annotation":
                        row["agent2_top1_annotation"],
                    "reference_cell_type_annotation": "",
                    "correctness_label_0_1": "",
                    "reviewer_note": "",
                })

    # --------------------------------------------------------
    # REPRODUCIBILITY MANIFEST
    # --------------------------------------------------------

    manifest = {
        "created_utc":
            datetime.now(timezone.utc).isoformat(),
        "total_clusters": len(consolidated),
        "dataset_counts": dict(
            Counter(row["dataset"] for row in consolidated)
        ),
        "original_unstandardized_features": list(FEATURES),
        "historical_correctness_labels":
            "Provisional manually assigned 0/1 labels",
        "cosmx_correctness_labels":
            "Not verified; left blank",
        "logistic_status":
            "Frozen exploratory logistic model; "
            "not calibrated biological confidence",
        "source_files": {
            name: {
                "path": str(path),
                "sha256": original_hashes[name],
            }
            for name, path in paths.items()
        },
        "master_csv_sha256": sha256_file(master_csv),
        "existing_pipeline_outputs_modified": False,
    }

    manifest_file = OUTPUT_DIR / "provenance.json"

    with manifest_file.open(
        "x",
        encoding="utf-8",
    ) as handle:

        json.dump(
            manifest,
            handle,
            indent=2,
        )

    # ========================================================
    # CONSOLE RESULTS
    # ========================================================

    print()
    print("=" * 72)
    print("ORIGINAL SIX ROUTER FEATURES — CONSOLIDATED")
    print("=" * 72)

    print("Total clusters:", len(consolidated))

    for dataset, count in Counter(
        row["dataset"] for row in consolidated
    ).items():
        print(f"{dataset}: {count}")

    labeled = sum(
        row["correctness_label_0_1"] in (0, 1)
        for row in consolidated
    )

    print("Correctness labels present:", labeled)
    print("Reference labels pending:", len(consolidated) - labeled)

    print(
        "Router V4 decisions:",
        dict(Counter(
            row["router_v4_decision"]
            for row in consolidated
        )),
    )

    print("Original six features: UNSTANDARDIZED")
    print("Frozen logistic coefficients: USED")
    print("Expert/reference labels used for scoring: NO")
    print("Existing pipeline files changed: NO")

    print()
    print("MASTER CSV:", master_csv)
    print("REFERENCE TEMPLATE:", reference_template)
    print("PROVENANCE:", manifest_file)

    print()
    print("STATUS: ROUTER_29_CLUSTER_TABLE_CREATED")


if __name__ == "__main__":
    main()
