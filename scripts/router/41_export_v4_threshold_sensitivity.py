
from pathlib import Path
from collections import Counter
import csv
import hashlib
import io
import json
import math

# Run from the project root in VS Code.
SOURCE = Path(
    "data/analysis/router_v4_uncertainty_export_v1/"
    "router_v4_uncertainty_indicators_29_clusters.csv"
)

OUTPUT_DIR = Path(
    "data/analysis/router_v4_threshold_sensitivity_v1"
)

EXPECTED_DATASETS = {
    "Breast Xenium": 8,
    "Lung Xenium": 12,
    "Breast CosMX": 9,
}

# Existing V4 broad-label vocabulary.
BROAD_LABELS = {
    "cancer cell",
    "epithelial cell",
    "immune cell",
    "lymphocyte",
    "myeloid cell",
    "stromal cell",
}

OUTPUT_COLUMNS = [
    "dataset",
    "platform",
    "cluster_id",
    "agent2_top1_annotation",
    "top1_supporting_gene_count",
    "top2_supporting_gene_count",
    "top1_top2_support_ratio",
    "candidate_cell_type_count",
    "broad_label_indicator",
    "agent2_review_flag",
]


def integer(value, name):
    number = float(value)

    if not math.isfinite(number) or number != int(number):
        raise ValueError(f"Invalid integer {name}: {value}")

    return int(number)


def binary(value, name):
    number = integer(value, name)

    if number not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1")

    return number


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_write(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(
                f"Existing output differs; refusing overwrite: {path}"
            )
        return

    path.write_bytes(data)


def main():

    if not SOURCE.is_file():
        raise FileNotFoundError(
            f"Task 1 source CSV not found:\n{SOURCE}"
        )

    with SOURCE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    required = {
        "dataset",
        "platform",
        "cluster_id",
        "agent2_top1_annotation",
        "v4_top_supporting_gene_count",
        "v4_second_supporting_gene_count",
        "v4_support_ratio",
        "v4_candidate_count",
        "signal_broad_or_nonspecific_leading_label",
        "signal_agent2_review_flag",
    }

    if not rows:
        raise ValueError("Source CSV is empty.")

    missing = required - set(rows[0])

    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")

    results = []
    dataset_counts = Counter()
    seen = set()

    for row in rows:

        dataset = row["dataset"].strip()
        cluster_id = integer(row["cluster_id"], "cluster_id")
        key = (dataset, cluster_id)

        if key in seen:
            raise ValueError(f"Duplicate cluster: {key}")

        seen.add(key)
        dataset_counts[dataset] += 1

        top1 = integer(
            row["v4_top_supporting_gene_count"],
            "Top-1 supporting gene count",
        )

        top2 = integer(
            row["v4_second_supporting_gene_count"],
            "Top-2 supporting gene count",
        )

        candidate_count = integer(
            row["v4_candidate_count"],
            "Candidate count",
        )

        if top1 < 0 or top2 < 0 or top2 > top1:
            raise ValueError(f"Invalid supporting counts: {key}")

        if candidate_count < 1:
            raise ValueError(f"Invalid candidate count: {key}")

        # Raw ratio for professor's threshold sensitivity analysis.
        ratio = top1 / top2 if top2 > 0 else None

        # Verify against the previously saved V4 ratio.
        saved_ratio = row["v4_support_ratio"].strip()

        if ratio is None:
            if saved_ratio:
                raise ValueError(f"Unexpected ratio: {key}")

        elif (
            not saved_ratio
            or not math.isclose(
                float(saved_ratio),
                ratio,
                rel_tol=0,
                abs_tol=5.1e-7,
            )
        ):
            raise ValueError(f"Saved V4 ratio mismatch: {key}")

        top1_label = row["agent2_top1_annotation"].strip()

        # Raw broad-label indicator (without the candidate-count rule).
        broad_indicator = int(
            top1_label.casefold() in BROAD_LABELS
        )

        saved_broad_signal = binary(
            row["signal_broad_or_nonspecific_leading_label"],
            "V4 broad-label signal",
        )

        # Original V4 condition additionally requires >= 3 candidates.
        expected_broad_signal = int(
            broad_indicator == 1 and candidate_count >= 3
        )

        if saved_broad_signal != expected_broad_signal:
            raise ValueError(
                f"Broad-label rule mismatch: {key}"
            )

        review_flag = binary(
            row["signal_agent2_review_flag"],
            "Agent 2 review flag",
        )

        results.append({
            "dataset": dataset,
            "platform": row["platform"],
            "cluster_id": cluster_id,
            "agent2_top1_annotation": top1_label,
            "top1_supporting_gene_count": top1,
            "top2_supporting_gene_count": top2,
            "top1_top2_support_ratio": (
                f"{ratio:.12g}" if ratio is not None else ""
            ),
            "candidate_cell_type_count": candidate_count,
            "broad_label_indicator": broad_indicator,
            "agent2_review_flag": review_flag,
        })

    if dict(dataset_counts) != EXPECTED_DATASETS:
        raise ValueError(
            f"Unexpected dataset counts: {dict(dataset_counts)}"
        )

    if len(results) != 29:
        raise ValueError(
            f"Expected 29 clusters, found {len(results)}"
        )

    dataset_order = list(EXPECTED_DATASETS)

    results.sort(
        key=lambda row: (
            dataset_order.index(row["dataset"]),
            row["cluster_id"],
        )
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_csv = (
        OUTPUT_DIR /
        "router_v4_threshold_sensitivity_29_clusters.csv"
    )

    buffer = io.StringIO(newline="")

    writer = csv.DictWriter(
        buffer,
        fieldnames=OUTPUT_COLUMNS,
        lineterminator="\n",
    )

    writer.writeheader()
    writer.writerows(results)

    safe_write(
        output_csv,
        buffer.getvalue().encode("utf-8-sig"),
    )

    audit = {
        "source": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "output": str(output_csv),
        "output_sha256": sha256(output_csv),
        "dataset_counts": dict(dataset_counts),
        "total_clusters": len(results),
        "ratio": "Top-1 count / Top-2 count",
        "zero_top2_policy": "Blank ratio",
        "broad_indicator": "Top-1 label belongs to V4 broad-label vocabulary",
        "original_v4_broad_penalty_condition": (
            "Broad indicator == 1 and candidate count >= 3"
        ),
        "review_flag": "Existing Agent 2 review signal",
        "reference_labels_used": False,
        "original_router_modified": False,
    }

    audit_path = (
        OUTPUT_DIR /
        "router_v4_threshold_sensitivity_audit.json"
    )

    safe_write(
        audit_path,
        (json.dumps(audit, indent=2) + "\n").encode("utf-8"),
    )

    print("\n" + "=" * 65)
    print("ROUTER V4 THRESHOLD SENSITIVITY EXPORT")
    print("=" * 65)

    for dataset, expected in EXPECTED_DATASETS.items():
        print(f"PASS: {dataset}: {dataset_counts[dataset]}/{expected}")

    print("\nSTATUS: SUCCESS - ALL 29 CLUSTERS EXPORTED")
    print("PASS: Supporting gene counts and ratios verified")
    print("PASS: Broad-label indicators verified")
    print("PASS: Agent 2 review flags verified")

    print("\nCSV:", output_csv)
    print("Audit:", audit_path)

    print("\nExisting Agent 2 files modified: NONE")
    print("Existing Router V4 files modified: NONE")
    print("Model training performed: NONE")


if __name__ == "__main__":
    main()
