
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import time
import zipfile

from collections import defaultdict
from pathlib import Path


# ============================================================
# MERSCOPE BREAST CANCER — STANDARDIZED PREPROCESSING
# Same marker-selection rules as the validated CosMX run.
#
# Filters:
#   adjusted p-value <= 0.05
#   log2 fold change >= 0.5
#   top 50 markers per cluster
#
# No artificial marker padding.
# No reference/expert labels.
# Existing nonempty output directories are never overwritten.
# ============================================================

DATASET_ID = "merscope_brcr_validated_clean"

MAX_ADJUSTED_P = 0.05
MIN_LOG2FC = 0.5
TOP_K = 50

OUTPUT_COLUMNS = [
    "cluster_id",
    "gene",
    "rank_within_cluster",
    "score",
    "log_fold_change",
    "p_value",
    "adjusted_p_value",
    "pct_nz_group",
    "pct_nz_reference",
]

REQUIRED_COLUMNS = {
    "gene",
    "cluster",
    "p_val",
    "p_val_adj",
    "avg_log2FC",
    "pct.1",
    "pct.2",
}

CONTROL_PATTERN = re.compile(
    r"^(?:SystemControl\d+|Negative\d+)$",
    re.IGNORECASE,
)


def read_zip_csv(archive, filename):
    matches = [
        path
        for path in archive.namelist()
        if Path(path).name == filename
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {filename}; "
            f"found {len(matches)}."
        )

    content = archive.read(matches[0]).decode("utf-8-sig")

    return list(
        csv.DictReader(io.StringIO(content))
    )


def write_csv(path, columns, rows):
    with path.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Standardized MERSCOPE marker preprocessing"
    )

    parser.add_argument(
        "--input-zip",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    if not args.input_zip.is_file():
        raise FileNotFoundError(
            f"Input ZIP not found: {args.input_zip}"
        )

    if (
        args.output_dir.exists()
        and any(args.output_dir.iterdir())
    ):
        raise FileExistsError(
            "Refusing to overwrite existing outputs: "
            f"{args.output_dir}"
        )

    start = time.perf_counter()

    # ========================================================
    # 1. READ ORIGINAL MERSCOPE FILES
    # ========================================================

    with zipfile.ZipFile(args.input_zip) as archive:

        if archive.testzip() is not None:
            raise ValueError("ZIP integrity check failed.")

        raw_markers = read_zip_csv(
            archive,
            "MERSCOPE_all_cluster_markers.csv",
        )

        size_rows = read_zip_csv(
            archive,
            "MERSCOPE_BrCr_cluster_sizes.csv",
        )

        top10_rows = read_zip_csv(
            archive,
            "MERSCOPE_top10_markers_per_cluster.csv",
        )

    if (
        not raw_markers
        or not REQUIRED_COLUMNS.issubset(raw_markers[0])
    ):
        raise ValueError(
            "Missing required columns in marker file."
        )

    # ========================================================
    # 2. VALIDATE CLUSTER SIZES
    # ========================================================

    sizes = {
        str(int(row["cluster"])): int(row["n_cells"])
        for row in size_rows
    }

    if (
        len(sizes) != len(size_rows)
        or not sizes
        or any(count <= 0 for count in sizes.values())
    ):
        raise ValueError(
            "Invalid or duplicate cluster sizes."
        )

    top10 = defaultdict(list)

    for row in top10_rows:
        cluster = str(int(row["cluster"]))

        top10[cluster].append(
            row["gene"].strip()
        )

    # ========================================================
    # 3. FILTER ORIGINAL MARKERS
    # ========================================================

    seen = set()
    original_genes = set()

    original_cluster_counts = defaultdict(int)
    qualifying = defaultdict(list)

    controls_removed = 0

    for row_number, row in enumerate(
        raw_markers,
        start=2,
    ):
        cluster = str(int(row["cluster"]))
        gene = row["gene"].strip()

        if not gene or (cluster, gene) in seen:
            raise ValueError(
                f"Invalid or duplicated marker at "
                f"row {row_number}: {cluster}, {gene}"
            )

        seen.add((cluster, gene))
        original_genes.add(gene)

        original_cluster_counts[cluster] += 1

        values = [
            float(row[column])
            for column in (
                "p_val",
                "p_val_adj",
                "avg_log2FC",
                "pct.1",
                "pct.2",
            )
        ]

        if not all(math.isfinite(v) for v in values):
            raise ValueError(
                f"Non-finite statistics at row {row_number}"
            )

        if not all(
            0 <= values[index] <= 1
            for index in (0, 1, 3, 4)
        ):
            raise ValueError(
                f"Invalid probability/proportion at "
                f"row {row_number}"
            )

        adjusted_p = values[1]
        log2fc = values[2]

        # Same recognized control-name rule as CosMX.
        if CONTROL_PATTERN.fullmatch(gene):
            controls_removed += 1
            continue

        if (
            adjusted_p <= MAX_ADJUSTED_P
            and log2fc >= MIN_LOG2FC
        ):
            qualifying[cluster].append(row)

    if set(sizes) != set(original_cluster_counts):
        raise ValueError(
            "Cluster IDs differ between marker and size tables."
        )

    # ========================================================
    # 4. RETAIN TOP 50 WITHOUT PADDING
    # ========================================================

    final_markers = []
    cluster_counts = []

    for cluster in sorted(sizes, key=int):

        eligible = qualifying[cluster]

        if not eligible:
            raise ValueError(
                f"Cluster {cluster} has no qualifying markers."
            )

        # Verify existing top-10 ranking.
        expected_top10 = [
            row["gene"].strip()
            for row in eligible[:10]
        ]

        if expected_top10 != top10[cluster]:
            raise ValueError(
                f"Top-10 ranking mismatch in cluster {cluster}"
            )

        # Preserve original marker ranking.
        retained = eligible[:TOP_K]

        cluster_counts.append({
            "cluster_id": cluster,
            "n_cells": sizes[cluster],
            "raw_marker_rows": original_cluster_counts[cluster],
            "passing_marker_rows": len(eligible),
            "retained_marker_rows": len(retained),
        })

        for rank, row in enumerate(
            retained,
            start=1,
        ):
            final_markers.append({
                "cluster_id": cluster,
                "gene": row["gene"].strip(),
                "rank_within_cluster": rank,
                "score": "",
                "log_fold_change": row["avg_log2FC"],
                "p_value": row["p_val"],
                "adjusted_p_value": row["p_val_adj"],
                "pct_nz_group": row["pct.1"],
                "pct_nz_reference": row["pct.2"],
            })

    panel_genes = sorted({
        row["gene"]
        for row in final_markers
    })

    # ========================================================
    # 5. SAVE STANDARDIZED OUTPUTS
    # ========================================================

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    marker_csv = (
        args.output_dir
        / "merscope_brcr_top50_filtered_markers_clean.csv"
    )

    panel_csv = (
        args.output_dir
        / "merscope_brcr_unique_panel_genes.csv"
    )

    cluster_csv = (
        args.output_dir
        / "merscope_brcr_cluster_marker_counts.csv"
    )

    write_csv(
        marker_csv,
        OUTPUT_COLUMNS,
        final_markers,
    )

    write_csv(
        panel_csv,
        ["gene"],
        [{"gene": gene} for gene in panel_genes],
    )

    write_csv(
        cluster_csv,
        list(cluster_counts[0]),
        cluster_counts,
    )

    runtime = time.perf_counter() - start

    # ========================================================
    # 6. REPRODUCIBILITY SUMMARY
    # ========================================================

    summary = {
        "dataset_id": DATASET_ID,
        "input_zip": str(args.input_zip),
        "input_sha256": hashlib.sha256(
            args.input_zip.read_bytes()
        ).hexdigest(),
        "cell_count_from_cluster_sizes": sum(
            sizes.values()
        ),
        "cluster_count": len(sizes),
        "original_rows": len(raw_markers),
        "unique_original_genes": len(original_genes),
        "recognizable_controls_removed": controls_removed,
        "passing_rows": sum(
            len(rows)
            for rows in qualifying.values()
        ),
        "retained_rows": len(final_markers),
        "unique_retained_genes": len(panel_genes),
        "top10_check": "PASS_ALL_CLUSTERS",
        "filters": {
            "max_adjusted_p": MAX_ADJUSTED_P,
            "min_log2fc": MIN_LOG2FC,
            "top_k": TOP_K,
        },
        "score": "blank; not provided in source",
        "source_order_preserved": True,
        "marker_padding": False,
        "raw_cell_preprocessing_performed": False,
        "reference_label_used": False,
        "runtime_seconds": round(runtime, 6),
    }

    summary_json = (
        args.output_dir
        / "merscope_brcr_preprocessing_summary.json"
    )

    summary_json.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    # ========================================================
    # 7. DISPLAY RESULTS
    # ========================================================

    print("\n" + "=" * 60)
    print("MERSCOPE STANDARDIZED PREPROCESSING COMPLETED")
    print("=" * 60)

    print(f"Cells: {sum(sizes.values())}")
    print(f"Clusters: {len(sizes)}")
    print(f"Original marker rows: {len(raw_markers)}")
    print(f"Controls removed: {controls_removed}")

    print(
        "Passing filter rows:",
        summary["passing_rows"],
    )

    print(
        "Final marker rows:",
        len(final_markers),
    )

    print(
        "Unique panel genes:",
        len(panel_genes),
    )

    print("\nCluster marker counts:")

    for row in cluster_counts:
        print(
            f"Cluster {row['cluster_id']}: "
            f"{row['retained_marker_rows']} markers"
        )

    print(
        f"\nTop-10 validation: "
        f"{summary['top10_check']}"
    )

    print(f"Preprocessing runtime: {runtime:.6f} s")
    print(f"\nOutputs: {args.output_dir.resolve()}")
    print("Existing CosMX outputs: NOT MODIFIED")
    print("=" * 60)


if __name__ == "__main__":
    main()
