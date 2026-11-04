
"""Read-only inspection of the original MERSCOPE breast marker-input ZIP.

Does not extract ZIP members, run annotation, or modify existing datasets.
Every execution writes a new, separately named validation report directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import uuid
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

EXPECTED = {
    "MERSCOPE_all_cluster_markers.csv",
    "MERSCOPE_BrCr_cluster_sizes.csv",
    "MERSCOPE_top10_markers_per_cluster.csv",
    "MERSCOPE_umap_leiden.png",
}

REQUIRED_MARKER_COLUMNS = {
    "gene", "cluster", "p_val", "avg_log2FC",
    "pct.1", "pct.2", "p_val_adj"
}

CONTROL_PATTERN = re.compile(
    r"^(SystemControl\d+|Negative\d+)$",
    re.IGNORECASE,
)


def read_csv_bytes(blob: bytes, filename: str) -> list[dict[str, str]]:
    reader = csv.DictReader(
        io.StringIO(blob.decode("utf-8-sig"))
    )

    if not reader.fieldnames:
        raise ValueError(f"No CSV header: {filename}")

    if len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValueError(f"Duplicate CSV column names: {filename}")

    rows = list(reader)

    if any(None in row or None in row.values() for row in rows):
        raise ValueError(f"Malformed or missing values: {filename}")

    return rows


def parse_cluster(value: str) -> str:
    return str(int(value))


def parse_stats(row: dict[str, str]) -> tuple[float, float]:
    for key in (
        "p_val", "p_val_adj", "avg_log2FC", "pct.1", "pct.2"
    ):
        num = float(row[key])

        if not math.isfinite(num):
            raise ValueError(f"Non-finite value in {key}: {row}")

        if key in (
            "p_val", "p_val_adj", "pct.1", "pct.2"
        ) and not 0 <= num <= 1:
            raise ValueError(
                f"Value outside [0,1] in {key}: {row}"
            )

    return float(row["p_val_adj"]), float(row["avg_log2FC"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input-zip",
        type=Path,
        default=Path(
            "data/merscope_brcr/input/MERSCOPE_BrCr_info.zip"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "data/pipeline_runs/"
            "merscope_brcr_validated_clean/"
            "00_input_inspection"
        ),
    )

    args = parser.parse_args()
    source = args.input_zip

    if not source.is_file():
        raise FileNotFoundError(
            f"MERSCOPE ZIP not found: {source}\n"
            "Place MERSCOPE_BrCr_info.zip inside "
            "data/merscope_brcr/input/ or provide --input-zip."
        )

    started = perf_counter()
    zip_sha256 = hashlib.sha256(
        source.read_bytes()
    ).hexdigest()

    # Inspect ZIP in memory. No extraction or modification.
    with zipfile.ZipFile(source, "r") as archive:
        bad = archive.testzip()

        if bad is not None:
            raise ValueError(
                f"ZIP CRC verification failed: {bad}"
            )

        members = [
            m for m in archive.infolist()
            if not m.is_dir()
        ]

        matches = defaultdict(list)

        for member in members:
            matches[Path(member.filename).name].append(member)

        for filename in EXPECTED:
            if len(matches[filename]) != 1:
                raise ValueError(
                    f"Expected exactly one {filename}; "
                    f"found {len(matches[filename])}"
                )

        contents = {
            name: archive.read(matches[name][0])
            for name in EXPECTED
        }

        file_inventory = [
            {
                "member": member.filename,
                "size_bytes": member.file_size,
                "sha256": hashlib.sha256(
                    archive.read(member)
                ).hexdigest(),
            }
            for member in members
        ]

    markers = read_csv_bytes(
        contents["MERSCOPE_all_cluster_markers.csv"],
        "all cluster markers",
    )

    size_rows = read_csv_bytes(
        contents["MERSCOPE_BrCr_cluster_sizes.csv"],
        "cluster sizes",
    )

    top10_rows = read_csv_bytes(
        contents["MERSCOPE_top10_markers_per_cluster.csv"],
        "top 10 markers",
    )

    # Validate source columns before filtering.
    if (
        not markers
        or not REQUIRED_MARKER_COLUMNS.issubset(markers[0])
    ):
        raise ValueError("Missing required marker columns")

    if (
        not size_rows
        or not {"cluster", "n_cells"}.issubset(size_rows[0])
    ):
        raise ValueError("Missing cluster-size columns")

    if (
        not top10_rows
        or not REQUIRED_MARKER_COLUMNS.issubset(top10_rows[0])
    ):
        raise ValueError("Missing top-10 marker columns")

    sizes = {}

    for row in size_rows:
        cluster = parse_cluster(row["cluster"])
        count = int(row["n_cells"])

        if cluster in sizes or count <= 0:
            raise ValueError(
                f"Duplicate cluster or invalid size: {row}"
            )

        sizes[cluster] = count

    top10 = defaultdict(list)

    for row in top10_rows:
        cluster = parse_cluster(row["cluster"])
        top10[cluster].append(row["gene"].strip())

    all_by_cluster = defaultdict(list)
    qualifying_by_cluster = defaultdict(list)

    distinct_genes = set()
    seen = set()
    removed_controls = 0

    # Use the same statistical filtering criteria as CosMX.
    for row in markers:
        cluster = parse_cluster(row["cluster"])
        gene = row["gene"].strip()

        if not gene or (cluster, gene) in seen:
            raise ValueError(
                f"Blank/duplicated gene-cluster pair: "
                f"{cluster}, {gene}"
            )

        seen.add((cluster, gene))
        all_by_cluster[cluster].append(gene)
        distinct_genes.add(gene)

        p_adj, logfc = parse_stats(row)

        if CONTROL_PATTERN.fullmatch(gene):
            removed_controls += 1

        elif p_adj <= 0.05 and logfc >= 0.5:
            qualifying_by_cluster[cluster].append(gene)

    if (
        set(sizes) != set(all_by_cluster)
        or set(sizes) != set(top10)
    ):
        raise ValueError(
            "Cluster IDs disagree across input CSV files"
        )

    # Retain up to 50 qualifying markers per cluster.
    # Preserve original ranking. Never pad.
    cluster_report = []
    retained_gene_set = set()

    for cluster in sorted(sizes, key=int):
        eligible = qualifying_by_cluster[cluster]

        if not eligible:
            raise ValueError(
                f"No qualifying positive markers "
                f"for cluster {cluster}"
            )

        if top10[cluster] != eligible[:10]:
            raise ValueError(
                f"Top-10 source ranking mismatch "
                f"for cluster {cluster}"
            )

        retained = eligible[:50]
        retained_gene_set.update(retained)

        cluster_report.append({
            "cluster_id": cluster,
            "n_cells": sizes[cluster],
            "original_marker_rows": len(
                all_by_cluster[cluster]
            ),
            "passing_filter_rows": len(eligible),
            "retained_top50_rows": len(retained),
            "top10_check": "PASS",
            "top_3_retained_genes": " | ".join(
                retained[:3]
            ),
        })

    summary = {
        "dataset": "merscope_brcr_validated_clean",
        "status": "PASS_INPUT_INSPECTION",
        "source_zip": str(source.resolve()),
        "source_zip_sha256": zip_sha256,
        "source_data_type": (
            "precomputed cluster-level marker statistics"
        ),
        "has_raw_anndata": False,
        "reference_labels_used": False,
        "filters_checked": {
            "adjusted_p_max": 0.05,
            "log2fc_min": 0.5,
            "top_k": 50,
        },
        "source_order_preserved": True,
        "padding_used": False,
        "n_clusters": len(sizes),
        "n_cells_from_cluster_sizes": sum(
            sizes.values()
        ),
        "original_marker_rows": len(markers),
        "unique_genes_in_marker_table": len(
            distinct_genes
        ),
        "recognized_control_rows": removed_controls,
        "passing_filter_rows": sum(
            len(x) for x in qualifying_by_cluster.values()
        ),
        "retained_top50_rows": sum(
            row["retained_top50_rows"]
            for row in cluster_report
        ),
        "unique_genes_in_retained_set": len(
            retained_gene_set
        ),
        "top10_crosscheck": "PASS_ALL_CLUSTERS",
        "runtime_seconds": round(
            perf_counter() - started, 6
        ),
        "notes": [
            "Input inspection only; agents not executed.",
            "Cell counts come from the cluster-size CSV.",
            "Raw-cell QC and clustering are not reproduced.",
            "Confirm tissue and disease context before Agent 3.",
        ],
        "cluster_summary": cluster_report,
        "source_members": file_inventory,
    }

    # Always create a new directory, preserving previous reports.
    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    output = args.output_root / (
        f"inspection_{stamp}_{uuid.uuid4().hex[:8]}"
    )

    output.mkdir(parents=True, exist_ok=False)

    json_file = output / "merscope_input_inspection.json"

    json_file.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    csv_file = output / "merscope_cluster_inspection.csv"

    with csv_file.open(
        "x", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(cluster_report[0]),
        )
        writer.writeheader()
        writer.writerows(cluster_report)

    print("\nMERSCOPE INPUT INSPECTION: PASS")

    for key in (
        "n_cells_from_cluster_sizes",
        "n_clusters",
        "original_marker_rows",
        "unique_genes_in_marker_table",
        "recognized_control_rows",
        "passing_filter_rows",
        "retained_top50_rows",
        "unique_genes_in_retained_set",
        "top10_crosscheck",
        "runtime_seconds",
    ):
        print(f"{key}: {summary[key]}")

    print("\nCluster summary:")

    for row in cluster_report:
        print(
            f"  C{row['cluster_id']}: "
            f"{row['n_cells']} cells, "
            f"{row['retained_top50_rows']} retained markers"
        )

    print("\nReports saved to:", output.resolve())
    print("No existing pipeline files were modified.")


if __name__ == "__main__":
    main()
