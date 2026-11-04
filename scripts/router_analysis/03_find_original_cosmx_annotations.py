
from __future__ import annotations

import csv
import os
from pathlib import Path

ROOT = Path.cwd()

EXTENSIONS = {
    ".csv", ".tsv", ".xlsx",
    ".parquet", ".h5ad", ".h5",
}

IGNORE_DIRECTORIES = {
    ".git", ".venv", "__pycache__",
    ".pytest_cache", "node_modules",
}

LABEL_TERMS = (
    "annotation", "annotated",
    "cell_type", "celltype",
    "ground_truth", "groundtruth",
    "reference", "manual",
    "expert", "metadata",
    "cluster_label", "cluster_identity",
)

LABEL_COLUMNS = (
    "cell_type", "celltype",
    "annotation", "cell_label",
    "manual_label", "expert_label",
    "ground_truth", "cluster_identity",
)

CLUSTER_COLUMNS = (
    "cluster", "cluster_id",
    "leiden", "seurat_clusters",
)


def find_files():
    results = []

    for current, dirs, filenames in os.walk(ROOT):
        dirs[:] = [
            name for name in dirs
            if name not in IGNORE_DIRECTORIES
        ]

        for filename in filenames:
            path = Path(current) / filename

            if path.suffix.lower() not in EXTENSIONS:
                continue

            relative = path.relative_to(ROOT)
            location = str(relative).lower()
            name = filename.lower()

            is_cosmx = (
                "cosmx" in location
                or "nanostring" in location
            )

            has_label_name = any(
                term in name for term in LABEL_TERMS
            )

            is_anndata = path.suffix.lower() == ".h5ad"

            # Search original data and archived locations too.
            if not (
                (is_cosmx and (has_label_name or is_anndata))
                or is_anndata
                or has_label_name
            ):
                continue

            is_derived = (
                "pipeline_runs" in relative.parts
                or "analysis" in relative.parts
            )

            score = 0

            if is_cosmx:
                score += 10

            if has_label_name:
                score += 7

            if is_anndata:
                score += 5

            if is_derived:
                score -= 8

            if "marker" in name:
                score -= 8

            results.append({
                "score": score,
                "path": path,
                "derived": is_derived,
                "cosmx": is_cosmx,
            })

    return sorted(
        results,
        key=lambda x: (
            -x["score"],
            str(x["path"]).lower(),
        ),
    )


def inspect_tabular_header(path):
    delimiter = (
        "\t" if path.suffix.lower() == ".tsv"
        else ","
    )

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.DictReader(
                file,
                delimiter=delimiter,
            )

            columns = reader.fieldnames or []

            label_cols = [
                col for col in columns
                if any(
                    term in col.lower()
                    for term in LABEL_COLUMNS
                )
            ]

            cluster_cols = [
                col for col in columns
                if col.lower() in CLUSTER_COLUMNS
            ]

            print("  Cluster columns:", cluster_cols)
            print("  Possible cell-type columns:", label_cols)

            if cluster_cols and label_cols:
                print("  First five candidate mappings:")

                for _, row in zip(range(5), reader):
                    print(
                        "   ",
                        row[cluster_cols[0]],
                        "->",
                        row[label_cols[0]],
                    )

                print(
                    "  NOTE: Candidate mapping only; "
                    "reference provenance not verified."
                )

    except (OSError, UnicodeError, csv.Error) as error:
        print("  Header inspection error:", error)


def inspect_anndata(path):
    try:
        import h5py
    except ImportError:
        print("  h5py unavailable")
        return

    try:
        with h5py.File(path, "r") as file:
            if "obs" not in file:
                print("  No AnnData obs metadata")
                return

            obs = file["obs"]
            columns = list(obs.keys())

            print("  Available obs columns:", columns[:40])

            label_cols = [
                column for column in columns
                if any(
                    term in column.lower()
                    for term in LABEL_COLUMNS
                )
            ]

            cluster_cols = [
                column for column in columns
                if any(
                    term in column.lower()
                    for term in CLUSTER_COLUMNS
                )
            ]

            print("  Possible cluster fields:", cluster_cols)
            print("  Possible label fields:", label_cols)

            for column in label_cols[:5]:
                field = obs[column]

                if (
                    isinstance(field, h5py.Group)
                    and "categories" in field
                ):
                    categories = field["categories"][:]

                    values = [
                        item.decode("utf-8")
                        if isinstance(item, bytes)
                        else str(item)
                        for item in categories[:25]
                    ]

                    print(
                        f"  {column} categories:",
                        values,
                    )

            print(
                "  NOTE: Categories alone do not establish "
                "a verified cluster-to-reference mapping."
            )

    except (OSError, KeyError, ValueError) as error:
        print("  H5AD inspection error:", error)


def main():
    print("=" * 72)
    print("ORIGINAL COSMX REFERENCE SOURCE AUDIT")
    print("=" * 72)

    candidates = find_files()

    print("Potential metadata files:", len(candidates))
    print("Showing highest-priority 35 candidates")

    for index, item in enumerate(candidates[:35], 1):
        path = item["path"]

        print()
        print(f"[{index}] {path.relative_to(ROOT)}")
        print(
            "  Category:",
            "DERIVED_ANALYSIS"
            if item["derived"]
            else "POSSIBLE_SOURCE",
        )

        if path.suffix.lower() in (".csv", ".tsv"):
            inspect_tabular_header(path)

        elif path.suffix.lower() == ".h5ad":
            inspect_anndata(path)

        else:
            print(
                "  File found; no cluster-reference "
                "mapping has been verified."
            )

    print()
    print("=" * 72)
    print("STATUS: ORIGINAL_COSMX_REFERENCE_SEARCH_COMPLETE")
    print("FILES MODIFIED: NONE")
    print("REFERENCE LABELS AUTOMATICALLY ASSIGNED: NONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
