
from __future__ import annotations

import csv
import os
from pathlib import Path

ROOT = Path.cwd()
DATA_DIR = ROOT / "data"

EXTENSIONS = {
    ".csv", ".tsv", ".h5ad", ".xlsx",
    ".json", ".parquet",
}

LABEL_TERMS = (
    "annotation", "cell_type", "celltype",
    "reference", "ground_truth", "label",
    "cluster_identity", "identity",
)

IGNORE_DIRS = {
    "pipeline_runs",
    "analysis",
    "__pycache__",
    ".git",
    ".venv",
}

MAX_RESULTS = 40


def find_candidate_files():
    candidates = []

    if not DATA_DIR.exists():
        raise FileNotFoundError(DATA_DIR)

    for current, directories, filenames in os.walk(DATA_DIR):

        directories[:] = [
            name for name in directories
            if name not in IGNORE_DIRS
        ]

        for filename in filenames:
            path = Path(current) / filename

            if path.suffix.lower() not in EXTENSIONS:
                continue

            relative = path.relative_to(ROOT)
            text = str(relative).lower()

            cosmx_related = (
                "cosmx" in text
                or "nanostring" in text
            )

            label_related = any(
                term in text for term in LABEL_TERMS
            )

            reference_location = (
                "reference" in relative.parts
            )

            if not (
                cosmx_related
                or (reference_location and label_related)
            ):
                continue

            score = 0

            if cosmx_related:
                score += 4

            if label_related:
                score += 5

            if path.suffix.lower() == ".h5ad":
                score += 3

            if "processed" in text:
                score -= 1

            if "marker" in text or "panel" in text:
                score -= 3

            candidates.append((score, path))

    return sorted(
        candidates,
        key=lambda item: (
            -item[0],
            str(item[1]).lower(),
        ),
    )


def inspect_csv_header(path: Path):
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:

            reader = csv.reader(
                handle,
                delimiter=delimiter,
            )

            header = next(reader, [])

        print("  Columns:", ", ".join(header[:25]))

        matches = [
            column for column in header
            if any(
                term in column.lower()
                for term in LABEL_TERMS
            )
        ]

        if matches:
            print("  Possible label columns:", matches)

    except (UnicodeError, OSError, csv.Error) as exc:
        print("  Header inspection unavailable:", exc)


def inspect_h5ad(path: Path):
    try:
        import h5py
    except ImportError:
        print("  h5py unavailable; cannot inspect metadata")
        return

    try:
        with h5py.File(path, "r") as handle:
            if "obs" not in handle:
                print("  No AnnData obs metadata found")
                return

            obs = handle["obs"]
            columns = list(obs.keys())

            print("  AnnData obs columns:")
            for column in columns[:50]:
                print("   -", column)

            matching_columns = [
                column for column in columns
                if any(
                    term in column.lower()
                    for term in LABEL_TERMS
                )
            ]

            if matching_columns:
                print(
                    "  Possible annotation columns:",
                    matching_columns,
                )

                for column in matching_columns:
                    entry = obs[column]

                    # Categorical metadata may store its
                    # small vocabulary separately.
                    if (
                        isinstance(entry, h5py.Group)
                        and "categories" in entry
                    ):
                        categories = entry["categories"]
                        sample = categories[:20]

                        labels = [
                            value.decode("utf-8")
                            if isinstance(value, bytes)
                            else str(value)
                            for value in sample
                        ]

                        print(
                            f"  {column} categories:",
                            labels,
                        )

    except (OSError, KeyError, TypeError, ValueError) as exc:
        print("  H5AD inspection unavailable:", exc)


def main():
    print("=" * 72)
    print("COSMX REFERENCE ANNOTATION SOURCE DISCOVERY")
    print("=" * 72)

    candidates = find_candidate_files()

    print("Candidate files found:", len(candidates))
    print("Showing up to:", MAX_RESULTS)

    for index, (score, path) in enumerate(
        candidates[:MAX_RESULTS], start=1
    ):
        print()
        print(f"[{index}] {path.relative_to(ROOT)}")
        print("  Relevance score:", score)

        suffix = path.suffix.lower()

        if suffix in (".csv", ".tsv"):
            inspect_csv_header(path)

        elif suffix == ".h5ad":
            inspect_h5ad(path)

        elif suffix in (".xlsx", ".json", ".parquet"):
            print(
                "  File identified; contents not used "
                "as reference without separate validation."
            )

    print()
    print("=" * 72)
    print("STATUS: COSMX_REFERENCE_DISCOVERY_COMPLETE")
    print("Files modified: NONE")
    print("Reference annotations verified: NOT YET")
    print("=" * 72)


if __name__ == "__main__":
    main()
