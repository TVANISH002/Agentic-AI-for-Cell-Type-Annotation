from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


# paths.py is located at:
# project_root/src/marker_interpretation_agent_v2/paths.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
TESTS_DIR = PROJECT_ROOT / "tests"

AGENT2_V2_DATA_DIR = DATA_DIR / "agent2_v2"
PANEL_OUTPUT_DIR = AGENT2_V2_DATA_DIR / "panel"
CLUSTER_OUTPUT_DIR = AGENT2_V2_DATA_DIR / "clusters"
UNRESOLVED_OUTPUT_DIR = AGENT2_V2_DATA_DIR / "unresolved"


MARKER_FILE_CANDIDATES = (
    DATA_DIR / "processed" / "cluster_marker_genes_brcr_Agentic_AI.csv",
    DATA_DIR / "processed" / "brcr_top50_primary_marker_genes_by_cluster.csv",
    DATA_DIR / "cluster_marker_genes_brcr_Agentic_AI.csv",
    PROJECT_ROOT / "cluster_marker_genes_brcr_Agentic_AI.csv",
)

CELLMARKER_FILE_CANDIDATES = (
    DATA_DIR / "raw" / "Cell_marker_Human.xlsx",
    DATA_DIR / "reference" / "Cell_marker_Human.xlsx",
    DATA_DIR / "Cell_marker_Human.xlsx",
    PROJECT_ROOT / "Cell_marker_Human.xlsx",
)

PANGLAODB_FILE_CANDIDATES = (
    DATA_DIR / "raw" / "PanglaoDB_markers_27_Mar_2020.tsv",
    DATA_DIR / "reference" / "PanglaoDB_markers_27_Mar_2020.tsv",
    DATA_DIR / "PanglaoDB_markers_27_Mar_2020.tsv",
    PROJECT_ROOT / "PanglaoDB_markers_27_Mar_2020.tsv",
)

HPA_PANEL_FILE_CANDIDATES = (
    DATA_DIR / "supplemental" / "hpa_panel_evidence.json",
    DATA_DIR / "processed" / "hpa_panel_evidence.json",
    PANEL_OUTPUT_DIR / "hpa_panel_evidence.json",
    PROJECT_ROOT / "hpa_panel_evidence.json",
)

PANEL_CATALOG_JSON = PANEL_OUTPUT_DIR / "panel_evidence_catalog.json"
PANEL_CATALOG_CSV = PANEL_OUTPUT_DIR / "panel_evidence_catalog.csv"
PANEL_GENE_LIST_CSV = PANEL_OUTPUT_DIR / "panel_gene_list.csv"
PANEL_UNRESOLVED_GENES_CSV = UNRESOLVED_OUTPUT_DIR / "unresolved_panel_genes.csv"

ALL_CLUSTERS_SUMMARY_CSV = (
    AGENT2_V2_DATA_DIR / "all_clusters_agent2_summary.csv"
)

ALL_CLUSTERS_SELECTED_CANDIDATES_CSV = (
    AGENT2_V2_DATA_DIR / "all_clusters_selected_candidates.csv"
)


def normalize_path(path: str | Path) -> Path:
    """Return an absolute normalized Path object."""

    return Path(path).expanduser().resolve()


def get_environment_path(variable_name: str) -> Path | None:
    """Read an optional path from an environment variable."""

    value = os.getenv(variable_name)

    if not value:
        return None

    return normalize_path(value)


def find_existing_path(
    candidates: Iterable[str | Path],
    *,
    description: str,
    environment_variable: str | None = None,
    required: bool = True,
) -> Path | None:
    """
    Return the first existing path.

    An environment variable may override the default candidate locations.
    """

    checked_paths: list[Path] = []

    if environment_variable:
        environment_path = get_environment_path(environment_variable)

        if environment_path is not None:
            checked_paths.append(environment_path)

            if environment_path.exists():
                return environment_path

    for candidate in candidates:
        candidate_path = normalize_path(candidate)
        checked_paths.append(candidate_path)

        if candidate_path.exists():
            return candidate_path

    if not required:
        return None

    checked_text = "\n".join(f"  - {path}" for path in checked_paths)

    raise FileNotFoundError(
        f"Could not locate {description}.\n"
        f"Checked the following locations:\n{checked_text}\n\n"
        f"Place the file in one of these locations or set the "
        f"{environment_variable!r} environment variable."
    )


def get_marker_file(path: str | Path | None = None) -> Path:
    """Resolve the cluster marker-gene CSV."""

    if path is not None:
        resolved = normalize_path(path)

        if not resolved.exists():
            raise FileNotFoundError(f"Marker file does not exist: {resolved}")

        return resolved

    result = find_existing_path(
        MARKER_FILE_CANDIDATES,
        description="cluster marker-gene CSV",
        environment_variable="AGENT2_MARKER_FILE",
    )

    assert result is not None
    return result


def get_cellmarker_file(path: str | Path | None = None) -> Path:
    """Resolve the CellMarker human reference spreadsheet."""

    if path is not None:
        resolved = normalize_path(path)

        if not resolved.exists():
            raise FileNotFoundError(f"CellMarker file does not exist: {resolved}")

        return resolved

    result = find_existing_path(
        CELLMARKER_FILE_CANDIDATES,
        description="CellMarker human reference spreadsheet",
        environment_variable="AGENT2_CELLMARKER_FILE",
    )

    assert result is not None
    return result


def get_panglaodb_file(path: str | Path | None = None) -> Path:
    """Resolve the PanglaoDB reference TSV."""

    if path is not None:
        resolved = normalize_path(path)

        if not resolved.exists():
            raise FileNotFoundError(f"PanglaoDB file does not exist: {resolved}")

        return resolved

    result = find_existing_path(
        PANGLAODB_FILE_CANDIDATES,
        description="PanglaoDB reference TSV",
        environment_variable="AGENT2_PANGLAODB_FILE",
    )

    assert result is not None
    return result


def get_hpa_panel_file(
    path: str | Path | None = None,
    *,
    required: bool = True,
) -> Path | None:
    """Resolve the panel-wide HPA evidence JSON."""

    if path is not None:
        resolved = normalize_path(path)

        if resolved.exists():
            return resolved

        if required:
            raise FileNotFoundError(
                f"HPA panel evidence file does not exist: {resolved}"
            )

        return None

    return find_existing_path(
        HPA_PANEL_FILE_CANDIDATES,
        description="panel-wide HPA evidence JSON",
        environment_variable="AGENT2_HPA_PANEL_FILE",
        required=required,
    )


def get_panel_catalog_file(
    path: str | Path | None = None,
    *,
    required: bool = True,
) -> Path | None:
    """Resolve the reusable panel evidence catalog JSON."""

    resolved = normalize_path(path) if path is not None else PANEL_CATALOG_JSON

    if resolved.exists():
        return resolved

    if required:
        raise FileNotFoundError(
            "Panel evidence catalog does not exist yet:\n"
            f"  {resolved}\n\n"
            "Run the panel catalog builder before running cluster interpretation."
        )

    return None


def get_cluster_output_dir(cluster_id: str | int) -> Path:
    """Return the output directory for one cluster."""

    cluster_name = str(cluster_id).strip()

    if not cluster_name:
        raise ValueError("cluster_id cannot be empty.")

    output_dir = CLUSTER_OUTPUT_DIR / f"cluster_{cluster_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def ensure_output_directories() -> dict[str, Path]:
    """Create every Agent 2 V2 output directory."""

    directories = {
        "agent2_v2": AGENT2_V2_DATA_DIR,
        "panel": PANEL_OUTPUT_DIR,
        "clusters": CLUSTER_OUTPUT_DIR,
        "unresolved": UNRESOLVED_OUTPUT_DIR,
    }

    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    return directories


def path_status() -> dict[str, str | None]:
    """Return a simple diagnostic view of important project paths."""

    marker_file = find_existing_path(
        MARKER_FILE_CANDIDATES,
        description="cluster marker-gene CSV",
        environment_variable="AGENT2_MARKER_FILE",
        required=False,
    )

    cellmarker_file = find_existing_path(
        CELLMARKER_FILE_CANDIDATES,
        description="CellMarker reference spreadsheet",
        environment_variable="AGENT2_CELLMARKER_FILE",
        required=False,
    )

    panglaodb_file = find_existing_path(
        PANGLAODB_FILE_CANDIDATES,
        description="PanglaoDB reference TSV",
        environment_variable="AGENT2_PANGLAODB_FILE",
        required=False,
    )

    hpa_file = get_hpa_panel_file(required=False)

    return {
        "project_root": str(PROJECT_ROOT),
        "marker_file": str(marker_file) if marker_file else None,
        "cellmarker_file": str(cellmarker_file) if cellmarker_file else None,
        "panglaodb_file": str(panglaodb_file) if panglaodb_file else None,
        "hpa_panel_file": str(hpa_file) if hpa_file else None,
        "panel_catalog_json": str(PANEL_CATALOG_JSON),
        "cluster_output_dir": str(CLUSTER_OUTPUT_DIR),
    }