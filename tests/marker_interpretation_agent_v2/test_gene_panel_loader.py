from __future__ import annotations

import csv
from pathlib import Path

import pytest

from marker_interpretation_agent_v2.panel.gene_panel_loader import (
    GenePanelLoader,
)


FIELDNAMES = [
    "names",
    "scores",
    "logfoldchanges",
    "pvals",
    "pvals_adj",
    "pct_nz_group",
    "pct_nz_reference",
    "spatial_cluster",
]


def _write_marker_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> Path:
    """Write a minimal marker CSV for loader tests."""

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()
        writer.writerows(rows)

    return path


def _valid_rows() -> list[dict[str, object]]:
    """Return valid marker rows for two clusters."""

    return [
        {
            "names": "GATA3",
            "scores": 10.0,
            "logfoldchanges": 3.0,
            "pvals": 0.001,
            "pvals_adj": 0.01,
            "pct_nz_group": 0.80,
            "pct_nz_reference": 0.20,
            "spatial_cluster": 0,
        },
        {
            "names": "EPCAM",
            "scores": 9.0,
            "logfoldchanges": 2.5,
            "pvals": 0.002,
            "pvals_adj": 0.02,
            "pct_nz_group": 0.70,
            "pct_nz_reference": 0.15,
            "spatial_cluster": 0,
        },
        {
            "names": "COL1A1",
            "scores": 8.0,
            "logfoldchanges": 2.2,
            "pvals": 0.003,
            "pvals_adj": 0.03,
            "pct_nz_group": 0.65,
            "pct_nz_reference": 0.10,
            "spatial_cluster": 1,
        },
    ]


def test_loads_cluster_ids_and_markers(
    tmp_path: Path,
) -> None:
    """Load clusters and preserve within-cluster marker ranking."""

    marker_file = _write_marker_csv(
        tmp_path / "markers.csv",
        _valid_rows(),
    )

    loader = GenePanelLoader(
        marker_file
    )

    cluster_ids = {
        str(cluster_id)
        for cluster_id
        in loader.get_cluster_ids()
    }

    assert cluster_ids == {
        "0",
        "1",
    }

    cluster_zero = (
        loader.get_cluster_markers(
            "0"
        )
    )

    assert [
        marker.gene
        for marker in cluster_zero
    ] == [
        "GATA3",
        "EPCAM",
    ]

    assert [
        marker.rank_within_cluster
        for marker in cluster_zero
    ] == [
        1,
        2,
    ]

    assert (
        cluster_zero[0].score
        == pytest.approx(10.0)
    )

    assert (
        cluster_zero[0].log_fold_change
        == pytest.approx(3.0)
    )

    assert (
        cluster_zero[0].adjusted_p_value
        == pytest.approx(0.01)
    )

    assert (
        cluster_zero[0].pct_nz_group
        == pytest.approx(0.80)
    )

    assert (
        cluster_zero[0].pct_nz_reference
        == pytest.approx(0.20)
    )


def test_rejects_duplicate_gene_within_cluster(
    tmp_path: Path,
) -> None:
    """Reject duplicate cluster-gene combinations when loading."""

    rows = _valid_rows()

    rows.append(
        {
            **rows[0],
            "scores": 7.0,
        }
    )

    marker_file = _write_marker_csv(
        tmp_path
        / "duplicate_gene.csv",
        rows,
    )

    loader = GenePanelLoader(
        marker_file
    )

    with pytest.raises(
        ValueError,
        match="(?i)duplicate",
    ):
        loader.get_cluster_ids()


def test_rejects_invalid_group_proportion(
    tmp_path: Path,
) -> None:
    """Reject group proportions outside zero to one when loading."""

    rows = _valid_rows()

    rows[0][
        "pct_nz_group"
    ] = 1.20

    marker_file = _write_marker_csv(
        tmp_path
        / "invalid_group_proportion.csv",
        rows,
    )

    loader = GenePanelLoader(
        marker_file
    )

    with pytest.raises(
        ValueError
    ):
        loader.get_cluster_ids()


def test_rejects_invalid_reference_proportion(
    tmp_path: Path,
) -> None:
    """Reject reference proportions outside zero to one when loading."""

    rows = _valid_rows()

    rows[0][
        "pct_nz_reference"
    ] = -0.10

    marker_file = _write_marker_csv(
        tmp_path
        / "invalid_reference_proportion.csv",
        rows,
    )

    loader = GenePanelLoader(
        marker_file
    )

    with pytest.raises(
        ValueError
    ):
        loader.get_cluster_ids()


def test_generates_unique_ranks_within_cluster(
    tmp_path: Path,
) -> None:
    """Confirm generated ranks are unique within each cluster."""

    marker_file = _write_marker_csv(
        tmp_path / "markers.csv",
        _valid_rows(),
    )

    loader = GenePanelLoader(
        marker_file
    )

    cluster_zero = (
        loader.get_cluster_markers(
            "0"
        )
    )

    ranks = [
        marker.rank_within_cluster
        for marker in cluster_zero
    ]

    assert len(ranks) == len(
        set(ranks)
    )