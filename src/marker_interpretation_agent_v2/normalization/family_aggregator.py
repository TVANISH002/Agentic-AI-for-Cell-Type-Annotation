

"""Aggregate candidate labels into conservative subtype-level review groups.

Raw database labels remain unchanged. Closely related labels are aggregated at
the subtype level, while broader biological lineage is stored separately and
is not used as an automatic replacement for the detailed candidate ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isnan
from pathlib import Path
from typing import Iterable

import pandas as pd

from .family_mapping import (
    MAPPING_SCHEMA_VERSION,
    map_cell_type,
)


_REQUIRED_COLUMNS = {
    "cluster_id",
    "candidate_rank",
    "cell_type",
    "selected_for_downstream",
    "supporting_genes",
    "both_source_supporting_genes",
    "evidence_sources",
    "tissue_contexts",
    "disease_contexts",
    "cell_ontology_ids",
    "hpa_top_cell_types",
    "warnings",
}


@dataclass
class _SubtypeAccumulator:
    """Internal state for one conservative subtype group."""

    cluster_id: str
    subtype_group: str
    best_raw_candidate_rank: int

    broad_lineage_families: set[str] = field(
        default_factory=set
    )

    supporting_genes: set[str] = field(
        default_factory=set
    )

    both_source_supporting_genes: set[str] = field(
        default_factory=set
    )

    raw_cell_types: set[str] = field(
        default_factory=set
    )

    canonical_cell_types: set[str] = field(
        default_factory=set
    )

    selected_raw_cell_types: set[str] = field(
        default_factory=set
    )

    mapping_rules: set[str] = field(
        default_factory=set
    )

    evidence_sources: set[str] = field(
        default_factory=set
    )

    tissue_contexts: set[str] = field(
        default_factory=set
    )

    disease_contexts: set[str] = field(
        default_factory=set
    )

    cell_ontology_ids: set[str] = field(
        default_factory=set
    )

    hpa_top_cell_types: set[str] = field(
        default_factory=set
    )

    warnings: set[str] = field(
        default_factory=set
    )


def _is_missing(value: object) -> bool:
    """Return True for scalar missing values read from CSV files."""

    if value is None or value is pd.NA:
        return True

    if isinstance(value, float):
        return isnan(value)

    return str(value).strip().casefold() in {
        "",
        "nan",
        "none",
        "<na>",
        "nat",
    }


def _split_pipe_values(value: object) -> set[str]:
    """Convert a pipe-separated CSV value into a clean unique set."""

    if _is_missing(value):
        return set()

    return {
        item.strip()
        for item in str(value).split("|")
        if item.strip()
        and item.strip().casefold()
        not in {
            "nan",
            "none",
            "<na>",
        }
    }


def _join_sorted(values: Iterable[str]) -> str:
    """Return deterministic pipe-separated output."""

    return " | ".join(
        sorted(
            set(values),
            key=str.casefold,
        )
    )


def _as_bool(value: object) -> bool:
    """Convert a scalar CSV value into a Boolean."""

    if isinstance(value, bool):
        return value

    if _is_missing(value):
        return False

    return str(value).strip().casefold() in {
        "true",
        "1",
        "yes",
        "y",
    }


class CandidateFamilyAggregator:
    """Create subtype summaries without changing raw candidate outputs."""

    def aggregate_dataframe(
        self,
        candidates: pd.DataFrame,
    ) -> pd.DataFrame:
        """Aggregate one cluster's candidates by conservative subtype."""

        missing = sorted(
            _REQUIRED_COLUMNS.difference(
                candidates.columns
            )
        )

        if missing:
            raise ValueError(
                "Candidate table is missing required columns: "
                + ", ".join(missing)
            )

        if candidates.empty:
            return pd.DataFrame(
                columns=self.output_columns()
            )

        cluster_ids = {
            str(value).strip()
            for value in candidates[
                "cluster_id"
            ].dropna().tolist()
            if str(value).strip()
        }

        if len(cluster_ids) != 1:
            raise ValueError(
                "Expected exactly one cluster_id, "
                f"found {sorted(cluster_ids)}."
            )

        cluster_id = next(
            iter(cluster_ids)
        )

        accumulators: dict[
            str,
            _SubtypeAccumulator,
        ] = {}

        rows = candidates.to_dict(
            orient="records"
        )

        for row in rows:
            mapping = map_cell_type(
                str(row["cell_type"])
            )

            subtype_group = (
                mapping.subtype_group
            )

            rank = int(
                float(
                    str(
                        row["candidate_rank"]
                    )
                )
            )

            accumulator = accumulators.setdefault(
                subtype_group,
                _SubtypeAccumulator(
                    cluster_id=cluster_id,
                    subtype_group=subtype_group,
                    best_raw_candidate_rank=rank,
                ),
            )

            accumulator.best_raw_candidate_rank = min(
                accumulator.best_raw_candidate_rank,
                rank,
            )

            accumulator.broad_lineage_families.add(
                mapping.broad_lineage_family
            )

            accumulator.supporting_genes.update(
                _split_pipe_values(
                    row["supporting_genes"]
                )
            )

            accumulator.both_source_supporting_genes.update(
                _split_pipe_values(
                    row[
                        "both_source_supporting_genes"
                    ]
                )
            )

            accumulator.raw_cell_types.add(
                mapping.cleaned_cell_type
            )

            accumulator.canonical_cell_types.add(
                mapping.canonical_cell_type
            )

            accumulator.mapping_rules.add(
                mapping.mapping_rule
            )

            accumulator.evidence_sources.update(
                _split_pipe_values(
                    row["evidence_sources"]
                )
            )

            accumulator.tissue_contexts.update(
                _split_pipe_values(
                    row["tissue_contexts"]
                )
            )

            accumulator.disease_contexts.update(
                _split_pipe_values(
                    row["disease_contexts"]
                )
            )

            accumulator.cell_ontology_ids.update(
                _split_pipe_values(
                    row["cell_ontology_ids"]
                )
            )

            accumulator.hpa_top_cell_types.update(
                _split_pipe_values(
                    row["hpa_top_cell_types"]
                )
            )

            accumulator.warnings.update(
                _split_pipe_values(
                    row["warnings"]
                )
            )

            if _as_bool(
                row["selected_for_downstream"]
            ):
                accumulator.selected_raw_cell_types.add(
                    mapping.cleaned_cell_type
                )

        records = [
            self._to_record(accumulator)
            for accumulator
            in accumulators.values()
        ]

        summary = pd.DataFrame(
            records
        )

        summary = summary.sort_values(
            by=[
                "unique_supporting_gene_count",
                "both_source_supporting_gene_count",
                "selected_raw_candidate_count",
                "best_raw_candidate_rank",
                "subtype_group",
            ],
            ascending=[
                False,
                False,
                False,
                True,
                True,
            ],
            kind="stable",
        ).reset_index(
            drop=True
        )

        ranks = range(
            1,
            len(summary) + 1,
        )

        summary.insert(
            1,
            "family_rank",
            ranks,
        )

        summary.insert(
            2,
            "subtype_rank",
            range(
                1,
                len(summary) + 1,
            ),
        )

        return summary[
            self.output_columns()
        ]

    def aggregate_csv(
        self,
        input_csv: Path | str,
        output_csv: Path | str,
    ) -> pd.DataFrame:
        """Read raw candidates and write the subtype summary."""

        input_path = Path(
            input_csv
        )

        output_path = Path(
            output_csv
        )

        if not input_path.exists():
            raise FileNotFoundError(
                "Candidate CSV was not found: "
                f"{input_path}"
            )

        candidates = pd.read_csv(
            input_path
        )

        summary = self.aggregate_dataframe(
            candidates
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary.to_csv(
            output_path,
            index=False,
        )

        return summary

    @staticmethod
    def _to_record(
        accumulator: _SubtypeAccumulator,
    ) -> dict[str, object]:
        """Convert one subtype accumulator into an output record."""

        selected = (
            accumulator.selected_raw_cell_types
        )

        warnings = set(
            accumulator.warnings
        )

        if (
            len(
                accumulator.broad_lineage_families
            )
            > 1
        ):
            warnings.add(
                "multiple_broad_lineages_in_subtype_group"
            )

        broad_lineage_family = _join_sorted(
            accumulator.broad_lineage_families
        )

        return {
            "cluster_id": (
                accumulator.cluster_id
            ),

            # Backward-compatible field.
            # It now represents the conservative subtype group.
            "cell_type_family": (
                accumulator.subtype_group
            ),

            "subtype_group": (
                accumulator.subtype_group
            ),

            "broad_lineage_family": (
                broad_lineage_family
            ),

            "aggregation_level": (
                "subtype_group"
            ),

            "mapping_schema_version": (
                MAPPING_SCHEMA_VERSION
            ),

            "contains_selected_raw_candidate": bool(
                selected
            ),

            "unique_supporting_gene_count": len(
                accumulator.supporting_genes
            ),

            "supporting_genes": _join_sorted(
                accumulator.supporting_genes
            ),

            "both_source_supporting_gene_count": len(
                accumulator.both_source_supporting_genes
            ),

            "both_source_supporting_genes": _join_sorted(
                accumulator.both_source_supporting_genes
            ),

            "raw_candidate_count": len(
                accumulator.raw_cell_types
            ),

            "raw_cell_types": _join_sorted(
                accumulator.raw_cell_types
            ),

            "canonical_cell_types": _join_sorted(
                accumulator.canonical_cell_types
            ),

            "selected_raw_candidate_count": len(
                selected
            ),

            "selected_raw_cell_types": _join_sorted(
                selected
            ),

            "best_raw_candidate_rank": (
                accumulator.best_raw_candidate_rank
            ),

            "mapping_rules": _join_sorted(
                accumulator.mapping_rules
            ),

            "evidence_sources": _join_sorted(
                accumulator.evidence_sources
            ),

            "tissue_contexts": _join_sorted(
                accumulator.tissue_contexts
            ),

            "disease_contexts": _join_sorted(
                accumulator.disease_contexts
            ),

            "cell_ontology_ids": _join_sorted(
                accumulator.cell_ontology_ids
            ),

            "hpa_top_cell_types": _join_sorted(
                accumulator.hpa_top_cell_types
            ),

            "warnings": _join_sorted(
                warnings
            ),
        }

    @staticmethod
    def output_columns() -> list[str]:
        """Return the stable subtype-summary output schema."""

        return [
            "cluster_id",
            "family_rank",
            "subtype_rank",
            "cell_type_family",
            "subtype_group",
            "broad_lineage_family",
            "aggregation_level",
            "mapping_schema_version",
            "contains_selected_raw_candidate",
            "unique_supporting_gene_count",
            "supporting_genes",
            "both_source_supporting_gene_count",
            "both_source_supporting_genes",
            "raw_candidate_count",
            "raw_cell_types",
            "canonical_cell_types",
            "selected_raw_candidate_count",
            "selected_raw_cell_types",
            "best_raw_candidate_rank",
            "mapping_rules",
            "evidence_sources",
            "tissue_contexts",
            "disease_contexts",
            "cell_ontology_ids",
            "hpa_top_cell_types",
            "warnings",
        ]






