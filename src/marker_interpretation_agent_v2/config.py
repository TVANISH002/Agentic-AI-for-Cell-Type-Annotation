from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class Agent2Config:
    """Validated configuration for Marker Interpretation Agent V2."""

    species: str = "Human"
    tissue: str = "Breast"
    platform: str = "Xenium"
    top_k: int = 50

    near_tie_ratio: float = 1.10
    broad_cell_type_count: int = 10
    broad_ratio: float = 1.50
    trace_max_ncpm: float = 0.10
    trace_max_read_count: float = 1.0

    candidate_min_relative_drop: float = 0.30
    candidate_min_absolute_drop: int = 2
    candidate_elbow_search_limit: int = 5
    candidate_fallback_max_candidates: int = 3
    candidate_min_supporting_genes: int = 2

    candidate_reporting_min_supporting_genes: int = 2

    hpa_candidate_voting_weight: float = 0.0
    hpa_enabled: bool = True

    broad_candidate_labels: tuple[str, ...] = (
        "Cancer cell",
        "Epithelial cell",
    )

    direct_marker_warning_minimum: int = 3
    direct_marker_warning_fraction: float = 0.25

    panel_catalog_schema_name: str = "PanelEvidenceCatalog"
    panel_catalog_schema_version: str = "2.0"

    marker_evidence_schema_name: str = "MarkerEvidence"
    marker_evidence_schema_version: str = "2.0"

    def __post_init__(self) -> None:
        """Validate configuration values after construction."""

        for field_name in ("species", "tissue", "platform"):
            value = str(getattr(self, field_name)).strip()

            if not value:
                raise ValueError(f"{field_name} cannot be empty.")

            object.__setattr__(self, field_name, value)

        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if self.near_tie_ratio <= 0:
            raise ValueError("near_tie_ratio must be greater than zero.")

        if self.broad_cell_type_count <= 0:
            raise ValueError("broad_cell_type_count must be greater than zero.")

        if self.broad_ratio <= 0:
            raise ValueError("broad_ratio must be greater than zero.")

        if self.trace_max_ncpm < 0:
            raise ValueError("trace_max_ncpm cannot be negative.")

        if self.trace_max_read_count < 0:
            raise ValueError("trace_max_read_count cannot be negative.")

        if not 0 < self.candidate_min_relative_drop <= 1:
            raise ValueError(
                "candidate_min_relative_drop must be in the interval (0, 1]."
            )

        if self.candidate_min_absolute_drop <= 0:
            raise ValueError(
                "candidate_min_absolute_drop must be greater than zero."
            )

        if self.candidate_elbow_search_limit < 2:
            raise ValueError(
                "candidate_elbow_search_limit must be at least 2."
            )

        if self.candidate_fallback_max_candidates <= 0:
            raise ValueError(
                "candidate_fallback_max_candidates must be greater than zero."
            )

        if self.candidate_min_supporting_genes <= 0:
            raise ValueError(
                "candidate_min_supporting_genes must be greater than zero."
            )

        if self.candidate_reporting_min_supporting_genes < 0:
            raise ValueError(
                "candidate_reporting_min_supporting_genes cannot be negative."
            )

        if self.hpa_candidate_voting_weight != 0.0:
            raise ValueError(
                "hpa_candidate_voting_weight must remain 0.0 "
                "in the current baseline."
            )

        if not self.broad_candidate_labels:
            raise ValueError("broad_candidate_labels cannot be empty.")

        cleaned_labels = tuple(
            dict.fromkeys(
                " ".join(str(label).strip().split())
                for label in self.broad_candidate_labels
                if str(label).strip()
            )
        )

        if not cleaned_labels:
            raise ValueError(
                "broad_candidate_labels must contain at least one "
                "non-empty label."
            )

        object.__setattr__(
            self,
            "broad_candidate_labels",
            cleaned_labels,
        )

        if self.direct_marker_warning_minimum < 0:
            raise ValueError(
                "direct_marker_warning_minimum cannot be negative."
            )

        if not 0 <= self.direct_marker_warning_fraction <= 1:
            raise ValueError(
                "direct_marker_warning_fraction must be between 0 and 1."
            )

        for field_name in (
            "panel_catalog_schema_name",
            "panel_catalog_schema_version",
            "marker_evidence_schema_name",
            "marker_evidence_schema_version",
        ):
            value = str(getattr(self, field_name)).strip()

            if not value:
                raise ValueError(f"{field_name} cannot be empty.")

            object.__setattr__(self, field_name, value)

    def direct_marker_warning_threshold(
        self,
        total_input_genes: int,
    ) -> int:
        """
        Return the direct-marker count below which a warning is raised.

        Example:
        50 genes × 0.25 = 12.5
        Integer threshold = 12
        """

        if total_input_genes < 0:
            raise ValueError(
                "total_input_genes cannot be negative."
            )

        fraction_threshold = int(
            total_input_genes
            * self.direct_marker_warning_fraction
        )

        return max(
            self.direct_marker_warning_minimum,
            fraction_threshold,
        )

    def meets_candidate_reporting_threshold(
        self,
        supporting_gene_count: int,
    ) -> bool:
        """Return whether a candidate should appear in the threshold output."""

        if supporting_gene_count < 0:
            raise ValueError(
                "supporting_gene_count cannot be negative."
            )

        return (
            supporting_gene_count
            >= self.candidate_reporting_min_supporting_genes
        )

    def is_broad_candidate_label(
        self,
        cell_type: str,
    ) -> bool:
        """Return whether a candidate label is biologically broad."""

        normalized = " ".join(
            str(cell_type).strip().split()
        ).casefold()

        if not normalized:
            return False

        broad_labels = {
            label.casefold()
            for label in self.broad_candidate_labels
        }

        return normalized in broad_labels

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration snapshot."""

        payload = asdict(self)

        payload["broad_candidate_labels"] = list(
            self.broad_candidate_labels
        )

        return payload

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> "Agent2Config":
        """Create configuration from a dictionary-like object."""

        payload = dict(values)

        broad_labels = payload.get(
            "broad_candidate_labels"
        )

        if broad_labels is not None:
            if isinstance(broad_labels, str):
                payload["broad_candidate_labels"] = tuple(
                    value.strip()
                    for value in broad_labels.replace(",", "|").split("|")
                    if value.strip()
                )
            else:
                payload["broad_candidate_labels"] = tuple(
                    str(value).strip()
                    for value in broad_labels
                    if str(value).strip()
                )

        return cls(**payload)


DEFAULT_CONFIG = Agent2Config()