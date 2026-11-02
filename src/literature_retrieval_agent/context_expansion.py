from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic import Field, field_validator

from literature_retrieval_agent.schemas import StrictModel


def _clean_terms(values: list[str]) -> list[str]:
    """Normalize terms while preserving their original order."""

    cleaned: list[str] = []
    seen: set[str] = set()

    for value in values:
        term = " ".join(str(value).split())
        key = term.casefold()

        if term and key not in seen:
            seen.add(key)
            cleaned.append(term)

    return cleaned


class ContextExpansionTerms(StrictModel):
    """
    Runtime vocabulary used by context-expanded retrieval.

    Candidate, tissue, disease, and platform aliases live in JSON rather
    than Python so that a different cluster or dataset can provide a new
    vocabulary without changing source code.
    """

    schema_name: str = "ContextExpansionTerms"
    schema_version: str = "1.0"

    tissue_aliases: list[str] = Field(default_factory=list)
    disease_aliases: list[str] = Field(default_factory=list)
    platform_aliases: list[str] = Field(default_factory=list)
    candidate_aliases: dict[str, list[str]] = Field(default_factory=dict)

    @field_validator(
        "tissue_aliases",
        "disease_aliases",
        "platform_aliases",
    )
    @classmethod
    def clean_alias_list(cls, values: list[str]) -> list[str]:
        return _clean_terms(values)

    @field_validator("candidate_aliases")
    @classmethod
    def clean_candidate_aliases(
        cls,
        values: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        cleaned: dict[str, list[str]] = {}
        seen_keys: set[str] = set()

        for raw_candidate, raw_aliases in values.items():
            candidate = " ".join(str(raw_candidate).split())
            key = candidate.casefold()

            if not candidate:
                raise ValueError(
                    "candidate_aliases cannot contain an empty key."
                )

            if key in seen_keys:
                raise ValueError(
                    "candidate_aliases contains duplicate candidate keys "
                    f"after normalization: {candidate!r}."
                )

            seen_keys.add(key)
            cleaned[candidate] = _clean_terms(raw_aliases)

        return cleaned

    def aliases_for_candidate(self, candidate_name: str) -> list[str]:
        """Return aliases for one canonical candidate name."""

        target = " ".join(candidate_name.split()).casefold()

        for configured_name, aliases in self.candidate_aliases.items():
            if configured_name.casefold() == target:
                return list(aliases)

        return []


@dataclass(frozen=True, slots=True)
class ContextQueryConfig:
    """Technical controls for the generalized v2 query strategy."""

    marker_batch_size: int = 6
    max_marker_batches: int = 4
    include_cluster_marker_queries: bool = True
    include_platform_context_query: bool = True
    strategy_name: str = "context_expanded_v2"
    query_plan_schema_version: str = "2.0"
    retrieval_schema_version: str = "2.0"

    def __post_init__(self) -> None:
        if self.marker_batch_size <= 0:
            raise ValueError("marker_batch_size must be greater than zero.")

        if self.max_marker_batches <= 0:
            raise ValueError("max_marker_batches must be greater than zero.")

        for field_name in (
            "strategy_name",
            "query_plan_schema_version",
            "retrieval_schema_version",
        ):
            value = " ".join(str(getattr(self, field_name)).split())

            if not value:
                raise ValueError(f"{field_name} cannot be empty.")

            object.__setattr__(self, field_name, value)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ContextQueryAudit(StrictModel):
    """Machine-readable proof that the v2 query plan is balanced."""

    schema_name: str = "ContextQueryAudit"
    schema_version: str = "1.0"
    strategy_name: str
    cluster_id: str
    valid: bool
    total_query_count: int = Field(ge=0)
    candidate_query_counts: dict[str, int] = Field(default_factory=dict)
    candidate_marker_batch_counts: dict[str, int] = Field(
        default_factory=dict
    )
    candidate_alias_counts: dict[str, int] = Field(default_factory=dict)
    equal_candidate_query_counts: bool
    equal_candidate_marker_batch_counts: bool
    all_candidate_queries_context_constrained: bool
    expert_label_used_as_query_input: bool = False
    accepted_input_schema: str = "LiteratureRequest"
    accepted_input_fields: list[str] = Field(default_factory=list)
    unused_alias_config_candidates: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def load_context_expansion_terms(
    path: str | Path | None,
) -> ContextExpansionTerms:
    """Load and validate optional runtime context vocabulary."""

    if path is None:
        return ContextExpansionTerms()

    resolved = Path(path).expanduser().resolve()

    if not resolved.is_file():
        raise FileNotFoundError(
            f"Context-expansion terms JSON was not found: {resolved}"
        )

    try:
        payload = json.loads(resolved.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Context-expansion terms file is not valid JSON: {resolved}"
        ) from exc

    return ContextExpansionTerms.model_validate(payload)
