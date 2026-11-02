import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, cast

import numpy as np
import pandas as pd

from .context import CellTypeNormalizer, ContextMatcher
from .schemas import (
    DEGValidationSummary,
    GeneEvidence,
    InputGene,
    KnowledgeEvidence,
    SourceCoverage,
    SourceEvidence,
    SupportingCellTypeEvidence,
)

RetrievalMethod = Literal[
    "official_symbol",
    "alias",
]


@dataclass
class _CellTypeAccumulator:
    raw_labels: set[str] = field(default_factory=set)
    genes: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)
    tissues: set[str] = field(default_factory=set)
    diseases: set[str] = field(default_factory=set)
    ontology_ids: set[str] = field(default_factory=set)
    evidence_count: int = 0


class EvidenceBuilder:
    """Builds validated, context-aware KnowledgeEvidence objects."""

    @staticmethod
    def _records(
        dataframe: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        # Converts a DataFrame into row dictionaries.
        return cast(
            list[dict[str, Any]],
            dataframe.to_dict(orient="records"),
        )

    @staticmethod
    def _clean_value(value: Any) -> Any:
        # Converts missing and NumPy values into Python values.
        if value is None or value is pd.NA:
            return None

        try:
            missing = pd.isna(value)

            if (
                isinstance(missing, (bool, np.bool_))
                and bool(missing)
            ):
                return None

        except (TypeError, ValueError):
            pass

        if hasattr(value, "item"):
            return value.item()

        return value

    @classmethod
    def _clean_text(
        cls,
        value: Any,
    ) -> str | None:
        # Cleans and normalizes text values.
        cleaned = cls._clean_value(value)

        if cleaned is None:
            return None

        text = " ".join(
            str(cleaned).strip().split()
        )

        return text or None

    @classmethod
    def _clean_identifier(
        cls,
        value: Any,
    ) -> str | None:
        # Cleans identifier values for consistent output.
        cleaned = cls._clean_value(value)

        if cleaned is None:
            return None

        if (
            isinstance(cleaned, float)
            and cleaned.is_integer()
        ):
            return str(int(cleaned))

        text = str(cleaned).strip()

        if (
            text.endswith(".0")
            and text[:-2].isdigit()
        ):
            return text[:-2]

        return text or None

    @classmethod
    def _clean_ontology_id(
        cls,
        value: Any,
    ) -> str | None:
        # Standardizes Cell Ontology identifiers.
        text = cls._clean_text(value)

        if text is None:
            return None

        match = re.fullmatch(
            r"CL[_:](\d{7})",
            text,
            flags=re.IGNORECASE,
        )

        if match is not None:
            return f"CL:{match.group(1)}"

        return text

    @staticmethod
    def _unique_sorted(
        values: Iterable[str | None],
    ) -> list[str]:
        # Returns unique non-empty values in sorted order.
        unique_values = {
            value
            for value in values
            if value is not None
            and value != ""
        }

        return sorted(
            unique_values,
            key=str.casefold,
        )

    @classmethod
    def _split_aliases(
        cls,
        value: Any,
    ) -> list[str]:
        # Splits pipe-separated gene aliases.
        text = cls._clean_text(value)

        if text is None:
            return []

        return [
            item.strip()
            for item in text.split("|")
            if item.strip()
        ]

    @classmethod
    def _safe_bool(
        cls,
        value: Any,
    ) -> bool:
        # Safely converts a value to boolean.
        cleaned = cls._clean_value(value)

        if cleaned is None:
            return False

        return bool(cleaned)

    @staticmethod
    def _retrieval_method(
        value: Any,
    ) -> RetrievalMethod:
        # Normalizes the evidence retrieval method.
        if str(value) == "alias":
            return "alias"

        return "official_symbol"

    @staticmethod
    def _filter_cluster(
        dataframe: pd.DataFrame,
        cluster_id: int | str,
    ) -> pd.DataFrame:
        # Filters evidence rows for one cluster.
        if dataframe.empty:
            return dataframe.copy()

        if "spatial_cluster" not in dataframe.columns:
            raise ValueError(
                "Evidence table is missing 'spatial_cluster'."
            )

        cluster_mask = (
            dataframe["spatial_cluster"].astype(str)
            == str(cluster_id)
        )

        return dataframe.loc[
            cluster_mask
        ].copy()

    def _build_input_genes(
        self,
        markers: pd.DataFrame,
    ) -> list[InputGene]:
        # Builds validated input-gene records.
        input_genes: list[InputGene] = []

        sorted_markers = markers.sort_values(
            "rank_within_cluster"
        )

        for row in self._records(sorted_markers):
            raw_warnings = row.get(
                "deg_validation_warnings"
            )

            warnings = (
                [
                    str(item)
                    for item in raw_warnings
                ]
                if isinstance(raw_warnings, list)
                else []
            )

            input_genes.append(
                InputGene.model_validate(
                    {
                        "gene": str(
                            row["names"]
                        ).strip(),
                        "statistics": {
                            "rank_within_cluster": int(
                                row[
                                    "rank_within_cluster"
                                ]
                            ),
                            "score": float(
                                row["scores"]
                            ),
                            "log_fold_change": float(
                                row[
                                    "logfoldchanges"
                                ]
                            ),
                            "p_value": float(
                                row["pvals"]
                            ),
                            "adjusted_p_value": float(
                                row["pvals_adj"]
                            ),
                            "percent_expressed_group": float(
                                row[
                                    "pct_nz_group"
                                ]
                            ),
                            "percent_expressed_reference": float(
                                row[
                                    "pct_nz_reference"
                                ]
                            ),
                            "validation_warnings": warnings,
                        },
                    }
                )
            )

        return input_genes

    def _cellmarker_records(
        self,
        rows: pd.DataFrame,
        matcher: ContextMatcher,
    ) -> list[SourceEvidence]:
        # Builds structured CellMarker evidence records.
        records: list[SourceEvidence] = []

        for row in self._records(rows):
            tissue = self._clean_text(
                row.get("tissue_type")
            )

            disease = self._clean_text(
                row.get("cancer_type")
            )

            cell_type = self._clean_text(
                row.get("cell_name")
            )

            record = SourceEvidence.model_validate(
                {
                    "source": "CellMarker",
                    "retrieval_method": (
                        self._retrieval_method(
                            row.get(
                                "retrieval_match_method"
                            )
                        )
                    ),
                    "reported_marker": self._clean_text(
                        row.get("marker")
                    ),
                    "official_symbol": self._clean_text(
                        row.get("Symbol")
                    ),
                    "cell_type": cell_type,
                    "normalized_cell_type": (
                        CellTypeNormalizer.normalize(
                            cell_type
                        )
                    ),
                    "tissue": tissue,
                    "disease_context": disease,
                    "context_match_level": (
                        matcher.classify(
                            tissue,
                            disease,
                        )
                    ),
                    "cell_ontology_id": (
                        self._clean_ontology_id(
                            row.get(
                                "cellontology_id"
                            )
                        )
                    ),
                    "description": None,
                    "pmid": self._clean_identifier(
                        row.get("PMID")
                    ),
                    "evidence_title": self._clean_text(
                        row.get("Title")
                    ),
                    "metadata": {
                        "cell_category": self._clean_text(
                            row.get("cell_type")
                        ),
                        "marker_source": self._clean_text(
                            row.get(
                                "marker_source"
                            )
                        ),
                        "source_gene_name_raw": (
                            self._clean_text(
                                row.get("Genename")
                            )
                        ),
                        "description_policy": (
                            "excluded_from_"
                            "validated_descriptions"
                        ),
                    },
                }
            )

            records.append(record)

        return records

    def _panglaodb_records(
        self,
        rows: pd.DataFrame,
        matcher: ContextMatcher,
    ) -> list[SourceEvidence]:
        # Builds structured PanglaoDB evidence records.
        records: list[SourceEvidence] = []

        for row in self._records(rows):
            tissue = self._clean_text(
                row.get("organ")
            )

            cell_type = self._clean_text(
                row.get("cell type")
            )

            human_only_alias_safe = (
                self._safe_bool(
                    row.get(
                        "human_only_alias_safe"
                    )
                )
            )

            record = SourceEvidence.model_validate(
                {
                    "source": "PanglaoDB",
                    "retrieval_method": (
                        self._retrieval_method(
                            row.get(
                                "retrieval_match_method"
                            )
                        )
                    ),
                    "reported_marker": self._clean_text(
                        row.get(
                            "official gene symbol"
                        )
                    ),
                    "official_symbol": self._clean_text(
                        row.get(
                            "official gene symbol"
                        )
                    ),
                    "cell_type": cell_type,
                    "normalized_cell_type": (
                        CellTypeNormalizer.normalize(
                            cell_type
                        )
                    ),
                    "tissue": tissue,
                    "disease_context": None,
                    "context_match_level": (
                        matcher.classify(
                            tissue,
                            None,
                        )
                    ),
                    "cell_ontology_id": None,
                    "description": self._clean_text(
                        row.get(
                            "product description"
                        )
                    ),
                    "pmid": None,
                    "evidence_title": None,
                    "metadata": {
                        "species": self._clean_text(
                            row.get("species")
                        ),
                        "canonical_marker": (
                            self._clean_value(
                                row.get(
                                    "canonical marker"
                                )
                            )
                        ),
                        "germ_layer": self._clean_text(
                            row.get("germ layer")
                        ),
                        "gene_type": self._clean_text(
                            row.get("gene type")
                        ),
                        "nicknames_raw": self._clean_text(
                            row.get("nicknames")
                        ),
                        "aliases_promoted": (
                            human_only_alias_safe
                        ),
                    },
                }
            )

            records.append(record)

        return records

    def _select_official_symbol(
        self,
        gene: str,
        cellmarker_rows: pd.DataFrame,
        panglaodb_rows: pd.DataFrame,
    ) -> str | None:
        # Selects the best supported official gene symbol.
        exact_symbols: list[str | None] = []
        alias_symbols: list[str | None] = []

        for row in self._records(
            cellmarker_rows
        ):
            symbol = self._clean_text(
                row.get("Symbol")
            )

            if (
                row.get(
                    "retrieval_match_method"
                )
                == "official_symbol"
            ):
                exact_symbols.append(symbol)
            else:
                alias_symbols.append(symbol)

        for row in self._records(
            panglaodb_rows
        ):
            symbol = self._clean_text(
                row.get(
                    "official gene symbol"
                )
            )

            if (
                row.get(
                    "retrieval_match_method"
                )
                == "official_symbol"
            ):
                exact_symbols.append(symbol)
            else:
                alias_symbols.append(symbol)

        exact = self._unique_sorted(
            exact_symbols
        )

        for symbol in exact:
            if symbol.upper() == gene.upper():
                return symbol

        if exact:
            return exact[0]

        alias = self._unique_sorted(
            alias_symbols
        )

        if alias:
            return alias[0]

        return None

    def _build_gene_evidence(
        self,
        gene: str,
        cellmarker_rows: pd.DataFrame,
        panglaodb_rows: pd.DataFrame,
        matcher: ContextMatcher,
    ) -> GeneEvidence:
        # Combines source records into one gene-evidence object.
        official_symbol = (
            self._select_official_symbol(
                gene,
                cellmarker_rows,
                panglaodb_rows,
            )
        )

        cellmarker_records = (
            self._cellmarker_records(
                cellmarker_rows,
                matcher,
            )
        )

        panglaodb_records = (
            self._panglaodb_records(
                panglaodb_rows,
                matcher,
            )
        )

        records = (
            cellmarker_records
            + panglaodb_records
        )

        context_records = [
            record
            for record in records
            if record.context_match_level
            != "none"
        ]

        human_alias_candidates: list[
            str | None
        ] = []

        mixed_alias_candidates: list[
            str | None
        ] = []

        for row in self._records(
            cellmarker_rows
        ):
            human_alias_candidates.append(
                self._clean_text(
                    row.get("marker")
                )
            )

        for row in self._records(
            panglaodb_rows
        ):
            aliases = self._split_aliases(
                row.get("nicknames")
            )

            if self._safe_bool(
                row.get(
                    "human_only_alias_safe"
                )
            ):
                human_alias_candidates.extend(
                    aliases
                )
            else:
                mixed_alias_candidates.extend(
                    aliases
                )

        excluded_symbols = {
            gene.upper()
        }

        if official_symbol is not None:
            excluded_symbols.add(
                official_symbol.upper()
            )

        human_aliases = [
            alias
            for alias in self._unique_sorted(
                human_alias_candidates
            )
            if alias.upper()
            not in excluded_symbols
        ]

        mixed_aliases = [
            alias
            for alias in self._unique_sorted(
                mixed_alias_candidates
            )
            if alias.upper()
            not in excluded_symbols
        ]

        quality_warnings: list[str] = []

        if records and not context_records:
            quality_warnings.append(
                "No evidence record matched the requested "
                "tissue context; global evidence is "
                "preserved for audit."
            )

        mixed_species_count = sum(
            1
            for row in self._records(
                panglaodb_rows
            )
            if not self._safe_bool(
                row.get(
                    "human_only_alias_safe"
                )
            )
        )

        if mixed_species_count:
            quality_warnings.append(
                f"{mixed_species_count} PanglaoDB "
                "mixed-species record(s) supplied marker "
                "evidence; their nicknames were not "
                "promoted as human aliases."
            )

        descriptions = self._unique_sorted(
            record.description
            for record in records
            if record.source == "PanglaoDB"
        )

        if len(descriptions) > 1:
            quality_warnings.append(
                "Multiple PanglaoDB product descriptions "
                "were returned; all validated source "
                "descriptions are preserved."
            )

        return GeneEvidence.model_validate(
            {
                "gene": gene,
                "official_symbol": official_symbol,
                "aliases": self._unique_sorted([*human_aliases, *mixed_aliases]),
                "human_only_aliases": human_aliases,
                "mixed_species_aliases": mixed_aliases,
                "known_marker_cell_types": (
                    self._unique_sorted(
                        record.normalized_cell_type
                        for record in records
                    )
                ),
                "raw_marker_cell_types": (
                    self._unique_sorted(
                        record.cell_type
                        for record in records
                    )
                ),
                "context_matched_cell_types": (
                    self._unique_sorted(
                        record.normalized_cell_type
                        for record in context_records
                    )
                ),
                "tissue_specificity": (
                    self._unique_sorted(
                        record.tissue
                        for record in records
                    )
                ),
                "context_matched_tissues": (
                    self._unique_sorted(
                        record.tissue
                        for record in context_records
                    )
                ),
                "disease_contexts": (
                    self._unique_sorted(
                        record.disease_context
                        for record in records
                    )
                ),
                "context_matched_disease_contexts": (
                    self._unique_sorted(
                        record.disease_context
                        for record in context_records
                    )
                ),
                "cell_ontology_ids": (
                    self._unique_sorted(
                        record.cell_ontology_id
                        for record in records
                    )
                ),
                "context_matched_cell_ontology_ids": (
                    self._unique_sorted(
                        record.cell_ontology_id
                        for record in context_records
                    )
                ),
                "descriptions": descriptions,
                "evidence_sources": (
                    self._unique_sorted(
                        record.source
                        for record in records
                    )
                ),
                "database_evidence": records,
                "context_matched_database_evidence": (
                    context_records
                ),
                "atlas_evidence": [],
                "quality_warnings": quality_warnings,
            }
        )

    def _aggregate_cell_types(
        self,
        gene_evidence: list[GeneEvidence],
        context_only: bool,
    ) -> list[SupportingCellTypeEvidence]:
        # Aggregates gene evidence by normalized cell type.
        support: dict[
            str,
            _CellTypeAccumulator,
        ] = {}

        for gene_record in gene_evidence:
            records = (
                gene_record
                .context_matched_database_evidence
                if context_only
                else gene_record.database_evidence
            )

            for record in records:
                cell_type = (
                    record.normalized_cell_type
                )

                if cell_type is None:
                    continue

                accumulator = support.setdefault(
                    cell_type,
                    _CellTypeAccumulator(),
                )

                if record.cell_type is not None:
                    accumulator.raw_labels.add(
                        record.cell_type
                    )

                accumulator.genes.add(
                    gene_record.gene
                )

                accumulator.sources.add(
                    record.source
                )

                if record.tissue is not None:
                    accumulator.tissues.add(
                        record.tissue
                    )

                if (
                    record.disease_context
                    is not None
                ):
                    accumulator.diseases.add(
                        record.disease_context
                    )

                if (
                    record.cell_ontology_id
                    is not None
                ):
                    accumulator.ontology_ids.add(
                        record.cell_ontology_id
                    )

                accumulator.evidence_count += 1

        result: list[
            SupportingCellTypeEvidence
        ] = []

        for cell_type, accumulator in support.items():
            result.append(
                SupportingCellTypeEvidence.model_validate(
                    {
                        "cell_type": cell_type,
                        "raw_cell_type_labels": sorted(
                            accumulator.raw_labels,
                            key=str.casefold,
                        ),
                        "supporting_genes": sorted(
                            accumulator.genes,
                            key=str.casefold,
                        ),
                        "evidence_sources": sorted(
                            accumulator.sources,
                            key=str.casefold,
                        ),
                        "tissue_contexts": sorted(
                            accumulator.tissues,
                            key=str.casefold,
                        ),
                        "disease_contexts": sorted(
                            accumulator.diseases,
                            key=str.casefold,
                        ),
                        "cell_ontology_ids": sorted(
                            accumulator.ontology_ids,
                            key=str.casefold,
                        ),
                        "evidence_count": (
                            accumulator.evidence_count
                        ),
                    }
                )
            )

        return sorted(
            result,
            key=lambda item: (
                -len(item.supporting_genes),
                -item.evidence_count,
                item.cell_type.casefold(),
            ),
        )

    @staticmethod
    def _build_source_coverage(
        markers: pd.DataFrame,
    ) -> SourceCoverage:
        # Calculates gene-matching coverage across sources.
        total = len(markers)

        matched_either = (
            markers["matched_either"]
            .fillna(False)
            .astype(bool)
        )

        matched_alias = (
            markers["matched_alias"]
            .fillna(False)
            .astype(bool)
        )

        matched_final = (
            markers["matched_final"]
            .fillna(False)
            .astype(bool)
        )

        matched_cellmarker = (
            markers[
                "matched_cellmarker_final"
            ]
            .fillna(False)
            .astype(bool)
        )

        matched_panglaodb = (
            markers[
                "matched_panglaodb_final"
            ]
            .fillna(False)
            .astype(bool)
        )

        official_count = len(
            markers.loc[matched_either]
        )

        alias_count = len(
            markers.loc[
                (~matched_either)
                & matched_alias
            ]
        )

        final_count = len(
            markers.loc[matched_final]
        )

        cellmarker_count = len(
            markers.loc[
                matched_cellmarker
            ]
        )

        panglaodb_count = len(
            markers.loc[
                matched_panglaodb
            ]
        )

        both_count = len(
            markers.loc[
                matched_cellmarker
                & matched_panglaodb
            ]
        )

        coverage_percent = (
            round(
                final_count
                / total
                * 100,
                2,
            )
            if total
            else 0.0
        )

        return SourceCoverage.model_validate(
            {
                "total_input_genes": total,
                "matched_official_symbol": (
                    official_count
                ),
                "recovered_through_alias": (
                    alias_count
                ),
                "final_matched_genes": (
                    final_count
                ),
                "unresolved_genes": (
                    total
                    - final_count
                ),
                "coverage_percent": (
                    coverage_percent
                ),
                "matched_cellmarker": (
                    cellmarker_count
                ),
                "matched_panglaodb": (
                    panglaodb_count
                ),
                "matched_both": both_count,
            }
        )
#Per cluster evidence builder for the Biological Knowledge Retrieval Agent.

    def build_cluster(
        self,
        cluster_id: int | str,
        marker_status: pd.DataFrame,
        cellmarker_evidence: pd.DataFrame,
        panglaodb_evidence: pd.DataFrame,
        species: str,
        tissue: str,
        platform: str,
        context_matcher: ContextMatcher,
        deg_validation: DEGValidationSummary,
        data_quality_warnings: list[str],
    ) -> KnowledgeEvidence:
        # Builds the complete KnowledgeEvidence for one cluster.
        markers = self._filter_cluster(
            marker_status,
            cluster_id,
        )

        markers = markers.sort_values(
            "rank_within_cluster"
        ).copy()

        if markers.empty:
            raise ValueError(
                f"No marker genes found for "
                f"cluster {cluster_id}."
            )

        cellmarker = self._filter_cluster(
            cellmarker_evidence,
            cluster_id,
        )

        panglaodb = self._filter_cluster(
            panglaodb_evidence,
            cluster_id,
        )

        input_genes = self._build_input_genes(
            markers
        )

        gene_evidence: list[
            GeneEvidence
        ] = []

        for row in self._records(markers):
            gene = str(
                row["names"]
            ).strip()

            gene_key = str(
                row["gene_key"]
            )

            if cellmarker.empty:
                gene_cellmarker = (
                    cellmarker.copy()
                )
            else:
                cellmarker_mask = (
                    cellmarker[
                        "input_gene_key"
                    ].astype(str)
                    == gene_key
                )

                gene_cellmarker = (
                    cellmarker.loc[
                        cellmarker_mask
                    ].copy()
                )

            if panglaodb.empty:
                gene_panglaodb = (
                    panglaodb.copy()
                )
            else:
                panglaodb_mask = (
                    panglaodb[
                        "input_gene_key"
                    ].astype(str)
                    == gene_key
                )

                gene_panglaodb = (
                    panglaodb.loc[
                        panglaodb_mask
                    ].copy()
                )

            gene_evidence.append(
                self._build_gene_evidence(
                    gene,
                    gene_cellmarker,
                    gene_panglaodb,
                    context_matcher,
                )
            )

        input_names = [
            record.gene
            for record in input_genes
        ]

        evidence_names = [
            record.gene
            for record in gene_evidence
        ]

        if input_names != evidence_names:
            raise ValueError(
                "Every input gene must produce one "
                "ordered GeneEvidence record."
            )

        unresolved_mask = ~(
            markers["matched_final"]
            .fillna(False)
            .astype(bool)
        )

        unresolved_genes = (
            markers.loc[
                unresolved_mask
            ]
            .sort_values(
                "rank_within_cluster"
            )["names"]
            .astype(str)
            .tolist()
        )

        context_support = (
            self._aggregate_cell_types(
                gene_evidence,
                context_only=True,
            )
        )

        global_support = (
            self._aggregate_cell_types(
                gene_evidence,
                context_only=False,
            )
        )

        final_warnings = list(
            data_quality_warnings
        )

        if not context_support:
            final_warnings.append(
                "No context-matched cell-type evidence "
                "was available; supporting_cell_types "
                "is empty and "
                "global_supporting_cell_types remains "
                "available."
            )

        context_definition = {
            "requested_tissue": (
                context_matcher.requested_tissue
            ),
            "tissue_terms": sorted(
                set(
                    context_matcher.tissue_terms
                ),
                key=str.casefold,
            ),
            "disease_terms": sorted(
                set(
                    context_matcher.disease_terms
                ),
                key=str.casefold,
            ),
        }

        return KnowledgeEvidence.model_validate(
            {
                "cluster_id": cluster_id,
                "species": species,
                "tissue": tissue,
                "platform": platform,
                "context_definition": (
                    context_definition
                ),
                "input_genes": input_genes,
                "gene_evidence": (
                    gene_evidence
                ),
                "supporting_cell_types": (
                    context_support
                ),
                "global_supporting_cell_types": (
                    global_support
                ),
                "source_coverage": (
                    self._build_source_coverage(
                        markers
                    )
                ),
                "deg_validation": (
                    deg_validation
                ),
                "unresolved_genes": (
                    unresolved_genes
                ),
                "data_quality_warnings": sorted(
                    set(final_warnings),
                    key=str.casefold,
                ),
            }
        )





