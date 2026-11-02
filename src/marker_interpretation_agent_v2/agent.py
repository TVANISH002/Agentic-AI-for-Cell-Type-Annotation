
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from marker_interpretation_agent_v2.config import Agent2Config, DEFAULT_CONFIG
from marker_interpretation_agent_v2.interpretation.candidate_builder import (
    CandidateBuilder,
)
from marker_interpretation_agent_v2.interpretation.candidate_selector import (
    CandidateSelector,
)
from marker_interpretation_agent_v2.interpretation.evidence_classifier import (
    EvidenceClassifier,
)
from marker_interpretation_agent_v2.interpretation.explanation_builder import (
    ExplanationBuilder,
)
from marker_interpretation_agent_v2.interpretation.marker_ranker import (
    MarkerRanker,
)
from marker_interpretation_agent_v2.interpretation.review_flag_builder import (
    ReviewFlagBuilder,
)
from marker_interpretation_agent_v2.reporting.report_writer import (
    Agent2ReportWriter,
)
from marker_interpretation_agent_v2.schemas import (
    Agent2Summary,
    CandidateCellType,
    FollowUpGene,
    GeneInterpretation,
    InputGene,
    MarkerEvidence,
    PanelEvidenceCatalog,
    RankedMarkerGene,
)
from marker_interpretation_agent_v2.unresolved.follow_up_queue_builder import (
    FollowUpQueueBuilder,
)


class MarkerInterpretationAgent:
    """Orchestrate deterministic cluster-level marker interpretation."""

    def __init__(
        self,
        config: Agent2Config | None = None,
        *,
        classifier: EvidenceClassifier | None = None,
        marker_ranker: MarkerRanker | None = None,
        candidate_builder: CandidateBuilder | None = None,
        candidate_selector: CandidateSelector | None = None,
        review_flag_builder: ReviewFlagBuilder | None = None,
        explanation_builder: ExplanationBuilder | None = None,
        follow_up_queue_builder: FollowUpQueueBuilder | None = None,
        report_writer: Agent2ReportWriter | None = None,
    ) -> None:
        """Initialize Agent 2 and all deterministic components."""

        self.config = config or DEFAULT_CONFIG
        self.classifier = classifier or EvidenceClassifier(self.config)
        self.marker_ranker = marker_ranker or MarkerRanker()
        self.candidate_builder = candidate_builder or CandidateBuilder()
        self.candidate_selector = (
            candidate_selector
            or CandidateSelector(self.config)
        )
        self.review_flag_builder = (
            review_flag_builder
            or ReviewFlagBuilder(self.config)
        )
        self.explanation_builder = (
            explanation_builder
            or ExplanationBuilder()
        )
        self.follow_up_queue_builder = (
            follow_up_queue_builder
            or FollowUpQueueBuilder()
        )
        self.report_writer = (
            report_writer
            or Agent2ReportWriter()
        )

    @staticmethod
    def _normalize_cluster_id(
        cluster_id: str | int,
    ) -> str:
        """Normalize one cluster identifier."""

        normalized = str(cluster_id).strip()

        if not normalized:
            raise ValueError(
                "cluster_id cannot be empty."
            )

        return normalized

    @staticmethod
    def _normalize_gene(
        value: str,
    ) -> str:
        """Normalize one gene symbol."""

        gene = str(value).strip().upper()

        if not gene:
            raise ValueError(
                "Gene symbol cannot be empty."
            )

        return gene

    @classmethod
    def _prepare_input_genes(
        cls,
        input_genes: Sequence[InputGene],
        *,
        top_k: int,
    ) -> list[InputGene]:
        """Validate, order, and limit cluster marker inputs."""

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        prepared = [
            InputGene.model_validate(
                input_gene.model_dump()
            )
            for input_gene in input_genes
        ]

        genes = [
            cls._normalize_gene(
                input_gene.gene
            )
            for input_gene in prepared
        ]

        duplicate_genes = sorted(
            {
                gene
                for gene in genes
                if genes.count(gene) > 1
            }
        )

        if duplicate_genes:
            raise ValueError(
                "Cluster input contains duplicate genes: "
                f"{duplicate_genes}"
            )

        ranks = [
            input_gene.rank_within_cluster
            for input_gene in prepared
        ]

        duplicate_ranks = sorted(
            {
                rank
                for rank in ranks
                if ranks.count(rank) > 1
            }
        )

        if duplicate_ranks:
            raise ValueError(
                "Cluster input contains duplicate ranks: "
                f"{duplicate_ranks}"
            )

        ordered = sorted(
            prepared,
            key=lambda input_gene: (
                input_gene.rank_within_cluster,
                cls._normalize_gene(
                    input_gene.gene
                ),
            ),
        )

        return ordered[:top_k]

    def _validate_catalog_context(
        self,
        catalog: PanelEvidenceCatalog,
    ) -> None:
        """Require catalog and Agent 2 context to match."""

        expected = {
            "species": self.config.species,
            "tissue": self.config.tissue,
            "platform": self.config.platform,
        }

        actual = {
            "species": catalog.species,
            "tissue": catalog.tissue,
            "platform": catalog.platform,
        }

        mismatches = [
            (
                f"{field}: expected "
                f"{expected[field]!r}, received "
                f"{actual[field]!r}"
            )
            for field in expected
            if (
                str(expected[field])
                .strip()
                .casefold()
                != str(actual[field])
                .strip()
                .casefold()
            )
        ]

        if mismatches:
            raise ValueError(
                "Panel catalog context does not "
                "match Agent 2 configuration: "
                + "; ".join(mismatches)
            )

    @staticmethod
    def _build_summary(
        input_genes: Sequence[InputGene],
        interpretations: Sequence[
            GeneInterpretation
        ],
        ranked_markers: Sequence[
            RankedMarkerGene
        ],
        ranked_candidates: Sequence[
            CandidateCellType
        ],
        selected_candidates: Sequence[
            CandidateCellType
        ],
        follow_up_queue: Sequence[
            FollowUpGene
        ],
        *,
        human_review_required: bool,
    ) -> Agent2Summary:
        """Build the cluster-level numeric summary."""

        return Agent2Summary(
            total_input_genes=len(
                input_genes
            ),
            direct_marker_supported_genes=sum(
                1
                for interpretation
                in interpretations
                if (
                    interpretation
                    .direct_marker_supported
                )
            ),
            hpa_queried_genes=sum(
                1
                for interpretation
                in interpretations
                if interpretation.hpa.queried
            ),
            hpa_gene_matches=sum(
                1
                for interpretation
                in interpretations
                if (
                    interpretation.hpa
                    .gene_present_in_hpa
                )
            ),
            hpa_breast_expression_genes=sum(
                1
                for interpretation
                in interpretations
                if (
                    interpretation.hpa
                    .breast_expression_found
                )
            ),
            ranked_marker_gene_count=len(
                ranked_markers
            ),
            ranked_candidate_count=len(
                ranked_candidates
            ),
            selected_candidate_count=len(
                selected_candidates
            ),
            follow_up_gene_count=len(
                follow_up_queue
            ),
            leading_candidate=(
                selected_candidates[0].cell_type
                if selected_candidates
                else None
            ),
            human_review_required=(
                human_review_required
            ),
        )

    def interpret_cluster(
        self,
        *,
        cluster_id: str | int,
        input_genes: Sequence[InputGene],
        catalog: PanelEvidenceCatalog,
    ) -> MarkerEvidence:
        """Interpret one cluster using the reusable panel catalog."""

        normalized_cluster_id = (
            self._normalize_cluster_id(
                cluster_id
            )
        )

        self._validate_catalog_context(
            catalog
        )

        prepared_input_genes = (
            self._prepare_input_genes(
                input_genes,
                top_k=self.config.top_k,
            )
        )

        interpretations = (
            self.classifier.classify(
                prepared_input_genes,
                catalog,
            )
        )

        ranked_markers = (
            self.marker_ranker.rank(
                interpretations
            )
        )

        ranked_candidates = (
            self.candidate_builder.build(
                interpretations
            )
        )

        (
            selected_candidates,
            candidate_selection,
        ) = self.candidate_selector.select(
            ranked_candidates
        )

        review_flags = (
            self.review_flag_builder.build(
                interpretations,
                ranked_candidates,
                selected_candidates,
                candidate_selection,
            )
        )

        follow_up_queue = (
            self.follow_up_queue_builder.build(
                interpretations
            )
        )

        human_review_required = (
            self.review_flag_builder
            .human_review_required(
                review_flags
            )
        )

        summary = self._build_summary(
            prepared_input_genes,
            interpretations,
            ranked_markers,
            ranked_candidates,
            selected_candidates,
            follow_up_queue,
            human_review_required=(
                human_review_required
            ),
        )

        explanation = (
            self.explanation_builder.build(
                cluster_id=(
                    normalized_cluster_id
                ),
                interpretations=(
                    interpretations
                ),
                ranked_candidates=(
                    ranked_candidates
                ),
                selected_candidates=(
                    selected_candidates
                ),
                candidate_selection=(
                    candidate_selection
                ),
                review_flags=(
                    review_flags
                ),
            )
        )

        return MarkerEvidence.model_validate(
            {
                "cluster_id": (
                    normalized_cluster_id
                ),
                "species": catalog.species,
                "tissue": catalog.tissue,
                "platform": catalog.platform,
                "input_genes": (
                    prepared_input_genes
                ),
                "gene_interpretations": (
                    interpretations
                ),
                "ranked_marker_genes": (
                    ranked_markers
                ),
                "ranked_candidate_cell_types": (
                    ranked_candidates
                ),
                "selected_candidate_cell_types": (
                    selected_candidates
                ),
                "candidate_selection": (
                    candidate_selection
                ),
                "follow_up_gene_queue": (
                    follow_up_queue
                ),
                "review_flags": (
                    review_flags
                ),
                "summary": summary,
                "explanation": explanation,
                "config": self.config.to_dict(),
            }
        )

    def write_reports(
        self,
        result: MarkerEvidence,
        *,
        output_dir: str | Path | None = None,
    ) -> dict[str, Path]:
        """Write one validated cluster result."""

        return self.report_writer.write(
            result,
            output_dir=output_dir,
        )

    def run(
        self,
        *,
        cluster_id: str | int,
        input_genes: Sequence[InputGene],
        catalog: PanelEvidenceCatalog,
        output_dir: str | Path | None = None,
        write_reports: bool = True,
    ) -> tuple[
        MarkerEvidence,
        dict[str, Path],
    ]:
        """Interpret one cluster and optionally write reports."""

        result = self.interpret_cluster(
            cluster_id=cluster_id,
            input_genes=input_genes,
            catalog=catalog,
        )

        paths = (
            self.write_reports(
                result,
                output_dir=output_dir,
            )
            if write_reports
            else {}
        )

        return result, paths