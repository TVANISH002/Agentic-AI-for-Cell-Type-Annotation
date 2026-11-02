from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class HPAEvidenceConfig:
    """Configuration for the independent HPA evidence workflow."""

    # Canonical cleaned list of all 241 panel genes.
    panel_gene_list_path: Path = Path(
        "data/agent2_v2/panel/panel_gene_list.csv"
    )

    # Original cluster marker file used to map HPA evidence
    cluster_marker_path: Path = Path(
         "data/raw/"
         "cluster_marker_genes_brcr_Agentic_AI.csv"
        #"data/cluster_marker_genes_brcr_Agentic_AI.csv" 
    )

    # Local Human Protein Atlas single-cell RNA dataset.
    hpa_source_path: Path = Path(
           "data/reference/hpa/"
           "rna_single_cell_cluster_hpa_v25_1.tsv.zip"
       # "data/reference/hpa/rna_single_cell_cluster.tsv.zip" 
       # "data\reference/hpa/rna_single_cell_cluster_hpa_v25_1.tsv.zip"
        
    )

    # Independent output directory for this agent.
    output_dir: Path = Path(
        "data/hpa_evidence_agent"
    )

    # HPA biological context.
    query_tissue: str = "breast"
    source_name: str = "Human Protein Atlas"
    source_release: str = "25.1"
    evidence_type: str = "single_cell_rna_expression"
    evidence_scope: str = "healthy_breast_reference"

    # Panel gene-list column.
    panel_gene_column: str = "gene"

    # Original cluster marker CSV columns.
    marker_gene_column: str = "names"
    marker_cluster_column: str = "spatial_cluster"
    marker_score_column: str = "scores"
    marker_logfc_column: str = "logfoldchanges"
    marker_padj_column: str = "pvals_adj"

    # Human Protein Atlas source columns.
    hpa_gene_column: str = "Gene name"
    hpa_ensembl_column: str = "Gene"
    hpa_tissue_column: str = "Tissue"
    hpa_cell_type_column: str = "Cell type"
    hpa_cluster_column: str = "Cluster"
    hpa_read_count_column: str = "Read count"
    hpa_ncpm_column: str = "nCPM"

    # Cluster projection and expression settings.
    top_genes_per_cluster: int = 50
    minimum_positive_ncpm: float = 0.0

    def validate(self) -> None:
        """Validate configuration values before the workflow starts."""

        path_fields = {
            "panel_gene_list_path": self.panel_gene_list_path,
            "cluster_marker_path": self.cluster_marker_path,
            "hpa_source_path": self.hpa_source_path,
            "output_dir": self.output_dir,
        }

        for field_name, value in path_fields.items():
            if not isinstance(value, Path):
                raise TypeError(
                    f"{field_name} must be a pathlib.Path."
                )

        text_fields = {
            "query_tissue": self.query_tissue,
            "source_name": self.source_name,
            "source_release": self.source_release,
            "evidence_type": self.evidence_type,
            "evidence_scope": self.evidence_scope,
            "panel_gene_column": self.panel_gene_column,
            "marker_gene_column": self.marker_gene_column,
            "marker_cluster_column": self.marker_cluster_column,
            "marker_score_column": self.marker_score_column,
            "marker_logfc_column": self.marker_logfc_column,
            "marker_padj_column": self.marker_padj_column,
            "hpa_gene_column": self.hpa_gene_column,
            "hpa_ensembl_column": self.hpa_ensembl_column,
            "hpa_tissue_column": self.hpa_tissue_column,
            "hpa_cell_type_column": self.hpa_cell_type_column,
            "hpa_cluster_column": self.hpa_cluster_column,
            "hpa_read_count_column": self.hpa_read_count_column,
            "hpa_ncpm_column": self.hpa_ncpm_column,
        }

        for field_name, value in text_fields.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{field_name} must be a non-empty string."
                )

        if (
            isinstance(self.top_genes_per_cluster, bool)
            or not isinstance(self.top_genes_per_cluster, int)
            or self.top_genes_per_cluster <= 0
        ):
            raise ValueError(
                "top_genes_per_cluster must be a positive integer."
            )

        if self.minimum_positive_ncpm < 0:
            raise ValueError(
                "minimum_positive_ncpm cannot be negative."
            )


DEFAULT_CONFIG = HPAEvidenceConfig()
DEFAULT_CONFIG.validate()