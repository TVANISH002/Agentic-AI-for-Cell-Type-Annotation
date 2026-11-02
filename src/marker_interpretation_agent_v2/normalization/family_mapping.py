"""Conservative multi-level cell-type normalization for Agent 2 V2.

The mapping preserves raw database labels and creates three interpretation
levels:

1. Canonical label:
   Combines only genuine spelling variants, formatting differences, or
   closely equivalent names.

2. Subtype group:
   Groups closely related biological subtypes while preserving meaningful
   differences such as fibroblast versus mesenchymal cell.

3. Broad lineage:
   Provides a high-level review category only. It must not automatically
   replace the raw or canonical candidate ranking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


MAPPING_SCHEMA_VERSION = "2.0.0"


@dataclass(frozen=True)
class CellTypeMapping:
    """Normalized representation of one raw cell-type label."""

    raw_cell_type: str
    cleaned_cell_type: str
    canonical_cell_type: str
    subtype_group: str
    broad_lineage_family: str
    mapping_rule: str

    @property
    def cell_type_family(self) -> str:
        """Backward-compatible alias used by the current aggregator.

        This property temporarily returns the subtype group rather than the
        broad lineage so existing code becomes more conservative until the
        aggregator is upgraded to store both levels explicitly.
        """

        return self.subtype_group


# lookup key:
# (
#     canonical_cell_type,
#     subtype_group,
#     broad_lineage_family,
# )
EXACT_CELL_TYPE_MAPPINGS: dict[
    str,
    tuple[str, str, str],
] = {
    # ------------------------------------------------------------------
    # General epithelial populations
    # ------------------------------------------------------------------
    "epithelial cell": (
        "Epithelial cell",
        "Epithelial cell",
        "Epithelial",
    ),
    "mammary epithelial cell": (
        "Mammary epithelial cell",
        "Mammary epithelial cell",
        "Epithelial",
    ),

    
    # Basal and myoepithelial populations
   
    "basal cell": (
        "Basal epithelial cell",
        "Basal/Myoepithelial epithelial",
        "Epithelial",
    ),
    "basal epithelial cell": (
        "Basal epithelial cell",
        "Basal/Myoepithelial epithelial",
        "Epithelial",
    ),
    "myo-epithelial cell": (
        "Myoepithelial cell",
        "Basal/Myoepithelial epithelial",
        "Epithelial",
    ),
    "myoepithelial cell": (
        "Myoepithelial cell",
        "Basal/Myoepithelial epithelial",
        "Epithelial",
    ),
    "basal progenitor cell": (
        "Basal progenitor cell",
        "Basal progenitor cell",
        "Epithelial",
    ),

    # ------------------------------------------------------------------
    # Luminal epithelial populations
    # ------------------------------------------------------------------
    "luminal epithelial cell": (
        "Luminal epithelial cell",
        "Luminal epithelial cell",
        "Epithelial",
    ),
    "mature luminal cell": (
        "Mature luminal cell",
        "Luminal epithelial cell",
        "Epithelial",
    ),
    "luminal progenitor cell": (
        "Luminal progenitor cell",
        "Luminal progenitor cell",
        "Epithelial",
    ),

    # ------------------------------------------------------------------
    # Cytotoxic and CD8 T-cell populations
    # ------------------------------------------------------------------
    "cytotoxic t cell": (
        "Cytotoxic T cell",
        "CD8/Cytotoxic T cell",
        "T cell",
    ),
    "cytotoxic cd8+ t cell": (
        "Cytotoxic CD8 T cell",
        "CD8/Cytotoxic T cell",
        "T cell",
    ),
    "cd8+ t cell": (
        "CD8 T cell",
        "CD8 T cell",
        "T cell",
    ),
    "effector cd8 t cell": (
        "Effector CD8 T cell",
        "Effector CD8 T cell",
        "T cell",
    ),
    "exhausted cd8+ t cell": (
        "Exhausted CD8 T cell",
        "Exhausted CD8 T cell",
        "T cell",
    ),
    "memory cd8+ t cell": (
        "Memory CD8 T cell",
        "Memory CD8 T cell",
        "T cell",
    ),

    # ------------------------------------------------------------------
    # Other T-cell populations
    # ------------------------------------------------------------------
    "t cell": (
        "T cell",
        "T cell",
        "T cell",
    ),
    "tumor-infiltrating t cell": (
        "Tumor-infiltrating T cell",
        "Tumor-infiltrating T cell",
        "T cell",
    ),
    "regulatory t cell": (
        "Regulatory T cell",
        "Regulatory T cell",
        "T cell",
    ),
    "cd4+ t cell": (
        "CD4 T cell",
        "CD4 T cell",
        "T cell",
    ),
    "naive cd4 t cell": (
        "Naive CD4 T cell",
        "Naive CD4 T cell",
        "T cell",
    ),
    "t helper(th) cell": (
        "T helper cell",
        "CD4 T cell",
        "T cell",
    ),
    "t helper 1(th1) cell": (
        "T helper 1 cell",
        "CD4 T cell",
        "T cell",
    ),
    "conventional t(tconv) cell": (
        "Conventional T cell",
        "CD4 T cell",
        "T cell",
    ),

    # ------------------------------------------------------------------
    # Natural killer populations
    # ------------------------------------------------------------------
    "natural killer cell": (
        "Natural killer cell",
        "Natural killer cell",
        "Natural killer cell",
    ),
    "natural killer t(nkt) cell": (
        "Natural killer T cell",
        "Natural killer T cell",
        "Natural killer/T cell",
    ),
    "cytokine induced killer cell": (
        "Cytokine-induced killer cell",
        "Cytokine-induced killer cell",
        "Natural killer/T cell",
    ),

    # ------------------------------------------------------------------
    # B-cell and plasma-cell populations
    # ------------------------------------------------------------------
    "b cell": (
        "B cell",
        "B cell",
        "B cell",
    ),
    "plasma cell": (
        "Plasma cell",
        "Plasma cell",
        "B cell",
    ),

    # ------------------------------------------------------------------
    # Macrophage and monocyte populations
    # ------------------------------------------------------------------
    "macrophage": (
        "Macrophage",
        "Macrophage",
        "Myeloid",
    ),
    "pan-macrophage": (
        "Macrophage",
        "Macrophage",
        "Myeloid",
    ),
    "m1 macrophage": (
        "M1 macrophage",
        "Macrophage",
        "Myeloid",
    ),
    "tumor-associated macrophage (tam)": (
        "Tumor-associated macrophage",
        "Macrophage",
        "Myeloid",
    ),
    "monocyte": (
        "Monocyte",
        "Monocyte",
        "Myeloid",
    ),
    "myeloid cell": (
        "Myeloid cell",
        "General myeloid cell",
        "Myeloid",
    ),
    "mast cell": (
        "Mast cell",
        "Mast cell",
        "Myeloid",
    ),
    "plasmacytoid dendritic cell(pdc)": (
        "Plasmacytoid dendritic cell",
        "Plasmacytoid dendritic cell",
        "Dendritic cell",
    ),

    # ------------------------------------------------------------------
    # Fibroblast and stromal populations
    #
    # These labels share a broad lineage but remain separate subtype groups.
    # This prevents all stromal labels from automatically becoming one vote.
    # ------------------------------------------------------------------
    "fibroblast": (
        "Fibroblast",
        "Fibroblast",
        "Stromal",
    ),
    "cancer-associated fibroblast": (
        "Cancer-associated fibroblast",
        "Cancer-associated fibroblast",
        "Stromal",
    ),
    "antigen presentation cancer-associated fibroblast": (
        "Antigen-presenting cancer-associated fibroblast",
        "Cancer-associated fibroblast",
        "Stromal",
    ),
    "myofibroblast": (
        "Myofibroblast",
        "Myofibroblast",
        "Stromal",
    ),
    "stromal cell": (
        "Stromal cell",
        "General stromal cell",
        "Stromal",
    ),
    "mesenchymal cell": (
        "Mesenchymal cell",
        "Mesenchymal cell",
        "Stromal",
    ),

    # ------------------------------------------------------------------
    # Perivascular and smooth-muscle populations
    # ------------------------------------------------------------------
    "perivascular cell": (
        "Perivascular cell",
        "Perivascular/Pericyte",
        "Perivascular",
    ),
    "perivascular-like cell": (
        "Perivascular cell",
        "Perivascular/Pericyte",
        "Perivascular",
    ),
    "pericyte": (
        "Pericyte",
        "Perivascular/Pericyte",
        "Perivascular",
    ),
    "muscle cell": (
        "Muscle cell",
        "Muscle cell",
        "Muscle",
    ),
    "smooth muscle cell": (
        "Smooth muscle cell",
        "Smooth muscle cell",
        "Muscle",
    ),

    # ------------------------------------------------------------------
    # Endothelial populations
    # ------------------------------------------------------------------
    "endothelial cell": (
        "Endothelial cell",
        "Endothelial cell",
        "Endothelial",
    ),

    # ------------------------------------------------------------------
    # Stem and progenitor populations
    # ------------------------------------------------------------------
    "stem cell": (
        "Stem cell",
        "Stem cell",
        "Stem/Progenitor",
    ),
    "hematopoietic stem cell": (
        "Hematopoietic stem cell",
        "Hematopoietic stem cell",
        "Stem/Progenitor",
    ),
    "mammary stem cell": (
        "Mammary stem cell",
        "Mammary stem cell",
        "Stem/Progenitor",
    ),

    # ------------------------------------------------------------------
    # Cancer-associated populations
    # ------------------------------------------------------------------
    "cancer cell": (
        "Cancer cell",
        "Cancer cell",
        "Tumor",
    ),
    "cancer stem cell": (
        "Cancer stem cell",
        "Cancer stem cell",
        "Tumor/Stem-like",
    ),
    "epithelial-mesenchymal transition cancer stem cell": (
        "EMT cancer stem cell",
        "EMT cancer stem cell",
        "Tumor/Stem-like",
    ),
}


def clean_cell_type_label(cell_type: str) -> str:
    """Repair common source encoding and formatting problems."""

    value = str(cell_type or "").strip()

    replacements = {
        "ΓÇÉ": "-",
        "ΓÇô": "-",
        "ΓÇö": "-",
        "ΓÇô": "-",
        "–": "-",
        "—": "-",
        "‐": "-",
        "-": "-",
        "−": "-",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
    }

    for source, target in replacements.items():
        value = value.replace(source, target)

    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s*-\s*", "-", value)

    return value.strip()


def normalize_lookup_key(cell_type: str) -> str:
    """Create a stable case-insensitive key for exact mapping."""

    return clean_cell_type_label(cell_type).casefold()


def map_cell_type(cell_type: str) -> CellTypeMapping:
    """Map one source label without guessing unknown biological meanings."""

    raw_value = str(cell_type or "")
    cleaned_value = clean_cell_type_label(raw_value)
    lookup_key = normalize_lookup_key(cleaned_value)

    mapped = EXACT_CELL_TYPE_MAPPINGS.get(lookup_key)

    if mapped is None:
        canonical_cell_type = cleaned_value
        subtype_group = cleaned_value
        broad_lineage_family = cleaned_value
        mapping_rule = "preserved_unmapped_label"
    else:
        (
            canonical_cell_type,
            subtype_group,
            broad_lineage_family,
        ) = mapped

        mapping_rule = "curated_exact_mapping"

    return CellTypeMapping(
        raw_cell_type=raw_value,
        cleaned_cell_type=cleaned_value,
        canonical_cell_type=canonical_cell_type,
        subtype_group=subtype_group,
        broad_lineage_family=broad_lineage_family,
        mapping_rule=mapping_rule,
    )