import re
from dataclasses import dataclass
from typing import Literal

ContextMatchLevel = Literal["tissue_and_disease", "tissue", "disease", "none"]


class CellTypeNormalizer:
    """Conservative label harmonization without inventing new biology."""

    EXPLICIT_MAP: dict[str, str] = {
        "epithelial cells": "Epithelial cell",
        "luminal epithelial cells": "Luminal epithelial cell",
        "mammary epithelial cells": "Mammary epithelial cell",
        "mast cells": "Mast cell",
        "ductal cells": "Ductal cell",
        "cholangiocytes": "Cholangiocyte",
        "hepatocytes": "Hepatocyte",
        "oligodendrocytes": "Oligodendrocyte",
        "sertoli cells": "Sertoli cell",
        "mesothelial cells": "Mesothelial cell",
        "satellite cells": "Satellite cell",
        "basophils": "Basophil",
        "erythroblasts": "Erythroblast",
        "plasmacytoid dendritic cells": "Plasmacytoid dendritic cell",
        "principal cells": "Principal cell",
        "natural killer t cells": "Natural killer T cell",
        "hematopoietic stem cells": "Hematopoietic stem cell",
        "embryonic stem cells": "Embryonic stem cell",
        "adipocyte progenitor cells": "Adipocyte progenitor cell",
        "airway epithelial cells": "Airway epithelial cell",
        "epiblast cells": "Epiblast cell",
        "hepatoblasts": "Hepatoblast",
        "regulatory t(treg) cell": "Regulatory T cell",
        "regulatory t (treg) cell": "Regulatory T cell",
        "t helper 2(th2) cell": "T helper 2 cell",
        "t helper2 (th2) cell": "T helper 2 cell",
    }

    @classmethod
    def normalize(cls, value: str | None) -> str | None:
        if value is None:
            return None

        text = " ".join(value.strip().split())

        if not text:
            return None

        mapped = cls.EXPLICIT_MAP.get(text.casefold())

        if mapped is not None:
            return mapped

        if text.casefold().endswith(" cells"):
            return f"{text[:-6]} cell"

        return text


@dataclass(frozen=True)
class ContextMatcher:
    requested_tissue: str
    tissue_terms: tuple[str, ...]
    disease_terms: tuple[str, ...]

    @classmethod
    def from_request(cls, tissue: str) -> "ContextMatcher":
        tissue_groups: dict[str, tuple[str, ...]] = {
            "breast": ("breast", "mammary gland", "mammary", "milk"),
            "lung": ("lung", "airway", "bronchus", "bronchiole"),
            "liver": ("liver", "hepatic", "bile duct"),
            "pancreas": (
                "pancreas",
                "pancreatic",
                "pancreatic duct",
                "pancreatic islet",
            ),
            "kidney": (
                "kidney",
                "renal",
                "urine",
                "embryonic kidney",
                "fetal kidney",
            ),
            "colon": (
                "colon",
                "colorectum",
                "large intestine",
                "intestine",
            ),
            "prostate": ("prostate",),
            "ovary": ("ovary", "ovarian cortex"),
            "brain": ("brain", "central nervous system"),
            "skin": ("skin", "epidermis"),
        }

        disease_groups: dict[str, tuple[str, ...]] = {
            "breast": (
                "breast cancer",
                "metastatic breast cancer",
                "triple-negative breast cancer",
                "ductal carcinoma in situ",
                "breast cancer (circulating)",
            ),
            "lung": (
                "lung cancer",
                "lung adenocarcinoma",
                "non-small cell lung cancer",
            ),
            "liver": (
                "hepatocellular cancer",
                "hepatocellular carcinoma",
                "cholangiocarcinoma",
            ),
            "pancreas": (
                "pancreatic cancer",
                "pancreatic ductal adenocarcinoma",
            ),
            "kidney": (
                "kidney cancer",
                "renal cancer",
            ),
            "colon": (
                "colon cancer",
                "colorectal cancer",
            ),
            "prostate": (
                "prostate cancer",
                "prostate cancer (circulating)",
            ),
            "ovary": (
                "ovarian cancer",
                "serous ovarian cancer",
            ),
        }

        clean = " ".join(tissue.strip().split())
        lowered = clean.casefold()

        matched_group = next(
            (group for group in tissue_groups if group in lowered),
            None,
        )

        if matched_group is not None:
            tissue_terms = tissue_groups[matched_group]
            disease_terms = disease_groups.get(
                matched_group,
                (lowered,),
            )
        else:
            base = re.sub(
                r"\b(cancer|carcinoma|tumou?r|disease|normal)\b",
                "",
                lowered,
            )
            base = " ".join(base.split())

            tissue_terms = tuple(
                dict.fromkeys(
                    term
                    for term in (base, lowered)
                    if term
                )
            )
            disease_terms = (lowered,)

        return cls(
            requested_tissue=clean,
            tissue_terms=tuple(tissue_terms),
            disease_terms=tuple(disease_terms),
        )

    @staticmethod
    def _contains(
        value: str | None,
        terms: tuple[str, ...],
    ) -> bool:
        if value is None:
            return False

        normalized = " ".join(
            value.strip().casefold().split()
        )

        return any(term in normalized for term in terms)

    def classify(
        self,
        tissue: str | None,
        disease: str | None,
    ) -> ContextMatchLevel:
        tissue_match = self._contains(
            tissue,
            self.tissue_terms,
        )

        disease_match = self._contains(
            disease,
            self.disease_terms,
        )

        if tissue_match and disease_match:
            return "tissue_and_disease"

        if tissue_match:
            return "tissue"

        if disease_match:
            return "disease"

        return "none"

    def to_definition_data(
        self,
    ) -> dict[str, str | list[str]]:
        return {
            "requested_tissue": self.requested_tissue,
            "tissue_terms": sorted(
                set(self.tissue_terms),
                key=str.casefold,
            ),
            "disease_terms": sorted(
                set(self.disease_terms),
                key=str.casefold,
            ),
        }