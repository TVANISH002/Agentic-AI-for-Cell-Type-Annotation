from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def _clean(value: Any) -> str:
    return " ".join(
        str(value or "").split()
    )


def _unique(values):
    seen = set()
    output = []

    for value in values:
        value = _clean(value)

        if not value:
            continue

        key = value.casefold()

        if key in seen:
            continue

        seen.add(key)
        output.append(value)

    return tuple(output)


#
# These mappings normalize common species names.
#
# They are NOT tissue- or cluster-specific.
#
_SPECIES = {
    "human": {
        "canonical": "Human",
        "ncbi": "Homo sapiens",
        "terms": (
            "human",
            "humans",
            "Homo sapiens",
        ),
        "subject_terms": (
            "patient",
            "patients",
        ),
    },

    "homo sapiens": {
        "canonical": "Human",
        "ncbi": "Homo sapiens",
        "terms": (
            "human",
            "humans",
            "Homo sapiens",
        ),
        "subject_terms": (
            "patient",
            "patients",
        ),
    },

    "mouse": {
        "canonical": "Mouse",
        "ncbi": "Mus musculus",
        "terms": (
            "mouse",
            "mice",
            "Mus musculus",
        ),
        "subject_terms": (),
    },

    "mice": {
        "canonical": "Mouse",
        "ncbi": "Mus musculus",
        "terms": (
            "mouse",
            "mice",
            "Mus musculus",
        ),
        "subject_terms": (),
    },

    "mus musculus": {
        "canonical": "Mouse",
        "ncbi": "Mus musculus",
        "terms": (
            "mouse",
            "mice",
            "Mus musculus",
        ),
        "subject_terms": (),
    },

    "rat": {
        "canonical": "Rat",
        "ncbi": "Rattus norvegicus",
        "terms": (
            "rat",
            "rats",
            "Rattus norvegicus",
        ),
        "subject_terms": (),
    },

    "rats": {
        "canonical": "Rat",
        "ncbi": "Rattus norvegicus",
        "terms": (
            "rat",
            "rats",
            "Rattus norvegicus",
        ),
        "subject_terms": (),
    },

    "rattus norvegicus": {
        "canonical": "Rat",
        "ncbi": "Rattus norvegicus",
        "terms": (
            "rat",
            "rats",
            "Rattus norvegicus",
        ),
        "subject_terms": (),
    },
}


@dataclass(
    frozen=True,
)
class Agent3RuntimeContext:

    cluster_id: str
    species: str
    tissue: str
    disease_context: str
    platform: str

    ncbi_organism: str

    species_search_terms: tuple[str, ...]
    subject_search_terms: tuple[str, ...]

    @classmethod
    def from_mapping(
        cls,
        request: Mapping[str, Any],
    ) -> "Agent3RuntimeContext":

        cluster_id = _clean(
            request.get(
                "cluster_id"
            )
        )

        species_raw = _clean(
            request.get(
                "species"
            )
        )

        tissue = _clean(
            request.get(
                "tissue"
            )
        )

        disease = _clean(
            request.get(
                "disease_context"
            )
        )

        platform = _clean(
            request.get(
                "platform"
            )
        )

        if not cluster_id:
            raise ValueError(
                "LiteratureRequest is missing "
                "cluster_id."
            )

        if not species_raw:
            raise ValueError(
                "LiteratureRequest is missing "
                "species."
            )

        if not tissue:
            raise ValueError(
                "LiteratureRequest is missing "
                "tissue."
            )

        species_info = _SPECIES.get(
            species_raw.casefold()
        )

        if species_info is None:

            
            # Unknown species remain usable without inventing aliases.
            # The exact supplied species string is used for literature and NCBI queries.
            
            canonical_species = (
                species_raw
            )

            ncbi_organism = (
                species_raw
            )

            species_terms = (
                species_raw,
            )

            subject_terms = ()

        else:

            canonical_species = (
                species_info[
                    "canonical"
                ]
            )

            ncbi_organism = (
                species_info[
                    "ncbi"
                ]
            )

            species_terms = (
                species_info[
                    "terms"
                ]
            )

            subject_terms = (
                species_info[
                    "subject_terms"
                ]
            )

        return cls(
            cluster_id=cluster_id,
            species=canonical_species,
            tissue=tissue,
            disease_context=disease,
            platform=platform,
            ncbi_organism=ncbi_organism,
            species_search_terms=_unique(
                species_terms
            ),
            subject_search_terms=_unique(
                subject_terms
            ),
        )

    def context_terms(
        self,
    ) -> tuple[str, ...]:

        return _unique(
            (
                self.tissue,
                self.disease_context,
            )
        )

    def to_dict(
        self,
    ) -> dict:

        return {
            "cluster_id":
                self.cluster_id,

            "species":
                self.species,

            "tissue":
                self.tissue,

            "disease_context":
                self.disease_context,

            "platform":
                self.platform,

            "ncbi_organism":
                self.ncbi_organism,

            "species_search_terms":
                list(
                    self.species_search_terms
                ),

            "subject_search_terms":
                list(
                    self.subject_search_terms
                ),
        }
