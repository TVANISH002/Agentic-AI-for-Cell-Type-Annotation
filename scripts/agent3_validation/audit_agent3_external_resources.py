from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(".").resolve()

OUT = (
    ROOT
    / "data"
    / "analysis"
    / "agent3_validation"
    / "agent3_external_resource_audit"
)
OUT.mkdir(parents=True, exist_ok=True)

CELL_ONTOLOGY = (
    ROOT
    / "data"
    / "resources"
    / "cell_ontology"
    / "v2026-06-08"
    / "cl-base.json"
)

CELL_ONTOLOGY_EXPECTED_SHA256 = (
    "e07aa201fcc72c186005e9ab2ddb0e6f49333bcff3d458d5ad8b0ee67c1c0f08"
)

MODEL_ID = "NeuML/pubmedbert-base-embeddings"

EUROPE_CLIENT = (
    ROOT
    / "src"
    / "literature_retrieval_agent"
    / "europe_pmc_client.py"
)

GENE_CLIENT = (
    ROOT
    / "src"
    / "literature_retrieval_agent"
    / "rescue_gene_terminology.py"
)

RUNNER = (
    ROOT
    / "scripts"
    / "agent3_validation"
    / "run_agent3_standardized_full.py"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(
        encoding="utf-8-sig",
        errors="replace",
    )


def urls_from(path: Path) -> list[str]:
    text = read(path)
    return sorted(
        set(
            re.findall(
                r'https?://[^\s\'"<>]+',
                text,
            )
        )
    )


def package_version(name: str):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


# -------------------------------------------------------------------------
# Cell Ontology
# -------------------------------------------------------------------------

ontology_exists = CELL_ONTOLOGY.exists()

ontology_sha = (
    sha256(CELL_ONTOLOGY)
    if ontology_exists
    else None
)

ontology_hash_match = (
    ontology_sha
    == CELL_ONTOLOGY_EXPECTED_SHA256
)

# -------------------------------------------------------------------------
# Hugging Face model revision
# -------------------------------------------------------------------------

hf_revision = None
hf_revision_source = None
hf_error = None

try:
    from huggingface_hub import constants

    cache_repo = (
        Path(constants.HF_HUB_CACHE)
        / "models--NeuML--pubmedbert-base-embeddings"
    )

    ref_main = (
        cache_repo
        / "refs"
        / "main"
    )

    if ref_main.exists():
        hf_revision = ref_main.read_text(
            encoding="utf-8"
        ).strip()

        hf_revision_source = (
            "local_huggingface_cache"
        )

except Exception as exc:
    hf_error = str(exc)


# If local cache does not expose a revision,
# query model metadata WITHOUT downloading the model.
if hf_revision is None:
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(
            MODEL_ID
        )

        hf_revision = info.sha
        hf_revision_source = (
            "huggingface_model_metadata"
        )

    except Exception as exc:
        if hf_error:
            hf_error += " | " + str(exc)
        else:
            hf_error = str(exc)


runner_text = read(RUNNER)

model_present_in_runner = (
    MODEL_ID in runner_text
)


# -------------------------------------------------------------------------
# Runtime software
# -------------------------------------------------------------------------

packages = {}

for package in [
    "sentence-transformers",
    "transformers",
    "torch",
    "huggingface-hub",
    "requests",
    "pydantic",
    "numpy",
    "pandas",
]:
    packages[package] = (
        package_version(package)
    )


# -------------------------------------------------------------------------
# External endpoints found in production code
# -------------------------------------------------------------------------

europe_urls = urls_from(
    EUROPE_CLIENT
)

ncbi_urls = urls_from(
    GENE_CLIENT
)


audit = {
    "schema_name":
        "Agent3ExternalResourceAudit",

    "schema_version":
        "1.0",

    "audit_date":
        datetime.now().isoformat(
            timespec="seconds"
        ),

    "europe_pmc": {
        "resource":
            "Europe PMC RESTful Web Service",

        "production_source_file":
            str(EUROPE_CLIENT),

        "urls_detected_in_source":
            europe_urls,

        "paper_doi":
            "10.1093/nar/gku1061",
    },

    "cell_ontology": {
        "release":
            "2026-06-08",

        "asset":
            "cl-base.json",

        "local_path":
            str(CELL_ONTOLOGY),

        "exists":
            ontology_exists,

        "local_sha256":
            ontology_sha,

        "official_release_sha256":
            CELL_ONTOLOGY_EXPECTED_SHA256,

        "hash_matches_official_release":
            ontology_hash_match,

        "resource_paper_doi":
            "10.1186/s13326-016-0088-7",
    },

    "ncbi_gene": {
        "resource":
            "NCBI Gene",

        "production_source_file":
            str(GENE_CLIENT),

        "urls_detected_in_source":
            ncbi_urls,

        "resource_paper_doi":
            "10.1093/nar/gku1055",
    },

    "pubmedbert": {
        "model_id":
            MODEL_ID,

        "model_id_present_in_standardized_runner":
            model_present_in_runner,

        "resolved_revision":
            hf_revision,

        "revision_source":
            hf_revision_source,

        "revision_lookup_error":
            hf_error,

        "base_model_family":
            "PubMedBERT/BiomedBERT",

        "base_model_paper_doi":
            "10.1145/3458754",

        "interpretation":
            (
                "Sentence-embedding model used for "
                "semantic relevance; similarity is "
                "not annotation probability or "
                "biological confidence."
            ),
    },

    "software_versions":
        packages,
}


json_path = (
    OUT
    / "agent3_external_resource_audit.json"
)

json_path.write_text(
    json.dumps(
        audit,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# -------------------------------------------------------------------------
# Bibliography
# -------------------------------------------------------------------------

accessed = datetime.now().strftime(
    "%Y-%m-%d"
)

bib = rf"""
@article{{europepmc2015,
  author  = {{The Europe PMC Consortium}},
  title   = {{Europe PMC: a full-text literature database for the life sciences and platform for innovation}},
  journal = {{Nucleic Acids Research}},
  year    = {{2015}},
  volume  = {{43}},
  number  = {{D1}},
  pages   = {{D1042--D1048}},
  doi     = {{10.1093/nar/gku1061}}
}}

@misc{{europepmc_rest,
  author       = {{Europe PMC}},
  title        = {{Europe PMC RESTful Web Service}},
  howpublished = {{Europe PMC Developer Resources}},
  note         = {{Accessed {accessed}}}
}}

@article{{diehl2016cellontology,
  author  = {{Diehl, Alexander D. and Meehan, Terrence F. and
              Bradford, Yvonne M. and Brush, Matthew H. and
              Dahdul, Wasila M. and Dougall, David S. and
              He, Yongqun and Osumi-Sutherland, David and
              Ruttenberg, Alan and Sarntivijai, Sirarat and
              Van Slyke, Ceri E. and Vasilevsky, Nicole A. and
              Haendel, Melissa A. and Blake, Judith A. and
              Mungall, Christopher J.}},
  title   = {{The Cell Ontology 2016: enhanced content,
              modularization, and ontology interoperability}},
  journal = {{Journal of Biomedical Semantics}},
  year    = {{2016}},
  volume  = {{7}},
  pages   = {{44}},
  doi     = {{10.1186/s13326-016-0088-7}}
}}

@misc{{cellontology20260608,
  author       = {{Cell Ontology Consortium}},
  title        = {{Cell Ontology Release 2026-06-08}},
  year         = {{2026}},
  howpublished = {{Cell Ontology GitHub release}},
  note         = {{cl-base.json; SHA-256:
                   e07aa201fcc72c186005e9ab2ddb0e6f49333bcff3d458d5ad8b0ee67c1c0f08}}
}}

@article{{brown2015ncbi_gene,
  author  = {{Brown, Garth R. and Hem, Vichet and Katz, Kenneth S. and
              Ovetsky, Michael and Wallin, Craig and Ermolaeva, Olga and
              Tolstoy, Igor and Tatusova, Tatiana and Pruitt, Kim D. and
              Maglott, Donna R. and Murphy, Terence D.}},
  title   = {{Gene: a gene-centered information resource at NCBI}},
  journal = {{Nucleic Acids Research}},
  year    = {{2015}},
  volume  = {{43}},
  number  = {{D1}},
  pages   = {{D36--D42}},
  doi     = {{10.1093/nar/gku1055}}
}}

@article{{gu2021pubmedbert,
  author  = {{Gu, Yu and Tinn, Robert and Cheng, Hao and Lucas, Michael and
              Usuyama, Naoto and Liu, Xiaodong and Naumann, Tristan and
              Gao, Jianfeng and Poon, Hoifung}},
  title   = {{Domain-Specific Language Model Pretraining for Biomedical
              Natural Language Processing}},
  journal = {{ACM Transactions on Computing for Healthcare}},
  year    = {{2021}},
  volume  = {{3}},
  number  = {{1}},
  pages   = {{1--23}},
  doi     = {{10.1145/3458754}}
}}

@misc{{neuml_pubmedbert_embeddings,
  author       = {{NeuML}},
  title        = {{PubMedBERT Embeddings}},
  howpublished = {{Hugging Face model repository:
                   NeuML/pubmedbert-base-embeddings}},
  note         = {{Exact model artifact used for Agent 3 semantic
                   reranking; accessed {accessed}}}
}}
""".strip()


bib_path = (
    OUT
    / "agent3_external_resources.bib"
)

bib_path.write_text(
    bib + "\n",
    encoding="utf-8",
)


print("=" * 100)
print("AGENT 3 EXTERNAL RESOURCE AUDIT")
print("=" * 100)

print(
    "Cell Ontology exists:",
    ontology_exists,
)

print(
    "Cell Ontology SHA256:",
    ontology_sha,
)

print(
    "Matches official 2026-06-08 release:",
    ontology_hash_match,
)

print()

print(
    "PubMedBERT model:",
    MODEL_ID,
)

print(
    "Model present in standardized runner:",
    model_present_in_runner,
)

print(
    "Resolved model revision:",
    hf_revision,
)

print(
    "Revision source:",
    hf_revision_source,
)

print()

print(
    "Europe PMC URLs detected:"
)

for url in europe_urls:
    print(" ", url)

print()

print(
    "NCBI URLs detected:"
)

for url in ncbi_urls:
    print(" ", url)

print()

print("Software versions:")

for package, value in packages.items():
    print(
        f"  {package}: {value}"
    )

print()
print("JSON:", json_path)
print("BibTeX:", bib_path)

print()
print(
    "OVERALL RESOURCE AUDIT:",
    (
        "PASS"
        if (
            ontology_hash_match
            and model_present_in_runner
        )
        else "REVIEW"
    ),
)
print("=" * 100)
