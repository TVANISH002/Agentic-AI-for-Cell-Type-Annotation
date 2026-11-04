# Agentic AI for Cell-Type Annotation

**Evidence-grounded, uncertainty-aware research software prototype**  

## Overview

This repository implements a multi-component framework for **cell-type annotation from prepared cluster-marker and gene-panel data**. The framework combines curated marker knowledge, cluster-level marker interpretation, evidence-sufficiency routing, and literature-based evidence assessment. It records supporting evidence and uncertainty rather than forcing a definitive identity for every cluster.

The software is designed for **configurable research workflows**: a researcher supplies a compatible study configuration and local input files, then invokes a single entry point. The code is not restricted to a hardcoded Breast or Lung cluster list; however, compatibility with additional tissues, platforms, and Router policies must be established separately.

> **Scope and intended use.** This is a research prototype, not a clinical diagnostic system. The current public entry point starts with **prepared marker-gene and gene-panel CSVs**. It does **not** automatically process arbitrary raw `.h5ad`, Xenium output folders, FASTQ files, or other sequencing input formats. Configuration flexibility must not be interpreted as demonstrated biological generalization.

## 1. Framework and decision flow

```text
Study configuration + prepared marker/panel inputs
                         |
                         v
          Optional HPA panel-evidence preparation
                         |
                         v
          Agent 1: Biological Knowledge Retrieval
                         |
                         v
          Agent 2: Marker Interpretation
                         |
                         v
          Router V4: Evidence-Sufficiency Assessment
                    /             \
           No escalation       Escalation required
                 |                    |
       Retain Agent 2 state      Agent 3: Literature Retrieval
                                      |
                             Core evidence decision
                                      |
                            If core UNRESOLVED and
                            supplemental review is
                            enabled and applicable:
                            supplemental evidence audit
                    \             /
                         v
       Annotation states, evidence, audit outputs,
             execution logs, and run manifest
```

| Component | Role |
| --- | --- |
| **Agent 1 — Biological Knowledge Retrieval** | Retrieves structured gene-to-cell-type evidence from configured biological reference resources, including CellMarker and PanglaoDB. |
| **Agent 2 — Marker Interpretation** | Integrates cluster-marker and available supporting evidence, ranks candidate cell identities, and reports ambiguous or insufficient support. |
| **Router V4 — Evidence-Sufficiency Assessment** | Uses evidence and uncertainty features from the upstream workflow to decide whether additional literature review is required. Its routing score is **not** a calibrated probability of biological correctness. |
| **Agent 3 — Literature Retrieval** | Performs literature retrieval, ranking, PubMedBERT-based semantic evidence analysis, and conservative candidate-rescue/audit operations when applicable. Semantic similarity is **not** cell-type annotation probability. |
| **Supplemental unresolved-case audit** | Examines eligible cases that remain unresolved after the core Agent 3 decision. Its supplementary projection is recorded separately and must not silently replace the core decision. |

**Routing is conditional:** if the Router produces an empty escalation queue, Agent 3 is correctly skipped. The number and identity of escalated clusters depend on the evidence and the **explicitly selected Router policy**, not on a hardcoded cluster ID in the public entry point.

## 2. Repository organization

The existing source organization is retained to preserve imports and the tested pipeline. Four components describe the **logical architecture**, not an enforced four-directory physical layout.

```text
.
├── src/
│   ├── biological_knowledge_agent/          # Agent 1 knowledge retrieval
│   ├── cosmx_brcr_annotation_pipeline/      # Existing Agent 1 execution adapter
│   ├── marker_interpretation_agent_v2/      # Agent 2
│   ├── lung_annotation_pipeline/            # Lung workflow adapter
│   ├── annotation_router/                   # Router implementations
│   ├── literature_retrieval_agent/          # Agent 3
│   ├── supplemental_unresolved_audit_standardized/
│   ├── supplemental_evidence_retrieval/     # Supporting evidence retrieval
│   ├── hpa_evidence_agent/                  # HPA utilities
│   ├── annotation_orchestrator/             # Additional research orchestration
│   └── unresolved_recovery_v1/             # Retained development component
├── tools/
│   ├── run_pipeline.py                      # Researcher-facing entry point
│   └── run_annotation_pipeline.py           # Execution coordinator
├── configs/
│   ├── studies/                             # Study template and tested examples
│   └── router/                              # Study-specific Router configurations
├── scripts/                                 # Research runners and audits
├── tests/                                   # Component tests
├── benchmarks/                              # Benchmark-related materials
├── data/                                    # Local inputs/references; excluded from Git
└── pyproject.toml
```

Supporting/experimental modules are retained for traceability. Their presence does **not** imply that all historical workflows are part of the validated one-command interface.

## 3. Installation and execution requirements

**Software:** Python 3.11 or later (`pyproject.toml` requirement), [`uv`](https://docs.astral.sh/uv/), and the dependencies declared in `pyproject.toml`. The portability/regression testing described below used Python 3.11. The dependencies include libraries for data processing and biomedical semantic models; some Agent 3 operations additionally require network access, external literature services, and model downloads.

From the repository root, create the environment:

```powershell
uv sync
```

The project dependencies are declared in `pyproject.toml`. Running `uv sync` resolves and installs the required dependencies and generates a local `uv.lock` file. The lockfile is intentionally excluded from the repository. Consequently, dependency versions may differ between installations, and exact environment reproducibility is not guaranteed without a separately preserved dependency lock or equivalent environment record.

**Before running an annotation, provide the required inputs and reference resources locally.** The repository's private GitHub version is intended to exclude the contents of `data/`; the example studies will not run on an empty-data clone without those files. See Sections 4 and 5.

The **single researcher command**, after installation and configuration, is:

```powershell
uv run python tools/run_pipeline.py --config configs/studies/breast_xenium.json
```

The executable interface has been tested using the repository's Python virtual environment on the original project, and the HPA-enabled wrapper has been tested in an isolated portability copy. A separate fresh-clone installation and execution test, using the repository-tracked files and separately provisioned research inputs, is required before claiming that the repository runs without local setup adjustments.

### Agent 3: External services and usage limitations

Agent 3 uses external biomedical services for literature retrieval, gene terminology, and semantic evidence ranking. These resources serve different purposes and have specific access limitations.

**Hugging Face and PubMedBERT.** Agent 3 uses the pretrained [`NeuML/pubmedbert-base-embeddings`](https://huggingface.co/NeuML/pubmedbert-base-embeddings) model for semantic reranking of retrieved literature evidence. The standardized workflow does not use a generative LLM for annotation. The public model can generally be accessed without an `HF_TOKEN`, although Hugging Face may display an unauthenticated-access warning and apply lower request limits. Researchers may configure their own token if needed. Authentication credentials must never be committed to Git. A successful cached model load does not independently verify a fresh-machine download.

**NCBI Gene terminology.** Agent 3 uses NCBI Gene E-utilities to retrieve official gene symbols and alternative gene terminology for marker-evidence analysis. NCBI's default unauthenticated E-utilities limit is three requests per second per IP address; the default limit with an API key is ten requests per second. NCBI also recommends providing application identification through `tool` and `email` parameters.

The current implementation includes a 0.4-second delay between marker lookups and records fallback information when an exact NCBI Gene match cannot be obtained. This delay is not a strict request-level or IP-wide rate limiter, and the implementation does not currently configure NCBI `tool` and `email` request parameters. Larger or concurrent workloads may require additional request management. An NCBI API key is not mandatory for the current small-scale workflow.

**Europe PMC.** Agent 3 retrieves biomedical literature through Europe PMC. Retrieval depends on network and service availability, and results may change as the literature is updated.

**Reproducibility.** Researchers should record model identifiers and revisions, retrieval settings, external-resource versions, NCBI gene-resolution statuses, and execution manifests. PubMedBERT semantic similarity is not biological annotation confidence, and successful workflow execution does not establish independently validated annotation accuracy.

References: [NCBI E-utilities documentation](https://www.ncbi.nlm.nih.gov/books/NBK25497/) and [Hugging Face Hub rate limits](https://huggingface.co/docs/hub/rate-limits).

## 4. Input contract and reference resources

The entry point accepts **prepared CSV inputs** with an explicit study configuration.

| Required input | Expected format / role |
| --- | --- |
| **Cluster-marker file** | CSV containing `gene` and `cluster_id` (or recognized aliases, e.g., `names` and `spatial_cluster`). Preserve marker ranking, scores, fold changes, and statistical fields when available; requirements may differ across Agent 2 modes. |
| **Gene-panel file** | CSV with a `gene` header and **unique, nonempty** gene symbols corresponding to the study's panel. |
| **Study configuration** | JSON identifying input paths, biological context, Agent 2 mode, Router configuration, optional HPA settings, and output location. |
| **CellMarker / PanglaoDB references** | Local biological marker databases required by the existing Agent 1 implementation. |
| **Cell Ontology** | Compatible local `cl-base.json` release for ontology-grounded operations. |
| **HPA reference (optional)** | Local HPA single-cell archive, or compatible pre-generated evidence JSON, when `general-hpa` mode is selected. |

For the tested local configuration, the relevant reference locations are:

```text
data/
├── reference/
│   ├── Cell_marker_Human.xlsx
│   ├── PanglaoDB_markers_27_Mar_2020.tsv
│   └── hpa/
│       └── rna_single_cell_cluster_hpa_v25_1.tsv.zip
└── resources/
    └── cell_ontology/
        └── v2026-06-08/
            └── cl-base.json
```

The tested HPA archive corresponds to **HPA release 25.1**. Reference versions must be recorded as actually obtained; a provider's current download may differ from the frozen tested release. Check each resource's access and redistribution conditions. For any reproducibility study, record input provenance, reference versions, SHA256 checksums, and any literature/model settings used.

Illustrative *synthetic* CSV structure (not a complete validated study):

`cluster_markers.csv`:

```csv
cluster_id,gene,rank_within_cluster,log_fold_change,adjusted_p_value
0,EPCAM,1,2.4,0.00001
0,KRT8,2,2.1,0.00002
1,PTPRC,1,2.0,0.00001
```

`panel_genes.csv`:

```csv
gene
EPCAM
KRT8
PTPRC
```

**Data availability and privacy.** The GitHub upload is intended to contain an **empty `data/` directory** (for example, retained via `data/.gitkeep`), not the local confidential inputs, historical outputs, model caches, or downloaded reference archives. Existing research data can remain in the authors' local archive; excluding them from Git does not delete them. The publication inventory must be checked to ensure that *all* private data are excluded, not just selected subfolders. Users must supply their own authorized compatible inputs and references.

## 5. Configuring a study and running the pipeline

The researcher-facing interface is `tools/run_pipeline.py`. It reads a JSON file; researchers do not need to launch individual agent runners.

**Step A — Prepare a configuration.** Adapt `configs/studies/study_template.json` and save a study-specific JSON. The template deliberately contains placeholders and is **not executable unchanged**. The Breast and Lung JSON files are historical example configurations, not universal defaults.

| JSON field | Meaning |
| --- | --- |
| `dataset` | Study identifier, used in the generated run directory. |
| `markers`, `panel` | Paths to prepared marker and panel CSV files. |
| `tissue`, `platform`, `disease_context` | Explicit biological and technology context. |
| `agent2_mode` | `general-marker-only`, `general-hpa`, or `lung-marker-only` (the last is Lung-specific). |
| `router_config` | Explicit Router policy; **do not reuse a Breast/Lung historical policy for an unrelated tissue without scientific review**. |
| `cell_ontology` | Local Cell Ontology JSON path. |
| `hpa` | Reference/tissue settings (or existing compatible HPA evidence) for `general-hpa`. |
| `supplemental_review` | Enables conditional supplemental processing of eligible unresolved cases. |
| `no_expert` | Coordinator option for excluding expert-reference use; reference labels must remain separate from inference and routing. |
| `output_parent` | Directory for uniquely timestamped outputs **outside the repository**. |

Paths for study inputs can be relative to the repository root or absolute paths on the researcher's system. On Windows, choose a short output parent path: planning mode checks the coordinator's projected 240-character downstream path limit.

**Step B — Validate configuration without running agents:**

```powershell
uv run python tools/run_pipeline.py --config configs/studies/breast_xenium.json --plan
```

**Step C — Execute the full configured workflow with one command:**

```powershell
uv run python tools/run_pipeline.py --config configs/studies/breast_xenium.json
```

A second tested example configuration is:

```powershell
uv run python tools/run_pipeline.py --config configs/studies/lung_xenium_normal.json
```

For a new compatible study, substitute the path to its JSON file after `--config`. The command is **not limited to a single cluster**: Agent 1, Agent 2, and Router process the supplied study evidence, and Agent 3 is invoked for clusters in the resulting escalation queue.

**Scope of extensibility:** marker-level integration is configurable, but an unseen tissue, species, platform, marker schema, or Router policy may require validation or additional implementation. Direct raw-data preprocessing is not currently included in this entry point.

## 6. Results, evidence, and provenance

Each execution creates a **new timestamped directory** beneath `output_parent`. Depending on the mode and routing decisions, the directory includes:

```text
<output_parent>/<dataset>_<timestamp>/
├── panel_sources/                              # Generated HPA evidence, if requested
└── pipeline/
    ├── a2/agent2_output/all_clusters_summary.csv
    ├── router/router_decisions.csv
    ├── router/agent3_queue.csv
    ├── a3/                                    # Conditional cluster-level Agent 3 outputs
    ├── logs/                                  # Detailed execution logs
    └── run_manifest.json                      # Stage execution and integrity metadata
```

The workflow preserves selected live progress messages for Agent 3 in the console and writes full stage logs. An **empty** `agent3_queue.csv` is an expected result when the Router does not request escalation; supplemental outputs are also conditional and do not necessarily appear in every run.

The manifest reports stage status, execution times, and protected-input integrity information. A `Pipeline: PASS` message indicates **execution success**, not independently verified biological correctness. Preserve core and supplemental evidence states separately when interpreting or citing results.

## 7. Validation evidence and boundaries

The following describes **internal engineering regression tests**, not a clinical or biological accuracy benchmark.

| Tested workflow | Clusters | Engineering observation |
| --- | ---: | --- |
| Breast Xenium, historical reproducibility | 8 | Agent 2 summary and Router V4 decision files reproduced the prior tested outputs in the isolated portability environment. |
| Lung Normal Xenium, historical reproducibility | 5 | Agent 2 summary and Router V4 decision files reproduced prior outputs; empty Router queue appropriately skipped Agent 3. |
| Lung Tumor Xenium, historical reproducibility | 11 | Agent 2 summary and Router V4 decision files reproduced prior outputs; empty Router queue appropriately skipped Agent 3. |
| **Total historical cluster-level comparisons** | **24** | **24/24 Agent 2 and 24/24 Router decision comparisons** matched in the internal E: portability tests; these are not 24 independently validated biological annotations. |

Additional integration evidence:

- **HPA-enabled Breast workflow:** the configuration-based wrapper generated HPA evidence and completed Agent 1, Agent 2, Router V4, and a real conditional Agent 3 execution for the Router-escalated cluster in the isolated test environment.
- **Real-time reporting:** actual Agent 3 progress messages appeared while the complete execution log was retained.
- **Supplemental audit:** a five-stage frozen-evidence replay on a development case passed its integrity checks. This replay is distinct from a fresh end-to-end new-dataset validation.
- **Canonical repository integration:** the one-command wrapper executed Lung Normal successfully from the original project in VS Code, preserving the tested Agent 2 and Router outputs.

The tests above support **implementation reproducibility and conditional workflow execution**. They do **not** establish annotation accuracy, calibration, novel biological discovery, state-of-the-art performance, or generalization to arbitrary tissues. Development cases and historical comparisons must not be described as independent validation. Expert/reference labels are for evaluation and should not be used in inference-time routing or annotation decisions.

For a publication-quality evaluation on additional studies, separately document dataset identity/accession, training/development versus independent evaluation partitions, reference label harmonization, biological baselines, uncertainty/abstention behavior, resources/models with exact versions, and end-to-end outcomes. Private test artifacts are not automatically included in this repository.

## 8. Troubleshooting and reproducibility checklist

| Symptom | Action |
| --- | --- |
| Missing input or reference file | Supply the authorized file locally and adjust the relevant JSON path. The private repository intentionally excludes local `data/` contents. |
| Long Windows output path | Set a shorter `output_parent` outside the repository; rerun `--plan`. |
| Agent 3 not invoked | Inspect `pipeline/router/agent3_queue.csv`. An empty queue correctly skips it. |
| Literature retrieval or model errors | Check `pipeline/logs/` and `run_manifest.json`; network access and model availability may be required. |
| Previous results exist | The coordinator avoids overwriting an existing output directory; start a new timestamped run. |

When reporting an experiment, record the Git commit hash, the exact study JSON and Router policy, marker/panel file hashes, reference-database and Cell Ontology versions, biomedical model identifiers and versions, retrieval settings, the Python/environment dependency lock, evaluation protocol, and the run manifest. Do not report PubMedBERT similarity or a Router evidence score as a biological probability.

## 9. Availability, publication, and licensing

**Current status:** private repository under manuscript preparation, accessible only to authorized collaborators. Data and external reference distributions are governed by their respective permissions and licenses. Public release, a software license, author attribution, and a paper/DOI citation will be added only after they are finalized and approved; no publication or license is asserted here.

**Release note:** successful execution in the authors' configured environments does not yet constitute a verified fresh-clone installation. Before a broader release, confirm the exact one-command installation/run instructions against the Git-tracked repository with the intended empty-data layout and separately provisioned references.
