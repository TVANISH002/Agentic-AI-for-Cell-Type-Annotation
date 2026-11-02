from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_NAME = "StandardizedSupplementalEvidencePacket"
SCHEMA_VERSION = "1.0"

ALLOWED_CORE_DECISION = "UNRESOLVED"


# Generic utilities

def clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def gene_key(value: Any) -> str:
    """
    Exact-symbol matching with case normalization only.

    IMPORTANT:
    Composite probes such as TPSAB1/B2 or HLA-DQB1/2 are
    preserved exactly. They are NOT split or remapped here.
    """
    return clean(value).upper()


def sha256_file(path: str | Path) -> str:
    path = Path(path)

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"JSON file does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected JSON object in: {path}"
        )

    return payload


def load_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"CSV file does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(
                f"CSV has no header: {path}"
            )

        return [
            dict(row)
            for row in reader
        ]


def numeric_sort_value(value: Any) -> tuple[int, float | str]:
    """
    Sort numeric ranks numerically while remaining safe for
    non-numeric identifiers.
    """

    text = clean(value)

    try:
        return (0, float(text))
    except (TypeError, ValueError):
        return (1, text)


def verify_file_hash(
    *,
    path: str | Path,
    expected_sha256: str,
    artifact_name: str,
) -> None:

    path = Path(path)

    actual = sha256_file(path)

    if actual.lower() != clean(expected_sha256).lower():
        raise RuntimeError(
            "Frozen evidence integrity check failed for "
            f"{artifact_name}. "
            f"Expected SHA-256={expected_sha256}; "
            f"actual SHA-256={actual}; "
            f"path={path}"
        )


def validate_columns(
    *,
    rows: Sequence[Mapping[str, Any]],
    required_columns: set[str],
    artifact_name: str,
) -> None:

    if not rows:
        raise ValueError(
            f"No rows found in {artifact_name}."
        )

    observed = set(
        rows[0].keys()
    )

    missing = (
        required_columns
        - observed
    )

    if missing:
        raise ValueError(
            f"{artifact_name} is missing required columns: "
            + ", ".join(sorted(missing))
        )


# Request and manifest validation

def validate_request(
    request: Mapping[str, Any],
) -> None:

    decision = clean(
        request.get(
            "core_agent3_decision"
        )
    ).upper()

    if decision != ALLOWED_CORE_DECISION:
        raise RuntimeError(
            "Supplemental evidence extraction is permitted "
            "only for a frozen core Agent 3 UNRESOLVED case. "
            f"Received: {decision}"
        )

    if request.get(
        "human_trigger_confirmed"
    ) is not True:
        raise RuntimeError(
            "Human trigger was not confirmed."
        )

    if request.get(
        "reference_label_used"
    ) is not False:
        raise RuntimeError(
            "Reference labels are forbidden during "
            "supplemental inference."
        )

    if request.get(
        "expert_label_used"
    ) is not False:
        raise RuntimeError(
            "Expert labels are forbidden during "
            "supplemental inference."
        )

    if request.get(
        "core_agent3_output_changed"
    ) is not False:
        raise RuntimeError(
            "Core Agent 3 output must remain unchanged."
        )


def validate_manifest(
    manifest: Mapping[str, Any],
) -> None:

    if manifest.get(
        "reference_labels_allowed"
    ) is not False:
        raise RuntimeError(
            "Evidence manifest permits reference labels. "
            "This is forbidden."
        )

    if manifest.get(
        "expert_labels_allowed"
    ) is not False:
        raise RuntimeError(
            "Evidence manifest permits expert labels. "
            "This is forbidden."
        )

    status = clean(
        manifest.get(
            "manifest_status"
        )
    )

    if status != "EVIDENCE_INPUTS_FROZEN":
        raise RuntimeError(
            "Evidence inputs are not in the expected "
            "frozen state. "
            f"Received manifest_status={status}"
        )


def validate_request_manifest_identity(
    *,
    request: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:

    for field in (
        "dataset_id",
        "cluster_id",
        "species",
        "tissue",
        "platform",
        "disease_context",
    ):
        request_value = request.get(
            field
        )

        manifest_value = manifest.get(
            field
        )

        if request_value != manifest_value:
            raise RuntimeError(
                "Request/manifest mismatch for "
                f"{field}: "
                f"request={request_value!r}, "
                f"manifest={manifest_value!r}"
            )


# Frozen-artifact integrity

def verify_manifest_artifact(
    artifact: Mapping[str, Any],
    artifact_name: str,
) -> Path:

    path_text = clean(
        artifact.get("path")
    )

    expected_hash = clean(
        artifact.get("sha256")
    )

    if not path_text:
        raise ValueError(
            f"{artifact_name} has no path."
        )

    if not expected_hash:
        raise ValueError(
            f"{artifact_name} has no SHA-256 hash."
        )

    path = Path(
        path_text
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Frozen {artifact_name} does not exist: {path}"
        )

    verify_file_hash(
        path=path,
        expected_sha256=expected_hash,
        artifact_name=artifact_name,
    )

    return path


# Agent 2 artifact classification

def classify_agent2_csv(
    path: Path,
) -> tuple[str, list[dict[str, str]]]:

    rows = load_csv(
        path
    )

    if not rows:
        raise ValueError(
            f"Agent 2 evidence file is empty: {path}"
        )

    columns = set(
        rows[0].keys()
    )

    gene_signature = {
        "cluster_id",
        "gene",
        "rank_within_cluster",
    }

    candidate_signature = {
        "cluster_id",
        "candidate_rank",
        "cell_type",
    }

    if gene_signature.issubset(
        columns
    ):
        return (
            "GENE_EVIDENCE",
            rows,
        )

    if candidate_signature.issubset(
        columns
    ):
        return (
            "CANDIDATE_EVIDENCE",
            rows,
        )

    raise ValueError(
        "Unrecognized Agent 2 evidence schema: "
        f"{path}"
    )


# Cluster-specific extraction

def extract_cluster_markers(
    *,
    marker_path: Path,
    cluster_id: str,
) -> list[dict[str, str]]:

    rows = load_csv(
        marker_path
    )

    validate_columns(
        rows=rows,
        required_columns={
            "cluster_id",
            "rank_within_cluster",
            "gene",
        },
        artifact_name="marker evidence",
    )

    selected = [
        row
        for row in rows
        if clean(
            row.get("cluster_id")
        ) == cluster_id
    ]

    if not selected:
        raise ValueError(
            "No marker rows found for "
            f"cluster {cluster_id}."
        )

    selected.sort(
        key=lambda row: numeric_sort_value(
            row.get(
                "rank_within_cluster"
            )
        )
    )

    seen: set[str] = set()

    for row in selected:
        key = gene_key(
            row.get("gene")
        )

        if not key:
            raise ValueError(
                "Marker evidence contains an empty gene."
            )

        if key in seen:
            raise ValueError(
                "Duplicate marker gene found within cluster "
                f"{cluster_id}: {row.get('gene')}"
            )

        seen.add(
            key
        )

    return selected


def extract_agent2_evidence(
    *,
    agent2_artifacts: Sequence[
        Mapping[str, Any]
    ],
    cluster_id: str,
    marker_gene_keys: set[str],
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[str],
]:

    gene_tables: list[
        list[dict[str, str]]
    ] = []

    candidate_tables: list[
        list[dict[str, str]]
    ] = []

    for index, artifact in enumerate(
        agent2_artifacts,
        start=1,
    ):

        path = verify_manifest_artifact(
            artifact,
            f"Agent 2 evidence {index}",
        )

        artifact_type, rows = (
            classify_agent2_csv(
                path
            )
        )

        if artifact_type == "GENE_EVIDENCE":
            gene_tables.append(
                rows
            )

        elif artifact_type == "CANDIDATE_EVIDENCE":
            candidate_tables.append(
                rows
            )

    if len(gene_tables) != 1:
        raise RuntimeError(
            "Expected exactly one Agent 2 gene-evidence "
            "table. "
            f"Found: {len(gene_tables)}"
        )

    if len(candidate_tables) != 1:
        raise RuntimeError(
            "Expected exactly one Agent 2 candidate-evidence "
            "table. "
            f"Found: {len(candidate_tables)}"
        )

    gene_rows = [
        row
        for row in gene_tables[0]
        if clean(
            row.get("cluster_id")
        ) == cluster_id
    ]

    candidate_rows = [
        row
        for row in candidate_tables[0]
        if clean(
            row.get("cluster_id")
        ) == cluster_id
    ]

    if not gene_rows:
        raise ValueError(
            "No Agent 2 gene evidence found for "
            f"cluster {cluster_id}."
        )

    if not candidate_rows:
        raise ValueError(
            "No Agent 2 candidate evidence found for "
            f"cluster {cluster_id}."
        )

    # Keep only gene evidence corresponding to the frozen
    # cluster marker set.
    filtered_gene_rows = [
        row
        for row in gene_rows
        if gene_key(
            row.get("gene")
        ) in marker_gene_keys
    ]

    filtered_gene_rows.sort(
        key=lambda row: numeric_sort_value(
            row.get(
                "rank_within_cluster"
            )
        )
    )

    candidate_rows.sort(
        key=lambda row: numeric_sort_value(
            row.get(
                "candidate_rank"
            )
        )
    )

    observed_gene_keys = {
        gene_key(
            row.get("gene")
        )
        for row in filtered_gene_rows
    }

    missing_marker_keys = (
        marker_gene_keys
        - observed_gene_keys
    )

    missing_markers = sorted(
        missing_marker_keys
    )

    if missing_markers:
        raise RuntimeError(
            "Agent 2 gene evidence is missing frozen marker "
            "genes: "
            + ", ".join(
                missing_markers
            )
        )

    ignored_extra_genes = sorted(
        {
            gene_key(
                row.get("gene")
            )
            for row in gene_rows
            if gene_key(
                row.get("gene")
            )
            not in marker_gene_keys
        }
    )

    return (
        filtered_gene_rows,
        candidate_rows,
        ignored_extra_genes,
    )


def index_gene_objects(
    *,
    objects: Sequence[Mapping[str, Any]],
    source_name: str,
) -> dict[str, dict[str, Any]]:

    index: dict[
        str,
        dict[str, Any]
    ] = {}

    for obj in objects:
        gene = clean(
            obj.get("gene")
        )

        if not gene:
            raise ValueError(
                f"{source_name} contains an object "
                "without a gene field."
            )

        key = gene_key(
            gene
        )

        if key in index:
            raise ValueError(
                f"Duplicate gene in {source_name}: {gene}"
            )

        index[key] = dict(
            obj
        )

    return index


def extract_agent1_for_markers(
    *,
    agent1_path: Path | None,
    marker_genes: list[str],
) -> tuple[
    list[dict[str, Any]],
    list[str],
    dict[str, Any] | None,
]:

    if agent1_path is None:
        return (
            [],
            list(marker_genes),
            None,
        )

    payload = load_json(
        agent1_path
    )

    gene_evidence = payload.get(
        "gene_evidence"
    )

    if not isinstance(
        gene_evidence,
        list,
    ):
        raise ValueError(
            "Agent 1 JSON must contain a "
            "'gene_evidence' array."
        )

    index = index_gene_objects(
        objects=gene_evidence,
        source_name="Agent 1 gene_evidence",
    )

    matched: list[
        dict[str, Any]
    ] = []

    missing: list[
        str
    ] = []

    for gene in marker_genes:
        item = index.get(
            gene_key(gene)
        )

        if item is None:
            missing.append(
                gene
            )
        else:
            matched.append(
                item
            )

    source_metadata = {
        "species": payload.get(
            "species"
        ),
        "tissue": payload.get(
            "tissue"
        ),
        "platform": payload.get(
            "platform"
        ),
        "context_definition": payload.get(
            "context_definition"
        ),
        "source_coverage": payload.get(
            "source_coverage"
        ),
        "unresolved_genes": payload.get(
            "unresolved_genes"
        ),
        "data_quality_warnings": payload.get(
            "data_quality_warnings"
        ),
    }

    return (
        matched,
        missing,
        source_metadata,
    )


def extract_hpa_for_markers(
    *,
    hpa_path: Path | None,
    marker_genes: list[str],
) -> tuple[
    list[dict[str, Any]],
    list[str],
    dict[str, Any] | None,
]:

    if hpa_path is None:
        return (
            [],
            [],
            None,
        )

    payload = load_json(
        hpa_path
    )

    summaries = payload.get(
        "gene_summaries"
    )

    if not isinstance(
        summaries,
        list,
    ):
        raise ValueError(
            "HPA JSON must contain a "
            "'gene_summaries' array."
        )

    index = index_gene_objects(
        objects=summaries,
        source_name="HPA gene_summaries",
    )

    matched: list[
        dict[str, Any]
    ] = []

    missing: list[
        str
    ] = []

    for gene in marker_genes:
        item = index.get(
            gene_key(gene)
        )

        if item is None:
            missing.append(
                gene
            )
        else:
            matched.append(
                item
            )

    source_metadata = {
        "run_mode": payload.get(
            "run_mode"
        ),
        "query_tissue": payload.get(
            "query_tissue"
        ),
        "source": payload.get(
            "source"
        ),
        "source_release": payload.get(
            "source_release"
        ),
        "source_url": payload.get(
            "source_url"
        ),
        "source_sha256": payload.get(
            "source_sha256"
        ),
        "retrieved_at_utc": payload.get(
            "retrieved_at_utc"
        ),
        "summary": payload.get(
            "summary"
        ),
        "warnings": payload.get(
            "warnings"
        ),
    }

    return (
        matched,
        missing,
        source_metadata,
    )


# Main standardized evidence packet builder

def build_evidence_packet(
    *,
    request_json: str | Path,
    manifest_json: str | Path,
) -> dict[str, Any]:

    request_path = Path(
        request_json
    )

    manifest_path = Path(
        manifest_json
    )

    request = load_json(
        request_path
    )

    manifest = load_json(
        manifest_path
    )

    validate_request(
        request
    )

    validate_manifest(
        manifest
    )

    validate_request_manifest_identity(
        request=request,
        manifest=manifest,
    )

    dataset_id = clean(
        request.get(
            "dataset_id"
        )
    )

    cluster_id = clean(
        request.get(
            "cluster_id"
        )
    )

    if not dataset_id:
        raise ValueError(
            "Missing dataset_id."
        )

    if not cluster_id:
        raise ValueError(
            "Missing cluster_id."
        )

    # Verify frozen marker artifact

    marker_artifact = manifest.get(
        "marker_evidence"
    )

    if not isinstance(
        marker_artifact,
        Mapping,
    ):
        raise ValueError(
            "Manifest is missing marker_evidence."
        )

    marker_path = verify_manifest_artifact(
        marker_artifact,
        "marker evidence",
    )

    markers = extract_cluster_markers(
        marker_path=marker_path,
        cluster_id=cluster_id,
    )

    marker_genes = [
        clean(
            row.get("gene")
        )
        for row in markers
    ]

    marker_gene_keys = {
        gene_key(gene)
        for gene in marker_genes
    }

    # Verify frozen core Agent 3 decision

    core_artifact = manifest.get(
        "core_agent3_decision"
    )

    if not isinstance(
        core_artifact,
        Mapping,
    ):
        raise ValueError(
            "Manifest is missing core_agent3_decision."
        )

    core_path = verify_manifest_artifact(
        core_artifact,
        "core Agent 3 decision",
    )

    core_decision_payload = load_json(
        core_path
    )

    core_decision = clean(
        core_decision_payload.get(
            "decision_state"
        )
        or core_decision_payload.get(
            "decision"
        )
        or core_decision_payload.get(
            "agent3_decision"
        )
    ).upper()

    if core_decision != ALLOWED_CORE_DECISION:
        raise RuntimeError(
            "Frozen core Agent 3 decision is no longer "
            f"{ALLOWED_CORE_DECISION}. "
            f"Received: {core_decision}"
        )

    if core_decision_payload.get(
        "reference_label_used"
    ) is not False:
        raise RuntimeError(
            "Frozen Agent 3 result indicates reference-label "
            "use, which is not allowed."
        )

    if core_decision_payload.get(
        "expert_label_used"
    ) is not False:
        raise RuntimeError(
            "Frozen Agent 3 result indicates expert-label "
            "use, which is not allowed."
        )


    agent2_artifacts = manifest.get(
        "agent2_evidence"
    )

    if not isinstance(
        agent2_artifacts,
        list,
    ):
        raise ValueError(
            "Manifest agent2_evidence must be an array."
        )

    (
        agent2_gene_evidence,
        agent2_ranked_candidates,
        agent2_ignored_extra_genes,
    ) = extract_agent2_evidence(
        agent2_artifacts=agent2_artifacts,
        cluster_id=cluster_id,
        marker_gene_keys=marker_gene_keys,
    )


    agent1_artifact = manifest.get(
        "agent1_evidence"
    )

    agent1_path: Path | None = None

    if isinstance(
        agent1_artifact,
        Mapping,
    ):
        agent1_path = verify_manifest_artifact(
            agent1_artifact,
            "Agent 1 evidence",
        )

    (
        agent1_marker_evidence,
        agent1_missing_markers,
        agent1_source_metadata,
    ) = extract_agent1_for_markers(
        agent1_path=agent1_path,
        marker_genes=marker_genes,
    )


    hpa_artifact = manifest.get(
        "optional_hpa_evidence"
    )

    hpa_path: Path | None = None

    if isinstance(
        hpa_artifact,
        Mapping,
    ):
        hpa_path = verify_manifest_artifact(
            hpa_artifact,
            "optional HPA evidence",
        )

    (
        hpa_marker_summaries,
        hpa_missing_markers,
        hpa_source_metadata,
    ) = extract_hpa_for_markers(
        hpa_path=hpa_path,
        marker_genes=marker_genes,
    )


    packet: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,

        "dataset_id": dataset_id,
        "cluster_id": cluster_id,

        "species": request.get(
            "species"
        ),
        "tissue": request.get(
            "tissue"
        ),
        "platform": request.get(
            "platform"
        ),
        "disease_context": request.get(
            "disease_context"
        ),

        "trigger": {
            "trigger_source": request.get(
                "trigger_source"
            ),
            "human_trigger_required": True,
            "human_trigger_confirmed": True,
        },

        "safety": {
            "core_agent3_output_changed": False,
            "reference_label_used": False,
            "expert_label_used": False,
            "manual_cell_type_label_injected": False,
            "manual_gene_selection_used": False,
        },

        "core_agent3": {
            "decision_state": core_decision,
            "decision_summary": core_decision_payload,
        },

        "marker_set": {
            "marker_count": len(
                markers
            ),
            "marker_genes": marker_genes,
            "markers": markers,
            "padding_used": False,
        },

        "agent1": {
            "enabled": (
                agent1_path is not None
            ),
            "matched_marker_count": len(
                agent1_marker_evidence
            ),
            "missing_marker_count": len(
                agent1_missing_markers
            ),
            "missing_marker_genes": (
                agent1_missing_markers
            ),
            "source_metadata": (
                agent1_source_metadata
            ),
            "marker_evidence": (
                agent1_marker_evidence
            ),
        },

        "agent2": {
            "gene_evidence_count": len(
                agent2_gene_evidence
            ),
            "candidate_count": len(
                agent2_ranked_candidates
            ),
            "ignored_extra_gene_count": len(
                agent2_ignored_extra_genes
            ),
            "ignored_extra_genes": (
                agent2_ignored_extra_genes
            ),
            "gene_evidence": (
                agent2_gene_evidence
            ),
            "ranked_candidates": (
                agent2_ranked_candidates
            ),
        },

        "hpa": {
            "enabled": (
                hpa_path is not None
            ),
            "matched_marker_count": len(
                hpa_marker_summaries
            ),
            "missing_marker_count": len(
                hpa_missing_markers
            ),
            "missing_marker_genes": (
                hpa_missing_markers
            ),
            "source_metadata": (
                hpa_source_metadata
            ),
            "gene_summaries": (
                hpa_marker_summaries
            ),
        },

        "provenance": {
            "request_json": str(
                request_path
            ),
            "request_sha256": sha256_file(
                request_path
            ),

            "manifest_json": str(
                manifest_path
            ),
            "manifest_sha256": sha256_file(
                manifest_path
            ),

            "marker_evidence": (
                manifest.get(
                    "marker_evidence"
                )
            ),

            "agent1_evidence": (
                manifest.get(
                    "agent1_evidence"
                )
            ),

            "agent2_evidence": (
                manifest.get(
                    "agent2_evidence"
                )
            ),

            "optional_hpa_evidence": (
                manifest.get(
                    "optional_hpa_evidence"
                )
            ),

            "core_agent3_decision": (
                manifest.get(
                    "core_agent3_decision"
                )
            ),
        },

        "validation": {
            "frozen_hashes_verified": True,

            "core_decision_verified_unresolved": True,

            "reference_labels_excluded": True,

            "expert_labels_excluded": True,

            "marker_padding_used": False,

            "gene_matching_policy": (
                "Exact gene symbol after case normalization only; "
                "composite probe names are preserved and are not split."
            ),

            "biological_adjudication_performed": False,
        },

        "status": (
            "STANDARDIZED_EVIDENCE_PACKET_BUILT_"
            "NOT_YET_ADJUDICATED"
        ),
    }

    return packet


def write_evidence_packet(
    packet: Mapping[str, Any],
    output_json: str | Path,
    *,
    overwrite: bool = False,
) -> Path:

    output_path = Path(
        output_json
    )

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing standardized "
            f"evidence packet: {output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            dict(packet),
            handle,
            indent=2,
            ensure_ascii=False,
        )

        handle.write("\n")

    return output_path