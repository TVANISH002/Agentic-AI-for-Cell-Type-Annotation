
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


SCHEMA_NAME = "StandardizedSupplementalEvidenceManifest"
SCHEMA_VERSION = "1.1"


def clean(value: Any) -> str:
    return " ".join(str(value or "").split())


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


@dataclass(frozen=True)
class EvidenceArtifact:
    artifact_type: str
    path: str
    sha256: str
    required: bool


@dataclass(frozen=True)
class StandardizedEvidenceManifest:
    schema_name: str
    schema_version: str

    dataset_id: str
    cluster_id: str

    species: str | None
    tissue: str | None
    platform: str | None
    disease_context: str | None

    reference_labels_allowed: bool
    expert_labels_allowed: bool

    agent1_evidence: EvidenceArtifact | None

    # Multiple frozen Agent 2 artifacts are allowed because
    # gene-level evidence and candidate-level evidence are separate standardized upstream products.
    agent2_evidence: list[EvidenceArtifact]

    marker_evidence: EvidenceArtifact

    core_agent3_decision: EvidenceArtifact

    optional_hpa_evidence: EvidenceArtifact | None

    manifest_status: str


def make_artifact(
    *,
    artifact_type: str,
    path: str | Path | None,
    required: bool,
) -> EvidenceArtifact | None:

    if path is None:
        if required:
            raise ValueError(
                f"Required artifact missing: {artifact_type}"
            )

        return None

    path = Path(path)

    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Required artifact does not exist: {path}"
            )

        return None

    if not path.is_file():
        raise ValueError(
            f"Expected file for {artifact_type}: {path}"
        )

    return EvidenceArtifact(
        artifact_type=artifact_type,
        path=str(path),
        sha256=sha256_file(path),
        required=required,
    )


def make_artifact_list(
    *,
    artifact_type_prefix: str,
    paths: Sequence[str | Path],
) -> list[EvidenceArtifact]:

    artifacts: list[EvidenceArtifact] = []

    if not paths:
        raise ValueError(
            f"At least one {artifact_type_prefix} artifact is required."
        )

    seen_paths: set[str] = set()

    for index, raw_path in enumerate(
        paths,
        start=1,
    ):
        path = Path(raw_path)

        normalized = str(
            path.resolve()
        ).lower()

        if normalized in seen_paths:
            raise ValueError(
                "Duplicate evidence artifact supplied: "
                f"{path}"
            )

        seen_paths.add(
            normalized
        )

        artifact = make_artifact(
            artifact_type=(
                f"{artifact_type_prefix}_{index}"
            ),
            path=path,
            required=True,
        )

        if artifact is None:
            raise RuntimeError(
                "Unexpected missing required artifact: "
                f"{path}"
            )

        artifacts.append(
            artifact
        )

    return artifacts


def build_evidence_manifest(
    *,
    dataset_id: str,
    cluster_id: str | int,
    species: str | None,
    tissue: str | None,
    platform: str | None,
    disease_context: str | None,
    agent2_evidence: Sequence[str | Path],
    marker_evidence: str | Path,
    core_agent3_decision: str | Path,
    agent1_evidence: str | Path | None = None,
    hpa_evidence: str | Path | None = None,
) -> StandardizedEvidenceManifest:

    dataset_id = clean(
        dataset_id
    )

    cluster_id = clean(
        cluster_id
    )

    if not dataset_id:
        raise ValueError(
            "dataset_id must not be empty."
        )

    if not cluster_id:
        raise ValueError(
            "cluster_id must not be empty."
        )

    return StandardizedEvidenceManifest(
        schema_name=SCHEMA_NAME,
        schema_version=SCHEMA_VERSION,

        dataset_id=dataset_id,
        cluster_id=cluster_id,

        species=clean(species) or None,
        tissue=clean(tissue) or None,
        platform=clean(platform) or None,
        disease_context=clean(disease_context) or None,

        # Hard methodological restrictions.
        reference_labels_allowed=False,
        expert_labels_allowed=False,

        agent1_evidence=make_artifact(
            artifact_type="AGENT1_EVIDENCE",
            path=agent1_evidence,
            required=False,
        ),

        agent2_evidence=make_artifact_list(
            artifact_type_prefix="AGENT2_EVIDENCE",
            paths=agent2_evidence,
        ),

        marker_evidence=make_artifact(
            artifact_type="MARKER_EVIDENCE",
            path=marker_evidence,
            required=True,
        ),

        core_agent3_decision=make_artifact(
            artifact_type="CORE_AGENT3_DECISION",
            path=core_agent3_decision,
            required=True,
        ),

        optional_hpa_evidence=make_artifact(
            artifact_type="OPTIONAL_HPA_EVIDENCE",
            path=hpa_evidence,
            required=False,
        ),

        manifest_status="EVIDENCE_INPUTS_FROZEN",
    )


def write_evidence_manifest(
    manifest: StandardizedEvidenceManifest,
    output_json: str | Path,
    *,
    overwrite: bool = False,
) -> Path:

    output_path = Path(
        output_json
    )

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing evidence manifest: "
            f"{output_path}"
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
            asdict(manifest),
            handle,
            indent=2,
            ensure_ascii=False,
        )

        handle.write("\n")

    return output_path

