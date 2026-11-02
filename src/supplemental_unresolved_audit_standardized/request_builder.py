from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA_NAME = "StandardizedSupplementalUnresolvedAuditRequest"
SCHEMA_VERSION = "1.0"

ALLOWED_TRIGGER_DECISION = "UNRESOLVED"

DECISION_KEYS = (
    "decision_state",
    "decision",
    "agent3_decision",
)


def clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Required JSON file does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Expected a file but received: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected a JSON object in {path}."
        )

    return payload


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


def extract_core_decision(
    payload: Mapping[str, Any],
) -> str:
    """
    Extract the core Agent 3 decision conservatively.

    Only known decision fields are inspected.
    Reference labels or evaluation labels are never used.
    """

    containers: list[Mapping[str, Any]] = [payload]

    for container_key in (
        "decision_summary",
        "summary",
        "result",
    ):
        nested = payload.get(container_key)

        if isinstance(nested, Mapping):
            containers.append(nested)

    for container in containers:
        for key in DECISION_KEYS:
            value = clean(container.get(key))

            if value:
                return value.upper()

    raise ValueError(
        "Could not identify the Agent 3 decision from the "
        "decision-summary JSON. Expected one of: "
        + ", ".join(DECISION_KEYS)
    )


def extract_optional_cluster_id(
    payload: Mapping[str, Any],
) -> str | None:
    for key in (
        "cluster_id",
        "cluster",
    ):
        value = clean(payload.get(key))

        if value:
            return value

    return None


def extract_optional_dataset_id(
    payload: Mapping[str, Any],
) -> str | None:
    for key in (
        "dataset_id",
        "dataset",
    ):
        value = clean(payload.get(key))

        if value:
            return value

    return None


@dataclass(frozen=True)
class StandardizedUnresolvedAuditRequest:
    schema_name: str
    schema_version: str

    dataset_id: str
    cluster_id: str

    species: str | None
    tissue: str | None
    platform: str | None
    disease_context: str | None

    trigger_source: str
    core_agent3_decision: str

    human_trigger_required: bool
    human_trigger_confirmed: bool

    supplemental_audit_invoked: bool

    core_agent3_output_changed: bool

    reference_label_used: bool
    expert_label_used: bool

    core_agent3_decision_json: str
    core_agent3_decision_sha256: str

    request_created_utc: str

    audit_status: str


def build_request(
    *,
    dataset_id: str,
    cluster_id: str | int,
    core_agent3_decision_json: str | Path,
    species: str | None = None,
    tissue: str | None = None,
    platform: str | None = None,
    disease_context: str | None = None,
) -> StandardizedUnresolvedAuditRequest:
    """
    Build a human-triggered supplemental unresolved-case request.

    This function performs NO biological adjudication.

    It only verifies that:
      1. the supplied core Agent 3 decision exists;
      2. that decision is UNRESOLVED;
      3. dataset/cluster provenance is internally consistent;
      4. the original Agent 3 result remains read-only.
    """

    dataset_id = clean(dataset_id)
    cluster_id = clean(cluster_id)

    if not dataset_id:
        raise ValueError(
            "dataset_id must not be empty."
        )

    if not cluster_id:
        raise ValueError(
            "cluster_id must not be empty."
        )

    decision_path = Path(
        core_agent3_decision_json
    )

    payload = load_json(
        decision_path
    )

    core_decision = extract_core_decision(
        payload
    )

    if core_decision != ALLOWED_TRIGGER_DECISION:
        raise RuntimeError(
            "Supplemental unresolved-case audit is permitted "
            "only when the frozen core Agent 3 decision is "
            f"{ALLOWED_TRIGGER_DECISION}. "
            f"Received: {core_decision}"
        )

    payload_cluster_id = extract_optional_cluster_id(
        payload
    )

    if (
        payload_cluster_id is not None
        and payload_cluster_id != cluster_id
    ):
        raise ValueError(
            "Cluster mismatch between CLI/request and "
            "core Agent 3 decision JSON. "
            f"Requested cluster={cluster_id}; "
            f"decision JSON cluster={payload_cluster_id}."
        )

    payload_dataset_id = extract_optional_dataset_id(
        payload
    )

    if (
        payload_dataset_id is not None
        and payload_dataset_id != dataset_id
    ):
        raise ValueError(
            "Dataset mismatch between CLI/request and "
            "core Agent 3 decision JSON. "
            f"Requested dataset={dataset_id}; "
            f"decision JSON dataset={payload_dataset_id}."
        )

    return StandardizedUnresolvedAuditRequest(
        schema_name=SCHEMA_NAME,
        schema_version=SCHEMA_VERSION,

        dataset_id=dataset_id,
        cluster_id=cluster_id,

        species=clean(species) or None,
        tissue=clean(tissue) or None,
        platform=clean(platform) or None,
        disease_context=clean(disease_context) or None,

        trigger_source="HUMAN_MANUAL_INVOCATION",
        core_agent3_decision=core_decision,

        human_trigger_required=True,
        human_trigger_confirmed=True,

        supplemental_audit_invoked=True,

        # Critical invariant:
        # supplemental analysis NEVER changes the frozen
        # core Agent 3 output.
        core_agent3_output_changed=False,

        # Reference/expert labels are forbidden during
        # supplemental inference.
        reference_label_used=False,
        expert_label_used=False,

        core_agent3_decision_json=str(
            decision_path
        ),
        core_agent3_decision_sha256=sha256_file(
            decision_path
        ),

        request_created_utc=datetime.now(
            timezone.utc
        ).isoformat(),

        audit_status="REQUEST_CREATED_NOT_YET_ADJUDICATED",
    )


def write_request(
    request: StandardizedUnresolvedAuditRequest,
    output_json: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """
    Save the standardized request to a NEW supplemental
    output location.

    Existing files are protected by default.
    """

    output_path = Path(
        output_json
    )

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite an existing supplemental "
            f"audit request: {output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = asdict(
        request
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
        )

        handle.write("\n")

    return output_path