from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATASET = "cosmx_brcr_validated_clean"

RUN_ROOT = Path(
    "data/pipeline_runs/cosmx_brcr_orchestration_validation"
)

FREEZE_ROOT = RUN_ROOT / "final_freeze"

AGENT3_ROOT = RUN_ROOT / "agent3_primary"

SUPPLEMENTAL_ROOT = (
    RUN_ROOT
    / "supplemental_unresolved_audit"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def safe_labels(payload: dict[str, Any]) -> str:
    labels = payload.get(
        "possible_labels",
        [],
    )

    values = []

    for item in labels:
        if isinstance(item, str):
            values.append(item)

        elif isinstance(item, dict):
            for key in (
                "label",
                "cell_type",
                "candidate_cell_type",
            ):
                if item.get(key):
                    values.append(
                        str(item[key])
                    )
                    break

    return "; ".join(values)


def build_agent3_summary() -> list[dict[str, Any]]:
    rows = []

    if not AGENT3_ROOT.exists():
        return rows

    for cluster_dir in sorted(
        AGENT3_ROOT.glob("cluster_*"),
        key=lambda p: int(
            p.name.replace(
                "cluster_",
                "",
            )
        ),
    ):
        cluster_id = cluster_dir.name.replace(
            "cluster_",
            "",
        )

        decision = (
            cluster_dir
            / "result"
            / (
                f"cluster_{cluster_id}"
                "_agent3_decision_summary.json"
            )
        )

        if not decision.exists():
            rows.append(
                {
                    "dataset": DATASET,
                    "cluster_id": cluster_id,
                    "core_agent3_state": "MISSING",
                    "core_ambiguity_subtype": "",
                    "core_possible_labels": "",
                    "expert_label_used": "",
                    "reference_label_used": "",
                    "supplemental_state": "",
                    "supplemental_subtype": "",
                    "supplemental_possible_labels": "",
                    "core_agent3_output_changed": "",
                }
            )
            continue

        d = read_json(decision)

        row = {
            "dataset": DATASET,
            "cluster_id": cluster_id,
            "core_agent3_state": d.get(
                "decision_state",
                "",
            ),
            "core_ambiguity_subtype": (
                d.get(
                    "ambiguity_subtype"
                )
                or ""
            ),
            "core_possible_labels": safe_labels(
                d
            ),
            "expert_label_used": d.get(
                "expert_label_used",
                False,
            ),
            "reference_label_used": d.get(
                "reference_label_used",
                False,
            ),
            "supplemental_state": "",
            "supplemental_subtype": "",
            "supplemental_possible_labels": "",
            "core_agent3_output_changed": "",
        }

        supplemental = (
            SUPPLEMENTAL_ROOT
            / f"cluster_{cluster_id}"
            / f"cluster_{cluster_id}"
            / "adjudication"
            / (
                f"cluster_{cluster_id}"
                "_supplemental_adjudication.json"
            )
        )

        if supplemental.exists():
            s = read_json(
                supplemental
            )

            row[
                "supplemental_state"
            ] = s.get(
                "supplemental_decision_state",
                "",
            )

            row[
                "supplemental_subtype"
            ] = s.get(
                "supplemental_ambiguity_subtype",
                "",
            )

            row[
                "supplemental_possible_labels"
            ] = "; ".join(
                s.get(
                    "supplemental_possible_labels",
                    [],
                )
            )

            row[
                "core_agent3_output_changed"
            ] = s.get(
                "core_agent3_output_changed",
                "",
            )

        rows.append(row)

    return rows


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:

    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def build_file_manifest() -> list[dict[str, Any]]:
    rows = []

    important_roots = [
        Path(
            "configs/orchestrator/"
            "cosmx_brcr_context_validated_clean.json"
        ),
        Path(
            "configs/router/"
            "cosmx_brcr_router_v4_orchestration_validation.json"
        ),
        Path(
            "data/cosmx_brcr/processed/"
            "cosmx_brcr_top50_filtered_markers_clean.csv"
        ),
        RUN_ROOT,
    ]

    files: set[Path] = set()

    for item in important_roots:

        if item.is_file():
            files.add(item)

        elif item.is_dir():
            for path in item.rglob("*"):
                if path.is_file():
                    files.add(path)

    for path in sorted(
        files,
        key=lambda p: str(p).lower(),
    ):
        # Do not hash the freeze output into itself.
        if FREEZE_ROOT in path.parents:
            continue

        rows.append(
            {
                "relative_path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    return rows


def main() -> None:
    FREEZE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    agent3_rows = (
        build_agent3_summary()
    )

    summary_csv = (
        FREEZE_ROOT
        / "cosmx_final_agent3_summary.csv"
    )

    write_csv(
        summary_csv,
        agent3_rows,
    )

    manifest_rows = (
        build_file_manifest()
    )

    manifest_csv = (
        FREEZE_ROOT
        / "cosmx_final_file_manifest_sha256.csv"
    )

    write_csv(
        manifest_csv,
        manifest_rows,
    )

    manifest_json = (
        FREEZE_ROOT
        / "cosmx_final_freeze_manifest.json"
    )

    manifest = {
        "schema_name": (
            "CosMXFinalRunFreezeManifest"
        ),
        "schema_version": "1.0",
        "dataset": DATASET,
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "run_root": str(
            RUN_ROOT
        ),
        "agent3_cluster_count": len(
            agent3_rows
        ),
        "hashed_file_count": len(
            manifest_rows
        ),
        "reference_labels_used_for_inference": False,
        "expert_labels_used_for_inference": False,
        "supplemental_audit_policy": (
            "HUMAN_MANUAL_INVOCATION_ONLY"
        ),
        "core_agent3_outputs_overwritten_by_supplemental": False,
        "summary_csv": str(
            summary_csv
        ),
        "sha256_manifest_csv": str(
            manifest_csv
        ),
    }

    with manifest_json.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print(
        "\n"
        "=============================================="
    )
    print(
        "COSMX FINAL REPRODUCIBILITY FREEZE"
    )
    print(
        "=============================================="
    )

    print(
        f"Dataset: {DATASET}"
    )

    print(
        "Agent 3 clusters summarized:",
        len(agent3_rows),
    )

    print(
        "Files hashed:",
        len(manifest_rows),
    )

    print(
        "\nSummary:"
    )
    print(summary_csv)

    print(
        "\nSHA256 manifest:"
    )
    print(manifest_csv)

    print(
        "\nFreeze manifest:"
    )
    print(manifest_json)

    print(
        "\nSTATUS: COSMX_FINAL_RUN_FROZEN"
    )


if __name__ == "__main__":
    main()