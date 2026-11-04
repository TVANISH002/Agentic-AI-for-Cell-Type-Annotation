from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path



# Download and verify the exact Cell Ontology artifact that will later be used
# for reference-label harmonization and ontology relation traversal.



ONTOLOGY_NAME = "Cell Ontology"
ONTOLOGY_PREFIX = "CL"
RELEASE_VERSION = "2026-06-08"
RELEASE_TAG = "v2026-06-08"
ARTIFACT_NAME = "cl-base.json"

DOWNLOAD_URL = (
    "https://github.com/obophenotype/cell-ontology/releases/download/"
    "v2026-06-08/cl-base.json"
)

EXPECTED_SHA256 = (
    "e07aa201fcc72c186005e9ab2ddb0e6f49333bcff3d458d5ad8b0ee67c1c0f08"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def download_file(
    url: str,
    destination: Path,
) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Agentic-Cell-Annotation-Research/"
                "CellOntologyPinning"
            )
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=120,
    ) as response:
        with destination.open("wb") as handle:
            while True:
                chunk = response.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                handle.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download and cryptographically verify a pinned Cell Ontology "
            "release artifact."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/resources/cell_ontology/v2026-06-08"
        ),
    )

    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help=(
            "Redownload even if a verified local artifact already exists."
        ),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ontology_file = (
        args.output_dir
        / ARTIFACT_NAME
    )

    manifest_file = (
        args.output_dir
        / "cell_ontology_manifest.json"
    )

    if (
        ontology_file.exists()
        and not args.force_redownload
    ):
        existing_sha = sha256_file(
            ontology_file
        )

        if existing_sha == EXPECTED_SHA256:
            print(
                "Verified ontology artifact already exists; "
                "download skipped."
            )
        else:
            raise ValueError(
                "Existing ontology file has the WRONG SHA-256. "
                "Delete it or rerun with --force-redownload.\n"
                f"Expected: {EXPECTED_SHA256}\n"
                f"Found:    {existing_sha}"
            )

    else:
        temporary_file = (
            args.output_dir
            / f"{ARTIFACT_NAME}.download"
        )

        if temporary_file.exists():
            temporary_file.unlink()

        print(
            f"Downloading {ONTOLOGY_NAME} {RELEASE_TAG}..."
        )

        download_file(
            DOWNLOAD_URL,
            temporary_file,
        )

        downloaded_sha = sha256_file(
            temporary_file
        )

        if downloaded_sha != EXPECTED_SHA256:
            temporary_file.unlink(
                missing_ok=True
            )

            raise ValueError(
                "Downloaded ontology artifact failed SHA-256 verification.\n"
                f"Expected: {EXPECTED_SHA256}\n"
                f"Found:    {downloaded_sha}"
            )

        temporary_file.replace(
            ontology_file
        )

    actual_sha = sha256_file(
        ontology_file
    )

    if actual_sha != EXPECTED_SHA256:
        raise ValueError(
            "Final ontology artifact failed verification."
        )

    # Basic JSON readability check.
    with ontology_file.open(
        "r",
        encoding="utf-8",
    ) as handle:
        first_character = ""

        while True:
            character = handle.read(1)

            if not character:
                break

            if not character.isspace():
                first_character = character
                break

    if first_character not in {"{", "["}:
        raise ValueError(
            "Downloaded file does not appear to be JSON."
        )

    manifest = {
        "ontology_name": ONTOLOGY_NAME,
        "ontology_prefix": ONTOLOGY_PREFIX,
        "release_version": RELEASE_VERSION,
        "release_tag": RELEASE_TAG,
        "artifact_name": ARTIFACT_NAME,
        "download_url": DOWNLOAD_URL,
        "expected_sha256": EXPECTED_SHA256,
        "verified_sha256": actual_sha,
        "sha256_verification": "PASS",
        "local_file": str(
            ontology_file
        ),
        "download_or_verification_timestamp_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "intended_use": (
            "Evaluation-only reference-label harmonization and ontology "
            "relation traversal. The ontology is not a Router predictive "
            "feature."
        ),
        "label_mapping_performed": False,
        "ontology_relations_assigned": False,
        "correctness_targets_created": False,
        "router_model_trained": False,
        "router_threshold_selected": False,
        "routing_decisions_created": False,
        "existing_agent_outputs_modified": False,
    }

    manifest_file.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Pinned ontology artifact: {ontology_file}"
    )

    print(
        f"Ontology manifest: {manifest_file}"
    )

    print(
        f"Release: {RELEASE_TAG}"
    )

    print(
        f"SHA-256 verification: PASS"
    )

    print(
        f"SHA-256: {actual_sha}"
    )

    print(
        "No label mapping, correctness target, Router model, "
        "threshold, or routing decision was created."
    )

    print(
        "No existing Agent output or annotation was changed."
    )


if __name__ == "__main__":
    main()
