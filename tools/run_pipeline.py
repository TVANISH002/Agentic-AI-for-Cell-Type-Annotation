"""Researcher-facing entry point for the existing cell annotation pipeline.

This wrapper handles configuration and orchestration only.
It does not implement or modify biological decision rules.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def input_file(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required input missing: {path}")
    return path


def output_directory(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT / path
    path = path.resolve()

    # Keep generated runs separate from packaged source and references.
    if path == PROJECT or path.is_relative_to(PROJECT):
        raise ValueError(
            "Output parent must be outside the source project directory."
        )

    return path


def run_command(command, environment):
    print("\nEXECUTING:", Path(command[1]).name if len(command) > 1 else command[0],
          flush=True)

    subprocess.run(
        [str(part) for part in command],
        cwd=PROJECT,
        env=environment,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Configuration-driven cell-type annotation"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Validate and display the execution plan without running agents",
    )
    args = parser.parse_args()

    config_path = input_file(args.config)
    settings = json.loads(config_path.read_text(encoding="utf-8-sig"))

    required = (
        "dataset", "markers", "panel", "tissue", "platform",
        "disease_context", "agent2_mode", "router_config",
        "output_parent",
    )

    for key in required:
        if not isinstance(settings.get(key), str) or not settings[key].strip():
            raise ValueError(f"Missing or invalid configuration field: {key}")

    dataset = settings["dataset"]

    if not re.fullmatch(r"[A-Za-z0-9_-]+", dataset):
        raise ValueError("Invalid dataset identifier")

    mode = settings["agent2_mode"]
    valid_modes = (
        "lung-marker-only",
        "general-marker-only",
        "general-hpa",
    )

    if mode not in valid_modes:
        raise ValueError(f"Unsupported Agent 2 mode: {mode}")

    if mode == "lung-marker-only" and settings["tissue"].lower() != "lung":
        raise ValueError("lung-marker-only requires Lung tissue")

    markers = input_file(settings["markers"])
    panel = input_file(settings["panel"])
    router = input_file(settings["router_config"])
    ontology = input_file(
        settings.get(
            "cell_ontology",
            "data/resources/cell_ontology/v2026-06-08/cl-base.json",
        )
    )

    output_parent = output_directory(settings["output_parent"])

    hpa_reference = None
    existing_hpa = None

    if mode == "general-hpa":
        hpa_settings = settings.get("hpa", {})

        if not isinstance(hpa_settings, dict):
            raise ValueError("hpa must be a configuration object")

        if bool(hpa_settings.get("reference")) == bool(
            hpa_settings.get("existing_json")
        ):
            raise ValueError(
                "For general-hpa provide exactly one of "
                "hpa.reference or hpa.existing_json"
            )

        if hpa_settings.get("reference"):
            hpa_reference = input_file(hpa_settings["reference"])
            if not hpa_settings.get("hpa_tissue"):
                raise ValueError("hpa.hpa_tissue is required")
        else:
            existing_hpa = input_file(hpa_settings["existing_json"])
    elif settings.get("hpa"):
        raise ValueError("HPA settings require general-hpa mode")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_root = output_parent / f"{dataset}_{run_id}"

    evidence_root = run_root / "panel_sources"
    pipeline_root = run_root / "pipeline"
    # Validate the same projected path as the frozen coordinator.
    # This runs before plan mode returns or any agent starts.
    projected = (
        pipeline_root / "a3/cluster_1234/marker_rescue/confirmation/pubmedbert/"
        "cluster_1234_rescue_confirmation_pubmedbert_summary.csv"
    )

    if len(str(projected)) > 240:
        raise ValueError(
            f"Projected Agent 3 path is {len(str(projected))} characters; "
            "the supported limit is 240. "
            "Choose a shorter output_parent, for example E:/ac_runs."
        )

    supplemental = settings.get("supplemental_review", True)
    no_expert = settings.get("no_expert", False)

    if not isinstance(supplemental, bool):
        raise ValueError("supplemental_review must be true or false")
    if not isinstance(no_expert, bool):
        raise ValueError("no_expert must be true or false")

    builder = [
        sys.executable, "-u", "-B", "-m",
        "marker_interpretation_agent_v2.runners.build_panel_sources",
        "--panel-gene-list", str(panel),
        "--hpa-reference", str(hpa_reference) if hpa_reference else "",
        "--output-dir", str(evidence_root),
        "--species", "Human",
        "--tissue", settings["tissue"],
        "--platform", settings["platform"],
        "--hpa-tissue", (
            settings.get("hpa", {}).get("hpa_tissue", "")
        ),
    ]

    coordinator = [
        sys.executable, "-u", "-B",
        str(PROJECT / "tools" / "run_annotation_pipeline.py"),
        "--project-root", str(PROJECT),
        "--markers", str(markers),
        "--panel", str(panel),
        "--dataset", dataset,
        "--tissue", settings["tissue"],
        "--platform", settings["platform"],
        "--disease-context", settings["disease_context"],
        "--router-config", str(router),
        "--output-root", str(pipeline_root),
        "--agent2-mode", mode,
        "--cell-ontology", str(ontology),
    ]

    if mode == "general-hpa":
        hpa_json = (
            existing_hpa
            if existing_hpa is not None
            else evidence_root / "panel_hpa_evidence.json"
        )
        coordinator += ["--hpa-json", str(hpa_json)]

    if supplemental:
        coordinator.append("--supplemental-review")

    if no_expert:
        coordinator.append("--no-expert")

    print("\n=== RESEARCHER PIPELINE PLAN ===")
    print("Project:", PROJECT)
    print("Configuration:", config_path)
    print("Dataset:", dataset)
    print("Markers:", markers)
    print("Panel:", panel)
    print("Tissue:", settings["tissue"])
    print("Platform:", settings["platform"])
    print("Agent 2 mode:", mode)
    print("Router policy:", router)
    print("Cell Ontology:", ontology)
    print("Supplemental review:", supplemental)
    print("No-expert routing:", no_expert)
    print("New output:", run_root)

    if hpa_reference:
        print("HPA build:", "ENABLED")
        print("HPA reference:", hpa_reference)
        print("HPA query tissue:", settings["hpa"]["hpa_tissue"])
    elif existing_hpa:
        print("Existing HPA evidence:", existing_hpa)
    else:
        print("HPA build:", "DISABLED")

    print("Agent 3: CONDITIONAL ON ROUTER V4")
    print("Supplemental: CONDITIONAL ON CORE UNRESOLVED")

    if args.plan:
        print("\nPLAN VALIDATED — NO AGENTS EXECUTED")
        print("No output directory created.")
        return

    if run_root.exists():
        raise FileExistsError(f"Run directory already exists: {run_root}")

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT / "src")
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment.setdefault("HF_HOME", str(output_parent / "_hf_cache"))
    environment.setdefault("TORCH_HOME", str(output_parent / "_torch_cache"))

    run_root.mkdir(parents=True, exist_ok=False)

    if hpa_reference:
        run_command(builder, environment)
        if not (evidence_root / "panel_hpa_evidence.json").is_file():
            raise RuntimeError("HPA evidence generation did not produce JSON")

    run_command(coordinator, environment)

    manifest = pipeline_root / "run_manifest.json"
    if not manifest.is_file():
        raise RuntimeError("Coordinator manifest missing")

    result = json.loads(manifest.read_text(encoding="utf-8"))
    if result.get("status") != "PASS":
        raise RuntimeError("Coordinator did not report PASS")

    print("\n=== RESEARCHER PIPELINE COMPLETE ===")
    print("Status: PASS")
    print("Output:", run_root)
    print("Manifest:", manifest)


if __name__ == "__main__":
    main()