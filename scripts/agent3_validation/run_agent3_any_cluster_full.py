from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_command(label: str, cmd: list[str]) -> None:
    print()
    print("=" * 90)
    print(label)
    print("=" * 90)
    print(" ".join(cmd))

    subprocess.run(
        cmd,
        check=True,
    )


def safe_dataset_name(dataset: str) -> str:
    value = dataset.strip().lower()

    if value not in {"lung", "breast"}:
        raise ValueError(
            "For this standardized workflow, --dataset must be lung or breast."
        )

    return value


def default_cluster_root(dataset: str, cluster_id: str) -> Path:
    return (
        Path("data")
        / "analysis"
        / "agent3_validation"
        / "standardized_full_runs"
        / dataset
        / f"cluster_{cluster_id}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete standardized Agent 3 workflow for any Breast or Lung "
            "cluster, starting from Agent 2 outputs and ending with a frozen evidence packet."
        )
    )

    parser.add_argument("--dataset", required=True, help="lung or breast")
    parser.add_argument("--cluster-id", required=True)
    parser.add_argument("--species", required=True)
    parser.add_argument("--tissue", required=True)
    parser.add_argument("--disease-context", required=True)
    parser.add_argument("--platform", required=True)

    parser.add_argument(
        "--cluster-root",
        default=None,
        type=Path,
        help=(
            "Optional output root. If omitted, uses "
            "data/analysis/agent3_validation/standardized_full_runs/<dataset>/cluster_<id>."
        ),
    )

    parser.add_argument(
        "--candidate-source",
        default="selected",
        choices=["selected", "ranked"],
        help=(
            "Use selected Agent 2 candidates by default. "
            "Use ranked only for documented sensitivity experiments."
        ),
    )

    parser.add_argument(
        "--candidate-count",
        type=int,
        default=0,
        help=(
            "Candidate count used by the request builder. "
            "0 keeps all candidates from selected/ranked file."
        ),
    )

    parser.add_argument(
        "--max-markers",
        type=int,
        default=50,
        help="Maximum ranked marker genes included in the LiteratureRequest.",
    )

    parser.add_argument("--context-page-size", type=int, default=3)
    parser.add_argument("--rescue-page-size", type=int, default=10)
    parser.add_argument("--confirmation-page-size", type=int, default=10)
    parser.add_argument("--pubmedbert-candidate-count", type=int, default=3)
    parser.add_argument("--top-k-passages", type=int, default=10)
    parser.add_argument("--top-marker-count", type=int, default=25)
    parser.add_argument("--rescue-top-marker-limit", type=int, default=5)

    parser.add_argument(
        "--min-paper-relevance",
        default="medium",
        choices=["high", "medium", "low"],
    )

    args = parser.parse_args()

    dataset = safe_dataset_name(args.dataset)
    cluster_id = str(args.cluster_id).strip()

    if not cluster_id:
        raise ValueError("--cluster-id cannot be empty.")

    cluster_root = (
        args.cluster_root
        if args.cluster_root is not None
        else default_cluster_root(dataset, cluster_id)
    )

    cluster_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    request_json = (
        cluster_root
        / f"cluster_{cluster_id}_literature_request.json"
    )

    py = sys.executable

    request_builder = Path(
        "scripts/agent3_validation/build_agent3_literature_request_from_agent2.py"
    )

    pipeline_runner = Path(
        "scripts/agent3_validation/run_agent3_cluster_pipeline.py"
    )

    standardized_runner = Path(
        "scripts/agent3_validation/run_agent3_cluster_standardized.py"
    )

    for script in [
        request_builder,
        pipeline_runner,
        standardized_runner,
    ]:
        if not script.exists():
            raise FileNotFoundError(
                f"Required standardized script missing: {script}"
            )

    started_at = datetime.now().isoformat(timespec="seconds")

    print("=" * 90)
    print("AGENT 3 ANY-CLUSTER FULL STANDARDIZED WORKFLOW")
    print("=" * 90)
    print("Dataset:", dataset)
    print("Cluster:", cluster_id)
    print("Species:", args.species)
    print("Tissue:", args.tissue)
    print("Disease context:", args.disease_context)
    print("Platform:", args.platform)
    print("Cluster root:", cluster_root)
    print("Request JSON:", request_json)
    print("Started:", started_at)
    print()
    print("This workflow fetches papers fresh for this cluster.")
    print("It does not use expert/reference labels for inference.")
    print("It does not force a final annotation.")
    print("=" * 90)

    # 1. Build LiteratureRequest from Agent 2 outputs.
    run_command(
        "1. BUILD CLUSTER-SPECIFIC LITERATURE REQUEST FROM AGENT 2 OUTPUTS",
        [
            py,
            str(request_builder),
            "--dataset",
            dataset,
            "--cluster-id",
            cluster_id,
            "--species",
            args.species,
            "--tissue",
            args.tissue,
            "--disease-context",
            args.disease_context,
            "--platform",
            args.platform,
            "--candidate-source",
            args.candidate_source,
            "--candidate-count",
            str(args.candidate_count),
            "--max-markers",
            str(args.max_markers),
            "--output-json",
            str(request_json),
        ],
    )

    # 2. Run full Agent 3 pipeline.
    run_command(
        "2. RUN FULL STANDARDIZED AGENT 3 PIPELINE",
        [
            py,
            str(pipeline_runner),
            "--dataset",
            dataset,
            "--cluster-id",
            cluster_id,
            "--request-json",
            str(request_json),
            "--cluster-root",
            str(cluster_root),
            "--context-page-size",
            str(args.context_page_size),
            "--rescue-page-size",
            str(args.rescue_page_size),
            "--confirmation-page-size",
            str(args.confirmation_page_size),
            "--candidate-count",
            str(args.pubmedbert_candidate_count),
            "--top-k-passages",
            str(args.top_k_passages),
            "--top-marker-count",
            str(args.top_marker_count),
            "--rescue-top-marker-limit",
            str(args.rescue_top_marker_limit),
            "--min-paper-relevance",
            args.min_paper_relevance,
        ],
    )

    # 3. Run final standardized freeze audit again explicitly.
    run_command(
        "3. FINAL STANDARDIZED FREEZE AUDIT",
        [
            py,
            str(standardized_runner),
            "--dataset",
            dataset,
            "--cluster-id",
            cluster_id,
            "--cluster-root",
            str(cluster_root),
        ],
    )

    finished_at = datetime.now().isoformat(timespec="seconds")

    print()
    print("=" * 90)
    print("AGENT 3 ANY-CLUSTER FULL STANDARDIZED WORKFLOW COMPLETE")
    print("=" * 90)
    print("Dataset:", dataset)
    print("Cluster:", cluster_id)
    print("Cluster root:", cluster_root)
    print("Request JSON:", request_json)
    print("Started:", started_at)
    print("Finished:", finished_at)
    print("=" * 90)


if __name__ == "__main__":
    main()
