from __future__ import annotations

import argparse
from pathlib import Path

from ..extraction_job_builder import (
    ExtractionJobBuilder,
    ExtractionJobConfig,
)
from ..schemas import (
    EvidenceCorpusResult,
)


def _project_root() -> Path:
    """
    Return the existing project root.
    """

    return Path(__file__).resolve().parents[3]


def _cluster_directory(
    cluster_id: str,
) -> Path:
    """
    Return the standard Literature Agent directory
    for the requested cluster.
    """

    return (
        _project_root()
        / "data"
        / "literature_retrieval_agent"
        / "clusters"
        / f"cluster_{cluster_id}"
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create inspectable, candidate-specific "
            "grounded literature extraction jobs."
        )
    )

    parser.add_argument(
        "--cluster-id",
        required=True,
        help=(
            "Cluster identifier from the Agent 2 "
            "and Agent 3 handoff."
        ),
    )

    parser.add_argument(
        "--corpus-json",
        type=Path,
        default=None,
        help=(
            "Optional evidence-corpus JSON path. "
            "The standard cluster path is used "
            "when omitted."
        ),
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Optional output JSON path. "
            "The standard cluster path is used "
            "when omitted."
        ),
    )

    parser.add_argument(
        "--maximum-passages-per-job",
        type=int,
        default=10,
        help=(
            "Maximum number of passages supplied "
            "to each candidate-specific job."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()

    cluster_id = str(args.cluster_id)

    cluster_directory = _cluster_directory(
        cluster_id
    )

    if args.corpus_json is not None:
        corpus_path = (
            args.corpus_json.resolve()
        )
    else:
        corpus_path = (
            cluster_directory
            / (
                f"cluster_{cluster_id}"
                "_evidence_corpus.json"
            )
        )

    if args.output_json is not None:
        output_path = (
            args.output_json.resolve()
        )
    else:
        output_path = (
            cluster_directory
            / (
                f"cluster_{cluster_id}"
                "_extraction_jobs.json"
            )
        )

    if not corpus_path.is_file():
        raise FileNotFoundError(
            "Evidence corpus JSON was not found: "
            f"{corpus_path}"
        )

    corpus = (
        EvidenceCorpusResult
        .model_validate_json(
            corpus_path.read_text(
                encoding="utf-8-sig"
            )
        )
    )

    if str(corpus.cluster_id) != cluster_id:
        raise ValueError(
            "Evidence corpus cluster ID does not "
            "match --cluster-id."
        )

    config = ExtractionJobConfig(
        maximum_passages_per_job=(
            args.maximum_passages_per_job
        )
    )

    plan = ExtractionJobBuilder(
        config=config
    ).build(
        corpus=corpus,
        source_corpus_file=str(
            corpus_path
        ),
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        plan.model_dump_json(indent=2),
        encoding="utf-8",
    )

    print(
        "Literature extraction jobs completed "
        "successfully."
    )

    print(
        f"Cluster: {plan.cluster_id}"
    )

    print(
        f"Jobs created: {plan.job_count}"
    )

    for job in plan.jobs:
        print(
            f"{job.candidate_cell_type}: "
            f"passages={len(job.passage_ids)}, "
            f"prompt_characters="
            f"{job.prompt_character_count}"
        )

    print(
        f"Warnings: {len(plan.warnings)}"
    )

    print(
        f"JSON: {output_path}"
    )


if __name__ == "__main__":
    main()