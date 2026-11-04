from __future__ import annotations

import argparse
from pathlib import Path

from literature_retrieval_agent.request_builder import (
    build_literature_request,
    load_agent2_handoff,
)
from literature_retrieval_agent.schemas import LiteratureRequest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a schema-valid Agent 3 LiteratureRequest directly "
            "from the production Agent 2 MarkerEvidence JSON."
        )
    )

    parser.add_argument(
        "--agent2-json",
        required=True,
        type=Path,
        help="Complete cluster_X_marker_evidence.json produced by Agent 2.",
    )
    parser.add_argument(
        "--disease-context",
        required=True,
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=3,
        help=(
            "Number of highest-ranked Agent 2 candidates passed to Agent 3. "
            "Default 3 matches the validated Agent 3 configuration."
        ),
    )
    parser.add_argument(
        "--marker-count",
        type=int,
        default=0,
        help=(
            "0 preserves all input markers already supplied by Agent 2. "
            "Use a positive integer only for a documented sensitivity analysis."
        ),
    )
    parser.add_argument(
        "--output-json",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    if args.candidate_count <= 0:
        raise ValueError(
            "--candidate-count must be greater than zero."
        )

    payload, source = load_agent2_handoff(
        args.agent2_json
    )

    marker_count = (
        None
        if args.marker_count == 0
        else args.marker_count
    )

    request = build_literature_request(
        payload,
        source_file=source,
        disease_context=args.disease_context,
        candidate_count=args.candidate_count,
        marker_count=marker_count,
    )

    # Serialize only fields belonging to the frozen LiteratureRequest schema.
    serialized = request.model_dump_json(
        indent=2,
    )

    # Hard validation before writing.
    validated = LiteratureRequest.model_validate_json(
        serialized
    )

    args.output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output_json.write_text(
        serialized,
        encoding="utf-8",
    )

    print("=" * 90)
    print("PRODUCTION AGENT 2 -> AGENT 3 REQUEST BUILDER")
    print("=" * 90)
    print("VALIDATION: PASS")
    print("Agent 2 source:", source)
    print("Output:", args.output_json)
    print("Cluster:", validated.cluster_id)
    print("Species:", validated.species)
    print("Tissue:", validated.tissue)
    print("Disease context:", validated.disease_context)
    print("Platform:", validated.platform)
    print("Marker count:", len(validated.marker_genes))
    print("Candidate count:", len(validated.candidate_annotations))

    print()
    print("Candidates:")
    for candidate in validated.candidate_annotations:
        print(
            f" - rank {candidate.upstream_rank}: "
            f"{candidate.name} "
            f"({', '.join(candidate.supporting_genes)})"
        )

    print()
    print("Review flags:", len(validated.review_flags))

    for flag in validated.review_flags:
        print(
            f" - [{flag.severity}] {flag.code}: "
            f"{flag.message}"
        )

    print()
    print("Source Agent 2 file:")
    print(validated.source_agent2_file)

    print()
    print("No extra wrapper metadata was inserted into LiteratureRequest.")
    print("=" * 90)


if __name__ == "__main__":
    main()
