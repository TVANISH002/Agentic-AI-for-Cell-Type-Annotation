from __future__ import annotations

import argparse
from pathlib import Path

from ..grounded_synthesizer import synthesize_literature_evidence_packet


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the professor-aligned literature evidence packet "
            "from semantically audited claims."
        )
    )
    parser.add_argument("--cluster-id", required=True)
    parser.add_argument("--semantic-audit-json", type=Path, required=True)
    parser.add_argument("--corpus-json", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()
    cluster_id = str(args.cluster_id)
    output_directory = args.output_directory.resolve()

    output_json = (
        output_directory
        / f"cluster_{cluster_id}_literature_evidence_packet.json"
    )
    candidate_csv = (
        output_directory
        / f"cluster_{cluster_id}_candidate_evidence_summary.csv"
    )
    evidence_csv = (
        output_directory
        / f"cluster_{cluster_id}_reference_evidence.csv"
    )

    packet = synthesize_literature_evidence_packet(
        semantic_audit_file=args.semantic_audit_json,
        corpus_file=args.corpus_json,
        output_json=output_json,
        candidate_csv=candidate_csv,
        evidence_csv=evidence_csv,
    )

    if str(packet.get("cluster_id")) != cluster_id:
        raise ValueError("Output cluster ID does not match --cluster-id.")

    print("Literature evidence appraisal completed successfully.")
    print(f"Cluster: {packet['cluster_id']}")
    print(f"Candidates: {len(packet['candidate_evidence'])}")
    print(f"Confidence contribution: {packet['confidence_contribution']}")
    print(f"Conclusion: {packet['literature_based_conclusion']}")
    print(f"JSON: {output_json}")
    print(f"Candidate CSV: {candidate_csv}")
    print(f"Reference CSV: {evidence_csv}")


if __name__ == "__main__":
    main()
