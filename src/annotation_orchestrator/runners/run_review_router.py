from __future__ import annotations

import argparse
import json

from annotation_orchestrator.review_router import ReviewRouter


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Review Router for cell annotation pipeline.")
    parser.add_argument("--context-name", required=True)
    parser.add_argument("--agent2-output-csv", required=True)
    parser.add_argument("--reference-labels-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.60)
    parser.add_argument("--min-support-gap", type=float, default=1.0)

    args = parser.parse_args()

    router = ReviewRouter(
        context_name=args.context_name,
        agent2_output_csv=args.agent2_output_csv,
        reference_labels_csv=args.reference_labels_csv,
        output_dir=args.output_dir,
        low_confidence_threshold=args.low_confidence_threshold,
        min_support_gap=args.min_support_gap,
    )

    summary = router.run()
    print("Review Router completed.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()