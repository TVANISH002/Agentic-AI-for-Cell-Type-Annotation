from __future__ import annotations

import argparse
import json
from pathlib import Path

from annotation_orchestrator.orchestrator import AnnotationOrchestrator


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the annotation orchestrator without changing agent logic.")
    parser.add_argument("--context-profile", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--execute-stage-commands", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    orchestrator = AnnotationOrchestrator(
        context_profile=args.context_profile,
        project_root=args.project_root,
    )
    summary = orchestrator.run(
        execute_stage_commands=args.execute_stage_commands,
        force=args.force,
    )

    print("Orchestrator completed.")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())