
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False


SRC_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_ROOT),
    )

from literature_retrieval_agent.grounding_validator import (  # noqa: E402
    DEFAULT_GROUNDING_VALIDATION_CONFIG,
    validate_grounding,
)


ReasoningEffort = Literal[
    "none",
    "default",
    "low",
    "medium",
    "high",
]

ALLOWED_STANCES = {
    "supporting",
    "contradictory",
    "ambiguous",
    "context_only",
}

ALLOWED_CATEGORIES = {
    "positive_marker",
    "negative_marker",
    "marker_combination",
    "lineage",
    "cell_state",
    "tissue_expression",
    "disease_change",
    "spatial_localization",
    "cell_type_distinction",
    "marker_limitation",
    "other",
}

# Basic utilities

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def read_json(
    path: Path,
) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


def write_json(
    path: Path,
    data: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def safe_name(
    value: Any,
) -> str:
    cleaned = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        str(value),
    ).strip("_").lower()

    return cleaned or "model"


def compact_version(
    value: str,
) -> str:
    cleaned = value.strip()

    if re.fullmatch(
        r"\d+\.0",
        cleaned,
    ):
        cleaned = cleaned[:-2]

    return safe_name(
        cleaned
    )

# Extraction-job reading and input audit

def get_jobs(
    payload: Any,
) -> list[dict[str, Any]]:
    if not isinstance(
        payload,
        dict,
    ):
        return []

    jobs = payload.get(
        "jobs",
        [],
    )

    if not isinstance(
        jobs,
        list,
    ):
        return []

    return [
        job
        for job in jobs
        if isinstance(
            job,
            dict,
        )
    ]


def extract_prompt_version(
    prompt: str,
) -> str | None:
    match = re.search(
        r"^prompt_version:\s*([^\s]+)\s*$",
        prompt,
        flags=re.MULTILINE,
    )

    if match is None:
        return None

    return (
        match
        .group(1)
        .strip()
    )


def extract_prompt_passages(
    prompt: str,
) -> dict[str, str]:
    """Extract passage IDs and passage texts embedded in a job prompt."""

    passages: dict[
        str,
        str,
    ] = {}

    blocks = re.split(
        r"\n(?=PASSAGE\s+\d+\s*\n)",
        prompt,
    )

    for block in blocks:
        if not re.match(
            r"^PASSAGE\s+\d+",
            block.strip(),
        ):
            continue

        passage_id_match = re.search(
            r"^passage_id:\s*(.+?)\s*$",
            block,
            flags=re.MULTILINE,
        )

        passage_text_match = re.search(
            r"^text:\s*(.*)\Z",
            block,
            flags=(
                re.MULTILINE
                | re.DOTALL
            ),
        )

        if (
            passage_id_match is None
            or passage_text_match is None
        ):
            continue

        passage_id = (
            passage_id_match
            .group(1)
            .strip()
        )

        passage_text = (
            passage_text_match
            .group(1)
            .strip()
        )

        passages[
            passage_id
        ] = passage_text

    return passages


def audit_input(
    payload: Any,
    expected_cluster_id: str,
    expected_marker_count: int | None,
    expected_passage_count: int | None,
    expected_prompt_version: str,
) -> dict[str, Any]:
    jobs = get_jobs(
        payload
    )

    global_errors: list[str] = []
    job_rows: list[
        dict[str, Any]
    ] = []

    if not jobs:
        global_errors.append(
            "No jobs were found in the extraction-jobs JSON."
        )

    declared_job_count: Any = None

    if isinstance(
        payload,
        dict,
    ):
        declared_job_count = payload.get(
            "job_count"
        )

    if (
        declared_job_count is not None
        and declared_job_count
        != len(jobs)
    ):
        global_errors.append(
            f"The file declares {declared_job_count} jobs, "
            f"but {len(jobs)} were found."
        )

    for index, job in enumerate(
        jobs,
        start=1,
    ):
        errors: list[str] = []

        job_id = str(
            job.get(
                "job_id",
                f"job_{index}",
            )
        )

        cluster_id = str(
            job.get(
                "cluster_id",
                "",
            )
        )

        candidate = str(
            job.get(
                "candidate_cell_type",
                "",
            )
        )

        markers: Any = job.get(
            "observed_marker_genes",
            [],
        )

        passage_ids: Any = job.get(
            "passage_ids",
            [],
        )

        prompt: Any = job.get(
            "prompt",
            "",
        )

        if not isinstance(
            markers,
            list,
        ):
            markers = []
            errors.append(
                "observed_marker_genes is not a list."
            )

        if not isinstance(
            passage_ids,
            list,
        ):
            passage_ids = []
            errors.append(
                "passage_ids is not a list."
            )

        if not isinstance(
            prompt,
            str,
        ):
            prompt = ""
            errors.append(
                "prompt is not text."
            )

        parsed_passages = (
            extract_prompt_passages(
                prompt
            )
        )

        prompt_version = (
            extract_prompt_version(
                prompt
            )
        )

        if cluster_id != expected_cluster_id:
            errors.append(
                f"Expected cluster {expected_cluster_id}, "
                f"but the job contains {cluster_id}."
            )

        if not candidate:
            errors.append(
                "candidate_cell_type is missing."
            )

        if prompt_version != expected_prompt_version:
            errors.append(
                f"Expected prompt version {expected_prompt_version}, "
                f"but found {prompt_version or 'none'}."
            )

        if not markers:
            errors.append(
                "No observed marker genes were supplied."
            )

        if (
            expected_marker_count is not None
            and len(markers) != expected_marker_count
        ):
            errors.append(
                f"Expected {expected_marker_count} observed markers, "
                f"but found {len(markers)}."
            )

        if not passage_ids:
            errors.append(
                "No passage IDs were supplied."
            )

        if (
            expected_passage_count is not None
            and len(passage_ids) != expected_passage_count
        ):
            errors.append(
                f"Expected {expected_passage_count} passage IDs, "
                f"but found {len(passage_ids)}."
            )

        if len(parsed_passages) != len(passage_ids):
            errors.append(
                "The number of passage texts inside the prompt "
                f"({len(parsed_passages)}) does not match the "
                f"number of passage IDs ({len(passage_ids)})."
            )

        normalized_passage_ids = [
            str(passage_id)
            for passage_id in passage_ids
        ]

        if len(set(normalized_passage_ids)) != len(normalized_passage_ids):
            errors.append(
                "Duplicate passage IDs were supplied."
            )

        missing_from_prompt = [
            str(passage_id)
            for passage_id in passage_ids
            if str(passage_id)
            not in parsed_passages
        ]

        if missing_from_prompt:
            errors.append(
                "These passage IDs were not found in the prompt: "
                + ", ".join(
                    missing_from_prompt
                )
            )

        expected_prompt_length = job.get(
            "prompt_character_count"
        )

        if (
            isinstance(
                expected_prompt_length,
                int,
            )
            and expected_prompt_length
            != len(prompt)
        ):
            errors.append(
                "The stored prompt_character_count does not "
                "match the actual prompt length."
            )

        job_rows.append(
            {
                "job_id": job_id,
                "candidate_cell_type": candidate,
                "prompt_version": prompt_version,
                "marker_count": len(
                    markers
                ),
                "passage_id_count": len(
                    passage_ids
                ),
                "parsed_passage_count": len(
                    parsed_passages
                ),
                "prompt_character_count": len(
                    prompt
                ),
                "valid": not errors,
                "errors": errors,
            }
        )

    valid = (
        bool(jobs)
        and not global_errors
        and all(
            row["valid"]
            for row in job_rows
        )
    )

    return {
        "valid": valid,
        "cluster_id": expected_cluster_id,
        "expected_prompt_version": expected_prompt_version,
        "job_count": len(
            jobs
        ),
        "errors": global_errors,
        "jobs": job_rows,
    }


# Model call and model-output JSON parsing

def parse_json_object_from_model_text(
    content: str,
) -> dict[str, Any]:
    """
    Parse a JSON object from model output.

    GPT-OSS can use Groq provider-side strict JSON mode. Qwen may fail
    provider-side JSON validation, so for Qwen we let the model return text
    and validate JSON locally.
    """

    if (
        content is None
        or not content.strip()
    ):
        raise ValueError(
            "The model returned empty content."
        )

    text = content.strip()

    try:
        parsed = json.loads(
            text
        )

        if isinstance(
            parsed,
            dict,
        ):
            return parsed

        raise ValueError(
            "The model response is JSON, "
            "but it is not a JSON object."
        )

    except json.JSONDecodeError:
        pass

    if text.startswith("```"):
        lines = text.splitlines()

        if (
            len(lines) >= 3
            and lines[0].strip().startswith("```")
            and lines[-1].strip().startswith("```")
        ):
            text = "\n".join(
                lines[1:-1]
            ).strip()

            try:
                parsed = json.loads(
                    text
                )

                if isinstance(
                    parsed,
                    dict,
                ):
                    return parsed

            except json.JSONDecodeError:
                pass

    decoder = json.JSONDecoder()

    for index, character in enumerate(
        text
    ):
        if character != "{":
            continue

        try:
            parsed, _ = decoder.raw_decode(
                text[index:]
            )

        except json.JSONDecodeError:
            continue

        if isinstance(
            parsed,
            dict,
        ):
            return parsed

    preview = text[:500].replace(
        "\n",
        " ",
    )

    raise ValueError(
        "Could not parse a JSON object from model output. "
        f"Output preview: {preview}"
    )


def call_model(
    client: Any,
    model: str,
    prompt: str,
    max_completion_tokens: int,
    reasoning_effort: ReasoningEffort,
) -> tuple[
    dict[str, Any],
    float,
    str | None,
]:
    started = time.perf_counter()

    model_id_lower = model.lower()

    use_provider_strict_json = not (
        "qwen" in model_id_lower
    )

    request: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0,
        "max_completion_tokens": max_completion_tokens,
        "stream": False,
    }

    if use_provider_strict_json:
        request["response_format"] = {
            "type": "json_object"
        }

    if model.startswith("openai/gpt-oss"):
        request["reasoning_effort"] = reasoning_effort
        request["include_reasoning"] = False

    response = (
        client
        .chat
        .completions
        .create(**request)
    )

    latency = round(
        time.perf_counter() - started,
        3,
    )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    finish_value = (
        response
        .choices[0]
        .finish_reason
    )

    finish_reason = (
        str(finish_value)
        if finish_value is not None
        else None
    )

    if (
        content is None
        or not content.strip()
    ):
        raise ValueError(
            "The model returned empty content. "
            f"finish_reason={finish_reason!r}"
        )

    parsed = parse_json_object_from_model_text(
        content
    )

    return (
        parsed,
        latency,
        finish_reason,
    )

# Model output validation

def validate_schema(
    result: dict[str, Any],
    expected_candidate: str,
) -> tuple[
    bool,
    list[str],
]:
    errors: list[str] = []

    required_types: dict[
        str,
        type[Any],
    ] = {
        "candidate_cell_type": str,
        "claims": list,
        "candidate_summary": str,
        "insufficient_evidence": bool,
        "ambiguities": list,
        "missing_discriminating_markers": list,
    }

    for field, expected_type in required_types.items():
        if field not in result:
            errors.append(
                f"Missing required field: {field}."
            )

        elif not isinstance(
            result[field],
            expected_type,
        ):
            errors.append(
                f"Field {field} has the wrong type."
            )

    returned_candidate = str(
        result.get(
            "candidate_cell_type",
            "",
        )
    )

    if returned_candidate != expected_candidate:
        errors.append(
            "candidate_cell_type does not match the extraction job."
        )

    raw_claims: Any = result.get(
        "claims",
        [],
    )

    claims = (
        raw_claims
        if isinstance(
            raw_claims,
            list,
        )
        else []
    )

    for index, claim in enumerate(
        claims,
        start=1,
    ):
        if not isinstance(
            claim,
            dict,
        ):
            errors.append(
                f"Claim {index} is not a JSON object."
            )
            continue

        required_claim_fields = (
            "stance",
            "category",
            "claim_text",
            "evidence_quote",
            "passage_id",
        )

        for field in required_claim_fields:
            value = claim.get(
                field
            )

            if (
                not isinstance(
                    value,
                    str,
                )
                or not value.strip()
            ):
                errors.append(
                    f"Claim {index} has an invalid {field}."
                )

        stance = str(
            claim.get(
                "stance",
                "",
            )
        ).strip().lower()

        category = str(
            claim.get(
                "category",
                "",
            )
        ).strip().lower()

        if stance not in ALLOWED_STANCES:
            errors.append(
                f"Claim {index} has invalid stance {stance!r}."
            )

        if category not in ALLOWED_CATEGORIES:
            errors.append(
                f"Claim {index} has invalid category {category!r}."
            )

        markers: Any = claim.get(
            "markers"
        )

        if (
            not isinstance(
                markers,
                list,
            )
            or not all(
                isinstance(marker, str)
                for marker in markers
            )
        ):
            errors.append(
                f"Claim {index} has an invalid markers list."
            )

    insufficient = result.get(
        "insufficient_evidence"
    )

    if isinstance(
        insufficient,
        bool,
    ):
        if not claims and not insufficient:
            errors.append(
                "claims is empty, so insufficient_evidence must be true."
            )

        if claims and insufficient:
            errors.append(
                "claims is not empty, so insufficient_evidence should be false."
            )

    return (
        not errors,
        errors,
    )


# Output helpers

def write_summary_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    columns = [
        "model_name",
        "model_id",
        "prompt_version",
        "job_id",
        "candidate_cell_type",
        "success",
        "schema_valid",
        "fully_grounded",
        "claim_count",
        "grounded_claim_count",
        "grounding_rate",
        "insufficient_evidence",
        "latency_seconds",
        "finish_reason",
        "error",
    ]

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def run_api_test(
    client: Any,
    model: str,
    reasoning_effort: ReasoningEffort,
) -> None:
    request: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Reply with exactly OK",
            }
        ],
        "temperature": 0,
        "max_completion_tokens": 128,
        "stream": False,
    }

    if model.startswith("openai/gpt-oss"):
        request["reasoning_effort"] = reasoning_effort
        request["include_reasoning"] = False

    response = (
        client
        .chat
        .completions
        .create(**request)
    )

    print(
        "Content:",
        repr(
            response
            .choices[0]
            .message
            .content
        ),
    )

    print(
        "Finish reason:",
        response
        .choices[0]
        .finish_reason,
    )

# Command-line interface

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a Groq model on literature evidence-extraction jobs."
        )
    )

    parser.add_argument(
        "--cluster-id",
        required=True,
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Open-weight model ID. Overrides GROQ_LLM_MODEL."
        ),
    )

    parser.add_argument(
        "--jobs-file",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--api-test",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--run-label",
        default=None,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help=(
            "Optional parent directory for model-run output folders."
        ),
    )

    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=65,
    )

    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=2500,
    )

    parser.add_argument(
        "--reasoning-effort",
        choices=(
            "none",
            "default",
            "low",
            "medium",
            "high",
        ),
        default="low",
    )

    parser.add_argument(
        "--expected-marker-count",
        type=int,
        default=None,
        help=(
            "Optional exact marker count. Omit for datasets with dynamic marker counts."
        ),
    )

    parser.add_argument(
        "--expected-passage-count",
        type=int,
        default=None,
        help=(
            "Optional exact passage count. Omit when candidate jobs have different counts."
        ),
    )

    parser.add_argument(
        "--expected-prompt-version",
        default="3.0",
    )

    return parser.parse_args()


# Main benchmark runner

def main() -> None:
    load_dotenv()

    args = parse_arguments()

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    model = (
        args.model
        or os.getenv("GROQ_LLM_MODEL")
    )

    reasoning_effort = cast(
        ReasoningEffort,
        args.reasoning_effort,
    )

    if args.api_test:
        if not api_key:
            raise SystemExit(
                "GROQ_API_KEY is missing from .env"
            )

        if not model:
            raise SystemExit(
                "GROQ_LLM_MODEL is missing from .env"
            )

        from groq import Groq

        run_api_test(
            Groq(
                api_key=api_key
            ),
            model,
            reasoning_effort,
        )

        return

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    cluster_folder = (
        project_root
        / "data"
        / "literature_retrieval_agent"
        / "clusters"
        / f"cluster_{args.cluster_id}"
    )

    jobs_file = (
        args.jobs_file
        if args.jobs_file is not None
        else (
            cluster_folder
            / (
                f"cluster_{args.cluster_id}"
                "_extraction_jobs.json"
            )
        )
    )

    if not jobs_file.exists():
        raise SystemExit(
            f"Extraction-jobs file was not found: {jobs_file}"
        )

    payload = read_json(
        jobs_file
    )

    jobs = get_jobs(
        payload
    )

    audit = audit_input(
        payload=payload,
        expected_cluster_id=str(
            args.cluster_id
        ),
        expected_marker_count=args.expected_marker_count,
        expected_passage_count=args.expected_passage_count,
        expected_prompt_version=args.expected_prompt_version,
    )

    print(
        json.dumps(
            {
                "input_audit": audit
            },
            indent=2,
        )
    )

    if not audit["valid"]:
        raise SystemExit(
            "Input audit failed. No model request was sent."
        )

    if args.dry_run:
        audit_path = (
            jobs_file.parent
            / (
                f"cluster_{args.cluster_id}"
                "_input_audit.json"
            )
        )

        write_json(
            audit_path,
            audit,
        )

        print(
            "Dry run passed. No model request was sent."
        )

        print(
            f"Input audit JSON: {audit_path}"
        )

        return

    if not api_key:
        raise SystemExit(
            "GROQ_API_KEY is missing from .env"
        )

    if not model:
        raise SystemExit(
            "GROQ_LLM_MODEL is missing from .env"
        )

    from groq import Groq

    client = Groq(
        api_key=api_key
    )

    model_name = safe_name(
        model.split("/")[-1]
    )

    prompt_version = str(
        args.expected_prompt_version
    )

    default_run_label = (
        f"{model_name}_prompt_v"
        f"{compact_version(prompt_version)}"
    )

    run_label = safe_name(
        args.run_label
        or default_run_label
    )

    output_parent = (
        args.output_directory.resolve()
        if args.output_directory is not None
        else cluster_folder / "model_benchmark"
    )

    output_folder = (
        output_parent
        / run_label
    )

    if (
        output_folder.exists()
        and not args.overwrite
    ):
        raise SystemExit(
            f"Output folder already exists: {output_folder}\n"
            "Nothing was overwritten. Use --overwrite only if you intentionally "
            "want to replace this run."
        )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    started_at = utc_now()

    run_configuration = {
        "run_label": run_label,
        "provider": "Groq",
        "model_name": model_name,
        "model_id": model,
        "cluster_id": str(
            args.cluster_id
        ),
        "prompt_version": prompt_version,
        "input_file": str(
            jobs_file
        ),
        "job_count": len(
            jobs
        ),
        "expected_marker_count": args.expected_marker_count,
        "expected_passage_count": args.expected_passage_count,
        "temperature": 0,
        "max_completion_tokens": args.max_completion_tokens,
        "reasoning_effort": reasoning_effort,
        "wait_seconds": args.wait_seconds,
        "grounding_validation": (
            DEFAULT_GROUNDING_VALIDATION_CONFIG
            .to_dict()
        ),
        "started_at_utc": started_at,
    }

    write_json(
        output_folder
        / "run_configuration.json",
        run_configuration,
    )

    write_json(
        output_folder
        / "input_audit.json",
        audit,
    )

    write_json(
        output_folder
        / (
            f"cluster_{args.cluster_id}"
            "_extraction_jobs_input.json"
        ),
        payload,
    )

    detailed_results: list[
        dict[str, Any]
    ] = []

    summary_rows: list[
        dict[str, Any]
    ] = []

    for index, job in enumerate(
        jobs
    ):
        job_id = str(
            job.get(
                "job_id",
                f"job_{index + 1}",
            )
        )

        candidate = str(
            job.get(
                "candidate_cell_type",
                "",
            )
        )

        prompt = str(
            job.get(
                "prompt",
                "",
            )
        )

        raw_passage_ids: Any = job.get(
            "passage_ids",
            [],
        )

        passage_ids = [
            str(value)
            for value in raw_passage_ids
        ]

        prompt_passages = (
            extract_prompt_passages(
                prompt
            )
        )

        print(
            f"Running {model_name}: {candidate}..."
        )

        write_json(
            output_folder
            / f"{safe_name(job_id)}.json",
            job,
        )

        detail: dict[str, Any] = {
            "model_name": model_name,
            "model_id": model,
            "prompt_version": prompt_version,
            "job_id": job_id,
            "cluster_id": str(
                job.get(
                    "cluster_id",
                    args.cluster_id,
                )
            ),
            "candidate_cell_type": candidate,
            "success": False,
            "schema_valid": False,
            "schema_errors": [],
            "fully_grounded": False,
            "claim_count": 0,
            "grounded_claim_count": 0,
            "grounding_rate": None,
            "grounding_checks": [],
            "insufficient_evidence": None,
            "latency_seconds": None,
            "finish_reason": None,
            "response": None,
            "error": "",
        }

        try:
            result, latency, finish_reason = call_model(
                client=client,
                model=model,
                prompt=prompt,
                max_completion_tokens=args.max_completion_tokens,
                reasoning_effort=reasoning_effort,
            )

            schema_valid, schema_errors = validate_schema(
                result=result,
                expected_candidate=candidate,
            )

            (
                fully_grounded,
                claim_count,
                grounded_count,
                grounding_checks,
            ) = validate_grounding(
                result=result,
                prompt_passages=prompt_passages,
                allowed_passage_ids=passage_ids,
            )

            grounding_rate = (
                round(
                    grounded_count / claim_count,
                    4,
                )
                if claim_count > 0
                else None
            )

            detail.update(
                {
                    "success": True,
                    "schema_valid": schema_valid,
                    "schema_errors": schema_errors,
                    "fully_grounded": fully_grounded,
                    "claim_count": claim_count,
                    "grounded_claim_count": grounded_count,
                    "grounding_rate": grounding_rate,
                    "grounding_checks": grounding_checks,
                    "insufficient_evidence": result.get(
                        "insufficient_evidence"
                    ),
                    "latency_seconds": latency,
                    "finish_reason": finish_reason,
                    "response": result,
                }
            )

        except Exception as error:
            detail["error"] = (
                f"{type(error).__name__}: {error}"
            )

        detailed_results.append(
            detail
        )

        write_json(
            output_folder
            / (
                f"{safe_name(job_id)}"
                "_result.json"
            ),
            detail,
        )

        summary_rows.append(
            {
                "model_name": model_name,
                "model_id": model,
                "prompt_version": prompt_version,
                "job_id": job_id,
                "candidate_cell_type": candidate,
                "success": detail["success"],
                "schema_valid": detail["schema_valid"],
                "fully_grounded": detail["fully_grounded"],
                "claim_count": detail["claim_count"],
                "grounded_claim_count": detail["grounded_claim_count"],
                "grounding_rate": detail["grounding_rate"],
                "insufficient_evidence": detail["insufficient_evidence"],
                "latency_seconds": detail["latency_seconds"],
                "finish_reason": detail["finish_reason"],
                "error": detail["error"],
            }
        )

        if (
            index < len(jobs) - 1
            and args.wait_seconds > 0
        ):
            print(
                f"Waiting {args.wait_seconds} seconds for the API limit..."
            )

            time.sleep(
                args.wait_seconds
            )

    completed_at = utc_now()

    write_json(
        output_folder
        / "benchmark_results.json",
        {
            **run_configuration,
            "completed_at_utc": completed_at,
            "input_audit": audit,
            "results": detailed_results,
        },
    )

    write_summary_csv(
        output_folder
        / "benchmark_summary.csv",
        summary_rows,
    )

    print(
        f"Completed. Results saved to: {output_folder}"
    )

    print(
        json.dumps(
            summary_rows,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()











    