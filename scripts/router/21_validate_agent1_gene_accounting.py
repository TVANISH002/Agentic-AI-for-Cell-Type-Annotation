from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd



REQUIRED_MANIFEST_COLUMNS = {
    "dataset",
    "marker_file",
    "gene_column",
    "agent1_json",
}


def extract_gene(item: object) -> str:
    if isinstance(item, str):
        return item.strip()

    if isinstance(item, dict):
        value = item.get("gene", "")
        return str(value).strip()

    return ""


def normalize_gene(value: object) -> str:
    return str(value).strip().upper()


def load_agent1(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        obj = json.load(handle)

    if not isinstance(obj, dict):
        raise ValueError(
            f"Agent 1 JSON must be an object: {path}"
        )

    return obj


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/analysis/pipeline_validation/agent1_gene_accounting"
        ),
    )

    args = parser.parse_args()

    manifest = pd.read_csv(
        args.manifest
    )

    missing = (
        REQUIRED_MANIFEST_COLUMNS
        - set(manifest.columns)
    )

    if missing:
        raise ValueError(
            "Manifest missing columns: "
            + ", ".join(sorted(missing))
        )

    if manifest["dataset"].duplicated().any():
        raise ValueError(
            "Dataset names must be unique."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_reports = []
    mismatch_rows = []

    top_level_schemas = {}
    gene_evidence_schemas = {}

    for _, config in manifest.iterrows():
        dataset = str(
            config["dataset"]
        ).strip()

        marker_path = Path(
            str(config["marker_file"])
        )

        agent1_path = Path(
            str(config["agent1_json"])
        )

        if not marker_path.is_absolute():
            marker_path = (
                args.project_root
                / marker_path
            )

        if not agent1_path.is_absolute():
            agent1_path = (
                args.project_root
                / agent1_path
            )

        gene_column = str(
            config["gene_column"]
        ).strip()

        if not marker_path.exists():
            raise FileNotFoundError(
                f"{dataset}: marker file not found: {marker_path}"
            )

        if not agent1_path.exists():
            raise FileNotFoundError(
                f"{dataset}: Agent 1 JSON not found: {agent1_path}"
            )

        markers = pd.read_csv(
            marker_path
        )

        if gene_column not in markers.columns:
            raise ValueError(
                f"{dataset}: gene column '{gene_column}' "
                f"not found in marker file."
            )

        marker_genes_raw = (
            markers[gene_column]
            .dropna()
            .astype(str)
            .str.strip()
        )

        marker_genes_raw = marker_genes_raw.loc[
            marker_genes_raw.ne("")
        ]

        unique_marker_genes = list(
            dict.fromkeys(
                normalize_gene(gene)
                for gene in marker_genes_raw
            )
        )

        agent1 = load_agent1(
            agent1_path
        )

        required_agent_fields = {
            "input_genes",
            "gene_evidence",
            "unresolved_genes",
            "source_coverage",
            "deg_validation",
        }

        missing_agent_fields = (
            required_agent_fields
            - set(agent1)
        )

        if missing_agent_fields:
            raise ValueError(
                f"{dataset}: Agent 1 output missing fields: "
                + ", ".join(
                    sorted(missing_agent_fields)
                )
            )

        input_items = agent1.get(
            "input_genes",
            []
        ) or []

        evidence_items = agent1.get(
            "gene_evidence",
            []
        ) or []

        unresolved_items = agent1.get(
            "unresolved_genes",
            []
        ) or []

        input_genes = [
            normalize_gene(
                extract_gene(item)
            )
            for item in input_items
            if extract_gene(item)
        ]

        evidence_genes = [
            normalize_gene(
                extract_gene(item)
            )
            for item in evidence_items
            if extract_gene(item)
        ]

        unresolved_genes = [
            normalize_gene(
                extract_gene(item)
            )
            for item in unresolved_items
            if extract_gene(item)
        ]

        marker_set = set(
            unique_marker_genes
        )

        input_set = set(
            input_genes
        )

        evidence_set = set(
            evidence_genes
        )

        unresolved_set = set(
            unresolved_genes
        )

        missing_from_agent_input = sorted(
            marker_set - input_set
        )

        extra_in_agent_input = sorted(
            input_set - marker_set
        )

        missing_from_evidence = sorted(
            input_set - evidence_set
        )

        extra_in_evidence = sorted(
            evidence_set - input_set
        )

        unresolved_not_in_input = sorted(
            unresolved_set - input_set
        )

        duplicate_agent_input = (
            len(input_genes)
            - len(input_set)
        )

        duplicate_gene_evidence = (
            len(evidence_genes)
            - len(evidence_set)
        )

        coverage = agent1.get(
            "source_coverage",
            {}
        ) or {}

        coverage_total = coverage.get(
            "total_input_genes"
        )

        coverage_unresolved = coverage.get(
            "unresolved_genes"
        )

        coverage_final_matched = coverage.get(
            "final_matched_genes"
        )

        expected_resolved = (
            len(input_set)
            - len(unresolved_set)
        )

        source_coverage_total_matches = (
            int(coverage_total)
            == len(input_set)
            if coverage_total is not None
            else False
        )

        source_coverage_unresolved_matches = (
            int(coverage_unresolved)
            == len(unresolved_set)
            if coverage_unresolved is not None
            else False
        )

        source_coverage_resolved_matches = (
            int(coverage_final_matched)
            == expected_resolved
            if coverage_final_matched is not None
            else False
        )

        all_markers_accounted = (
            not missing_from_agent_input
            and not extra_in_agent_input
        )

        one_evidence_record_per_input = (
            not missing_from_evidence
            and not extra_in_evidence
            and duplicate_gene_evidence == 0
            and len(evidence_genes)
            == len(input_genes)
        )

        unresolved_explicit_and_valid = (
            not unresolved_not_in_input
        )

        dataset_pass = all(
            [
                all_markers_accounted,
                one_evidence_record_per_input,
                unresolved_explicit_and_valid,
                duplicate_agent_input == 0,
                source_coverage_total_matches,
                source_coverage_unresolved_matches,
                source_coverage_resolved_matches,
            ]
        )

        top_level_schemas[
            dataset
        ] = sorted(
            agent1.keys()
        )

        if evidence_items and isinstance(
            evidence_items[0],
            dict,
        ):
            gene_evidence_schemas[
                dataset
            ] = sorted(
                evidence_items[0].keys()
            )
        else:
            gene_evidence_schemas[
                dataset
            ] = []

        dataset_reports.append(
            {
                "dataset": dataset,
                "marker_row_count": int(
                    len(markers)
                ),
                "unique_marker_gene_count": int(
                    len(marker_set)
                ),
                "agent1_input_gene_count": int(
                    len(input_genes)
                ),
                "agent1_gene_evidence_count": int(
                    len(evidence_genes)
                ),
                "agent1_unresolved_gene_count": int(
                    len(unresolved_genes)
                ),
                "expected_resolved_gene_count": int(
                    expected_resolved
                ),
                "duplicate_agent_input_genes": int(
                    duplicate_agent_input
                ),
                "duplicate_gene_evidence_genes": int(
                    duplicate_gene_evidence
                ),
                "missing_from_agent1_input_count": int(
                    len(missing_from_agent_input)
                ),
                "extra_in_agent1_input_count": int(
                    len(extra_in_agent_input)
                ),
                "missing_from_gene_evidence_count": int(
                    len(missing_from_evidence)
                ),
                "extra_in_gene_evidence_count": int(
                    len(extra_in_evidence)
                ),
                "unresolved_not_in_input_count": int(
                    len(unresolved_not_in_input)
                ),
                "source_coverage_total_matches": bool(
                    source_coverage_total_matches
                ),
                "source_coverage_unresolved_matches": bool(
                    source_coverage_unresolved_matches
                ),
                "source_coverage_resolved_matches": bool(
                    source_coverage_resolved_matches
                ),
                "all_unique_marker_genes_accounted_for": bool(
                    all_markers_accounted
                ),
                "one_gene_evidence_record_per_input_gene": bool(
                    one_evidence_record_per_input
                ),
                "unresolved_genes_explicitly_retained": bool(
                    unresolved_explicit_and_valid
                ),
                "gene_accounting_status": (
                    "PASS"
                    if dataset_pass
                    else "FAIL"
                ),
            }
        )

        for gene in missing_from_agent_input:
            mismatch_rows.append(
                {
                    "dataset": dataset,
                    "category": "MISSING_FROM_AGENT1_INPUT",
                    "gene": gene,
                }
            )

        for gene in extra_in_agent_input:
            mismatch_rows.append(
                {
                    "dataset": dataset,
                    "category": "EXTRA_IN_AGENT1_INPUT",
                    "gene": gene,
                }
            )

        for gene in missing_from_evidence:
            mismatch_rows.append(
                {
                    "dataset": dataset,
                    "category": "MISSING_FROM_GENE_EVIDENCE",
                    "gene": gene,
                }
            )

        for gene in extra_in_evidence:
            mismatch_rows.append(
                {
                    "dataset": dataset,
                    "category": "EXTRA_IN_GENE_EVIDENCE",
                    "gene": gene,
                }
            )

        for gene in unresolved_not_in_input:
            mismatch_rows.append(
                {
                    "dataset": dataset,
                    "category": "UNRESOLVED_NOT_IN_INPUT",
                    "gene": gene,
                }
            )

    same_top_level_schema = (
        len(
            {
                tuple(value)
                for value
                in top_level_schemas.values()
            }
        )
        == 1
    )

    same_gene_evidence_schema = (
        len(
            {
                tuple(value)
                for value
                in gene_evidence_schemas.values()
            }
        )
        == 1
    )

    overall_pass = (
        all(
            item[
                "gene_accounting_status"
            ]
            == "PASS"
            for item
            in dataset_reports
        )
        and same_top_level_schema
        and same_gene_evidence_schema
    )

    report = {
        "audit_role": (
            "READ_ONLY_AGENT1_GENE_ACCOUNTING_VALIDATION"
        ),
        "dataset_count": int(
            len(dataset_reports)
        ),
        "datasets": dataset_reports,
        "same_top_level_schema_across_datasets": bool(
            same_top_level_schema
        ),
        "same_gene_evidence_schema_across_datasets": bool(
            same_gene_evidence_schema
        ),
        "overall_gene_accounting_status": (
            "PASS"
            if overall_pass
            else "FAIL"
        ),
        "biological_correctness_evaluated": False,
        "provenance_content_evaluated": False,
        "router_involved": False,
        "existing_outputs_modified": False,
    }

    report_path = (
        args.output_dir
        / "agent1_gene_accounting_audit_v1.json"
    )

    mismatch_path = (
        args.output_dir
        / "agent1_gene_accounting_mismatches_v1.csv"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        mismatch_rows,
        columns=[
            "dataset",
            "category",
            "gene",
        ],
    ).to_csv(
        mismatch_path,
        index=False,
    )

    print(
        f"Agent 1 gene-accounting audit: {report_path}"
    )

    print(
        f"Mismatch table: {mismatch_path}"
    )

    print()

    for result in dataset_reports:
        print(
            f"{result['dataset'].upper()}:"
        )

        print(
            f"  Marker rows: "
            f"{result['marker_row_count']}"
        )

        print(
            f"  Unique marker genes: "
            f"{result['unique_marker_gene_count']}"
        )

        print(
            f"  Agent 1 input genes: "
            f"{result['agent1_input_gene_count']}"
        )

        print(
            f"  Agent 1 evidence records: "
            f"{result['agent1_gene_evidence_count']}"
        )

        print(
            f"  Unresolved genes: "
            f"{result['agent1_unresolved_gene_count']}"
        )

        print(
            "  All unique markers accounted for: "
            f"{result['all_unique_marker_genes_accounted_for']}"
        )

        print(
            "  One evidence record per input gene: "
            f"{result['one_gene_evidence_record_per_input_gene']}"
        )

        print(
            "  Source coverage counts consistent: "
            f"{all([result['source_coverage_total_matches'], result['source_coverage_unresolved_matches'], result['source_coverage_resolved_matches']])}"
        )

        print(
            f"  STATUS: "
            f"{result['gene_accounting_status']}"
        )

        print()

    print(
        "Same top-level schema across datasets: "
        f"{same_top_level_schema}"
    )

    print(
        "Same gene-evidence schema across datasets: "
        f"{same_gene_evidence_schema}"
    )

    print(
        "OVERALL AGENT 1 GENE-ACCOUNTING STATUS: "
        f"{report['overall_gene_accounting_status']}"
    )

    print()
    print(
        "Biological correctness and detailed provenance were NOT "
        "evaluated in this step."
    )

    print(
        "No existing Agent, Router, or annotation output was changed."
    )


if __name__ == "__main__":
    main()
