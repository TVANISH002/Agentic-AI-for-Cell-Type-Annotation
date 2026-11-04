from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd



# Validates INTERNAL provenance consistency across all Agent 1 gene records:
# 1. Only approved structured sources appear:
#       CellMarker
#       PanglaoDB
# 2. gene.evidence_sources agrees with the source names in database_evidence.
# 3. context_matched_database_evidence is traceable to database_evidence.
# 4. Context-matched records do not carry context_match_level == "none".
# 5. Aggregated context-matched cell types and ontology IDs are traceable to their underlying database records.
# 6. Unresolved genes remain explicit and contain no fabricated DB evidence.
# 7. source_coverage source counts agree with gene-level evidence.
# 8. CellMarker citation preservation is summarized (PMID/title availability).


APPROVED_SOURCES = {
    "CellMarker",
    "PanglaoDB",
}

REQUIRED_MANIFEST_COLUMNS = {
    "dataset",
    "agent1_json",
}


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    return [value]


def strings(value: Any) -> list[str]:
    out: list[str] = []

    for item in as_list(value):
        if item is None:
            continue

        text = str(item).strip()

        if text:
            out.append(text)

    return out


def gene_name(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()

    if isinstance(item, dict):
        return str(
            item.get("gene") or ""
        ).strip()

    return ""


def normalize_gene(value: Any) -> str:
    return str(value).strip().upper()


def record_fingerprint(
    record: dict[str, Any],
) -> tuple[Any, ...]:
    """
    Stable biological/provenance identity for checking whether a context
    record exists in the corresponding global database evidence.

    We intentionally use the principal provenance fields and not metadata
    serialization order.
    """

    fields = [
        "source",
        "retrieval_method",
        "reported_marker",
        "official_symbol",
        "cell_type",
        "normalized_cell_type",
        "tissue",
        "disease_context",
        "cell_ontology_id",
        "pmid",
        "evidence_title",
    ]

    return tuple(
        record.get(field)
        for field in fields
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        obj = json.load(handle)

    if not isinstance(
        obj,
        dict,
    ):
        raise ValueError(
            f"Expected Agent 1 JSON object: {path}"
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
            "data/analysis/pipeline_validation/agent1_provenance"
        ),
    )

    args = parser.parse_args()

    manifest = pd.read_csv(
        args.manifest
    )

    missing_manifest = (
        REQUIRED_MANIFEST_COLUMNS
        - set(
            manifest.columns
        )
    )

    if missing_manifest:
        raise ValueError(
            "Manifest missing columns: "
            + ", ".join(
                sorted(
                    missing_manifest
                )
            )
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_reports: list[
        dict[str, Any]
    ] = []

    issue_rows: list[
        dict[str, Any]
    ] = []

    for _, config in manifest.iterrows():
        dataset = str(
            config["dataset"]
        ).strip()

        json_path = Path(
            str(
                config["agent1_json"]
            )
        )

        if not json_path.is_absolute():
            json_path = (
                args.project_root
                / json_path
            )

        if not json_path.exists():
            raise FileNotFoundError(
                f"{dataset}: Agent 1 JSON not found: {json_path}"
            )

        agent1 = load_json(
            json_path
        )

        evidence_items = as_list(
            agent1.get(
                "gene_evidence"
            )
        )

        unresolved_names = {
            normalize_gene(
                gene_name(item)
            )
            for item in as_list(
                agent1.get(
                    "unresolved_genes"
                )
            )
            if gene_name(item)
        }

        observed_source_names: set[
            str
        ] = set()

        genes_with_cellmarker = 0
        genes_with_panglaodb = 0
        genes_with_both = 0

        database_record_count = 0
        context_record_count = 0

        context_records_not_traceable = 0
        context_records_with_none_level = 0
        evidence_source_mismatches = 0

        context_cell_type_traceability_failures = 0
        ontology_traceability_failures = 0
        context_ontology_traceability_failures = 0

        unresolved_with_database_evidence = 0
        unresolved_with_evidence_sources = 0
        resolved_without_database_evidence = 0

        cellmarker_record_count = 0
        cellmarker_records_with_pmid = 0
        cellmarker_records_with_title = 0
        cellmarker_records_with_both_citation_fields = 0
        cellmarker_records_missing_both = 0

        panglaodb_record_count = 0

        alias_or_reported_symbol_variation_records = 0

        nonempty_atlas_evidence_genes = 0

        for gene_record in evidence_items:
            if not isinstance(
                gene_record,
                dict,
            ):
                issue_rows.append(
                    {
                        "dataset": dataset,
                        "gene": "",
                        "issue": "NON_OBJECT_GENE_EVIDENCE_RECORD",
                        "detail": type(
                            gene_record
                        ).__name__,
                    }
                )
                continue

            gene = normalize_gene(
                gene_record.get(
                    "gene",
                    "",
                )
            )

            db_records = [
                item
                for item in as_list(
                    gene_record.get(
                        "database_evidence"
                    )
                )
                if isinstance(
                    item,
                    dict,
                )
            ]

            context_records = [
                item
                for item in as_list(
                    gene_record.get(
                        "context_matched_database_evidence"
                    )
                )
                if isinstance(
                    item,
                    dict,
                )
            ]

            database_record_count += len(
                db_records
            )

            context_record_count += len(
                context_records
            )

            declared_sources = set(
                strings(
                    gene_record.get(
                        "evidence_sources"
                    )
                )
            )

            actual_sources = {
                str(
                    record.get(
                        "source"
                    )
                    or ""
                ).strip()
                for record in db_records
                if str(
                    record.get(
                        "source"
                    )
                    or ""
                ).strip()
            }

            observed_source_names.update(
                actual_sources
            )

            if declared_sources != actual_sources:
                evidence_source_mismatches += 1

                issue_rows.append(
                    {
                        "dataset": dataset,
                        "gene": gene,
                        "issue": "EVIDENCE_SOURCE_SUMMARY_MISMATCH",
                        "detail": (
                            f"declared={sorted(declared_sources)}; "
                            f"records={sorted(actual_sources)}"
                        ),
                    }
                )

            unsupported = (
                actual_sources
                - APPROVED_SOURCES
            )

            if unsupported:
                issue_rows.append(
                    {
                        "dataset": dataset,
                        "gene": gene,
                        "issue": "UNAPPROVED_DATABASE_SOURCE",
                        "detail": " | ".join(
                            sorted(
                                unsupported
                            )
                        ),
                    }
                )

            has_cellmarker = (
                "CellMarker"
                in actual_sources
            )

            has_panglaodb = (
                "PanglaoDB"
                in actual_sources
            )

            genes_with_cellmarker += int(
                has_cellmarker
            )

            genes_with_panglaodb += int(
                has_panglaodb
            )

            genes_with_both += int(
                has_cellmarker
                and has_panglaodb
            )

            global_fingerprints = {
                record_fingerprint(
                    item
                )
                for item in db_records
            }

            for record in context_records:
                fingerprint = record_fingerprint(
                    record
                )

                if fingerprint not in global_fingerprints:
                    context_records_not_traceable += 1

                    issue_rows.append(
                        {
                            "dataset": dataset,
                            "gene": gene,
                            "issue": "CONTEXT_RECORD_NOT_IN_GLOBAL_EVIDENCE",
                            "detail": json.dumps(
                                record,
                                default=str,
                            ),
                        }
                    )

                match_level = str(
                    record.get(
                        "context_match_level"
                    )
                    or ""
                ).strip().casefold()

                if (
                    not match_level
                    or match_level == "none"
                ):
                    context_records_with_none_level += 1

                    issue_rows.append(
                        {
                            "dataset": dataset,
                            "gene": gene,
                            "issue": "INVALID_CONTEXT_MATCH_LEVEL",
                            "detail": match_level,
                        }
                    )

            underlying_context_cell_types = {
                str(
                    record.get(
                        "normalized_cell_type"
                    )
                    or record.get(
                        "cell_type"
                    )
                    or ""
                ).strip()
                for record in context_records
                if str(
                    record.get(
                        "normalized_cell_type"
                    )
                    or record.get(
                        "cell_type"
                    )
                    or ""
                ).strip()
            }

            declared_context_cell_types = set(
                strings(
                    gene_record.get(
                        "context_matched_cell_types"
                    )
                )
            )

            if not (
                declared_context_cell_types
                <= underlying_context_cell_types
            ):
                context_cell_type_traceability_failures += 1

                issue_rows.append(
                    {
                        "dataset": dataset,
                        "gene": gene,
                        "issue": "CONTEXT_CELL_TYPE_NOT_TRACEABLE",
                        "detail": (
                            f"declared={sorted(declared_context_cell_types)}; "
                            f"underlying={sorted(underlying_context_cell_types)}"
                        ),
                    }
                )

            underlying_ontology_ids = {
                str(
                    record.get(
                        "cell_ontology_id"
                    )
                ).strip()
                for record in db_records
                if record.get(
                    "cell_ontology_id"
                )
            }

            declared_ontology_ids = set(
                strings(
                    gene_record.get(
                        "cell_ontology_ids"
                    )
                )
            )

            if not (
                declared_ontology_ids
                <= underlying_ontology_ids
            ):
                ontology_traceability_failures += 1

                issue_rows.append(
                    {
                        "dataset": dataset,
                        "gene": gene,
                        "issue": "ONTOLOGY_ID_NOT_TRACEABLE",
                        "detail": (
                            f"declared={sorted(declared_ontology_ids)}; "
                            f"underlying={sorted(underlying_ontology_ids)}"
                        ),
                    }
                )

            underlying_context_ontology_ids = {
                str(
                    record.get(
                        "cell_ontology_id"
                    )
                ).strip()
                for record in context_records
                if record.get(
                    "cell_ontology_id"
                )
            }

            declared_context_ontology_ids = set(
                strings(
                    gene_record.get(
                        "context_matched_cell_ontology_ids"
                    )
                )
            )

            if not (
                declared_context_ontology_ids
                <= underlying_context_ontology_ids
            ):
                context_ontology_traceability_failures += 1

                issue_rows.append(
                    {
                        "dataset": dataset,
                        "gene": gene,
                        "issue": "CONTEXT_ONTOLOGY_ID_NOT_TRACEABLE",
                        "detail": (
                            f"declared={sorted(declared_context_ontology_ids)}; "
                            f"underlying={sorted(underlying_context_ontology_ids)}"
                        ),
                    }
                )

            is_unresolved = (
                gene
                in unresolved_names
            )

            if is_unresolved:
                if db_records:
                    unresolved_with_database_evidence += 1

                    issue_rows.append(
                        {
                            "dataset": dataset,
                            "gene": gene,
                            "issue": "UNRESOLVED_GENE_HAS_DATABASE_EVIDENCE",
                            "detail": str(
                                len(
                                    db_records
                                )
                            ),
                        }
                    )

                if declared_sources:
                    unresolved_with_evidence_sources += 1

                    issue_rows.append(
                        {
                            "dataset": dataset,
                            "gene": gene,
                            "issue": "UNRESOLVED_GENE_HAS_EVIDENCE_SOURCES",
                            "detail": " | ".join(
                                sorted(
                                    declared_sources
                                )
                            ),
                        }
                    )

            else:
                if not db_records:
                    resolved_without_database_evidence += 1

                    issue_rows.append(
                        {
                            "dataset": dataset,
                            "gene": gene,
                            "issue": "RESOLVED_GENE_WITHOUT_DATABASE_EVIDENCE",
                            "detail": "",
                        }
                    )

            if as_list(
                gene_record.get(
                    "atlas_evidence"
                )
            ):
                nonempty_atlas_evidence_genes += 1

            for record in db_records:
                source = str(
                    record.get(
                        "source"
                    )
                    or ""
                ).strip()

                reported_marker = str(
                    record.get(
                        "reported_marker"
                    )
                    or ""
                ).strip()

                official_symbol = str(
                    record.get(
                        "official_symbol"
                    )
                    or ""
                ).strip()

                if (
                    reported_marker
                    and official_symbol
                    and reported_marker.casefold()
                    != official_symbol.casefold()
                ):
                    alias_or_reported_symbol_variation_records += 1

                if source == "CellMarker":
                    cellmarker_record_count += 1

                    has_pmid = bool(
                        str(
                            record.get(
                                "pmid"
                            )
                            or ""
                        ).strip()
                    )

                    has_title = bool(
                        str(
                            record.get(
                                "evidence_title"
                            )
                            or ""
                        ).strip()
                    )

                    cellmarker_records_with_pmid += int(
                        has_pmid
                    )

                    cellmarker_records_with_title += int(
                        has_title
                    )

                    cellmarker_records_with_both_citation_fields += int(
                        has_pmid
                        and has_title
                    )

                    cellmarker_records_missing_both += int(
                        not has_pmid
                        and not has_title
                    )

                elif source == "PanglaoDB":
                    panglaodb_record_count += 1

        coverage = (
            agent1.get(
                "source_coverage"
            )
            or {}
        )

        coverage_cellmarker = coverage.get(
            "matched_cellmarker"
        )

        coverage_panglaodb = coverage.get(
            "matched_panglaodb"
        )

        coverage_both = coverage.get(
            "matched_both"
        )

        cellmarker_count_matches = (
            int(
                coverage_cellmarker
            )
            == genes_with_cellmarker
            if coverage_cellmarker
            is not None
            else False
        )

        panglaodb_count_matches = (
            int(
                coverage_panglaodb
            )
            == genes_with_panglaodb
            if coverage_panglaodb
            is not None
            else False
        )

        both_count_matches = (
            int(
                coverage_both
            )
            == genes_with_both
            if coverage_both
            is not None
            else False
        )

        unapproved_sources = sorted(
            observed_source_names
            - APPROVED_SOURCES
        )

        hard_failures = [
            bool(
                unapproved_sources
            ),
            evidence_source_mismatches > 0,
            context_records_not_traceable > 0,
            context_records_with_none_level > 0,
            context_cell_type_traceability_failures > 0,
            ontology_traceability_failures > 0,
            context_ontology_traceability_failures > 0,
            unresolved_with_database_evidence > 0,
            unresolved_with_evidence_sources > 0,
            resolved_without_database_evidence > 0,
            not cellmarker_count_matches,
            not panglaodb_count_matches,
            not both_count_matches,
        ]

        dataset_pass = not any(
            hard_failures
        )

        dataset_reports.append(
            {
                "dataset": dataset,
                "agent1_tissue": agent1.get(
                    "tissue"
                ),
                "agent1_species": agent1.get(
                    "species"
                ),
                "gene_evidence_count": int(
                    len(
                        evidence_items
                    )
                ),
                "unresolved_gene_count": int(
                    len(
                        unresolved_names
                    )
                ),
                "database_record_count": int(
                    database_record_count
                ),
                "context_matched_record_count": int(
                    context_record_count
                ),
                "observed_database_sources": sorted(
                    observed_source_names
                ),
                "unapproved_database_sources": unapproved_sources,
                "genes_with_cellmarker": int(
                    genes_with_cellmarker
                ),
                "genes_with_panglaodb": int(
                    genes_with_panglaodb
                ),
                "genes_with_both": int(
                    genes_with_both
                ),
                "source_coverage_cellmarker_matches": bool(
                    cellmarker_count_matches
                ),
                "source_coverage_panglaodb_matches": bool(
                    panglaodb_count_matches
                ),
                "source_coverage_both_matches": bool(
                    both_count_matches
                ),
                "evidence_source_summary_mismatches": int(
                    evidence_source_mismatches
                ),
                "context_records_not_traceable_to_global": int(
                    context_records_not_traceable
                ),
                "context_records_with_none_match_level": int(
                    context_records_with_none_level
                ),
                "context_cell_type_traceability_failures": int(
                    context_cell_type_traceability_failures
                ),
                "ontology_id_traceability_failures": int(
                    ontology_traceability_failures
                ),
                "context_ontology_id_traceability_failures": int(
                    context_ontology_traceability_failures
                ),
                "unresolved_with_database_evidence": int(
                    unresolved_with_database_evidence
                ),
                "unresolved_with_evidence_sources": int(
                    unresolved_with_evidence_sources
                ),
                "resolved_without_database_evidence": int(
                    resolved_without_database_evidence
                ),
                "cellmarker_record_count": int(
                    cellmarker_record_count
                ),
                "cellmarker_records_with_pmid": int(
                    cellmarker_records_with_pmid
                ),
                "cellmarker_records_with_title": int(
                    cellmarker_records_with_title
                ),
                "cellmarker_records_with_pmid_and_title": int(
                    cellmarker_records_with_both_citation_fields
                ),
                "cellmarker_records_missing_both_pmid_and_title": int(
                    cellmarker_records_missing_both
                ),
                "panglaodb_record_count": int(
                    panglaodb_record_count
                ),
                "reported_marker_official_symbol_variation_records": int(
                    alias_or_reported_symbol_variation_records
                ),
                "genes_with_nonempty_atlas_evidence": int(
                    nonempty_atlas_evidence_genes
                ),
                "provenance_internal_consistency_status": (
                    "PASS"
                    if dataset_pass
                    else "FAIL"
                ),
            }
        )

    overall_pass = all(
        item[
            "provenance_internal_consistency_status"
        ]
        == "PASS"
        for item
        in dataset_reports
    )

    report = {
        "audit_role": (
            "READ_ONLY_AGENT1_INTERNAL_PROVENANCE_VALIDATION"
        ),
        "approved_sources": sorted(
            APPROVED_SOURCES
        ),
        "datasets": dataset_reports,
        "overall_status": (
            "PASS"
            if overall_pass
            else "FAIL"
        ),
        "raw_source_row_reconciliation_performed": False,
        "biological_truth_evaluated": False,
        "router_involved": False,
        "existing_outputs_modified": False,
        "interpretation": (
            "PASS means Agent 1's structured evidence summaries, context "
            "subsets, source counts, ontology identifiers, unresolved status, "
            "and approved-source provenance are internally traceable to the "
            "gene-level database evidence stored in the same Agent 1 output. "
            "It does not yet independently compare every record against the "
            "original CellMarker/PanglaoDB source tables."
        ),
    }

    report_path = (
        args.output_dir
        / "agent1_provenance_audit_v1.json"
    )

    issue_path = (
        args.output_dir
        / "agent1_provenance_issues_v1.csv"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        issue_rows,
        columns=[
            "dataset",
            "gene",
            "issue",
            "detail",
        ],
    ).to_csv(
        issue_path,
        index=False,
    )

    print(
        f"Agent 1 provenance audit: {report_path}"
    )

    print(
        f"Provenance issues: {issue_path}"
    )

    print()

    for item in dataset_reports:
        print(
            f"{item['dataset'].upper()}:"
        )

        print(
            "  Sources observed: "
            + " | ".join(
                item[
                    "observed_database_sources"
                ]
            )
        )

        print(
            f"  Database records: "
            f"{item['database_record_count']}"
        )

        print(
            f"  Context-matched records: "
            f"{item['context_matched_record_count']}"
        )

        print(
            f"  CellMarker-supported genes: "
            f"{item['genes_with_cellmarker']}"
        )

        print(
            f"  PanglaoDB-supported genes: "
            f"{item['genes_with_panglaodb']}"
        )

        print(
            f"  Supported by both: "
            f"{item['genes_with_both']}"
        )

        print(
            "  Source-coverage counts match gene records: "
            f"{all([item['source_coverage_cellmarker_matches'], item['source_coverage_panglaodb_matches'], item['source_coverage_both_matches']])}"
        )

        print(
            f"  Context records not traceable: "
            f"{item['context_records_not_traceable_to_global']}"
        )

        print(
            f"  Ontology traceability failures: "
            f"{item['ontology_id_traceability_failures'] + item['context_ontology_id_traceability_failures']}"
        )

        print(
            f"  Unresolved genes with fabricated evidence: "
            f"{item['unresolved_with_database_evidence']}"
        )

        print(
            f"  CellMarker records with PMID + title: "
            f"{item['cellmarker_records_with_pmid_and_title']}/"
            f"{item['cellmarker_record_count']}"
        )

        print(
            f"  Non-empty atlas evidence genes: "
            f"{item['genes_with_nonempty_atlas_evidence']}"
        )

        print(
            f"  STATUS: "
            f"{item['provenance_internal_consistency_status']}"
        )

        print()

    print(
        "OVERALL AGENT 1 INTERNAL PROVENANCE STATUS: "
        f"{report['overall_status']}"
    )

    print()
    print(
        "This step did NOT independently reconcile every record against "
        "the original CellMarker/PanglaoDB source tables."
    )

    print(
        "No existing Agent, Router, orchestrator, or annotation output "
        "was changed."
    )


if __name__ == "__main__":
    main()
