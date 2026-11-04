from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# Prepare evaluation-only reference annotations for future Router development.


REQUIRED_METADATA_COLUMNS = {
    "row_id",
    "dataset",
    "cluster_id",
    "top1_candidate",
    "top1_cell_ontology_ids",
}


REQUIRED_REFERENCE_COLUMNS = {
    "dataset",
    "cluster_id",
    "primary_reference_label",
    "primary_reference_type",
    "use_for_primary_accuracy",
    "llm_label",
    "use_for_llm_agreement",
}


ALLOWED_PRIMARY_REFERENCE_TYPES = {
    "biology_expert",
    "published_reference",
    "expert_curated",
}


ONTOLOGY_RELATION_OPTIONS = [
    "exact",
    "predicted_child_of_reference",
    "predicted_parent_of_reference",
    "sibling",
    "biologically_compatible_nonhierarchical",
    "no_match",
    "unmapped",
]


def parse_bool(value: object) -> bool:
    if pd.isna(value):
        return False

    text = str(value).strip().casefold()

    if text in {"true", "1", "yes", "y"}:
        return True

    if text in {"false", "0", "no", "n", ""}:
        return False

    raise ValueError(
        f"Could not parse boolean value: {value!r}"
    )


def require_columns(
    df: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} missing required columns: "
            + ", ".join(sorted(missing))
        )


def build_worklist(
    metadata: pd.DataFrame,
    refs: pd.DataFrame,
) -> pd.DataFrame:

    merged = metadata.merge(
        refs,
        on=["dataset", "cluster_id"],
        how="left",
        validate="one_to_one",
    )

    if len(merged) != len(metadata):
        raise ValueError(
            "Reference merge changed the metadata row count."
        )

    use_primary = merged[
        "use_for_primary_accuracy"
    ].map(parse_bool)

    primary_type = (
        merged["primary_reference_type"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    has_primary_label = (
        merged["primary_reference_label"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    allowed_type = primary_type.isin(
        ALLOWED_PRIMARY_REFERENCE_TYPES
    )

    eligible = (
        use_primary
        & has_primary_label
        & allowed_type
    )

    out = pd.DataFrame()

   
    # IDENTIFIERS / CURRENT PREDICTION

    out["row_id"] = merged["row_id"]
    out["dataset"] = merged["dataset"]
    out["cluster_id"] = merged["cluster_id"]
    out["agent2_top1_candidate"] = (
        merged["top1_candidate"]
    )

    out["agent2_top1_cell_ontology_ids"] = (
        merged["top1_cell_ontology_ids"]
    )

  
    # PRIMARY EVALUATION REFERENCE

    out["reference_label_raw"] = (
        merged["primary_reference_label"]
    )

    out["reference_type"] = (
        merged["primary_reference_type"]
    )

    out["eligible_for_supervised_router"] = (
        eligible.astype(int)
    )

    # LLM LABELS: COMPARISON ONLY
   
    out["llm_comparison_label"] = (
        merged["llm_label"]
    )

    out["llm_label_role"] = (
        merged["use_for_llm_agreement"]
        .map(parse_bool)
        .map(
            {
                True: "DESCRIPTIVE_COMPARISON_ONLY",
                False: "NOT_USED",
            }
        )
    )

    # HARMONIZATION FIELDS
   
  
    out["reference_label_harmonized"] = ""
    out["reference_cell_ontology_id"] = ""
    out["ontology_relation"] = ""
    out["harmonization_evidence"] = ""
    out["harmonization_reviewer"] = ""
    out["harmonization_status"] = "NOT_REQUIRED"

    out.loc[
        eligible,
        "harmonization_status",
    ] = "PENDING_HARMONIZATION"

    # IMPORTANT:
    # We DO NOT generate agent2_top1_correct here.
    out["agent2_top1_correct"] = pd.NA

    out["correctness_status"] = (
        "NOT_ELIGIBLE_FOR_PRIMARY_TARGET"
    )

    out.loc[
        eligible,
        "correctness_status",
    ] = "PENDING_HARMONIZATION"

    out["notes"] = ""

    no_reference = ~has_primary_label

    out.loc[
        no_reference,
        "notes",
    ] = (
        "No independent primary reference is available. "
        "This row must not be used as a supervised correctness target."
    )

    llm_only = (
        no_reference
        & merged["llm_label"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    out.loc[
        llm_only,
        "notes",
    ] = (
        "LLM-only annotation is available for descriptive agreement, "
        "but it is not ground truth and must not be used to train or "
        "evaluate the primary Router correctness target."
    )

    return out


def build_protocol() -> dict[str, object]:
    return {
        "protocol_version": "reference_harmonization_v1",
        "status": "PRE_SPECIFIED_BEFORE_ROUTER_TARGET_CREATION",
        "purpose": (
            "Create biologically defensible evaluation targets without "
            "using naive exact-string label matching."
        ),
        "primary_reference_requirement": (
            "Only independent biology-expert, published-reference, or "
            "expert-curated labels may define the primary Router target."
        ),
        "llm_label_policy": (
            "LLM-only annotations are descriptive comparison labels only "
            "and cannot define supervised Router correctness."
        ),
        "ontology_framework": (
            "Prefer Cell Ontology identifiers and hierarchical relations "
            "when terms can be mapped."
        ),
        "allowed_ontology_relations": ONTOLOGY_RELATION_OPTIONS,
        "relation_interpretation": {
            "exact": (
                "Prediction and reference represent the same ontology term."
            ),
            "predicted_child_of_reference": (
                "Prediction is a biologically more specific descendant of "
                "the reference label."
            ),
            "predicted_parent_of_reference": (
                "Prediction is a biologically broader ancestor of the "
                "reference label."
            ),
            "sibling": (
                "Prediction and reference are related descendants of a "
                "shared parent but neither contains the other."
            ),
            "biologically_compatible_nonhierarchical": (
                "Expert-reviewed compatibility exists but a clean Cell "
                "Ontology hierarchical relation is unavailable."
            ),
            "no_match": (
                "Prediction and reference are biologically incompatible "
                "at the pre-specified evaluation resolution."
            ),
            "unmapped": (
                "One or both labels cannot currently be mapped reliably."
            ),
        },
        "critical_rule": (
            "The binary agent2_top1_correct target must NOT be generated "
            "until the project defines, documents, and freezes which "
            "ontology relations count as correct for the primary analysis."
        ),
        "recommended_reporting": [
            (
                "Report exact agreement separately from hierarchy-aware "
                "agreement."
            ),
            (
                "Report parent/child/sibling categories rather than hiding "
                "all disagreements inside a single binary accuracy number."
            ),
            (
                "Perform sensitivity analysis under at least one stricter "
                "and one hierarchy-aware correctness definition."
            ),
        ],
        "leakage_rule": (
            "Reference labels and harmonized correctness targets are "
            "evaluation/development metadata only and must never enter "
            "Agent 1, Agent 2, or Router inference features."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build evaluation-only reference harmonization worklist."
        )
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="router_feature_metadata_v1.csv",
    )

    parser.add_argument(
        "--reference-registry",
        type=Path,
        required=True,
        help=(
            "Reference registry containing expert/published and optional "
            "LLM comparison labels."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/analysis/router_reference_harmonization_v1"
        ),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = pd.read_csv(
        args.metadata
    )

    refs = pd.read_csv(
        args.reference_registry
    )

    require_columns(
        metadata,
        REQUIRED_METADATA_COLUMNS,
        "Metadata",
    )

    require_columns(
        refs,
        REQUIRED_REFERENCE_COLUMNS,
        "Reference registry",
    )

    if metadata.duplicated(
        subset=["dataset", "cluster_id"]
    ).any():
        raise ValueError(
            "Duplicate dataset/cluster rows in metadata."
        )

    if refs.duplicated(
        subset=["dataset", "cluster_id"]
    ).any():
        raise ValueError(
            "Duplicate dataset/cluster rows in reference registry."
        )

    worklist = build_worklist(
        metadata,
        refs,
    )

    protocol = build_protocol()

    worklist_file = (
        args.output_dir
        / "router_reference_harmonization_worklist_v1.csv"
    )

    protocol_file = (
        args.output_dir
        / "reference_harmonization_protocol_v1.json"
    )

    audit_file = (
        args.output_dir
        / "reference_harmonization_audit_v1.json"
    )

    worklist.to_csv(
        worklist_file,
        index=False,
    )

    protocol_file.write_text(
        json.dumps(
            protocol,
            indent=2,
        ),
        encoding="utf-8",
    )

    audit = {
        "rows_processed": int(len(worklist)),
        "eligible_primary_reference_rows": int(
            worklist[
                "eligible_for_supervised_router"
            ].sum()
        ),
        "noneligible_rows": int(
            (
                worklist[
                    "eligible_for_supervised_router"
                ] == 0
            ).sum()
        ),
        "datasets_with_primary_reference": sorted(
            worklist.loc[
                worklist[
                    "eligible_for_supervised_router"
                ] == 1,
                "dataset",
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        "router_target_created": False,
        "naive_string_matching_used": False,
        "llm_labels_used_as_ground_truth": False,
        "existing_agent_outputs_modified": False,
    }

    audit_file.write_text(
        json.dumps(
            audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Reference harmonization worklist: {worklist_file}"
    )

    print(
        f"Reference harmonization protocol: {protocol_file}"
    )

    print(
        f"Reference harmonization audit: {audit_file}"
    )

    print(
        f"Rows processed: {len(worklist)}"
    )

    print(
        "Naive exact-string correctness was NOT used."
    )

    print(
        "LLM-only labels were NOT treated as ground truth."
    )

    print(
        "No Router target, model, threshold, or routing decision was created."
    )

    print(
        "No existing Agent output or annotation was changed."
    )


if __name__ == "__main__":
    main()
