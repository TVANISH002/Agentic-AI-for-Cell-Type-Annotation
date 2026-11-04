from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def audit_dataset(
    *,
    path: Path,
    dataset: str,
    cluster_col: str,
    gene_col: str,
    padj_col: str,
    logfc_col: str,
    expected_count: int | None,
) -> tuple[dict[str, object], pd.DataFrame]:

    df = pd.read_csv(path)

    required = {
        cluster_col,
        gene_col,
        padj_col,
        logfc_col,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{dataset}: missing columns: {sorted(missing)}"
        )

    working = df.copy()

    working[cluster_col] = pd.to_numeric(
        working[cluster_col],
        errors="raise",
    ).astype(int)

    working[padj_col] = pd.to_numeric(
        working[padj_col],
        errors="coerce",
    )

    working[logfc_col] = pd.to_numeric(
        working[logfc_col],
        errors="coerce",
    )

    cluster_ids = sorted(
        working[cluster_col]
        .dropna()
        .unique()
        .tolist()
    )

    # Rule written in the manuscript/current methods description.
    manuscript_eligible = working.loc[
        (working[padj_col] <= 0.05)
        & (working[logfc_col] >= 0.5)
    ].copy()

    manuscript_top50 = (
        manuscript_eligible
        .groupby(
            cluster_col,
            group_keys=False,
            sort=True,
        )
        .head(50)
        .copy()
    )

 
    # Diagnostic alternative:
    # p_adj only -> top 50 -> remove BLANK probes.
    #
    # This is NOT declared to be the correct method. It is tested only
    # because it helps explain the existing breast count.
    padj_eligible = working.loc[
        working[padj_col] <= 0.05
    ].copy()

    padj_top50 = (
        padj_eligible
        .groupby(
            cluster_col,
            group_keys=False,
            sort=True,
        )
        .head(50)
        .copy()
    )

    blank_mask = (
        padj_top50[gene_col]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.startswith("BLANK")
    )

    padj_top50_no_blank = (
        padj_top50.loc[
            ~blank_mask
        ].copy()
    )

    duplicate_cluster_gene = int(
        working.duplicated(
            subset=[
                cluster_col,
                gene_col,
            ]
        ).sum()
    )

    result = {
        "dataset": dataset,
        "source_file": str(path),
        "raw_rows": int(len(working)),
        "cluster_count": int(
            working[cluster_col].nunique()
        ),
        "cluster_ids": [
            int(x)
            for x in cluster_ids
        ],
        "missing_gene_names": int(
            working[gene_col].isna().sum()
        ),
        "duplicate_cluster_gene_rows": (
            duplicate_cluster_gene
        ),
        "manuscript_rule": {
            "padj_max": 0.05,
            "log2fc_min": 0.5,
            "max_markers_per_cluster": 50,
            "eligible_before_top50": int(
                len(manuscript_eligible)
            ),
            "selected_after_top50": int(
                len(manuscript_top50)
            ),
            "selected_per_cluster": {
                str(int(k)): int(v)
                for k, v in (
                    manuscript_top50
                    .groupby(cluster_col)
                    .size()
                    .to_dict()
                    .items()
                )
            },
        },
        "diagnostic_padj_only_top50": {
            "padj_max": 0.05,
            "max_markers_per_cluster": 50,
            "selected_before_blank_removal": int(
                len(padj_top50)
            ),
            "blank_probe_rows": int(
                blank_mask.sum()
            ),
            "blank_probe_names": (
                padj_top50.loc[
                    blank_mask,
                    gene_col,
                ]
                .astype(str)
                .tolist()
            ),
            "selected_after_blank_removal": int(
                len(padj_top50_no_blank)
            ),
        },
        "expected_current_pipeline_count": (
            expected_count
        ),
        "manuscript_rule_matches_expected": (
            None
            if expected_count is None
            else int(
                len(manuscript_top50)
            )
            == int(expected_count)
        ),
        "diagnostic_padj_only_no_blank_matches_expected": (
            None
            if expected_count is None
            else int(
                len(padj_top50_no_blank)
            )
            == int(expected_count)
        ),
    }

    selected = manuscript_top50[
        [
            cluster_col,
            gene_col,
            padj_col,
            logfc_col,
        ]
    ].copy()

    selected.insert(
        0,
        "dataset",
        dataset,
    )

    selected = selected.rename(
        columns={
            cluster_col: "cluster_id",
            gene_col: "gene",
            padj_col: "adjusted_p_value",
            logfc_col: "log2_fold_change",
        }
    )

    return result, selected


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--breast-markers",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--lung-markers",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/analysis/pipeline_validation/input_markers"
        ),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    breast_result, breast_selected = audit_dataset(
        path=args.breast_markers,
        dataset="breast",
        cluster_col="spatial_cluster",
        gene_col="names",
        padj_col="pvals_adj",
        logfc_col="logfoldchanges",
        expected_count=399,
    )

    lung_result, lung_selected = audit_dataset(
        path=args.lung_markers,
        dataset="lung",
        cluster_col="cluster",
        gene_col="gene",
        padj_col="p_val_adj",
        logfc_col="avg_log2FC",
        expected_count=456,
    )

    report = {
        "audit_role": (
            "READ_ONLY_INPUT_MARKER_REPRODUCIBILITY_AUDIT"
        ),
        "datasets": [
            breast_result,
            lung_result,
        ],
        "existing_outputs_modified": False,
        "important_interpretation": (
            "A mismatch means the written preprocessing rule does not "
            "reproduce the currently expected marker count. It does not "
            "by itself identify which historical preprocessing rule was "
            "actually used."
        ),
    }

    json_path = (
        args.output_dir
        / "input_marker_preprocessing_audit_v1.json"
    )

    csv_path = (
        args.output_dir
        / "markers_recomputed_under_stated_rule_v1.csv"
    )

    json_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    pd.concat(
        [
            breast_selected,
            lung_selected,
        ],
        ignore_index=True,
    ).to_csv(
        csv_path,
        index=False,
    )

    print(
        f"Input marker audit: {json_path}"
    )

    print(
        f"Recomputed stated-rule markers: {csv_path}"
    )

    for result in report["datasets"]:
        print()
        print(
            f"{result['dataset'].upper()}:"
        )
        print(
            f"  Raw rows: {result['raw_rows']}"
        )
        print(
            f"  Clusters: {result['cluster_count']}"
        )
        print(
            "  Missing genes: "
            f"{result['missing_gene_names']}"
        )
        print(
            "  Duplicate cluster/gene rows: "
            f"{result['duplicate_cluster_gene_rows']}"
        )
        print(
            "  Stated rule selected: "
            f"{result['manuscript_rule']['selected_after_top50']}"
        )
        print(
            "  Expected current count: "
            f"{result['expected_current_pipeline_count']}"
        )
        print(
            "  Stated rule reproduces expected count: "
            f"{result['manuscript_rule_matches_expected']}"
        )
        print(
            "  p_adj-only top50 after BLANK removal: "
            f"{result['diagnostic_padj_only_top50']['selected_after_blank_removal']}"
        )

    print()
    print(
        "No existing Agent output, Router output, or annotation was changed."
    )


if __name__ == "__main__":
    main()
