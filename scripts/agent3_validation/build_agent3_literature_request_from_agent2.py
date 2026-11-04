from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def clean(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, float) and math.isnan(value):
        return ""

    return str(value).strip()


def none_if_nan(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, float) and math.isnan(value):
        return None

    if pd.isna(value):
        return None

    return value


def split_pipe(value: Any) -> list[str]:
    text = clean(value)

    if not text:
        return []

    return [
        item.strip()
        for item in text.split("|")
        if item.strip()
    ]


def scalar_number(value: Any) -> Any:
    value = none_if_nan(value)

    if value is None:
        return None

    try:
        if isinstance(value, str) and not value.strip():
            return None

        f = float(value)

        if math.isnan(f):
            return None

        if f.is_integer():
            return int(f)

        return f

    except Exception:
        return value


def infer_default_paths(
    dataset: str,
    cluster_id: str,
) -> dict[str, Path]:
    dataset_key = clean(dataset).lower()

    if dataset_key == "lung":
        base = Path(
            r"data\pipeline_runs\lung_validated_clean"
        )
    elif dataset_key == "breast":
        base = Path(
            r"data\pipeline_runs\breast_xenium_validated_clean"
        )
    else:
        raise ValueError(
            "For automatic path inference, dataset must be 'lung' or 'breast'. "
            "Otherwise pass --marker-file and --candidate-file explicitly."
        )

    return {
        "marker_file": (
            base
            / "processed"
            / f"{dataset_key}_cluster_markers_pipeline.csv"
        ),
        "candidate_file": (
            base
            / "agent2_output"
            / "clusters"
            / f"cluster_{cluster_id}"
            / f"cluster_{cluster_id}_selected_candidate_cell_types.csv"
        ),
        "ranked_candidate_file": (
            base
            / "agent2_output"
            / "clusters"
            / f"cluster_{cluster_id}"
            / f"cluster_{cluster_id}_ranked_candidate_cell_types.csv"
        ),
        "router_file": (
            base
            / "router_output_v4"
            / "router_decisions.csv"
        ),
    }


def build_marker_genes(
    marker_file: Path,
    cluster_id: str,
    max_markers: int,
) -> list[dict[str, Any]]:
    df = pd.read_csv(marker_file)

    if "cluster_id" not in df.columns:
        raise ValueError(
            f"marker_file missing cluster_id column: {marker_file}"
        )

    cluster_df = df[
        df["cluster_id"].astype(str) == str(cluster_id)
    ].copy()

    if cluster_df.empty:
        raise ValueError(
            f"No markers found for cluster {cluster_id} in {marker_file}"
        )

    rank_col = (
        "rank_within_cluster"
        if "rank_within_cluster" in cluster_df.columns
        else "cluster_gene_rank"
    )

    if rank_col in cluster_df.columns:
        cluster_df = cluster_df.sort_values(
            by=rank_col,
            kind="stable",
        )

    if max_markers > 0:
        cluster_df = cluster_df.head(max_markers)

    markers = []

    for i, row in enumerate(
        cluster_df.to_dict(orient="records"),
        start=1,
    ):
        rank = row.get(
            "rank_within_cluster",
            row.get("cluster_gene_rank", i),
        )

        markers.append(
            {
                "gene": clean(row.get("gene")),
                "rank_within_cluster": int(scalar_number(rank) or i),
                "score": scalar_number(row.get("score")),
                "log_fold_change": scalar_number(row.get("log_fold_change")),
                "p_value": scalar_number(row.get("p_value")),
                "adjusted_p_value": scalar_number(row.get("adjusted_p_value")),
                "pct_nz_group": scalar_number(row.get("pct_nz_group")),
                "pct_nz_reference": scalar_number(row.get("pct_nz_reference")),
            }
        )

    markers = [
        marker
        for marker in markers
        if marker["gene"]
    ]

    if not markers:
        raise ValueError(
            f"No valid marker genes found for cluster {cluster_id}."
        )

    return markers


def build_candidate_annotations(
    candidate_file: Path,
    cluster_id: str,
    candidate_count: int,
) -> list[dict[str, Any]]:
    df = pd.read_csv(candidate_file)

    if "cluster_id" in df.columns:
        df = df[
            df["cluster_id"].astype(str) == str(cluster_id)
        ].copy()

    if df.empty:
        raise ValueError(
            f"No candidates found for cluster {cluster_id} in {candidate_file}"
        )

    rank_col = (
        "candidate_rank"
        if "candidate_rank" in df.columns
        else None
    )

    if rank_col is not None:
        df = df.sort_values(
            by=rank_col,
            kind="stable",
        )

    if candidate_count > 0:
        df = df.head(candidate_count)

    candidates = []

    for i, row in enumerate(
        df.to_dict(orient="records"),
        start=1,
    ):
        name = clean(
            row.get("cell_type", row.get("name"))
        )

        if not name:
            continue

        rank = row.get(
            "candidate_rank",
            i,
        )

        candidates.append(
            {
                "name": name,
                "upstream_rank": int(scalar_number(rank) or i),
                "supporting_genes": split_pipe(
                    row.get("supporting_genes")
                ),
                "evidence_sources": split_pipe(
                    row.get("evidence_sources")
                ),
            }
        )

    if not candidates:
        raise ValueError(
            f"No valid candidate annotations found for cluster {cluster_id}."
        )

    return candidates


def build_review_flags(
    router_file: Path | None,
    cluster_id: str,
) -> dict[str, Any]:
    if router_file is None or not router_file.exists():
        return {
            "router_available": False,
        }

    df = pd.read_csv(router_file)

    if "cluster_id" not in df.columns:
        return {
            "router_available": False,
            "router_warning": "router file missing cluster_id column",
        }

    row_df = df[
        df["cluster_id"].astype(str) == str(cluster_id)
    ]

    if row_df.empty:
        return {
            "router_available": False,
            "router_warning": f"cluster {cluster_id} not found in router file",
        }

    row = row_df.iloc[0].to_dict()

    return {
        "router_available": True,
        "router_decision": clean(row.get("router_decision")),
        "human_review_required": bool(row.get("human_review_required")),
        "router_confidence_score": scalar_number(
            row.get("router_confidence_score")
        ),
        "router_confidence_level": clean(
            row.get("router_confidence_level")
        ),
        "router_reasons": split_pipe(
            row.get("router_reasons")
        ),
        "critical_reasons": split_pipe(
            row.get("critical_reasons")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a cluster-specific Agent 3 LiteratureRequest JSON "
            "from Agent 2 marker and candidate outputs."
        )
    )

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cluster-id", required=True)
    parser.add_argument("--species", required=True)
    parser.add_argument("--tissue", required=True)
    parser.add_argument("--disease-context", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--output-json", required=True, type=Path)

    parser.add_argument("--marker-file", default=None, type=Path)
    parser.add_argument("--candidate-file", default=None, type=Path)
    parser.add_argument("--ranked-candidate-file", default=None, type=Path)
    parser.add_argument("--router-file", default=None, type=Path)

    parser.add_argument(
        "--candidate-source",
        default="selected",
        choices=["selected", "ranked"],
        help=(
            "Use selected Agent 2 candidates by default. "
            "Use ranked only for documented sensitivity checks."
        ),
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=0,
        help="0 means keep all candidates in the chosen candidate file.",
    )
    parser.add_argument(
        "--max-markers",
        type=int,
        default=50,
        help="Maximum ranked cluster marker genes to include.",
    )

    args = parser.parse_args()

    cluster_id = str(args.cluster_id)

    defaults = infer_default_paths(
        args.dataset,
        cluster_id,
    )

    marker_file = args.marker_file or defaults["marker_file"]

    if args.candidate_source == "selected":
        candidate_file = args.candidate_file or defaults["candidate_file"]
    else:
        candidate_file = (
            args.ranked_candidate_file
            or defaults["ranked_candidate_file"]
        )

    router_file = args.router_file or defaults["router_file"]

    for label, path in [
        ("marker_file", marker_file),
        ("candidate_file", candidate_file),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found: {path}"
            )

    marker_genes = build_marker_genes(
        marker_file=marker_file,
        cluster_id=cluster_id,
        max_markers=args.max_markers,
    )

    candidate_annotations = build_candidate_annotations(
        candidate_file=candidate_file,
        cluster_id=cluster_id,
        candidate_count=args.candidate_count,
    )

    review_flags = build_review_flags(
        router_file=router_file,
        cluster_id=cluster_id,
    )

    question = (
        f"Using literature evidence, evaluate the candidate cell-type "
        f"annotations for cluster {cluster_id} in human {args.tissue} "
        f"under {args.disease_context} context. Preserve uncertainty "
        f"when evidence is incomplete."
    )

    request = {
        "schema_name": "Agent3LiteratureRequest",
        "schema_version": "1.0",
        "cluster_id": cluster_id,
        "species": args.species,
        "tissue": args.tissue,
        "disease_context": args.disease_context,
        "platform": args.platform,
        "marker_genes": marker_genes,
        "candidate_annotations": candidate_annotations,
        "review_flags": review_flags,
        "spatial_context": None,
        "question": question,
        "upstream_schema_name": "Agent2ToAgent3Handoff",
        "upstream_schema_version": "1.0",
        "source_agent2_file": str(candidate_file),
        "source_marker_file": str(marker_file),
        "source_router_file": str(router_file) if router_file else None,
        "reference_label_used": False,
        "expert_label_used": False,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "builder_note": (
            "This request was generated from Agent 2 marker/candidate outputs. "
            "No reference or expert annotation label is used."
        ),
    }

    args.output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output_json.write_text(
        json.dumps(
            request,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 90)
    print("AGENT 3 LITERATURE REQUEST BUILDER")
    print("=" * 90)
    print("Output:", args.output_json)
    print("Dataset:", args.dataset)
    print("Cluster:", cluster_id)
    print("Species:", args.species)
    print("Tissue:", args.tissue)
    print("Disease context:", args.disease_context)
    print("Platform:", args.platform)
    print("Marker file:", marker_file)
    print("Candidate file:", candidate_file)
    print("Router file:", router_file)
    print("Marker count:", len(marker_genes))
    print("Candidate count:", len(candidate_annotations))
    print("Candidates:")
    for cand in candidate_annotations:
        print(
            f" - rank {cand['upstream_rank']}: "
            f"{cand['name']} "
            f"({', '.join(cand['supporting_genes'])})"
        )
    print("Reference label used:", request["reference_label_used"])
    print("Expert label used:", request["expert_label_used"])
    print("=" * 90)


if __name__ == "__main__":
    main()
