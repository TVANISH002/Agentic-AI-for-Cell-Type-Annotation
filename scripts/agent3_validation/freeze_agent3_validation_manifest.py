from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


FILES = {
    #
    # Cluster 6 final evidence
    #
    "cluster6_pubmedbert": Path(
        "data/analysis/agent3_validation/"
        "pubmedbert_marker_sensitivity_25/"
        "cluster_6_pubmedbert_results.json"
    ),

    "cluster6_evidence_audit": Path(
        "data/analysis/agent3_validation/"
        "cluster6_pubmedbert_evidence_audit/"
        "cluster_6_pubmedbert_evidence_audit.json"
    ),

    "cluster6_final_packet": Path(
        "data/analysis/agent3_validation/"
        "cluster6_adjudication_final/"
        "cluster_6_adjudication_evidence_packet.json"
    ),

    #
    # Cluster 7 final evidence
    #
    "cluster7_gene_terminology": Path(
        "data/analysis/agent3_validation/"
        "cluster7_rescue_gene_terminology/"
        "cluster_7_rescue_gene_terminology.json"
    ),

    "cluster7_grounding_audit": Path(
        "data/analysis/agent3_validation/"
        "cluster7_rescue_grounding_audit/"
        "cluster_7_rescue_grounding_audit.json"
    ),

    "cluster7_pubmedbert": Path(
        "data/analysis/agent3_validation/"
        "cluster7_rescue_confirmation_pubmedbert/"
        "cluster_7_rescue_confirmation_pubmedbert_results.json"
    ),

    "cluster7_final_packet": Path(
        "data/analysis/agent3_validation/"
        "cluster7_rescue_final/"
        "cluster_7_rescue_evidence_packet.json"
    ),

    #
    # Validation scripts defining frozen behavior
    #
    "cluster6_packet_builder": Path(
        "scripts/agent3_validation/"
        "build_cluster6_adjudication_evidence_packet.py"
    ),

    "cluster7_rescue_query_builder": Path(
        "scripts/agent3_validation/"
        "build_marker_rescue_queries.py"
    ),

    "cluster7_candidate_discovery": Path(
        "scripts/agent3_validation/"
        "discover_rescue_candidates.py"
    ),

    "cluster7_confirmation_query_builder": Path(
        "scripts/agent3_validation/"
        "build_rescue_confirmation_queries.py"
    ),

    "cluster7_confirmation_pubmedbert": Path(
        "scripts/agent3_validation/"
        "run_rescue_confirmation_pubmedbert.py"
    ),

    "cluster7_grounding_audit_script": Path(
        "scripts/agent3_validation/"
        "audit_rescue_confirmation_grounding.py"
    ),

    "cluster7_packet_builder": Path(
        "scripts/agent3_validation/"
        "build_rescue_evidence_packet.py"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )


missing = [
    str(path)
    for path in FILES.values()
    if not path.exists()
]

if missing:
    raise RuntimeError(
        "Missing frozen files:\n"
        + "\n".join(missing)
    )


c6 = load_json(
    FILES["cluster6_final_packet"]
)

c7 = load_json(
    FILES["cluster7_final_packet"]
)


#
# --------------------------------------------------
# Scientific invariants
# --------------------------------------------------
#

checks = {
    "cluster6_id_is_6":
        str(c6.get("cluster_id")) == "6",

    "cluster7_id_is_7":
        str(c7.get("cluster_id")) == "7",

    "cluster6_reference_label_unused":
        c6.get("reference_label_used") is False,

    "cluster6_expert_label_unused":
        c6.get("expert_label_used") is False,

    "cluster7_reference_label_unused":
        c7.get("reference_label_used") is False,

    "cluster7_expert_label_unused":
        c7.get("expert_label_used") is False,

    "cluster6_acceptance_uncalibrated":
        c6.get("automatic_acceptance")
        == "NOT_CALIBRATED",

    "cluster7_acceptance_uncalibrated":
        c7.get("automatic_acceptance")
        == "NOT_CALIBRATED",

    "cluster6_candidate_count_is_3":
        int(c6.get("candidate_count", 0)) == 3,

    "cluster7_candidate_count_is_48":
        int(c7.get("candidate_count", 0)) == 48,

    "cluster6_top_markers_is_25":
        int(c6.get("top_marker_count", 0)) == 25,

    "cluster6_top_k_is_10":
        int(c6.get("top_k_passages", 0)) == 10,

    "cluster6_model_is_pubmedbert":
        c6.get("model_id")
        == "NeuML/pubmedbert-base-embeddings",
}


failed = [
    key
    for key, value in checks.items()
    if not value
]

if failed:
    raise RuntimeError(
        "Final Agent 3 invariant failure:\n"
        + "\n".join(failed)
    )


#
# --------------------------------------------------
# Extract compact frozen summaries
# --------------------------------------------------
#

cluster6_candidates = []

for item in c6.get(
    "candidates",
    [],
):
    cluster6_candidates.append(
        {
            "candidate":
                item.get("candidate"),

            "mean_similarity":
                item.get(
                    "pubmedbert_mean_similarity"
                ),

            "supporting_genes_retrieved":
                (
                    f"{item.get('distinct_supporting_markers_retrieved')}"
                    f"/"
                    f"{item.get('candidate_supporting_gene_count')}"
                ),

            "candidate_supporting_marker_passages":
                item.get(
                    "candidate_and_supporting_marker_passages"
                ),
        }
    )


mast = next(
    (
        item
        for item in c7.get(
            "candidates",
            [],
        )
        if str(
            item.get(
                "candidate",
                ""
            )
        ).casefold()
        == "mast cell"
    ),
    None,
)

if mast is None:
    raise RuntimeError(
        "Mast-cell record missing from "
        "Cluster 7 final packet."
    )


cluster7_mast_summary = {
    "candidate":
        mast.get("candidate"),

    "ontology_id":
        mast.get("ontology_id"),

    "is_novel_to_agent2":
        mast.get(
            "is_novel_to_agent2"
        ),

    "candidate_marker_papers":
        mast.get(
            "confirmation_candidate_marker_papers"
        ),

    "same_sentence_genes":
        mast.get(
            "same_sentence_linked_genes"
        ),

    "adjacent_genes":
        mast.get(
            "adjacent_sentence_linked_genes"
        ),

    "within_three_genes":
        mast.get(
            "within_three_sentence_linked_genes"
        ),

    "target_context_candidate_papers":
        mast.get(
            "target_context_candidate_papers"
        ),

    "target_context_marker_papers":
        mast.get(
            "target_context_candidate_marker_papers"
        ),

    "pubmedbert_unique_papers":
        mast.get(
            "pubmedbert_eligible_unique_papers"
        ),

    "pubmedbert_mean_similarity":
        mast.get(
            "pubmedbert_mean_similarity"
        ),

    "reporting_state":
        mast.get(
            "reporting_state"
        ),

    "automatic_acceptance":
        mast.get(
            "automatic_acceptance"
        ),
}


#
# --------------------------------------------------
# Hash manifest
# --------------------------------------------------
#

artifacts = {}

for name, path in FILES.items():
    artifacts[name] = {
        "path":
            str(path),

        "sha256":
            sha256(path),

        "size_bytes":
            path.stat().st_size,
    }


manifest = {
    "schema_name":
        "Agent3FrozenValidationManifest",

    "schema_version":
        "1.0",

    "created_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "scientific_invariants":
        checks,

    "cluster6": {
        "behavior":
            "candidate_adjudication",

        "status":
            c6.get(
                "adjudication_status"
            ),

        "automatic_acceptance":
            c6.get(
                "automatic_acceptance"
            ),

        "candidates":
            cluster6_candidates,
    },

    "cluster7": {
        "behavior":
            "open_set_candidate_rescue",

        "mast_cell":
            cluster7_mast_summary,
    },

    "artifacts":
        artifacts,
}


output = Path(
    "data/analysis/agent3_validation/"
    "agent3_frozen_manifest/"
    "agent3_frozen_validation_manifest.json"
)

output.parent.mkdir(
    parents=True,
    exist_ok=True,
)

output.write_text(
    json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


print(
    "FINAL AGENT 3 FREEZE AUDIT: PASS"
)

print("=" * 100)

print(
    "Scientific invariants:",
    len(checks),
    "/",
    len(checks),
    "PASS",
)

print()

print(
    "Cluster 6:",
    c6.get(
        "adjudication_status"
    )
)

for item in cluster6_candidates:
    print(
        " ",
        item
    )

print()

print(
    "Cluster 7 Mast cell:"
)

for key, value in (
    cluster7_mast_summary.items()
):
    print(
        f"  {key}: {value}"
    )

print()

print(
    "Hashed frozen artifacts:",
    len(artifacts)
)

print(
    "Manifest:",
    output
)
