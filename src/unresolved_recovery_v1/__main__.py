"""Conservative unresolved-case recovery. Never overwrites core decisions."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()

p.add_argument("--root", type=Path, required=True)
p.add_argument("--dataset", required=True)
p.add_argument("--router-features", type=Path, required=True)
p.add_argument("--clusters", required=True)
p.add_argument("--out", type=Path, required=True)

a = p.parse_args()

ids = [int(x.strip()) for x in a.clusters.split(",")]

if not ids or len(set(ids)) != len(ids):
    p.error("Duplicate or empty cluster IDs")

if a.out.exists():
    p.error("Output already exists. Refusing overwrite.")

if a.out.resolve().parent != a.root.resolve():
    p.error("Output must be a new direct child of the dataset root")


def load(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}")

    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)

    return h.hexdigest()


# Read-only Router features: never use as new annotations

feature_columns = (
    "direct_evidence_coverage",
    "top1_support_fraction",
    "top1_top2_support_gap",
    "cross_database_support_fraction",
    "first_direct_supported_marker_rank_norm",
    "rank_weighted_unexplained_marker_score"
)

rf = rows(a.router_features)

router_ids = [int(x["cluster_id"]) for x in rf]

if (
    len(router_ids) != len(set(router_ids))
    or not set(ids).issubset(set(router_ids))
):
    raise ValueError("Missing or duplicate Router cluster IDs")

router = {int(x["cluster_id"]): x for x in rf}

v2file = a.root / "lineage_recovery_v2" / "all_clusters_summary.csv"
v2rows = rows(v2file)

used = [a.router_features, v2file]
results = []


# Evaluate only requested clusters

for cid in ids:

    sources = {
        "addendum":
            a.root / "agent3_recovery_addendum_v1" /
            f"cluster_{cid}_evidence_addendum.json",

        "review":
            a.root / "lineage_review_v1" /
            f"cluster_{cid}_review.json",

        "agent2_candidates":
            a.root / "agent2_output" / "clusters" /
            f"cluster_{cid}" /
            f"cluster_{cid}_ranked_candidate_cell_types.csv",

        "core":
            a.root / "agent3_output" / "primary" /
            f"cluster_{cid}" / "result" /
            f"cluster_{cid}_agent3_decision_summary.json",

        "freeze":
            a.root / "agent3_output" / "primary" /
            f"cluster_{cid}" / "freeze" /
            f"agent3_{a.dataset}_cluster_{cid}_freeze_manifest.json"
    }

    used.extend(sources.values())

    addendum = load(sources["addendum"])
    review = load(sources["review"])
    core = load(sources["core"])
    freeze = load(sources["freeze"])

    candidates = rows(sources["agent2_candidates"])

    # Verify frozen inputs
    if (
        str(addendum.get("cluster")) != str(cid)
        or str(review.get("cluster")) != str(cid)
        or core.get("decision_state") != "UNRESOLVED"
        or freeze.get("overall_pass") is not True
    ):
        raise ValueError(
            f"Cluster {cid}: frozen input mismatch"
        )

    if not isinstance(addendum.get("candidate_evidence"), list):
        raise ValueError(
            f"Cluster {cid}: recovery addendum schema mismatch"
        )

    matches = [
        x for x in v2rows
        if int(x["cluster"]) == cid
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Cluster {cid}: V2 evidence missing or duplicated"
        )

    significant_degs = int(matches[0]["significant_DEGs"])

    selected_count = sum(
        str(r.get("selected_for_downstream", "")).lower() == "true"
        for r in candidates
    )

    # Existing candidate evidence: no hardcoded cell-type labels

    chosen = []

    for c in review.get("evidence", []):

        q = float(c["fdr"])
        pct = float(c["median_detection_pct"])

        genes = {
            str(g).strip().upper()
            for g in c.get("genes", [])
            if str(g).strip()
        }

        if not (
            0 <= q <= 1
            and 0 <= pct <= 100
            and len(genes) >= 1
        ):
            raise ValueError(
                f"Cluster {cid}: invalid marker evidence"
            )

        ontology_ids = c.get("ontology_ids", [])

        chosen.append({
            "label": c["label"],
            "cell_ontology_ids": ontology_ids,
            "fdr": q,
            "median_detection_pct": pct,
            "genes": sorted(genes),
            "enriched": q <= 0.05 and len(genes) >= 3
        })

    mapped_enriched = [
        x for x in chosen
        if x["enriched"]
        and len(x["cell_ontology_ids"]) == 1
    ]

    # Check for independent evidence from two distinct identities
    independent = False

    if len(mapped_enriched) >= 2:

        x, y = mapped_enriched[:2]

        gx = set(x["genes"])
        gy = set(y["genes"])

        independent = (
            len(gx - gy) >= 2
            and len(gy - gx) >= 2
            and x["cell_ontology_ids"][0]
                != y["cell_ontology_ids"][0]
        )

    # Standardized conservative review rules

    if significant_degs <= 4 and not mapped_enriched:

        status = "SPARSE_MARKER_ABSTENTION"

    elif independent:

        if any(
            x["median_detection_pct"] < 10
            for x in mapped_enriched[:2]
        ):
            status = "COMPETING_LOW_DETECTION_REVIEW"
        else:
            status = "COMPETING_SIGNATURE_REVIEW"

    elif len(mapped_enriched) == 1:

        status = "SINGLE_SIGNATURE_REVIEW"

    elif chosen:

        status = "CANDIDATE_ONLY_INSUFFICIENT_EVIDENCE"

    else:

        status = "NO_CANDIDATE_ABSTENTION"

    # Preserve six frozen Router diagnostics

    router_row = router[cid]

    six = {
        key: float(router_row[key])
        for key in feature_columns
    }

    if any(not 0 <= v <= 1 for v in six.values()):
        raise ValueError(
            f"Cluster {cid}: invalid Router feature values"
        )

    results.append({
        "cluster": cid,
        "original_decision": "UNRESOLVED",
        "recovery_review_state": status,
        "candidate_labels": [
            x["label"] for x in chosen
        ],
        "reference_enrichment": chosen,
        "significant_degs": significant_degs,
        "selected_agent2_candidates": selected_count,
        "zero_candidate_abstention": selected_count == 0,
        "router_features_exploratory": six,
        "router_top2_absent":
            str(router_row["top2_missing"]).lower() == "true",
        "new_final_annotation": None,
        "original_decision_changed": False
    })

# Only create output after ALL input checks succeed

a.out.mkdir(parents=True, exist_ok=False)

for result in results:

    path = a.out / f"cluster_{result['cluster']}.json"

    with path.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

summary_path = a.out / "summary.csv"

with summary_path.open(
    "x", encoding="utf-8", newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "cluster",
        "original_decision",
        "recovery_review_state",
        "candidate_labels",
        "significant_degs",
        "eligible_agent2_candidates",
        "router_f1",
        "router_f4",
        "router_f6",
        "router_top2_absent"
    ])

    for result in results:

        features = result["router_features_exploratory"]

        writer.writerow([
            result["cluster"],
            result["original_decision"],
            result["recovery_review_state"],
            "; ".join(result["candidate_labels"]),
            result["significant_degs"],
            result["selected_agent2_candidates"],
            features[feature_columns[0]],
            features[feature_columns[3]],
            features[feature_columns[5]],
            result["router_top2_absent"]
        ])

# Reproducibility record
manifest = {
    "method": "isolated_unresolved_recovery_v1",
    "dataset": a.dataset,
    "clusters": ids,
    "criteria": {
        "max_fdr": 0.05,
        "min_overlapping_markers": 3,
        "sparse_deg_max": 4,
        "independent_markers_per_lineage": 2,
        "low_detection_pct": 10
    },
    "router_features_used_for_annotation": False,
    "top2_absence_policy": "RETAIN_EXPLORATORY_FLAG_ONLY",
    "source_sha256": {
        str(path): sha(path) for path in used
    },
    "original_files_modified": False,
    "new_final_annotations_assigned": False
}

with (a.out / "manifest.json").open(
    "x", encoding="utf-8"
) as f:
    json.dump(manifest, f, indent=2)

print("\n========== ISOLATED RECOVERY RESULTS ==========")

for result in results:

    print(
        f"Cluster {result['cluster']}: "
        f"{result['recovery_review_state']} | "
        f"candidates: "
        f"{', '.join(result['candidate_labels']) or 'none'} | "
        f"Agent2 selected: "
        f"{result['selected_agent2_candidates']}"
    )

print("\nSaved:", a.out)
print("Original Agent 1/2/3/Router results were not written.")





