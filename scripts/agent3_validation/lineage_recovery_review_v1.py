import argparse
import csv
import json
import hashlib
from pathlib import Path

p = argparse.ArgumentParser()

for arg in ("addendum", "primary", "out"):
    p.add_argument("--" + arg, required=True, type=Path)

p.add_argument("--dataset", required=True)
p.add_argument("--clusters", required=True)

a = p.parse_args()

ids = [str(int(x.strip())) for x in a.clusters.split(",")]

if len(ids) != len(set(ids)) or not ids:
    p.error("Duplicate or empty cluster IDs")

if a.out.exists():
    p.error("Refusing to overwrite existing output")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)

    return h.hexdigest()


def parse_candidate(x):
    q = float(x.get("fdr", x.get("FDR", 1)))
    n = int(x.get("marker_count", 0))
    pct = float(x.get("median_detection_pct", 0))

    genes = x.get("supporting_genes", [])

    if isinstance(genes, str):
        genes = genes.split(";")

    genes = sorted({
        str(g).strip().upper()
        for g in genes if str(g).strip()
    })

    if not (
        0 <= q <= 1
        and 0 <= pct <= 100
        and n >= 2
        and n == len(genes)
    ):
        raise ValueError("Invalid reference evidence")

    label = str(x.get("cell_type", "")).strip()

    if not label:
        raise ValueError("Missing candidate cell type")

    cl = x.get("ontology_ids", []) or []

    if not isinstance(cl, list):
        raise ValueError("Invalid ontology IDs")

    return {
        "label": label,
        "ontology_ids": cl,
        "fdr": q,
        "genes": genes,
        "marker_count": n,
        "median_detection_pct": pct,
        "significant": q <= 0.05 and n >= 3,
        "source": x.get("source", "")
    }


sources = []
results = []

for cid in ids:

    files = [
        a.addendum / f"cluster_{cid}_evidence_addendum.json",

        a.primary / f"cluster_{cid}" / "result" /
        f"cluster_{cid}_agent3_decision_summary.json",

        a.primary / f"cluster_{cid}" / "freeze" /
        f"agent3_{a.dataset}_cluster_{cid}_freeze_manifest.json"
    ]

    for f in files:
        if not f.is_file():
            p.error(f"Missing frozen input: {f}")

    sources += files

    d, core, freeze = map(load, files)

    if (
        str(d.get("cluster")) != cid
        or core.get("decision_state") != "UNRESOLVED"
        or freeze.get("overall_pass") is not True
    ):
        p.error(f"Cluster {cid}: frozen input mismatch")

    if d.get(
        "final_annotation_assigned",
        d.get("final_label_assigned", False)
    ) is not False:
        p.error(f"Cluster {cid}: final annotation already assigned")

    raw = d.get(
        "candidate_evidence",
        d.get("candidates", [])
    )

    if not isinstance(raw, list):
        p.error(f"Cluster {cid}: invalid candidate list")

    candidates = sorted(
        (parse_candidate(x) for x in raw),
        key=lambda x: (
            x["fdr"],
            -x["marker_count"],
            x["label"]
        )
    )

    # Deduplicate equivalent Cell Ontology identities
    by_cl = {}

    for c in candidates:
        if c["significant"] and len(c["ontology_ids"]) == 1:
            by_cl.setdefault(c["ontology_ids"][0], c)

    mapped = sorted(
        by_cl.values(),
        key=lambda x: (x["fdr"], -x["marker_count"])
    )

    # Find up to two distinct supported identities
    chosen = []

    for c in mapped:

        if not chosen:
            chosen.append(c)

        elif len(chosen) == 1:
            s = set(chosen[0]["genes"])
            t = set(c["genes"])

            if len(s - t) >= 2 and len(t - s) >= 2:
                chosen.append(c)

        if len(chosen) == 2:
            break

    # Standardized evidence-review rules
    if len(chosen) == 2:

        if any(
            x["median_detection_pct"] < 10
            for x in chosen
        ):
            status = "COMPETING_SIGNATURES_LOW_DETECTION_REVIEW"
        else:
            status = "COMPETING_SIGNATURES_REVIEW"

    elif len(chosen) == 1:

        if chosen[0]["median_detection_pct"] < 10:
            status = "SINGLE_SIGNATURE_LOW_DETECTION_REVIEW"
        else:
            status = "SINGLE_SIGNATURE_REVIEW"

    elif any(c["significant"] for c in candidates):
        status = "SIGNIFICANT_BUT_UNMAPPED_OR_OVERLAPPING"

    else:
        status = (
            "NON_SIGNIFICANT_CANDIDATE_ONLY"
            if candidates else "NO_CANDIDATES"
        )

    hypotheses = chosen if chosen else candidates[:1]

    results.append({
        "cluster": cid,
        "frozen_agent3": "UNRESOLVED",
        "review_status": status,
        "provisional_candidates": [
            x["label"] for x in hypotheses
        ],
        "evidence": hypotheses,
        "mapped_enriched_ontology_ids": len(by_cl),
        "final_annotation": None,
        "decision_changed": False
    })


# All inputs validated before writing results
a.out.mkdir(parents=True, exist_ok=False)

for r in results:

    file = a.out / f"cluster_{r['cluster']}_review.json"

    with file.open("x", encoding="utf-8") as f:
        json.dump(r, f, indent=2)


with (a.out / "summary.csv").open(
    "x", newline="", encoding="utf-8"
) as f:

    columns = [
        "cluster",
        "frozen_agent3",
        "review_status",
        "provisional_candidates",
        "mapped_enriched_ontology_ids"
    ]

    w = csv.DictWriter(f, fieldnames=columns)
    w.writeheader()

    for r in results:
        w.writerow({
            k: (
                "; ".join(r[k])
                if k == "provisional_candidates"
                else r[k]
            )
            for k in columns
        })


with (a.out / "manifest.json").open(
    "x", encoding="utf-8"
) as f:

    json.dump({
        "method": "frozen_lineage_recovery_review_v1",
        "dataset": a.dataset,
        "inputs_sha256": {
            str(s): sha(s) for s in sources
        },
        "fdr": 0.05,
        "min_overlap": 3,
        "low_detection_pct": 10,
        "min_distinct_marker_genes_each": 2,
        "original_results_modified": False,
        "final_annotations_assigned": False
    }, f, indent=2)


print("[PASS] Frozen Agent 3 results preserved")

for r in results:
    print(
        "Cluster {} | {} | {}".format(
            r["cluster"],
            r["review_status"],
            "; ".join(r["provisional_candidates"])
        )
    )

print("Saved:", a.out)