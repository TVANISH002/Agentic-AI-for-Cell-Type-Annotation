"""Frozen Agent 3 full-DEG evidence addendum. No automatic labels."""

import argparse
import csv
import json
import hashlib
from pathlib import Path
from collections import defaultdict

p = argparse.ArgumentParser()

for field in ("deg", "v2", "agent2", "agent3", "ontology", "out"):
    p.add_argument("--" + field, required=True, type=Path)

p.add_argument("--dataset", required=True)
p.add_argument("--clusters", required=True)

a = p.parse_args()

ids = [str(int(x)) for x in a.clusters.split(",")]

if not ids or len(set(ids)) != len(ids):
    p.error("Cluster IDs must be unique and nonempty")


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize(text):
    return " ".join(str(text or "").lower().split())


# ------------------------------------------------------------
# Validate all inputs BEFORE creating the output directory
# ------------------------------------------------------------

inputs = [
    a.deg,
    a.ontology,
    a.v2 / "all_signature_tests.csv",
    a.v2 / "all_clusters_summary.csv",
    a.v2 / "manifest.json"
]

for cid in ids:
    inputs.extend([
        a.agent2 / "clusters" / f"cluster_{cid}" /
        f"cluster_{cid}_master_gene_evidence.csv",

        a.agent3 / f"cluster_{cid}" / "result" /
        f"cluster_{cid}_agent3_decision_summary.json",

        a.agent3 / f"cluster_{cid}" / "freeze" /
        f"agent3_{a.dataset}_cluster_{cid}_freeze_manifest.json"
    ])

if a.out.exists():
    p.error(f"Output exists. Refusing overwrite: {a.out}")

for path in inputs:
    if not path.is_file():
        p.error(f"Missing input: {path}")

# Check V2 and original DEG use the same input

manifest = read_json(a.v2 / "manifest.json")

hashes = manifest.get(
    "sha256",
    manifest.get("inputs_sha256", {})
)

matching = [
    value for key, value in hashes.items()
    if str(key).replace("\\", "/").split("/")[-1] == a.deg.name
]

if matching != [sha256(a.deg)]:
    p.error("DEG SHA256 differs from V2. Refusing to mix datasets.")

# ------------------------------------------------------------
# Load frozen Cell Ontology
# ------------------------------------------------------------

ontology = defaultdict(set)

for graph in read_json(a.ontology).get("graphs", []):
    for node in graph.get("nodes", []):

        iri = node.get("id", "")

        if "/obo/CL_" not in iri:
            continue

        ontology_id = "CL:" + iri.rsplit("CL_", 1)[-1]

        terms = [node.get("lbl", "")]

        terms.extend([
            x.get("val", "")
            for x in node.get("meta", {}).get("synonyms", [])
            if x.get("pred") == "hasExactSynonym"
        ])

        for term in terms:
            if normalize(term):
                ontology[normalize(term)].add(ontology_id)


def map_ontology(label):
    key = normalize(label)

    if ontology.get(key):
        return sorted(ontology[key]), "EXACT_OR_EXACT_SYNONYM"

    if key.endswith(" cells") and ontology.get(key[:-1]):
        return sorted(ontology[key[:-1]]), "PLURAL_CELL_NORMALIZED"

    return [], "UNMAPPED"


# ------------------------------------------------------------
# Load full DEG and V2 evidence
# ------------------------------------------------------------

signatures = defaultdict(list)

for row in read_rows(a.v2 / "all_signature_tests.csv"):
    if row["cluster"] in ids:
        signatures[row["cluster"]].append(row)

summaries = {
    row["cluster"]: row
    for row in read_rows(a.v2 / "all_clusters_summary.csv")
}

full_deg = defaultdict(dict)

for row in read_rows(a.deg):
    cid = str(row["cluster"])

    if cid not in ids:
        continue

    gene = str(row["gene"]).strip().upper()

    if gene in full_deg[cid]:
        raise ValueError(f"Duplicate DEG: cluster {cid}, gene {gene}")

    full_deg[cid][gene] = row


# ------------------------------------------------------------
# Analyze frozen unresolved clusters
# ------------------------------------------------------------

results = {}
summary = []

for cid in ids:

    decision_file = (
        a.agent3 / f"cluster_{cid}" / "result" /
        f"cluster_{cid}_agent3_decision_summary.json"
    )

    freeze_file = (
        a.agent3 / f"cluster_{cid}" / "freeze" /
        f"agent3_{a.dataset}_cluster_{cid}_freeze_manifest.json"
    )

    decision = read_json(decision_file)
    freeze = read_json(freeze_file)

    if (
        decision.get("decision_state") != "UNRESOLVED"
        or freeze.get("overall_pass") is not True
    ):
        raise ValueError(
            f"Cluster {cid} is not freeze-verified UNRESOLVED"
        )

    marker_file = (
        a.agent2 / "clusters" / f"cluster_{cid}" /
        f"cluster_{cid}_master_gene_evidence.csv"
    )

    original_panel = {
        row["gene"].strip().upper()
        for row in read_rows(marker_file)
    }

    significant = {
        gene for gene, row in full_deg[cid].items()
        if float(row["avg_log2FC"]) >= 0.5
        and float(row["p_val_adj"]) <= 0.05
    }

    if (
        cid not in summaries
        or len(significant) != int(summaries[cid]["significant_DEGs"])
    ):
        raise ValueError(f"Cluster {cid}: inconsistent V2 evidence")

    candidates = []

    for row in signatures[cid]:

        genes = sorted({
            g.strip().upper()
            for g in row["supporting_genes"].split(";")
            if g.strip()
        })

        if len(genes) < 2:
            continue

        if (
            len(genes) != int(row["marker_overlap"])
            or not set(genes).issubset(significant)
        ):
            raise ValueError(
                f"Cluster {cid}: reference signature mismatch"
            )

        fdr = float(row["fdr_bh"])
        detection = float(row["median_detection_pct"])

        enriched = (
            len(genes) >= 3
            and fdr <= 0.05
        )

        ontology_ids, mapping = map_ontology(
            row["reference_cell_type"]
        )

        if enriched and detection < 10:
            status = "ENRICHED_LOW_DETECTION_REVIEW"
        elif enriched:
            status = "ENRICHED_REVIEW"
        else:
            status = "EXPLORATORY_NOT_ENRICHED"

        candidates.append({
            "source": row["source"],
            "cell_type": row["reference_cell_type"],
            "fdr": fdr,
            "marker_count": len(genes),
            "median_detection_pct": detection,
            "evidence_status": status,
            "ontology_ids": ontology_ids,
            "ontology_mapping": mapping,
            "supporting_genes": genes,
            "new_genes_outside_agent2": sorted(
                set(genes) - original_panel
            ),
            "marker_details": [
                {
                    "gene": gene,
                    "log2FC": float(
                        full_deg[cid][gene]["avg_log2FC"]
                    ),
                    "padj": float(
                        full_deg[cid][gene]["p_val_adj"]
                    ),
                    "pct_in_cluster": 100 * float(
                        full_deg[cid][gene]["pct.1"]
                    )
                }
                for gene in genes
            ]
        })

    candidates.sort(
        key=lambda x: (
            x["fdr"],
            -x["marker_count"],
            x["cell_type"]
        )
    )

    enriched_count = sum(
        c["evidence_status"].startswith("ENRICHED_")
        for c in candidates
    )

    if enriched_count > 1:
        review = "COMPETING_SIGNATURES_REVIEW"
    elif enriched_count == 1:
        review = "SINGLE_ENRICHED_SIGNATURE_REVIEW"
    else:
        review = "INSUFFICIENT_ENRICHMENT_KEEP_UNRESOLVED"

    results[cid] = {
        "cluster": cid,
        "frozen_agent3_decision": "UNRESOLVED",
        "review_status": review,
        "candidate_evidence": candidates,
        "final_annotation_assigned": False
    }

    summary.append({
        "cluster": cid,
        "frozen_agent3": "UNRESOLVED",
        "significant_DEGs": len(significant),
        "significant_genes_outside_agent2": len(
            significant - original_panel
        ),
        "enriched_signatures": enriched_count,
        "review_status": review
    })


# ------------------------------------------------------------
# Save NEW results only after validation succeeds
# ------------------------------------------------------------

a.out.mkdir(parents=True, exist_ok=False)

for cid, payload in results.items():

    path = a.out / f"cluster_{cid}_evidence_addendum.json"

    with path.open("x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

with (a.out / "summary.csv").open(
    "x", encoding="utf-8", newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(summary[0])
    )

    writer.writeheader()
    writer.writerows(summary)

with (a.out / "manifest.json").open(
    "x", encoding="utf-8"
) as f:

    json.dump({
        "method": "agent3_frozen_evidence_addendum_v1",
        "dataset": a.dataset,
        "input_sha256": {
            str(path): sha256(path)
            for path in inputs
        },
        "thresholds": {
            "min_log2FC": 0.5,
            "max_padj": 0.05,
            "max_enrichment_fdr": 0.05,
            "min_enriched_overlap": 3
        },
        "original_outputs_modified": False,
        "automatic_final_annotations": False
    }, f, indent=2)

print("\n[PASS] AGENT 3 EVIDENCE ADDENDUM CREATED")

for row in summary:
    print(
        f"Cluster {row['cluster']}: "
        f"DEGs={row['significant_DEGs']}, "
        f"outside Agent2={row['significant_genes_outside_agent2']}, "
        f"enriched={row['enriched_signatures']}, "
        f"status={row['review_status']}"
    )

print("Output:", a.out)
print("Original Agent 3 decisions unchanged.")
