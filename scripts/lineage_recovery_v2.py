"""Full-DEG lineage recovery: exploratory evidence, never final annotation."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from scipy.stats import hypergeom

p = argparse.ArgumentParser()

for name in ("deg", "panglao", "cellmarker", "primary", "shadow", "out"):
    p.add_argument("--" + name, type=Path, required=True)

p.add_argument("--tissue", required=True)
p.add_argument("--expected-rows", type=int, default=102020)
p.add_argument("--expected-genes", type=int, default=5101)
p.add_argument("--expected-clusters", type=int, default=20)

a = p.parse_args()


# Verify inputs before creating any output


inputs = (a.deg, a.panglao, a.cellmarker)

for f in inputs:
    if not f.is_file():
        p.error(f"Missing input: {f}")

for f in (a.primary, a.shadow):
    if not f.is_dir():
        p.error(f"Missing Agent 3 directory: {f}")

if a.out.exists():
    p.error(f"Output exists; refusing overwrite: {a.out}")


# Read the original full DEG table


deg = pd.read_csv(a.deg, low_memory=False)

required = {
    "cluster", "gene", "avg_log2FC",
    "p_val_adj", "pct.1", "pct.2"
}

if not required.issubset(deg.columns):
    raise ValueError(f"Missing DEG columns: {required - set(deg.columns)}")

deg["gene"] = (
    deg["gene"].fillna("").astype(str).str.strip().str.upper()
)
deg["cluster"] = deg["cluster"].astype(str).str.strip()

if (deg["gene"] == "").any():
    raise ValueError("Blank gene symbols found")

if deg[["cluster", "gene"]].duplicated().any():
    raise ValueError("Duplicate cluster-gene entries found")

for col in ("avg_log2FC", "p_val_adj", "pct.1", "pct.2"):
    deg[col] = pd.to_numeric(deg[col], errors="raise")
    if deg[col].isna().any():
        raise ValueError(f"Missing numeric data: {col}")

if not deg["p_val_adj"].between(0, 1).all():
    raise ValueError("Invalid adjusted p-values")

if not deg["pct.1"].between(0, 1).all():
    raise ValueError("Invalid pct.1")

if not deg["pct.2"].between(0, 1).all():
    raise ValueError("Invalid pct.2")

panel = set(deg["gene"])
N = len(panel)

actual = (
    len(deg),
    N,
    deg["cluster"].nunique()
)

expected = (
    a.expected_rows,
    a.expected_genes,
    a.expected_clusters
)

if actual != expected:
    raise ValueError(
        f"Dataset mismatch. Expected {expected}, found {actual}"
    )


# Read existing marker reference databases


pang = pd.read_csv(a.panglao, sep="\t", low_memory=False)

pang_required = {
    "species", "official gene symbol", "cell type"
}

if not pang_required.issubset(pang.columns):
    raise ValueError("Unexpected PanglaoDB columns")

pang = pang[
    pang["species"].fillna("").astype(str).str.contains(
        "Hs", regex=False
    )
].copy()

pang["gene"] = (
    pang["official gene symbol"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
)

cell = pd.read_excel(
    a.cellmarker,
    usecols=["tissue_type", "cell_name", "Symbol"]
)

cell = cell[
    cell["tissue_type"].fillna("").astype(str).str.contains(
        a.tissue, case=False, regex=False
    )
].copy()

cell["gene"] = (
    cell["Symbol"].fillna("").astype(str).str.strip().str.upper()
)


# Build marker signatures


signatures = {}

for source, table, label_col in [
    ("PanglaoDB_Human", pang, "cell type"),
    ("CellMarker_Tissue", cell, "cell_name")
]:
    filtered = table[table["gene"].isin(panel)]

    for name, group in filtered.groupby(label_col):
        genes = set(group["gene"])

        if len(genes) >= 3:
            signatures[(source, str(name))] = genes

if not signatures:
    raise ValueError("No valid marker signatures found")



def bh(values):
    m = len(values)
    order = sorted(range(m), key=lambda i: values[i])

    q = [1.0] * m
    smallest = 1.0

    for rank in range(m, 0, -1):
        i = order[rank - 1]
        smallest = min(
            smallest,
            values[i] * m / rank
        )
        q[i] = smallest

    return q



all_tests = []
all_summaries = []

for cid in sorted(
    deg["cluster"].unique(),
    key=lambda x: int(x)
):
    table = deg[
        deg["cluster"] == cid
    ].set_index("gene")

    up = set(
        table[
            (table["avg_log2FC"] >= 0.5) &
            (table["p_val_adj"] <= 0.05)
        ].index
    )

    K = len(up)
    records = []

    for (source, name), genes in sorted(signatures.items()):
        hits = sorted(up & genes)
        k = len(hits)

        pvalue = (
            float(hypergeom.sf(k - 1, N, len(genes), K))
            if k else 1.0
        )

        detection = (
            round(
                float(table.loc[hits, "pct.1"].median() * 100),
                3
            )
            if hits else 0.0
        )

        records.append({
            "cluster": cid,
            "source": source,
            "reference_cell_type": name,
            "signature_size_in_panel": len(genes),
            "significant_DEGs": K,
            "marker_overlap": k,
            "supporting_genes": "; ".join(hits),
            "median_detection_pct": detection,
            "p_value": pvalue
        })

    qvalues = bh([r["p_value"] for r in records])

    for record, q in zip(records, qvalues):
        record["fdr_bh"] = q
        record["enriched"] = (
            record["marker_overlap"] >= 3 and q <= 0.05
        )
        record["sparse_detection_flag"] = (
            record["enriched"] and
            record["median_detection_pct"] < 10
        )

    all_tests.extend(records)

    multi = sorted(
        [
            r for r in records
            if r["marker_overlap"] >= 2
        ],
        key=lambda r: (
            r["fdr_bh"],
            -r["marker_overlap"],
            r["source"],
            r["reference_cell_type"]
        )
    )

    enriched = [
        r for r in multi if r["enriched"]
    ]

    if len(enriched) > 1:
        status = "MULTIPLE_ENRICHED_SIGNATURES_REVIEW"
    elif len(enriched) == 1:
        status = "ONE_ENRICHED_SIGNATURE_REVIEW"
    elif multi:
        status = "EXPLORATORY_ONLY_NOT_SIGNIFICANT"
    else:
        status = "NO_MULTI_MARKER_REFERENCE_OVERLAP"

    # Read frozen Agent 3 decisions; never modify them
    decision_paths = [
        (
            kind,
            root / f"cluster_{cid}" / "result" /
            f"cluster_{cid}_agent3_decision_summary.json"
        )
        for kind, root in [
            ("PRIMARY", a.primary),
            ("SHADOW", a.shadow)
        ]
    ]

    found = [
        (kind, path)
        for kind, path in decision_paths
        if path.is_file()
    ]

    if len(found) != 1:
        raise ValueError(
            f"Cluster {cid}: expected one frozen Agent 3 decision, "
            f"found {len(found)}"
        )

    kind, path = found[0]

    original = json.loads(
        path.read_text(encoding="utf-8")
    ).get("decision_state")

    if original not in (
        "RESOLVED", "AMBIGUOUS", "UNRESOLVED"
    ):
        raise ValueError(
            f"Cluster {cid}: invalid frozen decision"
        )

    best = multi[0] if multi else None

    all_summaries.append({
        "cluster": cid,
        "core_agent3_run": kind,
        "frozen_core_agent3_decision": original,
        "significant_DEGs": K,
        "tested_signatures": len(records),
        "enriched_signatures": len(enriched),
        "low_detection_enriched_signatures": sum(
            r["sparse_detection_flag"] for r in enriched
        ),
        "recovery_evidence_status": status,
        "top_reference_candidate": (
            best["reference_cell_type"] if best else ""
        ),
        "top_reference_source": (
            best["source"] if best else ""
        ),
        "top_reference_fdr": (
            best["fdr_bh"] if best else ""
        ),
        "top_reference_marker_overlap": (
            best["marker_overlap"] if best else 0
        ),
        "top_reference_median_detection_pct": (
            best["median_detection_pct"] if best else ""
        ),
        "top_reference_supporting_genes": (
            best["supporting_genes"] if best else ""
        ),
        "annotation_assigned": False
    })



a.out.mkdir(parents=True, exist_ok=False)

pd.DataFrame(all_tests).to_csv(
    a.out / "all_signature_tests.csv",
    index=False
)

pd.DataFrame([
    r for r in all_tests
    if r["marker_overlap"] >= 2
]).to_csv(
    a.out / "multi_marker_candidates.csv",
    index=False
)

pd.DataFrame(all_summaries).to_csv(
    a.out / "all_clusters_summary.csv",
    index=False
)


def sha(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(block)

    return h.hexdigest()

manifest = {
    "method": "full_DEG_reference_signature_enrichment_exploratory_v2",
    "thresholds": {
        "min_log2FC": 0.5,
        "max_p_adj": 0.05,
        "min_signature_genes": 3,
        "enriched_min_overlap": 3,
        "max_bh_fdr": 0.05,
        "sparse_detection_pct": 10
    },
    "universe_genes": N,
    "deg_rows": len(deg),
    "clusters": len(all_summaries),
    "tested_signatures": len(signatures),
    "tissue_filter": a.tissue,
    "sha256": {
        str(f): sha(f) for f in inputs
    },
    "original_decisions_changed": False,
    "final_annotations_assigned": False
}

(a.out / "manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8"
)

print(
    f"[PASS] {len(all_summaries)} clusters | "
    f"{N} genes | {len(signatures)} marker signatures"
)

summary = pd.DataFrame(all_summaries)

print(
    summary[
        summary["cluster"].isin(["5", "13", "17"])
    ][[
        "cluster",
        "frozen_core_agent3_decision",
        "recovery_evidence_status",
        "top_reference_candidate",
        "top_reference_fdr",
        "top_reference_median_detection_pct"
    ]].to_string(index=False)
)

print("New output only:", a.out)
print("No old results modified.")
print("No final biological labels automatically assigned.")