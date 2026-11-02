from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path
from typing import Any


RESOLVED = "RESOLVED"
AMBIGUOUS = "AMBIGUOUS"
UNRESOLVED = "UNRESOLVED"

RELATED_STATES = "CLOSELY_RELATED_STATES"
POSSIBLE_MIXED_CLUSTER = "POSSIBLE_MIXED_CLUSTER"

TARGET_MARKER_GROUNDED = "TARGET_CONTEXT_MARKER_GROUNDED"
PARTIAL_CONTEXT_GROUNDED = (
    "TARGET_CONTEXT_SUPPORTED_WITH_CROSS_CONTEXT_MARKER_GROUNDING"
)

GENERAL_ANCESTORS = {
    "cell",
    "native cell",
    "eukaryotic cell",
    "animal cell",
    "motile cell",
    "somatic cell",
    "single nucleate cell",
    "nucleate cell",
    "leukocyte",
    "hematopoietic cell",
}


def clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def to_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def to_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def split_markers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw = value
    else:
        raw = str(value).replace(",", ";").split(";")
    return sorted({clean(x) for x in raw if clean(x)})


def normalize_cl_id(value: Any) -> str | None:
    text = clean(value)
    if not text:
        return None
    if text.startswith("CL:"):
        return text.replace(":", "_")
    if text.startswith("http://purl.obolibrary.org/obo/CL_"):
        return text.rsplit("/", 1)[-1]
    return None


class CellOntologyIndex:
    def __init__(self, labels_by_id, ids_by_label, parents_by_id):
        self.labels_by_id = labels_by_id
        self.ids_by_label = ids_by_label
        self.parents_by_id = parents_by_id

    @classmethod
    def from_json(cls, path: Path | None):
        if path is None or not path.exists():
            return None

        obj = json.loads(path.read_text(encoding="utf-8"))
        graph = obj.get("graphs", [{}])[0]

        labels_by_id = {}
        ids_by_label = {}

        for node in graph.get("nodes", []):
            cl_id = normalize_cl_id(node.get("id"))
            label = clean(node.get("lbl"))
            if cl_id and label:
                labels_by_id[cl_id] = label
                ids_by_label[label.casefold()] = cl_id

        parents_by_id = {}

        for edge in graph.get("edges", []):
            if clean(edge.get("pred")) != "is_a":
                continue
            child = normalize_cl_id(edge.get("sub"))
            parent = normalize_cl_id(edge.get("obj"))
            if child and parent:
                parents_by_id.setdefault(child, set()).add(parent)

        return cls(labels_by_id, ids_by_label, parents_by_id)

    def resolve_id(self, label: str, ontology_id: str | None = None):
        cl_id = normalize_cl_id(ontology_id)
        if cl_id:
            return cl_id
        return self.ids_by_label.get(label.casefold())

    def ancestor_distances(self, cl_id: str):
        distances = {cl_id: 0}
        queue = deque([(cl_id, 0)])

        while queue:
            current, dist = queue.popleft()
            for parent in self.parents_by_id.get(current, set()):
                if parent in distances:
                    continue
                distances[parent] = dist + 1
                queue.append((parent, dist + 1))

        return distances

    def closest_common_ancestor(self, left_id: str, right_id: str):
        left = self.ancestor_distances(left_id)
        right = self.ancestor_distances(right_id)
        common = set(left) & set(right)

        if not common:
            return None, None, None

        best = min(
            common,
            key=lambda x: (
                left[x] + right[x],
                max(left[x], right[x]),
                self.labels_by_id.get(x, x),
            ),
        )

        return best, self.labels_by_id.get(best), left[best] + right[best]

    def are_closely_related(self, left: dict, right: dict):
        left_id = left.get("_resolved_ontology_id")
        right_id = right.get("_resolved_ontology_id")

        if not left_id or not right_id:
            return False, {"reason": "ontology_id_missing"}

        ancestor_id, ancestor_label, distance = self.closest_common_ancestor(
            left_id,
            right_id,
        )

        if not ancestor_id:
            return False, {"reason": "no_common_cell_ontology_ancestor"}

        left_dist = self.ancestor_distances(left_id).get(ancestor_id)
        right_dist = self.ancestor_distances(right_id).get(ancestor_id)

        direct_relation = left_dist == 0 or right_dist == 0
        ancestor_label_folded = clean(ancestor_label).casefold()

        informative_ancestor = (
            bool(ancestor_label_folded)
            and ancestor_label_folded not in GENERAL_ANCESTORS
        )

        left_tokens = {
            t
            for t in left["label"].casefold().replace("-", " ").split()
            if (len(t) > 1 or t in {"b", "t"}) and t != "cell"
        }
        right_tokens = {
            t
            for t in right["label"].casefold().replace("-", " ").split()
            if (len(t) > 1 or t in {"b", "t"}) and t != "cell"
        }
        ancestor_tokens = set(ancestor_label_folded.replace("-", " ").split())

        label_linked = bool(ancestor_tokens & (left_tokens | right_tokens))

        related = (
            direct_relation
            or (
                informative_ancestor
                and distance is not None
                and distance <= 3
                and label_linked
            )
        )

        return related, {
            "left_label": left["label"],
            "right_label": right["label"],
            "common_ancestor_id": ancestor_id.replace("_", ":"),
            "common_ancestor_label": ancestor_label,
            "ontology_distance": distance,
            "direct_relation": direct_relation,
            "informative_ancestor": informative_ancestor,
            "label_linked_ancestor": label_linked,
        }


def rescue_tier(row: dict):
    state = clean(row.get("evidence_state"))

    confirmation_marker = to_int(row.get("confirmation_candidate_marker_papers"))
    target_marker = to_int(row.get("target_context_candidate_marker_papers"))
    same_sentence = to_int(row.get("same_sentence_link_papers"))
    local_sentence = to_int(row.get("within_three_sentence_link_papers"))
    pubmedbert_unique = to_int(row.get("pubmedbert_eligible_unique_papers"))
    discovery_marker = to_int(row.get("discovery_marker_linked_papers"))
    target_candidate = to_int(row.get("target_context_candidate_papers"))
    upstream = bool(row.get("is_upstream_candidate"))

    if (
        state == TARGET_MARKER_GROUNDED
        and (target_marker >= 2 or confirmation_marker >= 2)
        and pubmedbert_unique >= 2
    ):
        return 3, "target-context marker-grounded evidence"

    if (
        state == PARTIAL_CONTEXT_GROUNDED
        and confirmation_marker >= 2
        and pubmedbert_unique >= 3
        and (same_sentence > 0 or local_sentence > 0)
    ):
        return 2, "target-context support with cross-context marker grounding"

    provenance_pairwise = to_int(row.get("provenance_pairwise_marker_papers"))
    provenance_target = to_int(row.get("provenance_target_context_papers") or row.get("provenance_breast_context_papers"))
    provenance_same_sentence = to_int(row.get("provenance_same_sentence_marker_papers"))

    if (
        upstream
        and provenance_pairwise >= 3
        and provenance_target >= 2
        and provenance_same_sentence >= 1
    ):
        return 2, "upstream candidate supported by rescue-marker provenance"

    if (
        upstream
        and target_candidate > 0
        and discovery_marker >= 2
        and pubmedbert_unique >= 2
    ):
        return 1, "upstream candidate retained with target-context support"

    return 0, "below possible-label evidence threshold"


def adjudication_tier(row: dict):
    candidate_support = to_int(row.get("candidate_and_supporting_marker_passages"))
    papers = to_int(row.get("independent_top_paper_count"))
    context = to_int(row.get("context_matched_passages"))
    distinct_support = to_int(row.get("distinct_supporting_markers_retrieved"))
    support_fraction = to_float(row.get("supporting_gene_retrieval_fraction"))

    if candidate_support >= 2 and papers >= 2 and context > 0:
        return 2, "candidate and supporting markers recovered in context-matched literature"

    if support_fraction >= 0.5 and distinct_support >= 2 and papers >= 2:
        return 1, "partial candidate-supporting marker recovery"

    return 0, "below possible-label evidence threshold"


def build_possible_label(row: dict, schema_name: str, ontology):
    label = clean(row.get("candidate"))
    if not label:
        return None

    if schema_name == "Agent3RescueEvidencePacket":
        tier, basis = rescue_tier(row)
        linked_markers = split_markers(row.get("discovery_linked_markers"))
    else:
        tier, basis = adjudication_tier(row)
        linked_markers = split_markers(row.get("retrieved_supporting_genes"))

    if tier <= 0:
        return None

    ontology_id = clean(row.get("ontology_id")) or None
    resolved_id = ontology.resolve_id(label, ontology_id) if ontology else None

    return {
        "label": label,
        "ontology_id": ontology_id or (resolved_id.replace("_", ":") if resolved_id else None),
        "_resolved_ontology_id": resolved_id,
        "evidence_tier": tier,
        "evidence_basis": basis,
        "evidence_state": row.get("evidence_state"),
        "reporting_state": row.get("reporting_state"),
        "is_upstream_candidate": bool(
            row.get("is_upstream_candidate")
            or row.get("candidate_already_upstream")
        ),
        "linked_markers": linked_markers,
        "support_summary": {
            "independent_papers": to_int(
                row.get("independent_top_paper_count")
                or row.get("discovery_independent_papers")
            ),
            "confirmation_candidate_marker_papers": to_int(
                row.get("confirmation_candidate_marker_papers")
            ),
            "target_context_candidate_marker_papers": to_int(
                row.get("target_context_candidate_marker_papers")
            ),
            "target_context_candidate_papers": to_int(
                row.get("target_context_candidate_papers")
            ),
            "candidate_supporting_marker_passages": to_int(
                row.get("candidate_and_supporting_marker_passages")
            ),
            "pubmedbert_eligible_unique_papers": to_int(
                row.get("pubmedbert_eligible_unique_papers")
            ),
            "pubmedbert_max_similarity": row.get("pubmedbert_max_similarity"),
        },
    }


def classify_ambiguity(possible_labels: list[dict], ontology):
    if len(possible_labels) <= 1:
        return None, []

    if ontology is None:
        return POSSIBLE_MIXED_CLUSTER, [
            {"reason": "cell_ontology_unavailable_for_relationship_check"}
        ]

    checks = []
    related_found = False

    for i, left in enumerate(possible_labels):
        for right in possible_labels[i + 1:]:
            related, evidence = ontology.are_closely_related(left, right)
            checks.append(evidence)
            related_found = related_found or related

    if related_found:
        return RELATED_STATES, checks

    return POSSIBLE_MIXED_CLUSTER, checks


def build_decision_summary(
    packet: dict,
    dataset: str | None = None,
    cluster_id: str | None = None,
    ontology_path: Path | None = None,
):
    schema_name = clean(packet.get("schema_name"))
    ontology = CellOntologyIndex.from_json(ontology_path)

    possible_labels = []

    for row in packet.get("candidates", []) or []:
        item = build_possible_label(row, schema_name, ontology)
        if item:
            possible_labels.append(item)

    possible_labels.sort(
        key=lambda x: (
            -to_int(x.get("evidence_tier")),
            -to_int(x.get("support_summary", {}).get("confirmation_candidate_marker_papers")),
            -to_int(x.get("support_summary", {}).get("target_context_candidate_marker_papers")),
            -to_int(x.get("support_summary", {}).get("pubmedbert_eligible_unique_papers")),
            x["label"].casefold(),
        )
    )

    if len(possible_labels) == 0:
        decision_state = UNRESOLVED
    elif len(possible_labels) == 1:
        decision_state = RESOLVED
    else:
        decision_state = AMBIGUOUS

    ambiguity_subtype, relationship_checks = classify_ambiguity(
        possible_labels,
        ontology,
    )

    if decision_state != AMBIGUOUS:
        ambiguity_subtype = None
        relationship_checks = []

    public_labels = []

    for item in possible_labels:
        item = dict(item)
        item.pop("_resolved_ontology_id", None)
        public_labels.append(item)

    return {
        "schema_name": "Agent3DecisionSummary",
        "schema_version": "1.0",
        "dataset": dataset,
        "cluster_id": cluster_id or clean(packet.get("cluster_id")),
        "source_packet_schema": schema_name,
        "decision_state": decision_state,
        "ambiguity_subtype": ambiguity_subtype,
        "possible_label_count": len(public_labels),
        "possible_labels": public_labels,
        "relationship_checks": relationship_checks,
        "decision_policy": (
            "Non-calibrated evidence-state summary. Agent 3 returns possible "
            "labels and a resolved/ambiguous/unresolved state; it does not "
            "force a single final cell-type annotation."
        ),
        "automatic_acceptance": "NOT_CALIBRATED",
        "final_annotation": None,
        "expert_label_used": False,
        "reference_label_used": False,
    }


def write_decision_outputs(summary: dict, output_json: Path):
    output_json.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_path = output_json.with_suffix(".csv")

    rows = []

    for label in summary.get("possible_labels", []):
        support = label.get("support_summary", {}) or {}
        rows.append(
            {
                "cluster_id": summary.get("cluster_id"),
                "decision_state": summary.get("decision_state"),
                "ambiguity_subtype": summary.get("ambiguity_subtype"),
                "label": label.get("label"),
                "ontology_id": label.get("ontology_id"),
                "evidence_tier": label.get("evidence_tier"),
                "evidence_basis": label.get("evidence_basis"),
                "is_upstream_candidate": label.get("is_upstream_candidate"),
                "linked_markers": "; ".join(label.get("linked_markers", []) or []),
                "independent_papers": support.get("independent_papers"),
                "confirmation_candidate_marker_papers": support.get(
                    "confirmation_candidate_marker_papers"
                ),
                "target_context_candidate_marker_papers": support.get(
                    "target_context_candidate_marker_papers"
                ),
                "target_context_candidate_papers": support.get(
                    "target_context_candidate_papers"
                ),
                "candidate_supporting_marker_passages": support.get(
                    "candidate_supporting_marker_passages"
                ),
                "pubmedbert_eligible_unique_papers": support.get(
                    "pubmedbert_eligible_unique_papers"
                ),
                "pubmedbert_max_similarity": support.get("pubmedbert_max_similarity"),
            }
        )

    if not rows:
        rows = [
            {
                "cluster_id": summary.get("cluster_id"),
                "decision_state": summary.get("decision_state"),
                "ambiguity_subtype": summary.get("ambiguity_subtype"),
                "label": None,
                "ontology_id": None,
                "evidence_tier": None,
                "evidence_basis": None,
                "is_upstream_candidate": None,
                "linked_markers": None,
                "independent_papers": None,
                "confirmation_candidate_marker_papers": None,
                "target_context_candidate_marker_papers": None,
                "target_context_candidate_papers": None,
                "candidate_supporting_marker_passages": None,
                "pubmedbert_eligible_unique_papers": None,
                "pubmedbert_max_similarity": None,
            }
        ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--cluster-id")
    parser.add_argument("--cell-ontology-json", type=Path)

    args = parser.parse_args()

    packet = json.loads(args.packet_json.read_text(encoding="utf-8-sig"))

    summary = build_decision_summary(
        packet,
        dataset=args.dataset,
        cluster_id=args.cluster_id,
        ontology_path=args.cell_ontology_json,
    )

    csv_path = write_decision_outputs(summary, args.output_json)

    print("Agent 3 decision summary completed.")
    print("Decision:", summary["decision_state"])
    print("Ambiguity subtype:", summary["ambiguity_subtype"])
    print("Possible labels:", summary["possible_label_count"])
    print("JSON:", args.output_json)
    print("CSV:", csv_path)


if __name__ == "__main__":
    main()

