from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def required(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def write_supplemental_markers(source, destination):
    """Create validated canonical marker evidence without changing source."""
    aliases = {
        "cluster_id": (
            "cluster_id", "cluster", "spatial_cluster", "leiden", "group"
        ),
        "gene": (
            "gene", "genes", "name", "names",
            "gene_name", "gene_symbol", "symbol"
        ),
        "rank_within_cluster": (
            "rank_within_cluster", "gene_rank", "marker_rank", "rank"
        ),
    }

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)

    if not fields or not rows:
        raise ValueError("Supplemental marker CSV is empty")

    if any(not field.strip() for field in fields):
        raise ValueError("Supplemental marker CSV has an empty header")

    if len({field.strip().casefold() for field in fields}) != len(fields):
        raise ValueError("Duplicate marker column headers")

    mapping = {}

    for canonical, accepted in aliases.items():
        matches = [
            field for field in fields
            if field.strip().casefold() in accepted
        ]

        if not matches:
            raise ValueError(
                f"Supplemental markers missing {canonical}; "
                f"available columns: {fields}"
            )

        chosen = next(
            field
            for alias in accepted
            for field in fields
            if field.strip().casefold() == alias
        )

        for row in rows:
            values = {
                str(row.get(field) or "").strip()
                for field in matches
            }
            if len(values) > 1:
                raise ValueError(
                    f"Conflicting marker aliases for {canonical}"
                )

        mapping[canonical] = chosen

    seen = set()

    for row_number, row in enumerate(rows, start=2):
        cluster = str(row[mapping["cluster_id"]] or "").strip()
        gene = str(row[mapping["gene"]] or "").strip()
        rank = str(row[mapping["rank_within_cluster"]] or "").strip()

        if not cluster or not gene or not rank:
            raise ValueError(
                f"Missing required marker value at row {row_number}"
            )

        try:
            numeric_rank = float(rank)
        except ValueError as exc:
            raise ValueError(
                f"Invalid marker rank at row {row_number}"
            ) from exc

        if not (0 < numeric_rank < float("inf")):
            raise ValueError(
                f"Nonpositive or nonfinite rank at row {row_number}"
            )

        key = (cluster, gene.casefold())

        if key in seen:
            raise ValueError(
                f"Duplicate marker gene {gene} in cluster {cluster}"
            )

        seen.add(key)

    output_fields = list(fields)

    for canonical in aliases:
        if canonical not in output_fields:
            output_fields.append(canonical)

    with destination.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()

        for row in rows:
            normalized = dict(row)

            for canonical, original_field in mapping.items():
                normalized[canonical] = row[original_field]

            writer.writerow(normalized)


def main():
    parser = argparse.ArgumentParser(
        description="Experimental fresh-run cell annotation coordinator"
    )

    for flag in (
        "project-root", "markers", "panel", "dataset",
        "tissue", "platform", "disease-context",
        "router-config", "output-root"
    ):
        parser.add_argument("--" + flag, required=True)

    parser.add_argument(
        "--agent2-mode",
        required=True,
        choices=["lung-marker-only", "general-marker-only", "general-hpa"]
    )
    parser.add_argument("--hpa-json")
    parser.add_argument("--cell-ontology")
    parser.add_argument("--no-expert", action="store_true")
    parser.add_argument("--supplemental-review", action="store_true")
    parser.add_argument("--plan", action="store_true")

    args = parser.parse_args()

    project = Path(args.project_root).resolve()
    markers = required(Path(args.markers).resolve())
    panel = required(Path(args.panel).resolve())
    router_policy = required(Path(args.router_config).resolve())
    output = Path(args.output_root).resolve()

    ontology = required(
        Path(args.cell_ontology).resolve()
        if args.cell_ontology
        else project / "data/resources/cell_ontology/v2026-06-08/cl-base.json"
    )

    hpa = required(Path(args.hpa_json).resolve()) if args.hpa_json else None

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.dataset):
        parser.error("Dataset ID must contain only letters, digits, _ or -")

    if args.agent2_mode == "general-hpa" and not hpa:
        parser.error("general-hpa requires --hpa-json")

    if args.agent2_mode != "general-hpa" and hpa:
        parser.error("--hpa-json is only accepted with general-hpa")

    if args.agent2_mode == "lung-marker-only" and args.tissue.lower() != "lung":
        parser.error("lung-marker-only requires Lung tissue")

    if output.exists():
        parser.error("Fresh output directory already exists")

    if output == project or project in output.parents:
        parser.error("Prototype outputs must remain outside the project")

    for file in (markers, panel):
        parts = [part.lower() for part in file.parts]
        if "pipeline_runs" in parts or "published_results" in parts:
            parser.error("Do not use historical results as marker/panel inputs")

    projected = (
        output / "a3/cluster_1234/marker_rescue/confirmation/pubmedbert/"
        "cluster_1234_rescue_confirmation_pubmedbert_summary.csv"
    )
    if len(str(projected)) > 240:
        parser.error("Choose a shorter output directory")

    for rel in (
        "data/reference/Cell_marker_Human.xlsx",
        "data/reference/PanglaoDB_markers_27_Mar_2020.tsv",
        "src/cosmx_brcr_annotation_pipeline/run_agent1_orchestrated.py",
        "src/annotation_router/runners/run_scored_router.py",
        "scripts/agent3_validation/run_agent3_from_router_queue.py",
        "scripts/agent3_validation/run_agent3_standardized_full_with_decision.py"
    ):
        required(project / rel)

    if args.agent2_mode == "lung-marker-only":
        required(project / "src/lung_annotation_pipeline/run_agent2.py")
    else:
        required(
            project /
            "src/marker_interpretation_agent_v2/runners/build_panel_catalog.py"
        )
        required(
            project /
            "src/marker_interpretation_agent_v2/runners/run_all_clusters.py"
        )

    supplemental_scripts = (
        project / "scripts/supplemental_unresolved_audit_standardized"
    )

    supplemental_stages = (
        "run_unresolved_audit",
        "run_candidate_evidence_audit",
        "run_candidate_relationship_audit",
        "run_component_evidence_audit",
        "run_supplemental_adjudication"
    )

    if args.supplemental_review:
        for name in supplemental_stages:
            required(supplemental_scripts / (name + ".py"))

    config = read_json(router_policy)

    for field in (
        "dataset_name", "agent2_summary_csv",
        "agent2_candidates_csv", "output_dir"
    ):
        if field not in config:
            parser.error("Router configuration missing: " + field)

    if args.plan:
        print("PREFLIGHT PASS")
        print("Dataset:", args.dataset)
        print("Agent 2 mode:", args.agent2_mode)
        print("Router policy:", router_policy)
        print("Supplemental enabled:", args.supplemental_review)
        print("No-expert policy:", args.no_expert)
        print("Output:", output)
        print("No pipeline executed or files created")
        return 0

    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(project / "src"),
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1"
    })

    protected = [markers, panel, router_policy, ontology]
    if hpa:
        protected.append(hpa)

    before = {str(p): digest(p) for p in protected}

    manifest = {
        "dataset": args.dataset,
        "status": "STARTED",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "agent2_mode": args.agent2_mode,
        "router_policy_source": str(router_policy),
        "queue_policy": "no_expert" if args.no_expert else "default",
        "supplemental_requested": args.supplemental_review,
        "reference_labels_passed_to_stages": False,
        "original_input_hashes": before,
        "stages": [],
        "agent3": [],
        "supplemental": []
    }

    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs").mkdir()
    (output / "logs").mkdir()

    input_markers = output / "inputs/markers.csv"
    input_panel = output / "inputs/panel.csv"

    shutil.copyfile(markers, input_markers)
    shutil.copyfile(panel, input_panel)

    supplemental_markers = None
    if args.supplemental_review:
        supplemental_markers = output / "inputs/supplemental_markers.csv"
        write_supplemental_markers(
            input_markers, supplemental_markers
        )

    a1 = output / "a1"
    a2 = output / "a2"
    a2root = a2 / "agent2_output"
    router = output / "router"
    a3 = output / "a3"

    def save():
        (output / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def execute(name, command):
        log = output / "logs" / (name + ".log")
        print("RUNNING:", name, flush=True)
        start = time.perf_counter()

        with log.open("w", encoding="utf-8") as stream:
            if name == "04_agent3_dispatch":
                live_env = dict(env)
                live_env["PYTHONUNBUFFERED"] = "1"

                with subprocess.Popen(
                    [sys.executable, "-u", "-B", *map(str, command)],
                    cwd=project,
                    env=live_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                ) as child:
                    for line in child.stdout:
                        stream.write(line)
                        stream.flush()

                        message = line.strip()

                        if (
                            message.startswith((
                                "CLUSTER ",
                                "Completed cluster",
                                "Running standardized Agent 3",
                                "Context-expanded retrieval completed",
                                "Literature paper ranking completed",
                                "Agent 3 decision summary added",
                                "Decision:",
                                "Unique papers retrieved:",
                                "Papers ranked:",
                                "STATUS:",
                                "ROUTER-DRIVEN AGENT 3 COMPLETE",
                            ))
                            or (
                                len(message) < 160
                                and "completed" in message.lower()
                            )
                        ):
                            print("  [Agent 3]", message, flush=True)

                    result_code = child.wait()

                result = subprocess.CompletedProcess(
                    command, result_code
                )
            else:
                result = subprocess.run(
                    [sys.executable, "-B", *map(str, command)],
                    cwd=project,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT
                )

        seconds = round(time.perf_counter() - start, 3)

        manifest["stages"].append({
            "name": name,
            "return_code": result.returncode,
            "elapsed_seconds": seconds,
            "log": str(log)
        })
        save()

        if result.returncode:
            lines = log.read_text(
                encoding="utf-8",
                errors="replace"
            ).splitlines()
            print("\n".join(lines[-25:]))
            raise RuntimeError(f"{name} failed. Log: {log}")

        print(f"PASS: {name} ({seconds}s)", flush=True)

    try:
        execute("01_agent1", [
            "-m", "cosmx_brcr_annotation_pipeline.run_agent1_orchestrated",
            "--panel-gene-list", input_panel,
            "--output-dir", a1,
            "--species", "Human",
            "--tissue", args.tissue,
            "--platform", args.platform
        ])

        agent1 = required(a1 / "panel_agent1_knowledge_evidence.json")

        if args.agent2_mode == "lung-marker-only":
            execute("02_agent2_lung", [
                "-c",
                "from pathlib import Path; "
                "from lung_annotation_pipeline.run_agent2 import run_agent2; "
                "import sys; "
                "run_agent2(Path(sys.argv[1]),Path(sys.argv[2]),"
                "Path(sys.argv[3]),Path(sys.argv[4]),top_k=50)",
                input_markers, input_panel, agent1, a2
            ])
        else:
            catalog = a2 / "agent2_input/panel_evidence_catalog.json"

            command = [
                "-m", "marker_interpretation_agent_v2.runners.build_panel_catalog",
                "--agent1-json", agent1,
                "--panel-gene-list", input_panel,
                "--output-json", catalog,
                "--output-csv", a2 / "agent2_input/panel_evidence_catalog.csv",
                "--unresolved-csv", a2 / "agent2_input/unresolved_panel_genes.csv"
            ]

            if hpa:
                command += ["--hpa-json", hpa]
            else:
                command += ["--marker-only"]

            execute("02a_agent2_catalog", command)

            command = [
                "-m", "marker_interpretation_agent_v2.runners.run_all_clusters",
                "--marker-file", input_markers,
                "--panel-catalog", required(catalog),
                "--output-root", a2root,
                "--species", "Human",
                "--tissue", args.tissue,
                "--platform", args.platform,
                "--top-k", "50"
            ]

            if not hpa:
                command.append("--no-hpa")

            execute("02b_agent2_clusters", command)

        summary = required(a2root / "all_clusters_summary.csv")
        candidates = required(
            a2root / "all_clusters_threshold_candidates.csv"
        )

        summary_rows = read_csv(summary)

        if (
            not summary_rows
            or len({r["cluster_id"] for r in summary_rows}) != len(summary_rows)
        ):
            raise ValueError("Invalid Agent 2 cluster summary")

        config["dataset_name"] = args.dataset
        config["agent2_summary_csv"] = str(summary)
        config["agent2_candidates_csv"] = str(candidates)
        config["output_dir"] = str(router)

        run_config = output / "router_config_for_this_run.json"
        run_config.write_text(
            json.dumps(config, indent=2),
            encoding="utf-8"
        )

        execute("03_router_v4", [
            "-m", "annotation_router.runners.run_scored_router",
            "--router-config", run_config
        ])

        decisions = read_csv(required(router / "router_decisions.csv"))

        if len(decisions) != len(summary_rows):
            raise ValueError("Router/Agent 2 cluster count mismatch")

        expected = {
            str(r["cluster_id"])
            for r in decisions
            if r["router_decision"] == "route_to_agent3"
        }

        queue_name = (
            "agent3_queue_if_no_expert.csv"
            if args.no_expert
            else "agent3_queue.csv"
        )

        if args.no_expert:
            expected.update({
                str(r["cluster_id"])
                for r in decisions
                if r["router_decision"] == "monitor_uncertainty"
            })

        queue = required(router / queue_name)
        queued = [str(r["cluster_id"]) for r in read_csv(queue)]

        if len(queued) != len(set(queued)) or set(queued) != expected:
            raise ValueError("Router queue mismatch")

        if queued:
            execute("04_agent3_dispatch", [
                project / "scripts/agent3_validation/run_agent3_from_router_queue.py",
                "--dataset", args.dataset,
                "--queue-csv", queue,
                "--agent2-root", a2root,
                "--output-root", a3,
                "--disease-context", args.disease_context,
                "--cell-ontology", ontology,
                "--wait-seconds", 0
            ])

            batch = read_json(required(a3 / "agent3_batch_manifest.json"))
            records = batch.get("clusters", [])

            if (
                {str(r.get("cluster_id")) for r in records} != set(queued)
                or any(r.get("status") != "COMPLETE" for r in records)
            ):
                raise ValueError("Agent 3 batch manifest incomplete")

            for cid in queued:
                cluster = a3 / f"cluster_{cid}"
                decision = read_json(required(
                    cluster / "result" /
                    f"cluster_{cid}_agent3_decision_summary.json"
                ))
                freeze = read_json(required(
                    cluster / "freeze" /
                    f"agent3_{args.dataset}_cluster_{cid}_freeze_manifest.json"
                ))

                state = decision.get("decision_state")

                if (
                    freeze.get("overall_pass") is not True
                    or state not in {"RESOLVED", "AMBIGUOUS", "UNRESOLVED"}
                ):
                    raise ValueError(f"Agent 3 result invalid: cluster {cid}")

                manifest["agent3"].append({
                    "cluster_id": cid,
                    "decision": state,
                    "freeze_pass": True
                })

            save()
        else:
            manifest["agent3_status"] = "SKIPPED_EMPTY_ROUTER_QUEUE"
            print("PASS: Agent 3 correctly skipped (empty Router queue)")

        if args.supplemental_review:
            for record in manifest["agent3"]:
                if record["decision"] != "UNRESOLVED":
                    continue

                cid = record["cluster_id"]
                prefix = f"sup_{cid}"
                sup_root = output / "sup"
                cluster = sup_root / f"cluster_{cid}"

                core = required(
                    a3 / f"cluster_{cid}/result/"
                    f"cluster_{cid}_agent3_decision_summary.json"
                )
                frozen_core_hash = digest(core)

                gene = required(
                    a2root / f"clusters/cluster_{cid}/"
                    f"cluster_{cid}_master_gene_evidence.csv"
                )
                ranked = required(
                    a2root / f"clusters/cluster_{cid}/"
                    f"cluster_{cid}_ranked_candidate_cell_types.csv"
                )

                packet = cluster / (
                    f"evidence/cluster_{cid}_standardized_evidence_packet.json"
                )
                ca = cluster / (
                    f"candidate_audit/cluster_{cid}_candidate_evidence_audit.json"
                )
                ra = cluster / (
                    f"relationship_audit/cluster_{cid}_candidate_relationship_audit.json"
                )
                ea = cluster / (
                    f"component_audit/cluster_{cid}_component_evidence_audit.json"
                )
                final = cluster / (
                    f"adjudication/cluster_{cid}_supplemental_adjudication.json"
                )

                command = [
                    supplemental_scripts / "run_unresolved_audit.py",
                    "--dataset", args.dataset,
                    "--cluster-id", cid,
                    "--core-agent3-decision-json", core,
                    "--agent2-evidence", gene,
                    "--agent2-evidence", ranked,
                    "--marker-evidence", supplemental_markers,
                    "--agent1-evidence", agent1,
                    "--species", "Human",
                    "--tissue", args.tissue,
                    "--platform", args.platform,
                    "--disease-context", args.disease_context,
                    "--output-root", sup_root
                ]

                if hpa:
                    command += ["--hpa-evidence", hpa]

                execute(prefix + "_01_evidence", command)

                execute(prefix + "_02_candidate", [
                    supplemental_scripts / "run_candidate_evidence_audit.py",
                    "--evidence-packet", required(packet),
                    "--output-dir", cluster / "candidate_audit"
                ])

                execute(prefix + "_03_ontology", [
                    supplemental_scripts / "run_candidate_relationship_audit.py",
                    "--candidate-audit", required(ca),
                    "--cell-ontology-json", ontology,
                    "--output-dir", cluster / "relationship_audit"
                ])

                execute(prefix + "_04_components", [
                    supplemental_scripts / "run_component_evidence_audit.py",
                    "--candidate-audit", required(ca),
                    "--relationship-audit", required(ra),
                    "--output-dir", cluster / "component_audit"
                ])

                execute(prefix + "_05_adjudication", [
                    supplemental_scripts / "run_supplemental_adjudication.py",
                    "--component-audit", required(ea),
                    "--output-dir", cluster / "adjudication"
                ])

                result = read_json(required(final))

                if digest(core) != frozen_core_hash:
                    raise ValueError("Frozen core Agent 3 decision changed")

                if (
                    result.get("core_agent3_decision") != "UNRESOLVED"
                    or result.get("core_agent3_output_changed") is not False
                    or result.get("supplemental_decision_state")
                    not in {"RESOLVED", "AMBIGUOUS", "UNRESOLVED"}
                ):
                    raise ValueError("Invalid supplemental adjudication")

                policy = result.get("adjudication_policy", {})

                if (
                    policy.get("reference_label_used") is not False
                    or policy.get("expert_label_used") is not False
                ):
                    raise ValueError("Supplemental label-isolation check failed")

                manifest["supplemental"].append({
                    "cluster_id": cid,
                    "core_decision": "UNRESOLVED",
                    "supplemental_decision": result["supplemental_decision_state"]
                })

                save()

            if not manifest["supplemental"]:
                manifest["supplemental_status"] = (
                    "NO_UNRESOLVED_ROUTED_CLUSTERS"
                )

        manifest["status"] = "PASS"

    except Exception as exc:
        manifest["status"] = "REVIEW_REQUIRED"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        print("REVIEW REQUIRED:", manifest["error"])

    finally:
        changed = [
            str(p) for p in protected
            if not p.is_file() or digest(p) != before[str(p)]
        ]

        manifest["protected_input_sha256_pass"] = not changed
        manifest["protected_input_changes"] = changed
        manifest["completed_utc"] = datetime.now(
            timezone.utc
        ).isoformat()

        save()

        print("\n=== FINAL VALIDATION ===")
        print("Pipeline:", manifest["status"])
        print(
            "Original input integrity:",
            "PASS" if not changed else "REVIEW REQUIRED"
        )
        print("Run manifest:", output / "run_manifest.json")

    return (
        0 if (
            manifest["status"] == "PASS"
            and manifest["protected_input_sha256_pass"]
        ) else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
