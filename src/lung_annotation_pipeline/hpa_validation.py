from __future__ import annotations

import csv
import hashlib
import io
import math
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class HPAConfig:
    query_tissue: str = "lung"
    source_name: str = "Human Protein Atlas"
    source_release: str = "25.1"
    evidence_scope: str = "healthy_lung_single_cell_rna_expression"
    minimum_supporting_genes: int = 2
    minimum_positive_ncpm: float = 0.0

    def validate(self) -> None:
        if not self.query_tissue.strip():
            raise ValueError("query_tissue cannot be empty.")
        if self.minimum_supporting_genes < 1:
            raise ValueError("minimum_supporting_genes must be at least 1.")
        if self.minimum_positive_ncpm < 0:
            raise ValueError("minimum_positive_ncpm cannot be negative.")


@dataclass(frozen=True, slots=True)
class MarkerGene:
    cluster_id: str
    rank: int
    gene: str
    log_fold_change: float
    adjusted_p_value: float


@dataclass(frozen=True, slots=True)
class HPAGeneEvidence:
    gene: str
    top_cell_type: str
    top_max_ncpm: float
    top_max_read_count: float
    top_positive_record_count: int
    second_cell_type: str
    second_max_ncpm: float
    detected_cell_type_count: int
    top_to_second_ratio: float | None
    specificity_fraction: float


@dataclass(frozen=True, slots=True)
class HPAScanResult:
    evidence_by_gene: dict[str, HPAGeneEvidence]
    genes_present_anywhere: frozenset[str]
    archive_member: str
    source_sha256: str
    scanned_row_count: int
    matched_selected_gene_row_count: int
    positive_lung_row_count: int


@dataclass(frozen=True, slots=True)
class HPACandidate:
    cluster_id: str
    rank: int
    cell_type: str
    broad_lineage: str
    selected_for_downstream: bool
    supporting_genes: tuple[str, ...]
    specificity_weighted_support: float
    mean_marker_rank: float
    mean_log_fold_change: float
    mean_top_ncpm: float
    max_top_ncpm: float


def _finite_float(
    value: str | None,
    field: str,
    row_number: int,
) -> float:
    raw = "" if value is None else value.strip()

    try:
        parsed = float(raw)
    except ValueError as error:
        raise ValueError(
            f"{field} is not numeric at row {row_number}: {raw!r}."
        ) from error

    if not math.isfinite(parsed):
        raise ValueError(
            f"{field} must be finite at row {row_number}."
        )

    return parsed


def _cluster_sort_key(
    value: str,
) -> tuple[int, int | str]:
    try:
        return 0, int(value)
    except ValueError:
        return 1, value.casefold()


def normalize_label(value: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        value.casefold(),
    ).strip()


def label_key(value: str) -> str:
    normalized = normalize_label(value)

    if normalized.endswith(" cells"):
        return normalized[:-1]

    return normalized


def broad_lineage(label: str) -> str:
    value = normalize_label(label)

    if "mast" in value:
        return "Mast cell"

    if "plasma" in value or "plasmablast" in value:
        return "Plasma cell"

    if (
        value in {"b cell", "b cells"}
        or "follicular b" in value
    ):
        return "B cell"

    if (
        "t cell" in value
        or value in {"t cells", "t cell"}
    ):
        return "T cell"

    if (
        "natural killer" in value
        or value in {"nk", "nk cells"}
    ):
        return "Natural killer cell"

    if "macrophage" in value:
        return "Macrophage"

    if "monocyte" in value:
        return "Monocyte"

    if "neutrophil" in value:
        return "Neutrophil"

    if (
        value in {"cdc", "pdcs", "pdc"}
        or "dendritic" in value
    ):
        return "Dendritic cell"

    if "endothelial" in value:
        return "Endothelial cell"

    if "fibroblast" in value:
        return "Fibroblast"

    if (
        "pericyte" in value
        or "smooth muscle" in value
    ):
        return "Mural cell"

    epithelial_terms = (
        "alveolar",
        "respiratory",
        "ionocyte",
        "glandular",
        "epithelial",
    )

    if any(term in value for term in epithelial_terms):
        return "Epithelial cell"

    if "mesothelial" in value:
        return "Mesothelial cell"

    if "schwann" in value:
        return "Schwann cell"

    if "innate lymphoid" in value:
        return "Innate lymphoid cell"

    if "myeloid" in value:
        return "Myeloid cell"

    if "stromal" in value:
        return "Stromal cell"

    return label.strip()


def load_processed_markers(
    path: str | Path,
) -> dict[str, list[MarkerGene]]:
    """
    Load the exact processed genes and ranks already used by Agent 2.

    HPA must not independently select or reorder the marker genes.
    """

    marker_path = Path(path)

    if not marker_path.is_file():
        raise FileNotFoundError(
            f"Processed marker CSV not found: {marker_path}"
        )

    required = {
        "cluster_id",
        "rank_within_cluster",
        "gene",
        "log_fold_change",
        "adjusted_p_value",
    }

    markers: dict[str, list[MarkerGene]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()

    with marker_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        missing = sorted(
            required - set(reader.fieldnames or [])
        )

        if missing:
            raise ValueError(
                "Processed marker CSV is missing columns: "
                + ", ".join(missing)
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            cluster_id = (
                row.get("cluster_id") or ""
            ).strip()

            gene = (
                row.get("gene") or ""
            ).strip().upper()

            if not cluster_id or not gene:
                raise ValueError(
                    "Missing cluster_id or gene at "
                    f"row {row_number}."
                )

            pair = (cluster_id, gene)

            if pair in seen:
                raise ValueError(
                    "Duplicate cluster/gene pair at "
                    f"row {row_number}: {pair}"
                )

            seen.add(pair)

            rank_value = _finite_float(
                row.get("rank_within_cluster"),
                "rank_within_cluster",
                row_number,
            )

            if (
                not rank_value.is_integer()
                or rank_value < 1
            ):
                raise ValueError(
                    "rank_within_cluster must be a positive "
                    f"integer at row {row_number}."
                )

            markers[cluster_id].append(
                MarkerGene(
                    cluster_id=cluster_id,
                    rank=int(rank_value),
                    gene=gene,
                    log_fold_change=_finite_float(
                        row.get("log_fold_change"),
                        "log_fold_change",
                        row_number,
                    ),
                    adjusted_p_value=_finite_float(
                        row.get("adjusted_p_value"),
                        "adjusted_p_value",
                        row_number,
                    ),
                )
            )

    if not markers:
        raise ValueError(
            "Processed marker CSV contains no marker rows."
        )

    ordered: dict[str, list[MarkerGene]] = {}

    for cluster_id in sorted(
        markers,
        key=_cluster_sort_key,
    ):
        cluster_markers = sorted(
            markers[cluster_id],
            key=lambda item: item.rank,
        )

        expected_ranks = list(
            range(1, len(cluster_markers) + 1)
        )

        actual_ranks = [
            item.rank
            for item in cluster_markers
        ]

        if actual_ranks != expected_ranks:
            raise ValueError(
                f"Cluster {cluster_id} ranks must be "
                "consecutive from 1; "
                f"found {actual_ranks}."
            )

        ordered[cluster_id] = cluster_markers

    return ordered


def load_csv_rows(
    path: str | Path,
    required_columns: set[str],
    description: str,
) -> list[dict[str, str]]:
    csv_path = Path(path)

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"{description} not found: {csv_path}"
        )

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        missing = sorted(
            required_columns - set(reader.fieldnames or [])
        )

        if missing:
            raise ValueError(
                f"{description} is missing columns: "
                + ", ".join(missing)
            )

        rows = [dict(row) for row in reader]

    if not rows:
        raise ValueError(
            f"{description} contains no rows."
        )

    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _select_tsv_member(
    archive: zipfile.ZipFile,
) -> str:
    members = [
        item.filename
        for item in archive.infolist()
        if (
            not item.is_dir()
            and item.filename.casefold().endswith(
                (".tsv", ".txt")
            )
        )
    ]

    if len(members) != 1:
        raise ValueError(
            "HPA ZIP must contain exactly one TSV/TXT "
            f"file; found {len(members)}."
        )

    return members[0]


def scan_hpa(
    path: str | Path,
    selected_genes: set[str],
    config: HPAConfig,
) -> HPAScanResult:
    """
    Stream the large HPA ZIP once.

    Only HPA rows for the selected marker genes are retained.
    The 1.47 GB TSV is never loaded completely into memory.
    """

    config.validate()

    hpa_path = Path(path)

    if not hpa_path.is_file():
        raise FileNotFoundError(
            f"HPA ZIP not found: {hpa_path}"
        )

    genes = {
        gene.strip().upper()
        for gene in selected_genes
        if gene.strip()
    }

    if not genes:
        raise ValueError(
            "selected_genes cannot be empty."
        )

    # gene -> cell type ->
    # [maximum nCPM, maximum read count, positive row count]
    aggregates: dict[
        str,
        dict[str, list[float]],
    ] = defaultdict(
        lambda: defaultdict(
            lambda: [0.0, 0.0, 0.0]
        )
    )

    genes_present: set[str] = set()
    scanned_rows = 0
    matched_rows = 0
    positive_lung_rows = 0

    with zipfile.ZipFile(
        hpa_path,
        "r",
    ) as archive:
        member = _select_tsv_member(archive)

        with archive.open(
            member,
            "r",
        ) as binary_handle:
            with io.TextIOWrapper(
                binary_handle,
                encoding="utf-8-sig",
                newline="",
            ) as text_handle:
                reader = csv.DictReader(
                    text_handle,
                    delimiter="\t",
                )

                required = {
                    "Gene",
                    "Gene name",
                    "Tissue",
                    "Cluster",
                    "Cell type",
                    "Read count",
                    "nCPM",
                }

                missing = sorted(
                    required - set(reader.fieldnames or [])
                )

                if missing:
                    raise ValueError(
                        "HPA TSV is missing columns: "
                        + ", ".join(missing)
                    )

                for row_number, row in enumerate(
                    reader,
                    start=2,
                ):
                    scanned_rows += 1

                    gene = (
                        row.get("Gene name") or ""
                    ).strip().upper()

                    if gene not in genes:
                        continue

                    matched_rows += 1
                    genes_present.add(gene)

                    tissue = (
                        row.get("Tissue") or ""
                    ).strip().casefold()

                    if tissue != config.query_tissue.casefold():
                        continue

                    ncpm = _finite_float(
                        row.get("nCPM"),
                        "nCPM",
                        row_number,
                    )

                    read_count = _finite_float(
                        row.get("Read count"),
                        "Read count",
                        row_number,
                    )

                    if ncpm < 0 or read_count < 0:
                        raise ValueError(
                            "HPA nCPM/read count cannot be "
                            f"negative at row {row_number}."
                        )

                    if ncpm <= config.minimum_positive_ncpm:
                        continue

                    cell_type = (
                        row.get("Cell type") or ""
                    ).strip()

                    if not cell_type:
                        raise ValueError(
                            "HPA Cell type is empty at "
                            f"row {row_number}."
                        )

                    positive_lung_rows += 1

                    stats = aggregates[gene][cell_type]
                    stats[0] = max(stats[0], ncpm)
                    stats[1] = max(stats[1], read_count)
                    stats[2] += 1

    evidence_by_gene: dict[
        str,
        HPAGeneEvidence,
    ] = {}

    for gene, cell_types in aggregates.items():
        ranked = sorted(
            cell_types.items(),
            key=lambda item: (
                -item[1][0],
                -item[1][1],
                item[0].casefold(),
            ),
        )

        top_cell_type, top = ranked[0]

        if len(ranked) > 1:
            second_cell_type, second = ranked[1]
        else:
            second_cell_type = ""
            second = [0.0, 0.0, 0.0]

        total_max_ncpm = sum(
            values[0]
            for values in cell_types.values()
        )

        ratio = (
            top[0] / second[0]
            if second[0] > 0
            else None
        )

        evidence_by_gene[gene] = HPAGeneEvidence(
            gene=gene,
            top_cell_type=top_cell_type,
            top_max_ncpm=top[0],
            top_max_read_count=top[1],
            top_positive_record_count=int(top[2]),
            second_cell_type=second_cell_type,
            second_max_ncpm=second[0],
            detected_cell_type_count=len(ranked),
            top_to_second_ratio=ratio,
            specificity_fraction=(
                top[0] / total_max_ncpm
                if total_max_ncpm
                else 0.0
            ),
        )

    return HPAScanResult(
        evidence_by_gene=evidence_by_gene,
        genes_present_anywhere=frozenset(
            genes_present
        ),
        archive_member=member,
        source_sha256=_sha256(hpa_path),
        scanned_row_count=scanned_rows,
        matched_selected_gene_row_count=matched_rows,
        positive_lung_row_count=positive_lung_rows,
    )


def build_candidates(
    markers_by_cluster: dict[
        str,
        list[MarkerGene],
    ],
    scan: HPAScanResult,
    config: HPAConfig,
) -> tuple[
    list[HPACandidate],
    list[dict[str, object]],
]:
    """
    Convert gene-level HPA evidence into cluster candidates.

    Each marker gene contributes one vote: the healthy-lung
    cell type in which that gene has its highest HPA nCPM.
    """

    all_candidates: list[HPACandidate] = []
    gene_rows: list[dict[str, object]] = []

    for cluster_id, markers in markers_by_cluster.items():
        grouped: dict[
            str,
            list[
                tuple[
                    MarkerGene,
                    HPAGeneEvidence,
                ]
            ],
        ] = defaultdict(list)

        for marker in markers:
            evidence = scan.evidence_by_gene.get(
                marker.gene
            )

            if evidence is not None:
                grouped[
                    evidence.top_cell_type
                ].append(
                    (marker, evidence)
                )

                status = (
                    "hpa_lung_expression_found"
                )

            elif marker.gene in scan.genes_present_anywhere:
                status = (
                    "gene_found_without_positive_lung_expression"
                )

            else:
                status = "gene_not_found_in_hpa"

            gene_rows.append(
                {
                    "cluster_id": cluster_id,
                    "marker_rank": marker.rank,
                    "gene": marker.gene,
                    "log_fold_change": (
                        marker.log_fold_change
                    ),
                    "adjusted_p_value": (
                        marker.adjusted_p_value
                    ),
                    "hpa_status": status,
                    "hpa_top_cell_type": (
                        evidence.top_cell_type
                        if evidence
                        else ""
                    ),
                    "hpa_top_broad_lineage": (
                        broad_lineage(
                            evidence.top_cell_type
                        )
                        if evidence
                        else ""
                    ),
                    "hpa_top_max_ncpm": (
                        evidence.top_max_ncpm
                        if evidence
                        else ""
                    ),
                    "hpa_second_cell_type": (
                        evidence.second_cell_type
                        if evidence
                        else ""
                    ),
                    "hpa_second_max_ncpm": (
                        evidence.second_max_ncpm
                        if evidence
                        else ""
                    ),
                    "hpa_top_to_second_ratio": (
                        evidence.top_to_second_ratio
                        if (
                            evidence
                            and evidence.top_to_second_ratio
                            is not None
                        )
                        else ""
                    ),
                    "hpa_detected_cell_type_count": (
                        evidence.detected_cell_type_count
                        if evidence
                        else 0
                    ),
                    "hpa_specificity_fraction": (
                        evidence.specificity_fraction
                        if evidence
                        else ""
                    ),
                    "evidence_scope": (
                        config.evidence_scope
                    ),
                }
            )

        unranked: list[HPACandidate] = []

        for cell_type, pairs in grouped.items():
            pairs.sort(
                key=lambda item: item[0].rank
            )

            count = len(pairs)

            unranked.append(
                HPACandidate(
                    cluster_id=cluster_id,
                    rank=0,
                    cell_type=cell_type,
                    broad_lineage=broad_lineage(
                        cell_type
                    ),
                    selected_for_downstream=(
                        count
                        >= config.minimum_supporting_genes
                    ),
                    supporting_genes=tuple(
                        marker.gene
                        for marker, _ in pairs
                    ),
                    specificity_weighted_support=sum(
                        evidence.specificity_fraction
                        for _, evidence in pairs
                    ),
                    mean_marker_rank=(
                        sum(
                            marker.rank
                            for marker, _ in pairs
                        )
                        / count
                    ),
                    mean_log_fold_change=(
                        sum(
                            marker.log_fold_change
                            for marker, _ in pairs
                        )
                        / count
                    ),
                    mean_top_ncpm=(
                        sum(
                            evidence.top_max_ncpm
                            for _, evidence in pairs
                        )
                        / count
                    ),
                    max_top_ncpm=max(
                        evidence.top_max_ncpm
                        for _, evidence in pairs
                    ),
                )
            )

        unranked.sort(
            key=lambda item: (
                -len(item.supporting_genes),
                -item.specificity_weighted_support,
                item.mean_marker_rank,
                -item.mean_log_fold_change,
                item.cell_type.casefold(),
            )
        )

        for rank, candidate in enumerate(
            unranked,
            start=1,
        ):
            all_candidates.append(
                HPACandidate(
                    cluster_id=candidate.cluster_id,
                    rank=rank,
                    cell_type=candidate.cell_type,
                    broad_lineage=(
                        candidate.broad_lineage
                    ),
                    selected_for_downstream=(
                        candidate.selected_for_downstream
                    ),
                    supporting_genes=(
                        candidate.supporting_genes
                    ),
                    specificity_weighted_support=(
                        candidate
                        .specificity_weighted_support
                    ),
                    mean_marker_rank=(
                        candidate.mean_marker_rank
                    ),
                    mean_log_fold_change=(
                        candidate.mean_log_fold_change
                    ),
                    mean_top_ncpm=(
                        candidate.mean_top_ncpm
                    ),
                    max_top_ncpm=(
                        candidate.max_top_ncpm
                    ),
                )
            )

    return all_candidates, gene_rows