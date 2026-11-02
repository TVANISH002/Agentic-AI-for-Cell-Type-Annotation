
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from literature_retrieval_agent.runtime_context import (
    Agent3RuntimeContext,
)


EUTILS = (
    "https://eutils.ncbi.nlm.nih.gov/"
    "entrez/eutils"
)


def clean(value: Any) -> str:
    return " ".join(
        str(value or "").split()
    )


def extract_gene(value: Any) -> str:

    if isinstance(
        value,
        Mapping,
    ):
        for key in (
            "GENE",
            "gene",
            "SYMBOL",
            "symbol",
            "MARKER",
            "marker",
        ):
            gene = clean(
                value.get(key)
            )

            if gene:
                return gene.upper()

        return ""

    return clean(
        value
    ).upper()


def split_terms(
    value: str,
) -> list[str]:

    value = clean(value)

    if not value:
        return []

    pieces = []

    for section in value.split("|"):
        for item in section.split(","):

            item = clean(item)

            if item:
                pieces.append(item)

    return pieces


def unique_terms(
    values,
) -> list[str]:

    seen = set()
    output = []

    for value in values:

        value = clean(value)

        if not value:
            continue

        key = value.casefold()

        if key in seen:
            continue

        seen.add(key)
        output.append(value)

    return output


def build_gene_search_term(
    symbol: str,
    organism: str,
) -> str:

    symbol = clean(symbol)
    organism = clean(organism)

    if not symbol:
        raise ValueError(
            "Gene symbol is required."
        )

    if not organism:
        raise ValueError(
            "NCBI organism is required."
        )

    return (
        f'{symbol}[Gene Name] '
        f'AND "{organism}"[Organism]'
    )


def fetch_json(
    endpoint: str,
    params: dict,
) -> dict:

    url = (
        f"{EUTILS}/{endpoint}?"
        + urlencode(params)
    )

    request = Request(
        url,
        headers={
            "User-Agent":
                "AgenticCellAnnotation/1.0"
        },
    )

    with urlopen(
        request,
        timeout=30,
    ) as response:

        return json.loads(
            response.read().decode(
                "utf-8"
            )
        )


def fetch_gene_record(
    symbol: str,
    organism: str,
) -> dict:

    search = fetch_json(
        "esearch.fcgi",
        {
            "db": "gene",

            "term":
                build_gene_search_term(
                    symbol,
                    organism,
                ),

            "retmode":
                "json",

            "retmax":
                20,
        },
    )

    ids = (
        search
        .get(
            "esearchresult",
            {},
        )
        .get(
            "idlist",
            [],
        )
    )

    if not ids:
        raise RuntimeError(
            f"No NCBI Gene record found "
            f"for {symbol} in {organism}."
        )

    # Do not trust the first NCBI search result.
    # Require an exact official-symbol match.
    summary = fetch_json(
        "esummary.fcgi",
        {
            "db":
                "gene",

            "id":
                ",".join(ids),

            "retmode":
                "json",
        },
    )

    result = summary.get(
        "result",
        {},
    )

    exact_matches = []

    for candidate_id in ids:

        candidate_doc = result.get(
            str(candidate_id),
            {},
        )

        if not isinstance(
            candidate_doc,
            dict,
        ):
            continue

        candidate_symbol = clean(
            candidate_doc.get(
                "nomenclaturesymbol"
            )
            or candidate_doc.get(
                "name"
            )
        )

        if (
            candidate_symbol
            and candidate_symbol.casefold()
            == symbol.casefold()
        ):
            exact_matches.append(
                (
                    str(candidate_id),
                    candidate_doc,
                )
            )

    if not exact_matches:

        returned_symbols = []

        for candidate_id in ids:

            candidate_doc = result.get(
                str(candidate_id),
                {},
            )

            if not isinstance(
                candidate_doc,
                dict,
            ):
                continue

            candidate_symbol = clean(
                candidate_doc.get(
                    "nomenclaturesymbol"
                )
                or candidate_doc.get(
                    "name"
                )
            )

            if candidate_symbol:
                returned_symbols.append(
                    candidate_symbol
                )

        raise RuntimeError(
            f"No exact official NCBI Gene "
            f"symbol match found for {symbol} "
            f"in {organism}. "
            f"Returned symbols: "
            f"{returned_symbols}"
        )

    if len(exact_matches) > 1:
        raise RuntimeError(
            f"Multiple exact official-symbol "
            f"matches found for {symbol} "
            f"in {organism}: "
            f"{[x[0] for x in exact_matches]}"
        )

    gene_id, doc = (
        exact_matches[0]
    )

    official_symbol = clean(
        doc.get(
            "nomenclaturesymbol"
        )
        or doc.get(
            "name"
        )
        or symbol
    )

    official_name = clean(
        doc.get(
            "nomenclaturename"
        )
        or doc.get(
            "description"
        )
    )

    aliases = split_terms(
        doc.get(
            "otheraliases",
            "",
        )
    )

    designations = split_terms(
        doc.get(
            "otherdesignations",
            "",
        )
    )

    terminology = unique_terms(
        [
            official_symbol,
            official_name,
            doc.get(
                "description"
            ),
            *aliases,
            *designations,
        ]
    )

    return {
        "query_symbol":
            symbol,

        "ncbi_gene_id":
            str(
                gene_id
            ),

        "official_symbol":
            official_symbol,

        "official_full_name":
            official_name,

        "aliases":
            aliases,

        "other_designations":
            designations,

        "terminology":
            terminology,

        "exact_official_symbol_match":
            True,

        "resolution_status":
            "EXACT_NCBI_MATCH",

        "lookup_warning":
            None,
    }


def build_fallback_record(
    symbol: str,
    warning: str,
) -> dict:
    """
    Preserve the original marker when it cannot be mapped
    to one exact official NCBI Gene symbol.

    Composite or panel-derived markers are not split or
    rewritten into a more specific gene.
    """

    symbol = clean(
        symbol
    ).upper()

    return {
        "query_symbol":
            symbol,

        "ncbi_gene_id":
            None,

        # Retained for downstream compatibility.
        # The resolution_status explicitly indicates that this is NOT an exact official NCBI Gene match.
        "official_symbol":
            symbol,

        "official_full_name":
            None,

        "aliases":
            [],

        "other_designations":
            [],

        "terminology":
            [symbol],

        "exact_official_symbol_match":
            False,

        "resolution_status":
            "FALLBACK_ORIGINAL_MARKER",

        "lookup_warning":
            clean(
                warning
            ),
    }


class RescueGeneTerminologyBuilder:

    def build(
        self,
        rescue_plan: Mapping[
            str,
            Any,
        ],
        literature_request: Mapping[
            str,
            Any,
        ],
        *,
        request_delay_seconds: float = 0.4,
    ) -> dict:

        context = (
            Agent3RuntimeContext
            .from_mapping(
                literature_request
            )
        )

        if str(
            rescue_plan.get(
                "cluster_id"
            )
        ) != str(
            context.cluster_id
        ):
            raise ValueError(
                "Cluster mismatch between "
                "rescue plan and literature request."
            )

        markers = []

        for item in rescue_plan.get(
            "rescue_markers",
            [],
        ):

            gene = extract_gene(
                item
            )

            if (
                gene
                and gene not in markers
            ):
                markers.append(
                    gene
                )

        if not markers:
            raise RuntimeError(
                "No rescue markers found."
            )

        records = []

        for gene in markers:

            try:
                record = fetch_gene_record(
                    gene,
                    context.ncbi_organism,
                )

            except (
                HTTPError,
                URLError,
                TimeoutError,
                RuntimeError,
            ) as exc:

                print(
                    f"Warning: NCBI gene terminology "
                    f"lookup failed for {gene}; "
                    f"using fallback original marker."
                )

                record = build_fallback_record(
                    gene,
                    str(exc),
                )

            records.append(
                record
            )

            if (
                request_delay_seconds
                > 0
            ):
                time.sleep(
                    request_delay_seconds
                )

        fallback_count = sum(
            1
            for record in records
            if not record.get(
                "exact_official_symbol_match",
                False,
            )
        )

        return {
            "schema_name":
                "RescueMarkerGeneTerminology",

            "schema_version":
                "2.1",

            "cluster_id":
                str(
                    context.cluster_id
                ),

            "species":
                context.species,

            "source":
                "NCBI Gene",

            "organism":
                context.ncbi_organism,

            "retrieved_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "reference_label_used":
                False,

            "expert_label_used":
                False,

            "marker_count":
                len(
                    records
                ),

            "exact_match_count":
                len(records)
                - fallback_count,

            "fallback_marker_count":
                fallback_count,

            "genes":
                records,
        }
    