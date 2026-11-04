from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


EUTILS = (
    "https://eutils.ncbi.nlm.nih.gov/"
    "entrez/eutils"
)


def clean(value) -> str:
    return " ".join(
        str(value or "").split()
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


def extract_gene(value) -> str:
    if isinstance(value, dict):
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

    return clean(value).upper()


def split_terms(value: str) -> list[str]:
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


def unique_terms(values):
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


def fetch_gene_record(
    symbol: str,
) -> dict:
    search = fetch_json(
        "esearch.fcgi",
        {
            "db": "gene",
            "term": (
                f'{symbol}[Gene Name] '
                'AND "Homo sapiens"[Organism]'
            ),
            "retmode": "json",
            "retmax": 20,
        },
    )

    ids = (
        search
        .get("esearchresult", {})
        .get("idlist", [])
    )

    if not ids:
        raise RuntimeError(
            f"No human NCBI Gene record "
            f"found for {symbol}."
        )

    #
    # Do NOT trust the first ESearch hit.
    #
    # NCBI Gene search may return records where the
    # requested symbol occurs only as an alias or
    # alternate designation. For annotation evidence
    # provenance we require an exact official-symbol
    # match.
    #
    summary = fetch_json(
        "esummary.fcgi",
        {
            "db": "gene",
            "id": ",".join(ids),
            "retmode": "json",
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
            f"No exact official NCBI Gene symbol "
            f"match found for {symbol}. "
            f"Returned symbols: "
            f"{returned_symbols}"
        )

    if len(exact_matches) > 1:
        raise RuntimeError(
            f"Multiple exact official-symbol "
            f"matches found for {symbol}: "
            f"{[item[0] for item in exact_matches]}"
        )

    gene_id, doc = exact_matches[0]

    official_symbol = clean(
        doc.get(
            "nomenclaturesymbol"
        )
        or doc.get("name")
        or symbol
    )

    official_name = clean(
        doc.get(
            "nomenclaturename"
        )
        or doc.get("description")
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
            doc.get("description"),
            *aliases,
            *designations,
        ]
    )

    return {
        "query_symbol":
            symbol,

        "ncbi_gene_id":
            str(gene_id),

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
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rescue-plan",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-json",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    plan = json.loads(
        args.rescue_plan.read_text(
            encoding="utf-8-sig"
        )
    )

    markers = []

    for item in plan.get(
        "rescue_markers",
        [],
    ):
        gene = extract_gene(item)

        if (
            gene
            and gene not in markers
        ):
            markers.append(gene)

    if not markers:
        raise RuntimeError(
            "No rescue markers found."
        )

    records = []

    for index, gene in enumerate(
        markers,
        start=1,
    ):
        print(
            f"[{index}/{len(markers)}] "
            f"NCBI Gene: {gene}"
        )

        records.append(
            fetch_gene_record(
                gene
            )
        )

        #
        # Remain comfortably below
        # unauthenticated NCBI request limits.
        #
        time.sleep(0.4)

    output = {
        "schema_name":
            "RescueMarkerGeneTerminology",

        "schema_version":
            "1.0",

        "cluster_id":
            str(
                plan.get(
                    "cluster_id",
                    "",
                )
            ),

        "source":
            "NCBI Gene",

        "organism":
            "Homo sapiens",

        "retrieved_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "reference_label_used":
            False,

        "expert_label_used":
            False,

        "marker_count":
            len(records),

        "genes":
            records,
    }

    args.output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output_json.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Gene terminology retrieval "
        "completed successfully."
    )

    for record in records:
        print()
        print(
            record["official_symbol"],
            "=>",
            record["official_full_name"],
        )

        print(
            "NCBI Gene ID:",
            record["ncbi_gene_id"],
        )

        print(
            "Terminology:",
            "; ".join(
                record["terminology"][:15]
            ),
        )

    print()
    print(
        "Saved:",
        args.output_json,
    )


if __name__ == "__main__":
    main()
