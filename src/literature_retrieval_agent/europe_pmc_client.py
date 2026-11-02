from __future__ import annotations

import html
import re
import time
from typing import Any, Mapping

import requests

from literature_retrieval_agent.config import (
    DEFAULT_CONFIG,
    LiteratureRetrievalConfig,
)
from literature_retrieval_agent.schemas import (
    LiteratureQuery,
    PublicationRecord,
    QueryRetrievalResult,
)


def _as_bool(value: Any) -> bool:
    """Convert Europe PMC truth values into booleans."""

    return str(value or "").strip().casefold() in {
        "true",
        "t",
        "yes",
        "y",
        "1",
    }


def _clean_markup(value: Any) -> str | None:
    """Remove HTML tags and normalize whitespace."""

    text = html.unescape(
        str(value or "")
    )

    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    text = " ".join(
        text.split()
    )

    return text or None


def _clean_doi(value: Any) -> str | None:
    """Normalize a DOI into its stable identifier."""

    doi = str(value or "").strip()

    if not doi:
        return None

    lowered = doi.casefold()

    prefixes = (
        "https://doi.org/",
        "http://doi.org/",
        "doi:",
    )

    for prefix in prefixes:
        if lowered.startswith(prefix):
            doi = doi[len(prefix):].strip()
            break

    return doi or None


class EuropePMCClient:
    """
    Retrying client for the official Europe PMC
    REST search API.
    """

    def __init__(
        self,
        config: LiteratureRetrievalConfig = DEFAULT_CONFIG,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config

        self.session = (
            session
            if session is not None
            else requests.Session()
        )

        self._owns_session = session is None

        self.session.headers.update(
            {
                "User-Agent": config.user_agent,
            }
        )

    def close(self) -> None:
        """Close an internally created HTTP session."""

        if self._owns_session:
            self.session.close()

    def __enter__(self) -> "EuropePMCClient":
        return self

    def __exit__(
        self,
        *_: object,
    ) -> None:
        self.close()

    def _execute(
        self,
        params: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Execute one request with retry and backoff."""

        last_error: Exception | None = None

        for attempt in range(
            1,
            self.config.max_retries + 1,
        ):
            try:
                response = self.session.get(
                    self.config.europe_pmc_endpoint,
                    params=dict(params),
                    timeout=self.config.timeout_seconds,
                )

                response.raise_for_status()

                payload = response.json()

                if not isinstance(payload, dict):
                    raise ValueError(
                        "Europe PMC returned a "
                        "non-object JSON root."
                    )

                return payload

            except (
                requests.RequestException,
                ValueError,
            ) as exc:
                last_error = exc

                if attempt == self.config.max_retries:
                    break

                delay = (
                    self.config.retry_backoff_seconds
                    * (2 ** (attempt - 1))
                )

                if delay > 0:
                    time.sleep(delay)

        assert last_error is not None

        raise RuntimeError(
            "Europe PMC request failed after "
            f"{self.config.max_retries} attempts: "
            f"{last_error}"
        ) from last_error

    @staticmethod
    def parse_publication(
        record: Mapping[str, Any],
    ) -> PublicationRecord:
        """Normalize one Europe PMC result."""

        source = str(
            record.get("source")
            or "UNKNOWN"
        ).strip()

        source_id = str(
            record.get("id")
            or record.get("pmid")
            or ""
        ).strip()

        if not source_id:
            raise ValueError(
                "Europe PMC record contains no "
                "stable source ID."
            )

        doi = _clean_doi(
            record.get("doi")
        )

        pmid = (
            str(
                record.get("pmid")
                or ""
            ).strip()
            or None
        )

        pmcid = (
            str(
                record.get("pmcid")
                or ""
            ).strip()
            or None
        )

        if doi:
            record_key = (
                f"doi:{doi.casefold()}"
            )
        elif pmid:
            record_key = f"pmid:{pmid}"
        elif pmcid:
            record_key = (
                f"pmcid:{pmcid.casefold()}"
            )
        else:
            record_key = (
                f"{source.casefold()}:"
                f"{source_id.casefold()}"
            )

        author_list = (
            record.get("authorList")
            or {}
        )

        if isinstance(author_list, Mapping):
            author_records = (
                author_list.get("author")
                or []
            )
        else:
            author_records = []

        authors: list[str] = []

        for author in author_records:
            if not isinstance(
                author,
                Mapping,
            ):
                continue

            name = str(
                author.get("fullName")
                or author.get("collectiveName")
                or ""
            ).strip()

            if name:
                authors.append(name)

        if not authors:
            author_string = str(
                record.get("authorString")
                or ""
            ).strip()

            if author_string:
                authors = [author_string]

        journal_info = (
            record.get("journalInfo")
            or {}
        )

        if isinstance(
            journal_info,
            Mapping,
        ):
            journal_record = (
                journal_info.get("journal")
                or {}
            )
        else:
            journal_record = {}

        journal: str | None = None

        if isinstance(
            journal_record,
            Mapping,
        ):
            journal = (
                str(
                    journal_record.get("title")
                    or journal_record.get(
                        "medlineAbbreviation"
                    )
                    or ""
                ).strip()
                or None
            )

        publication_type_data = (
            record.get("pubTypeList")
            or {}
        )

        if isinstance(
            publication_type_data,
            Mapping,
        ):
            publication_types_raw = (
                publication_type_data.get(
                    "pubType"
                )
                or []
            )
        else:
            publication_types_raw = []

        if isinstance(
            publication_types_raw,
            str,
        ):
            publication_types_raw = [
                publication_types_raw
            ]

        publication_types = [
            str(item).strip()
            for item in publication_types_raw
            if str(item).strip()
        ]

        raw_year = str(
            record.get("pubYear")
            or ""
        ).strip()

        publication_year = (
            int(raw_year)
            if raw_year.isdigit()
            else None
        )

        is_preprint = (
            source.casefold() == "ppr"
            or _as_bool(
                record.get("isPreprint")
            )
            or any(
                "preprint" in item.casefold()
                for item in publication_types
            )
        )

        title = _clean_markup(
            record.get("title")
        )

        if not title:
            raise ValueError(
                "Europe PMC record "
                f"{source}:{source_id} "
                "contains no title."
            )

        raw_citation_count = (
            record.get("citedByCount")
        )

        try:
            cited_by_count = max(
                0,
                int(raw_citation_count or 0),
            )
        except (
            TypeError,
            ValueError,
        ):
            cited_by_count = 0

        return PublicationRecord(
            record_key=record_key,
            source=source,
            source_id=source_id,
            pmid=pmid,
            pmcid=pmcid,
            doi=doi,
            title=title,
            authors=authors,
            journal=journal,
            publication_year=publication_year,
            first_publication_date=(
                str(
                    record.get(
                        "firstPublicationDate"
                    )
                    or ""
                ).strip()
                or None
            ),
            publication_types=publication_types,
            abstract=_clean_markup(
                record.get("abstractText")
            ),
            is_open_access=_as_bool(
                record.get("isOpenAccess")
            ),
            is_preprint=is_preprint,
            cited_by_count=cited_by_count,
        )

    def search(
        self,
        query: LiteratureQuery,
    ) -> QueryRetrievalResult:
        """Search Europe PMC for one planned query."""

        executed_query = query.query

        if self.config.open_access_only:
            executed_query = (
                f"({executed_query}) "
                "AND OPEN_ACCESS:Y"
            )

        payload = self._execute(
            {
                "query": executed_query,
                "format": "json",
                "resultType": (
                    self.config.result_type
                ),
                "pageSize": (
                    self.config.page_size
                ),
                "synonym": str(
                    self.config.synonym_expansion
                ).lower(),
            }
        )

        result_list = (
            payload.get("resultList")
            or {}
        )

        if isinstance(
            result_list,
            Mapping,
        ):
            raw_results = (
                result_list.get("result")
                or []
            )
        else:
            raw_results = []

        papers: list[
            PublicationRecord
        ] = []

        warnings: list[str] = []

        for record in raw_results:
            try:
                papers.append(
                    self.parse_publication(record)
                )
            except (
                TypeError,
                ValueError,
            ) as exc:
                warnings.append(
                    "Skipped malformed publication: "
                    f"{exc}"
                )

        return QueryRetrievalResult(
            query=query,
            executed_query=executed_query,
            hit_count=max(
                0,
                int(
                    payload.get("hitCount")
                    or 0
                ),
            ),
            returned_count=len(papers),
            next_cursor_mark=(
                str(
                    payload.get(
                        "nextCursorMark"
                    )
                    or ""
                ).strip()
                or None
            ),
            papers=papers,
            warnings=warnings,
        )