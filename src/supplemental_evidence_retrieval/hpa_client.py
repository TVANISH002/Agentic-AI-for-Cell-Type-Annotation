from __future__ import annotations

import zipfile
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class HPAReferenceClient:
    def __init__(
        self,
        source_url: str,
        cache_path: Path,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
    ) -> None:
        """Initializes the HPA reference downloader."""

        if not source_url.strip():
            raise ValueError(
                "source_url must not be empty."
            )

        if max_retries < 0:
            raise ValueError(
                "max_retries cannot be negative."
            )

        if backoff_factor < 0:
            raise ValueError(
                "backoff_factor cannot be negative."
            )

        self.source_url = source_url
        self.cache_path = cache_path

        self.session = self._build_session(
            max_retries=max_retries,
            backoff_factor=backoff_factor,
        )

    @staticmethod
    def _build_session(
        max_retries: int,
        backoff_factor: float,
    ) -> requests.Session:
        """Creates a reusable HTTP session."""

        retry_policy = Retry(
            total=max_retries,
            connect=max_retries,
            read=max_retries,
            status=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=(
                429,
                500,
                502,
                503,
                504,
            ),
            allowed_methods=frozenset(
                {"GET"}
            ),
            respect_retry_after_header=True,
            raise_on_status=False,
        )

        adapter = HTTPAdapter(
            max_retries=retry_policy
        )

        session = requests.Session()

        session.headers.update(
            {
                "User-Agent": (
                    "AgenticCellAnnotation-"
                    "HPA-Retrieval/1.0"
                )
            }
        )

        session.mount(
            "https://",
            adapter,
        )

        return session

    def download(
        self,
        force_download: bool = False,
    ) -> Path:
        """Downloads or reuses the cached HPA ZIP."""

        self.cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if (
            self.cache_path.exists()
            and not force_download
            and zipfile.is_zipfile(
                self.cache_path
            )
        ):
            return self.cache_path

        if self.cache_path.exists():
            self.cache_path.unlink()

        temporary_path = (
            self.cache_path.with_suffix(
                self.cache_path.suffix + ".part"
            )
        )

        temporary_path.unlink(
            missing_ok=True
        )

        try:
            with self.session.get(
                self.source_url,
                stream=True,
                timeout=(15, 300),
            ) as response:
                response.raise_for_status()

                with temporary_path.open(
                    "wb"
                ) as handle:
                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                    ):
                        if chunk:
                            handle.write(chunk)

            if (
                not temporary_path.exists()
                or temporary_path.stat().st_size == 0
            ):
                raise ValueError(
                    "The downloaded HPA file is empty."
                )

            if not zipfile.is_zipfile(
                temporary_path
            ):
                raise ValueError(
                    "The downloaded HPA file is not "
                    "a valid ZIP archive."
                )

            temporary_path.replace(
                self.cache_path
            )

            return self.cache_path

        except Exception:
            temporary_path.unlink(
                missing_ok=True
            )

            raise

    def close(self) -> None:
        """Closes the HTTP session."""

        self.session.close()