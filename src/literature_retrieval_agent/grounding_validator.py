from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


INVALID_CLAIM_OBJECT = "invalid_claim_object"
INVALID_PASSAGE_ID = "invalid_passage_id"
PASSAGE_NOT_PARSED = "passage_not_parsed"
EMPTY_QUOTE = "empty_quote"
QUOTE_NOT_FOUND = "quote_not_found"
QUOTE_TOO_SHORT = "quote_too_short"
INVALID_MARKERS = "invalid_markers"
MARKER_NOT_IN_QUOTE = "marker_not_in_quote"
DUPLICATE_CLAIM = "duplicate_claim"


@dataclass(frozen=True, slots=True)
class GroundingValidationConfig:
    """Deterministic rules for checking extracted evidence claims."""

    minimum_quote_characters: int = 30
    minimum_quote_words: int = 6

    def __post_init__(self) -> None:
        if self.minimum_quote_characters <= 0:
            raise ValueError(
                "minimum_quote_characters must be greater than zero."
            )

        if self.minimum_quote_words <= 0:
            raise ValueError(
                "minimum_quote_words must be greater than zero."
            )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


DEFAULT_GROUNDING_VALIDATION_CONFIG = (
    GroundingValidationConfig()
)


def normalize_text(value: Any) -> str:
    """Normalize whitespace and case without changing word order."""

    return re.sub(
        r"\s+",
        " ",
        str(value or ""),
    ).strip().casefold()


def _word_count(value: str) -> int:
    return len(
        re.findall(
            r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*",
            value,
        )
    )


def _contains_marker(
    quote: str,
    marker: str,
) -> bool:
    """Check a marker as a complete token, ignoring letter case."""

    cleaned_marker = marker.strip()

    if not cleaned_marker:
        return False

    pattern = re.compile(
        rf"(?<![A-Za-z0-9])"
        rf"{re.escape(cleaned_marker)}"
        rf"(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )

    return pattern.search(quote) is not None


def _claim_signature(
    claim: dict[str, Any],
) -> tuple[str, str, str]:
    return (
        normalize_text(
            claim.get(
                "claim_text",
                "",
            )
        ),
        normalize_text(
            claim.get(
                "evidence_quote",
                "",
            )
        ),
        normalize_text(
            claim.get(
                "passage_id",
                "",
            )
        ),
    )


def validate_grounding(
    result: dict[str, Any],
    prompt_passages: dict[str, str],
    allowed_passage_ids: list[str],
    config: GroundingValidationConfig = (
        DEFAULT_GROUNDING_VALIDATION_CONFIG
    ),
) -> tuple[
    bool,
    int,
    int,
    list[dict[str, Any]],
]:
    """
    Validate passage IDs, quotations, markers, and duplicate claims.

    This checks source traceability only. It does not determine whether
    a quotation semantically proves the biological claim.
    """

    raw_claims: Any = result.get(
        "claims",
        [],
    )

    claims = (
        raw_claims
        if isinstance(raw_claims, list)
        else []
    )

    allowed_ids = {
        str(passage_id)
        for passage_id in allowed_passage_ids
    }

    checks: list[dict[str, Any]] = []
    grounded_count = 0

    seen_signatures: set[
        tuple[str, str, str]
    ] = set()

    for index, claim in enumerate(
        claims,
        start=1,
    ):
        if not isinstance(claim, dict):
            checks.append(
                {
                    "claim_number": index,
                    "passage_id": "",
                    "grounded": False,
                    "reason_codes": [
                        INVALID_CLAIM_OBJECT
                    ],
                    "reasons": [
                        "The claim is not a JSON object."
                    ],
                    "reason": (
                        "The claim is not a JSON object."
                    ),
                    "quote_character_count": 0,
                    "quote_word_count": 0,
                    "markers_checked": [],
                    "markers_missing_from_quote": [],
                }
            )
            continue

        passage_id = str(
            claim.get(
                "passage_id",
                "",
            )
        ).strip()

        evidence_quote = str(
            claim.get(
                "evidence_quote",
                "",
            )
        ).strip()

        source_text = prompt_passages.get(
            passage_id
        )

        reason_codes: list[str] = []
        reasons: list[str] = []

        if passage_id not in allowed_ids:
            reason_codes.append(
                INVALID_PASSAGE_ID
            )
            reasons.append(
                "passage_id is not one of the IDs "
                "supplied to this candidate job."
            )

        elif source_text is None:
            reason_codes.append(
                PASSAGE_NOT_PARSED
            )
            reasons.append(
                "passage_id was not parsed from "
                "the supplied prompt."
            )

        if not evidence_quote:
            reason_codes.append(
                EMPTY_QUOTE
            )
            reasons.append(
                "evidence_quote is empty."
            )

        else:
            if (
                source_text is not None
                and normalize_text(
                    evidence_quote
                )
                not in normalize_text(
                    source_text
                )
            ):
                reason_codes.append(
                    QUOTE_NOT_FOUND
                )
                reasons.append(
                    "evidence_quote was not found "
                    "as a continuous quotation in "
                    "the cited passage."
                )

            quote_word_count = _word_count(
                evidence_quote
            )

            if (
                len(evidence_quote)
                < config.minimum_quote_characters
                or quote_word_count
                < config.minimum_quote_words
            ):
                reason_codes.append(
                    QUOTE_TOO_SHORT
                )
                reasons.append(
                    "evidence_quote is too short "
                    "to provide meaningful "
                    "biological context."
                )

        raw_markers: Any = claim.get(
            "markers",
            [],
        )

        if (
            not isinstance(
                raw_markers,
                list,
            )
            or not all(
                isinstance(marker, str)
                for marker in raw_markers
            )
        ):
            markers: list[str] = []
            missing_markers: list[str] = []

            reason_codes.append(
                INVALID_MARKERS
            )
            reasons.append(
                "markers must be a list of "
                "gene-symbol strings."
            )

        else:
            markers = [
                marker.strip()
                for marker in raw_markers
                if marker.strip()
            ]

            missing_markers = [
                marker
                for marker in markers
                if not _contains_marker(
                    evidence_quote,
                    marker,
                )
            ]

            if missing_markers:
                reason_codes.append(
                    MARKER_NOT_IN_QUOTE
                )
                reasons.append(
                    "These reported markers do not "
                    "occur in the quotation: "
                    + ", ".join(
                        missing_markers
                    )
                    + "."
                )

        signature = _claim_signature(
            claim
        )

        if signature in seen_signatures:
            reason_codes.append(
                DUPLICATE_CLAIM
            )
            reasons.append(
                "The same claim, quotation, and "
                "passage_id were already reported."
            )

        else:
            seen_signatures.add(
                signature
            )

        grounded = not reason_codes

        if grounded:
            grounded_count += 1
            reasons.append(
                "passage_id, quotation length, "
                "quotation text, and reported "
                "markers passed deterministic "
                "validation."
            )

        checks.append(
            {
                "claim_number": index,
                "passage_id": passage_id,
                "grounded": grounded,
                "reason_codes": reason_codes,
                "reasons": reasons,
                "reason": " ".join(
                    reasons
                ),
                "quote_character_count": len(
                    evidence_quote
                ),
                "quote_word_count": _word_count(
                    evidence_quote
                ),
                "markers_checked": markers,
                "markers_missing_from_quote": (
                    missing_markers
                ),
            }
        )

    claim_count = len(
        claims
    )

    insufficient_evidence = (
        result.get(
            "insufficient_evidence"
        )
        is True
    )

    fully_grounded = (
        grounded_count == claim_count
        and (
            claim_count > 0
            or insufficient_evidence
        )
    )

    return (
        fully_grounded,
        claim_count,
        grounded_count,
        checks,
    )