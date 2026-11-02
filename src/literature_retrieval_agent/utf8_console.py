from __future__ import annotations

import sys


def configure_utf8_console() -> None:
    """
    Make command-line output robust across operating systems.

    This prevents crashes when paper titles contain Greek letters
    or other non-ASCII biomedical characters.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
