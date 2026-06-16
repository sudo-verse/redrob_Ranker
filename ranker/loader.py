"""Stage 0 — efficient, memory-safe streaming JSONL loader.

The candidate pool is ~465 MB. We never hold the full parsed pool in memory:
``stream_candidates`` yields one decoded record at a time, and the pipeline keeps
only compact per-candidate feature objects. Gzipped (``.jsonl.gz``) input is
supported transparently so the same code runs on the shipped bundle.
"""

from __future__ import annotations

import gzip
import json
import logging
from datetime import date
from pathlib import Path
from typing import Iterator, TextIO

from .dates import parse_date

logger = logging.getLogger(__name__)


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def stream_candidates(path: str | Path) -> Iterator[dict]:
    """Yield one candidate dict per non-blank line. Bad lines are skipped+logged."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Candidates file not found: {p}")

    bad = 0
    with _open_text(p) as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                if bad <= 5:
                    logger.warning("Skipping malformed JSON on line %d", lineno)
    if bad:
        logger.warning("Skipped %d malformed line(s) total", bad)


def detect_reference_date(path: str | Path, fallback: date = date(2026, 6, 1)) -> date:
    """Streaming pass to find max ``last_active_date`` across the pool.

    Used as "today" for honeypot elapsed-time math when config requests "auto".
    Cheap relative to feature extraction and keeps the detector data-driven.
    """
    latest: date | None = None
    for cand in stream_candidates(path):
        raw = (cand.get("redrob_signals") or {}).get("last_active_date")
        d = parse_date(raw)
        if d is not None and (latest is None or d > latest):
            latest = d
    if latest is None:
        logger.warning("No parsable last_active_date found; using fallback %s", fallback)
        return fallback
    logger.info("Auto reference date (max last_active_date) = %s", latest)
    return latest
