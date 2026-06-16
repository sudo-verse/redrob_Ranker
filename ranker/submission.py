"""Stage 5 — submission CSV generation.

Sorting key ``(-score, candidate_id)`` guarantees the two properties the official
validator enforces: scores are monotonically non-increasing with rank, and any
two equal scores are ordered by ``candidate_id`` ascending. Exactly ``top_n``
rows are emitted.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

HEADER = ["candidate_id", "rank", "score", "reasoning"]


@dataclass
class ScoredCandidate:
    candidate_id: str
    score: float
    reasoning: str


def rank_and_select(candidates: list[ScoredCandidate], top_n: int) -> list[ScoredCandidate]:
    """Deterministically order all eligible candidates and take the top ``top_n``."""
    ordered = sorted(candidates, key=lambda c: (-c.score, c.candidate_id))
    return ordered[:top_n]


def write_submission(
    candidates: list[ScoredCandidate],
    out_path: str | Path,
    top_n: int,
    score_decimals: int,
) -> list[ScoredCandidate]:
    selected = rank_and_select(candidates, top_n)
    if len(selected) < top_n:
        raise ValueError(
            f"Only {len(selected)} eligible candidates after honeypot rejection; "
            f"need {top_n}. Loosen vetoes or check the input."
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        for rank, cand in enumerate(selected, start=1):
            writer.writerow(
                [cand.candidate_id, rank, f"{cand.score:.{score_decimals}f}", cand.reasoning]
            )
    logger.info("Wrote %d rows to %s", len(selected), out)
    return selected
