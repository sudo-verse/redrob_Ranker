"""End-to-end orchestration of Stages 0-5."""

from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path

from . import features as feat
from . import loader, scoring
from .config import Config
from .dates import parse_date
from .explain import build_reasoning
from .honeypot import HoneypotDetector
from .submission import ScoredCandidate, write_submission

logger = logging.getLogger(__name__)


def resolve_reference_date(candidates_path: str | Path, cfg: Config) -> date:
    ref_cfg = cfg.honeypot.reference_date
    if isinstance(ref_cfg, str) and ref_cfg.lower() == "auto":
        return loader.detect_reference_date(candidates_path)
    parsed = parse_date(ref_cfg)
    if parsed is None:
        raise ValueError(f"Invalid honeypot.reference_date: {ref_cfg!r}")
    logger.info("Using fixed reference date %s", parsed)
    return parsed


def run_pipeline(candidates_path: str | Path, out_path: str | Path, cfg: Config) -> dict:
    """Run the full ranking pipeline and write the submission CSV.

    Returns a small stats dict for logging/tests.
    """
    t0 = time.perf_counter()
    reference_date = resolve_reference_date(candidates_path, cfg)
    detector = HoneypotDetector(reference_date, cfg.honeypot)

    total = 0
    honeypots = 0
    scored: list[ScoredCandidate] = []

    for candidate in loader.stream_candidates(candidates_path):
        total += 1
        verdict = detector.screen(candidate)
        if verdict.final_rank_excluded:
            honeypots += 1
            continue
        f = feat.extract(candidate, cfg.scoring, reference_date)
        result = scoring.score_candidate(f, cfg)
        scored.append(
            ScoredCandidate(
                candidate_id=f.candidate_id,
                score=result.score,
                reasoning=build_reasoning(f, result),
            )
        )

    selected = write_submission(
        scored, out_path, cfg.output.top_n, cfg.output.score_decimals
    )

    elapsed = time.perf_counter() - t0
    stats = {
        "total_candidates": total,
        "honeypots_rejected": honeypots,
        "eligible": len(scored),
        "selected": len(selected),
        "reference_date": reference_date.isoformat(),
        "top_score": selected[0].score if selected else None,
        "cutoff_score": selected[-1].score if selected else None,
        "elapsed_seconds": round(elapsed, 2),
    }
    logger.info("Pipeline stats: %s", stats)
    return stats
