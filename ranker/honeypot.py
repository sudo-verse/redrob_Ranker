"""Stage 1 — honeypot rejection.

Honeypots are forced to relevance tier 0 in the hidden ground truth, and a
honeypot rate > 10% in the top 100 is an automatic disqualification. We use only
the *tight, zero-false-positive* cross-field contradictions validated in
CANDIDATE_POOL_INTELLIGENCE.md §7:

    A. A role's ``duration_months`` exceeds the months elapsed since its
       ``start_date`` (you cannot have worked a job longer than it has existed).
    B. A skill claimed at "expert" proficiency with ``duration_months == 0``.
    C. Summed career months greatly exceed ``years_of_experience * 12``.

Loose signals (skill-duration > career, low assessment scores) are deliberately
NOT used here — they fire on tens of thousands of legitimate profiles. They are
handled as soft down-weights in the scoring stage instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

from .config import HoneypotConfig
from .dates import months_between, parse_date

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HoneypotVerdict:
    """Result of screening a single candidate."""

    is_honeypot: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def honeypot_score(self) -> int:
        # Per the brief: failing candidates receive honeypot_score = 1.
        return 1 if self.is_honeypot else 0

    @property
    def final_rank_excluded(self) -> bool:
        return self.is_honeypot


class HoneypotDetector:
    """Reusable detector applying the three hard vetoes (A, B, C)."""

    def __init__(self, reference_date: date, config: HoneypotConfig) -> None:
        self.reference_date = reference_date
        self.tenure_slack = max(0, int(config.tenure_slack_months))
        self.yoe_buffer = max(0, int(config.yoe_excess_buffer_months))

    # --- individual vetoes ------------------------------------------------- #

    def _veto_impossible_tenure(self, candidate: dict) -> list[str]:
        reasons: list[str] = []
        for role in candidate.get("career_history") or []:
            start = parse_date(role.get("start_date"))
            if start is None:
                continue
            duration = role.get("duration_months")
            if not isinstance(duration, (int, float)):
                continue
            elapsed = months_between(start, self.reference_date)
            if duration > elapsed + self.tenure_slack:
                reasons.append(
                    f"impossible_tenure:{role.get('company', '?')}"
                    f"({int(duration)}mo>{elapsed}mo_since_{role.get('start_date')})"
                )
        return reasons

    def _veto_expert_zero_months(self, candidate: dict) -> list[str]:
        reasons: list[str] = []
        for skill in candidate.get("skills") or []:
            if (
                skill.get("proficiency") == "expert"
                and skill.get("duration_months") == 0
            ):
                reasons.append(f"expert_zero_months:{skill.get('name', '?')}")
        return reasons

    def _veto_yoe_inconsistency(self, candidate: dict) -> list[str]:
        yoe = candidate.get("profile", {}).get("years_of_experience")
        if not isinstance(yoe, (int, float)):
            return []
        total = sum(
            role.get("duration_months", 0)
            for role in candidate.get("career_history") or []
            if isinstance(role.get("duration_months"), (int, float))
        )
        if total > yoe * 12 + self.yoe_buffer:
            return [f"yoe_excess:{int(total)}mo_vs_yoe_{yoe}"]
        return []

    # --- public API -------------------------------------------------------- #

    def screen(self, candidate: dict) -> HoneypotVerdict:
        reasons: list[str] = []
        reasons += self._veto_impossible_tenure(candidate)
        reasons += self._veto_expert_zero_months(candidate)
        reasons += self._veto_yoe_inconsistency(candidate)
        return HoneypotVerdict(is_honeypot=bool(reasons), reasons=tuple(reasons))
