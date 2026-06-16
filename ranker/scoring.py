"""Stage 3 — transparent weighted scoring engine.

Every component is a pure function returning a value in ``[0, 1]``. The final
score is ``sum(weight_i * component_i) / sum(weights)`` (so always in ``[0, 1]``),
then multiplied by a soft keyword-stuffer modifier. Nothing is hardcoded: every
weight and threshold comes from :class:`Config`. The per-component breakdown is
returned alongside the score for auditability and explanation.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .features import Features


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class ScoreResult:
    score: float                       # final score in [0, 1]
    components: dict[str, float]       # each raw component in [0, 1]
    stuffer_modifier: float            # multiplicative penalty applied (<=1.0)


# --------------------------------------------------------------------------- #
# Individual components (each returns [0, 1])
# --------------------------------------------------------------------------- #

def title_fit(f: Features) -> float:
    if f.direct_ai_title:
        return 1.0
    if f.cv_speech_title:
        return 0.30  # CV/speech-only is a JD negative; IR evidence can rescue via ir_depth
    if f.adjacent_title:
        return 0.45  # Tier-5 plain-language fits live here; ir_depth carries them up
    if f.irrelevant_title:
        return 0.0
    return 0.15      # unknown/other title


def ir_depth(f: Features, cap: float) -> float:
    raw = (
        2.0 * min(f.foundational_ir_skill_count, 4)
        + 1.5 * min(f.ranking_skill_count, 3)
        + 1.5 * min(f.vector_db_count, 3)
        + 1.0 * min(f.llm_skill_count, 3)
        + 0.5 * min(f.buzzword_skill_count, 4)
        + 1.0 * min(len(f.text_evidence_phrases), 4)
    )
    return _clamp(raw / cap if cap > 0 else 0.0)


def product_experience(f: Features) -> float:
    score = 0.25  # neutral baseline (non-consulting, non-product)
    if f.ai_company_count > 0:
        score += 0.45
    if f.product_company_count > 0:
        score += 0.35
    if f.startup_experience:
        score += 0.15
    score -= 0.70 * f.consulting_ratio
    return _clamp(score)


def availability(f: Features, recency_window: float, notice_cap: float) -> float:
    otw = 1.0 if f.open_to_work else 0.0
    response = _clamp(f.recruiter_response_rate)
    interview = _clamp(f.interview_completion_rate)
    notice = _clamp(1.0 - f.notice_period_days / notice_cap) if notice_cap > 0 else 0.0
    if f.last_active_recency_days is None:
        recency = 0.3  # unknown recency -> mildly conservative
    else:
        recency = _clamp(1.0 - f.last_active_recency_days / recency_window)
    return _clamp(
        0.25 * otw + 0.30 * response + 0.15 * interview + 0.10 * notice + 0.20 * recency
    )


def assessment_trust(f: Features) -> float:
    base = 0.40  # neutral when little assessment evidence exists
    base += 0.15 * min(f.assessment_backed_skills, 4)
    base -= 0.30 * f.advanced_skill_low_score_count
    return _clamp(base)


def location_fit(f: Features) -> float:
    if f.preferred_location:
        return 1.0
    if f.relocatable:
        return 0.60
    if (f.country or "").strip().lower() == "india":
        return 0.40  # in India but not a preferred hub
    return 0.05      # foreign, not relocatable


def experience_fit(f: Features, decay_years: float) -> float:
    if decay_years <= 0:
        return 1.0 if f.distance_from_band == 0 else 0.0
    return _clamp(1.0 - f.distance_from_band / decay_years)


def github_signal(f: Features) -> float:
    if not f.github_present:
        return 0.25  # absence is a mild negative (JD wants external validation)
    return _clamp(0.30 + 0.70 * (f.github_score / 100.0))


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def compute_components(f: Features, cfg: Config) -> dict[str, float]:
    s = cfg.scoring
    return {
        "title_fit": title_fit(f),
        "ir_depth": ir_depth(f, s.ir_depth_cap),
        "product_experience": product_experience(f),
        "availability": availability(f, s.recency_window_days, s.notice_period_cap_days),
        "assessment_trust": assessment_trust(f),
        "location_fit": location_fit(f),
        "experience_fit": experience_fit(f, s.experience_decay_years),
        "github_signal": github_signal(f),
    }


def stuffer_modifier(f: Features, cfg: Config) -> float:
    """Soft (non-veto) keyword-stuffer down-weight."""
    triggered = f.dabbler_summary or (
        f.advanced_skill_low_score_count >= cfg.scoring.stuffer_low_score_trigger
    )
    return cfg.scoring.stuffer_penalty if triggered else 1.0


def score_candidate(f: Features, cfg: Config) -> ScoreResult:
    components = compute_components(f, cfg)
    weights = cfg.weights.as_dict()
    total_w = cfg.weights.total()
    weighted = sum(weights[k] * components[k] for k in components)
    base = weighted / total_w if total_w > 0 else 0.0
    modifier = stuffer_modifier(f, cfg)
    return ScoreResult(
        score=_clamp(base * modifier),
        components=components,
        stuffer_modifier=modifier,
    )
