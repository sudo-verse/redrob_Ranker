"""Stage 4 — factual, grounded candidate explanations.

Every clause is generated *only* from evidence present in the extracted
:class:`Features` (which itself only records things found in the profile), so the
reasoning can never reference a skill, employer, or trait the candidate does not
have.

This module affects ONLY the ``reasoning`` text column. It does not compute or
influence scores or candidate ordering in any way — the ranking model is frozen.

Diversity (added for submission readiness):
  * 21 opening templates across 5 rank tiers (>=4 per tier).
  * Rank-aware wording: when a final ``rank`` is supplied, the opener and tone
    are keyed to the candidate's position in the top-100 (elite/strong/solid/
    moderate/borderline) rather than to absolute score, so the top-100 — whose
    scores are clustered — still reads with appropriate variation.
  * Deterministic per-candidate variety: the specific opener and a couple of body
    phrasings are selected by a stable hash of ``candidate_id``, so two adjacent
    ranks rarely share wording, yet output is fully reproducible.
  * Factual grounding, honest concerns, and rank-consistent tone are preserved.
"""

from __future__ import annotations

import hashlib

from .features import Features
from .scoring import ScoreResult


# --------------------------------------------------------------------------- #
# Opening templates by rank tier (>=4 each => 21 total). Wording is tone-matched
# to the tier so a rank-5 opener reads confident and a rank-95 opener reads
# hedged, keeping reasoning consistent with rank.
# --------------------------------------------------------------------------- #
OPENERS: dict[str, tuple[str, ...]] = {
    "elite": (
        "Top of the slate —",
        "Exceptional match —",
        "Headline candidate for this role —",
        "Among the strongest in the pool —",
        "Clear first-tier fit —",
    ),
    "strong": (
        "Strong fit —",
        "High-confidence pick —",
        "Well-aligned with the role —",
        "Comfortably above the bar —",
    ),
    "solid": (
        "Solid fit —",
        "Credible match —",
        "Meets the core requirements —",
        "Dependable mid-slate pick —",
    ),
    "moderate": (
        "Moderate fit —",
        "Reasonable but not standout —",
        "Partial alignment with the role —",
        "Fits, with reservations —",
    ),
    "borderline": (
        "Borderline inclusion —",
        "Filler at this depth of the shortlist —",
        "Tail of the top-100 —",
        "Included with caveats —",
    ),
}

# Body phrasings for the lead identity clause (facts identical; wording varies).
_IDENTITY_TEMPLATES: tuple[str, ...] = (
    "{title} with {yoe}",
    "{title}, {yoe} of experience",
    "{title} ({yoe})",
    "{title} bringing {yoe}",
)


def _stable_index(key: str, n: int) -> int:
    """Deterministic index in [0, n) from a string key (reproducible across runs)."""
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest, 16) % n if n > 0 else 0


def _pick(options: tuple[str, ...], key: str) -> str:
    return options[_stable_index(key, len(options))]


def _tier(score: float, rank: int | None, total: int | None) -> str:
    """Rank-aware tier when rank is known; otherwise fall back to absolute score."""
    if rank is not None and total:
        if rank <= 10:
            return "elite"
        frac = rank / total
        if frac <= 0.30:
            return "strong"
        if frac <= 0.60:
            return "solid"
        if frac <= 0.85:
            return "moderate"
        return "borderline"
    if score >= 0.80:
        return "elite"
    if score >= 0.65:
        return "strong"
    if score >= 0.50:
        return "solid"
    if score >= 0.35:
        return "moderate"
    return "borderline"


def _opening(f: Features, score: float, rank: int | None, total: int | None) -> str:
    tier = _tier(score, rank, total)
    return _pick(OPENERS[tier], f.candidate_id + tier)


def _join(items: list[str], limit: int = 3) -> str:
    items = items[:limit]
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _pretty_skill(name: str) -> str:
    special = {
        "bm25": "BM25",
        "rag": "RAG",
        "llms": "LLMs",
        "nlp": "NLP",
        "faiss": "FAISS",
        "lora": "LoRA",
        "qlora": "QLoRA",
        "peft": "PEFT",
        "pytorch": "PyTorch",
        "tensorflow": "TensorFlow",
        "opensearch": "OpenSearch",
    }
    return special.get(name, name.title())


def _factual_clauses(f: Features) -> list[str]:
    """Build the grounded fact clauses (identical facts regardless of wording)."""
    parts: list[str] = []

    # 1. Identity + experience (specific facts; phrasing varies by candidate).
    yoe = f"{f.years_of_experience:.1f} yrs" if f.years_of_experience else "experience n/a"
    title = f.current_title or "unlisted title"
    parts.append(_pick(_IDENTITY_TEMPLATES, f.candidate_id).format(title=title, yoe=yoe))

    # 2. IR / skill evidence (positive, JD-connected).
    ir_named = (
        f.matched_foundational
        + [s for s in f.matched_ranking if s not in f.matched_foundational]
        + [s for s in f.matched_vector_db if s not in f.matched_foundational]
    )
    if ir_named:
        pretty = [_pretty_skill(s) for s in ir_named]
        parts.append(f"foundational IR skills incl. {_join(pretty)}")
    elif f.text_evidence_phrases:
        parts.append(f"career text shows {_join(f.text_evidence_phrases)} work")
    elif f.buzzword_skill_count > 0:
        parts.append(f"{f.buzzword_skill_count} AI keyword(s) listed but no foundational IR depth")

    # 3. Company pedigree (JD R6).
    if f.matched_ai_companies:
        parts.append(f"AI-company experience ({_join([c.title() for c in f.matched_ai_companies])})")
    elif f.matched_product_companies:
        parts.append(f"product-company experience ({_join([c.title() for c in f.matched_product_companies])})")

    # 4. Availability (the multiplier signal).
    avail_bits: list[str] = []
    if f.open_to_work:
        avail_bits.append("open to work")
    if f.recruiter_response_rate >= 0.5:
        avail_bits.append(f"responsive (resp {f.recruiter_response_rate:.2f})")
    if f.last_active_recency_days is not None and f.last_active_recency_days <= 60:
        avail_bits.append("recently active")
    if avail_bits:
        parts.append(_join(avail_bits))

    # 5. Location (JD R8).
    if f.preferred_location:
        parts.append("in a preferred hub")
    elif f.relocatable:
        parts.append("willing to relocate")

    return parts


def _concern_clauses(f: Features) -> list[str]:
    """Honest concerns (Stage-4 explicitly rewards acknowledging gaps)."""
    concerns: list[str] = []
    if f.consulting_ratio >= 0.6:
        concerns.append(f"consulting-heavy career ({f.consulting_ratio:.0%} of roles)")
    if f.cv_speech_title and not f.matched_foundational:
        concerns.append("CV/speech background with limited NLP/IR signal")
    if f.notice_period_days >= 60:
        concerns.append(f"long notice period ({f.notice_period_days}d)")
    if f.recruiter_response_rate < 0.3:
        concerns.append(f"low recruiter response rate ({f.recruiter_response_rate:.2f})")
    if f.last_active_recency_days is not None and f.last_active_recency_days > 120:
        concerns.append(f"inactive for {f.last_active_recency_days}d")
    if f.advanced_skill_low_score_count > 0:
        concerns.append(f"{f.advanced_skill_low_score_count} claimed skill(s) unbacked by assessment")
    if f.dabbler_summary:
        concerns.append("summary reads as casual AI dabbler")
    return concerns


def build_reasoning(
    f: Features,
    result: ScoreResult,
    rank: int | None = None,
    total: int | None = None,
) -> str:
    """Compose a grounded, rank-aware reasoning string.

    ``rank``/``total`` are optional and affect ONLY wording — never the score or
    order. When omitted, tone falls back to the candidate's absolute score.
    """
    opener = _opening(f, result.score, rank, total)
    body = "; ".join(_factual_clauses(f)) + "."
    concerns = _concern_clauses(f)
    if concerns:
        body += " Concerns: " + _join(concerns, limit=2) + "."
    return f"{opener} {body}"
