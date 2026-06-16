"""Stage 2 — candidate feature extraction.

Produces a compact, fully-typed :class:`Features` object per candidate. Only
evidence actually present in the profile is recorded; matched skill/company names
are retained so the explanation stage (Stage 4) can cite them without
hallucinating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from . import vocab
from .config import ScoringConfig
from .dates import months_between, parse_date


@dataclass
class Features:
    candidate_id: str

    # --- title features ---
    current_title: str = ""
    direct_ai_title: bool = False
    adjacent_title: bool = False
    irrelevant_title: bool = False
    cv_speech_title: bool = False

    # --- skill features ---
    foundational_ir_skill_count: int = 0
    buzzword_skill_count: int = 0
    vector_db_count: int = 0
    ranking_skill_count: int = 0
    llm_skill_count: int = 0
    matched_foundational: list[str] = field(default_factory=list)
    matched_ranking: list[str] = field(default_factory=list)
    matched_vector_db: list[str] = field(default_factory=list)
    matched_llm: list[str] = field(default_factory=list)

    # --- career features ---
    product_company_count: int = 0
    ai_company_count: int = 0
    consulting_company_count: int = 0
    consulting_ratio: float = 0.0
    startup_experience: bool = False
    matched_product_companies: list[str] = field(default_factory=list)
    matched_ai_companies: list[str] = field(default_factory=list)
    text_evidence_phrases: list[str] = field(default_factory=list)

    # --- experience features ---
    years_of_experience: float = 0.0
    distance_from_band: float = 0.0

    # --- availability features ---
    open_to_work: bool = False
    recruiter_response_rate: float = 0.0
    interview_completion_rate: float = 0.0
    notice_period_days: int = 0
    last_active_recency_days: int | None = None  # None => unknown

    # --- location features ---
    preferred_location: bool = False
    relocatable: bool = False
    country: str = ""

    # --- trust features ---
    assessment_backed_skills: int = 0
    advanced_skill_low_score_count: int = 0
    github_present: bool = False
    github_score: float = -1.0
    dabbler_summary: bool = False


def _matched(names: set[str], vocab_set: frozenset[str]) -> list[str]:
    return sorted(names & vocab_set)


def _company_is_consulting(name_norm: str) -> bool:
    return any(kw in name_norm for kw in vocab.CONSULTING_KEYWORDS)


def extract(candidate: dict, scoring: ScoringConfig, reference_date: date) -> Features:
    profile = candidate.get("profile") or {}
    signals = candidate.get("redrob_signals") or {}
    feats = Features(candidate_id=candidate.get("candidate_id", ""))

    # ---- titles ----
    title = vocab.normalize(profile.get("current_title"))
    feats.current_title = profile.get("current_title", "") or ""
    feats.direct_ai_title = title in vocab.DIRECT_AI_TITLES
    feats.adjacent_title = title in vocab.ADJACENT_TITLES
    feats.irrelevant_title = title in vocab.IRRELEVANT_TITLES
    feats.cv_speech_title = title in vocab.CV_SPEECH_TITLES

    # ---- skills ----
    skill_names = {vocab.normalize(s.get("name")) for s in candidate.get("skills") or []}
    feats.matched_foundational = _matched(skill_names, vocab.FOUNDATIONAL_IR_SKILLS)
    feats.matched_ranking = _matched(skill_names, vocab.RANKING_SKILLS)
    feats.matched_vector_db = _matched(skill_names, vocab.VECTOR_DB_SKILLS)
    feats.matched_llm = _matched(skill_names, vocab.LLM_SKILLS)
    feats.foundational_ir_skill_count = len(feats.matched_foundational)
    feats.buzzword_skill_count = len(skill_names & vocab.BUZZWORD_SKILLS)
    feats.vector_db_count = len(feats.matched_vector_db)
    feats.ranking_skill_count = len(feats.matched_ranking)
    feats.llm_skill_count = len(feats.matched_llm)

    # ---- assessment trust ----
    assessment = signals.get("skill_assessment_scores") or {}
    assessment_norm = {vocab.normalize(k): v for k, v in assessment.items()}
    for skill in candidate.get("skills") or []:
        if skill.get("proficiency") not in ("advanced", "expert"):
            continue
        score = assessment_norm.get(vocab.normalize(skill.get("name")))
        if not isinstance(score, (int, float)):
            continue
        if score >= scoring.assessment_high_threshold:
            feats.assessment_backed_skills += 1
        elif score < scoring.assessment_low_threshold:
            feats.advanced_skill_low_score_count += 1

    # ---- career / companies / text evidence ----
    roles = candidate.get("career_history") or []
    companies_norm = [vocab.normalize(r.get("company")) for r in roles]
    companies_norm.append(vocab.normalize(profile.get("current_company")))
    seen_companies = set(c for c in companies_norm if c)

    feats.matched_product_companies = sorted(seen_companies & vocab.PRODUCT_COMPANIES)
    feats.matched_ai_companies = sorted(seen_companies & vocab.AI_COMPANIES)
    feats.product_company_count = len(feats.matched_product_companies)
    feats.ai_company_count = len(feats.matched_ai_companies)

    consulting_roles = sum(1 for c in companies_norm[:-1] if _company_is_consulting(c))
    feats.consulting_company_count = consulting_roles
    feats.consulting_ratio = consulting_roles / len(roles) if roles else 0.0

    small_sizes = {"1-10", "11-50", "51-200"}
    feats.startup_experience = any(
        (r.get("company_size") in small_sizes) for r in roles
    )

    # free-text build evidence (summary + role descriptions)
    blob = vocab.normalize(profile.get("summary"))
    for r in roles:
        blob += " " + vocab.normalize(r.get("description"))
    found_phrases = [p for p in vocab.IR_TEXT_EVIDENCE_PHRASES if p in blob]
    feats.text_evidence_phrases = found_phrases
    feats.dabbler_summary = all(m in blob for m in vocab.DABBLER_SUMMARY_MARKERS)

    # ---- experience ----
    yoe = profile.get("years_of_experience")
    feats.years_of_experience = float(yoe) if isinstance(yoe, (int, float)) else 0.0
    lo, hi = scoring.experience_band
    feats.distance_from_band = max(0.0, lo - feats.years_of_experience, feats.years_of_experience - hi)

    # ---- availability ----
    feats.open_to_work = bool(signals.get("open_to_work_flag"))
    rr = signals.get("recruiter_response_rate")
    feats.recruiter_response_rate = float(rr) if isinstance(rr, (int, float)) else 0.0
    ic = signals.get("interview_completion_rate")
    feats.interview_completion_rate = float(ic) if isinstance(ic, (int, float)) else 0.0
    np_days = signals.get("notice_period_days")
    feats.notice_period_days = int(np_days) if isinstance(np_days, (int, float)) else 90
    last_active = parse_date(signals.get("last_active_date"))
    if last_active is not None:
        feats.last_active_recency_days = max(0, (reference_date - last_active).days)

    # ---- location ----
    location = vocab.normalize(profile.get("location"))
    feats.country = profile.get("country", "") or ""
    feats.preferred_location = any(city in location for city in vocab.PREFERRED_LOCATIONS)
    feats.relocatable = bool(signals.get("willing_to_relocate"))

    # ---- github ----
    gh = signals.get("github_activity_score")
    feats.github_score = float(gh) if isinstance(gh, (int, float)) else -1.0
    feats.github_present = feats.github_score >= 0.0

    return feats
