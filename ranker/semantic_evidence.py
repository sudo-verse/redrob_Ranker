"""V2 Stage D + E — semantic evidence extraction and bounded boost score.

Stage D mines each retrieved candidate's free text for concrete capability
evidence (recommendation systems, search, ranking, retrieval, embeddings,
relevance, personalization, marketplaces, matching). Stage E turns that evidence
into ``semantic_evidence_score`` in [0, 1] driven by the *evidence itself*
(volume, category diversity, career-history support, title support) — explicitly
NOT by cosine similarity. This is the value that becomes a small new scoring
component; the embedding only decided eligibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import vocab

# Stage D — evidence categories and their surface phrases (lower-cased).
EVIDENCE_CATEGORIES: dict[str, tuple[str, ...]] = {
    "recommendation": ("recommendation system", "recommender system", "recommendation engine"),
    "search": ("search engine", "search infrastructure", "search relevance", "full text search"),
    "ranking": ("ranking model", "ranking system", "learning to rank", "rank candidates", "result ranking"),
    "retrieval": ("retrieval", "information retrieval", "nearest neighbor", "nearest neighbour", "bm25"),
    "embeddings": ("embedding", "vector search", "semantic search", "sentence transformer", "vector database"),
    "relevance": ("relevance", "relevance optimization", "relevance tuning", "ctr", "click through"),
    "personalization": ("personalization", "personalisation", "personalized feed", "user preferences"),
    "marketplace": ("marketplace", "two sided market", "matching engine", "candidate matching", "job matching"),
    "matching": ("matching", "match candidates", "semantic matching", "profile matching"),
}

# Title terms that count as "title support" for semantic evidence.
TITLE_SUPPORT_TERMS: tuple[str, ...] = (
    "search", "ml", "machine learning", "ai", "data scien", "applied scien",
    "nlp", "recommendation", "ranking", "relevance",
)


@dataclass
class Snippet:
    category: str
    phrase: str
    source: str  # "headline" | "summary" | "title" | "career"


@dataclass
class EvidenceResult:
    candidate_id: str
    snippets: list[Snippet] = field(default_factory=list)
    categories: set[str] = field(default_factory=set)
    career_support: bool = False
    title_support: bool = False
    semantic_similarity: float = 0.0

    @property
    def snippet_count(self) -> int:
        return len(self.snippets)


def _scan(text: str, source: str, out: EvidenceResult) -> None:
    if not text:
        return
    low = text.lower()
    for category, phrases in EVIDENCE_CATEGORIES.items():
        for phrase in phrases:
            if phrase in low:
                out.snippets.append(Snippet(category, phrase, source))
                out.categories.add(category)
                if source == "career":
                    out.career_support = True


def extract_evidence(candidate: dict, semantic_similarity: float) -> EvidenceResult:
    profile = candidate.get("profile") or {}
    res = EvidenceResult(
        candidate_id=candidate.get("candidate_id", ""),
        semantic_similarity=semantic_similarity,
    )
    _scan(profile.get("headline", ""), "headline", res)
    _scan(profile.get("summary", ""), "summary", res)

    titles = [profile.get("current_title", "")]
    for role in candidate.get("career_history") or []:
        titles.append(role.get("title", ""))
        _scan(role.get("title", ""), "title", res)
        _scan(role.get("description", ""), "career", res)

    title_blob = " ".join(t.lower() for t in titles if t)
    res.title_support = any(term in title_blob for term in TITLE_SUPPORT_TERMS)
    return res


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def semantic_evidence_score(evidence: EvidenceResult) -> float:
    """Stage E — bounded [0, 1] score from evidence, NOT from cosine similarity."""
    diversity_term = min(len(evidence.categories), 4) / 4.0     # breadth of capability
    volume_term = min(evidence.snippet_count, 6) / 6.0          # weight of evidence
    career_term = 1.0 if evidence.career_support else 0.0       # shown in real roles
    title_term = 1.0 if evidence.title_support else 0.0         # role identity backs it
    return _clamp(
        0.40 * diversity_term + 0.30 * volume_term + 0.20 * career_term + 0.10 * title_term
    )


def top_snippets(evidence: EvidenceResult, limit: int = 3) -> list[str]:
    """Distinct, human-readable evidence phrases for reasoning/analysis."""
    seen: list[str] = []
    for snip in evidence.snippets:
        label = f"{snip.phrase} ({snip.source})"
        if snip.phrase not in [s.split(" (")[0] for s in seen]:
            seen.append(label)
        if len(seen) >= limit:
            break
    return seen
