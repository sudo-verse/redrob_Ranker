"""Domain vocabularies derived from CANDIDATE_POOL_INTELLIGENCE.md.

These lists encode the evidence findings (title strata, the two-tier AI-skill
structure, product/AI/consulting company clusters). They are intentionally kept
in code (not config.yaml) because they are *facts about the pool*, not tunable
weights. config.yaml holds only the knobs you would sweep.

All matching is case-insensitive; helpers below return normalized frozensets.
"""

from __future__ import annotations

from typing import Iterable

# --------------------------------------------------------------------------- #
# Titles (intelligence report §2-4)
# --------------------------------------------------------------------------- #

# Tier-A: directly relevant AI/ML/Search/RecSys roles.
DIRECT_AI_TITLES: frozenset[str] = frozenset(
    {
        "ml engineer",
        "machine learning engineer",
        "senior machine learning engineer",
        "staff machine learning engineer",
        "junior ml engineer",
        "applied ml engineer",
        "ai engineer",
        "senior ai engineer",
        "lead ai engineer",
        "ai research engineer",
        "ai specialist",
        "data scientist",
        "senior data scientist",
        "nlp engineer",
        "senior nlp engineer",
        "search engineer",
        "recommendation systems engineer",
        "applied scientist",
        "senior applied scientist",
        "senior software engineer (ml)",
    }
)

# CV/speech-only roles: JD explicitly de-prioritizes (negative N7). Treated as a
# weak title, not a direct match, unless IR evidence rescues them.
CV_SPEECH_TITLES: frozenset[str] = frozenset(
    {"computer vision engineer", "speech engineer", "robotics engineer"}
)

# Tier-B: adjacent engineering roles that can be Tier-5 "plain-language" fits
# *if* their free text shows retrieval/ranking/recsys work.
ADJACENT_TITLES: frozenset[str] = frozenset(
    {
        "data engineer",
        "senior data engineer",
        "analytics engineer",
        "backend engineer",
        "software engineer",
        "senior software engineer",
        "full stack developer",
        "cloud engineer",
        "devops engineer",
    }
)

# Obviously irrelevant title groups (the ~69% bulk distractors).
IRRELEVANT_TITLES: frozenset[str] = frozenset(
    {
        "hr manager",
        "accountant",
        "graphic designer",
        "sales executive",
        "content writer",
        "customer support",
        "operations manager",
        "project manager",
        "business analyst",
        "marketing manager",
        "mechanical engineer",
        "civil engineer",
        "java developer",
        ".net developer",
        "mobile developer",
        "frontend engineer",
        "qa engineer",
    }
)

# --------------------------------------------------------------------------- #
# Skills: the two-tier AI/IR structure (intelligence report §6.3)
# --------------------------------------------------------------------------- #

# High-trust foundational / ops skills (~1,380x each in the pool).
FOUNDATIONAL_IR_SKILLS: frozenset[str] = frozenset(
    {
        "pytorch",
        "tensorflow",
        "learning to rank",
        "bm25",
        "elasticsearch",
        "opensearch",
        "weaviate",
        "qdrant",
        "milvus",
        "nlp",
        "lora",
        "qlora",
        "peft",
    }
)

# "Buzzword" band (~5,000x each): cheap to list, low trust on their own.
BUZZWORD_SKILLS: frozenset[str] = frozenset(
    {
        "pinecone",
        "rag",
        "embeddings",
        "faiss",
        "semantic search",
        "llms",
        "recommendation systems",
        "information retrieval",
        "sentence transformers",
        "vector search",
        "fine-tuning llms",
    }
)

# Vector databases / search infra (JD must-have R2).
VECTOR_DB_SKILLS: frozenset[str] = frozenset(
    {"pinecone", "weaviate", "qdrant", "milvus", "faiss", "elasticsearch", "opensearch"}
)

# Ranking / IR skills (JD must-have R3).
RANKING_SKILLS: frozenset[str] = frozenset(
    {
        "learning to rank",
        "bm25",
        "information retrieval",
        "recommendation systems",
        "semantic search",
        "vector search",
    }
)

# LLM-centric skills (JD nice-to-have R10).
LLM_SKILLS: frozenset[str] = frozenset(
    {"llms", "rag", "fine-tuning llms", "lora", "qlora", "peft"}
)

# --------------------------------------------------------------------------- #
# Companies (intelligence report §8-9)
# --------------------------------------------------------------------------- #

# Real Indian product companies / consumer startups (positive R6).
PRODUCT_COMPANIES: frozenset[str] = frozenset(
    {
        "swiggy",
        "razorpay",
        "cred",
        "zomato",
        "flipkart",
        "meesho",
        "nykaa",
        "inmobi",
        "zoho",
        "ola",
        "vedantu",
        "byju's",
        "policybazaar",
        "paytm",
        "freshworks",
        "upgrad",
        "pharmeasy",
        "phonepe",
        "dream11",
        "unacademy",
    }
)

# AI-focused firms (strongest R1-R4 + R6 signal).
AI_COMPANIES: frozenset[str] = frozenset(
    {
        "genpact ai",
        "glance",
        "rephrase.ai",
        "sarvam ai",
        "krutrim",
        "aganitha",
        "niramai",
        "saarthi.ai",
        "wysa",
        "mad street den",
        "haptik",
        "verloop.io",
    }
)

# Consulting / IT-services keywords (negative N6). Substring matched because some
# appear with suffixes ("Tata Consultancy Services").
CONSULTING_KEYWORDS: tuple[str, ...] = (
    "tcs",
    "tata consultancy",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "hcl",
    "tech mahindra",
    "mindtree",
    "mphasis",
    "l&t infotech",
    "lti",
)

# --------------------------------------------------------------------------- #
# Locations (JD R8)
# --------------------------------------------------------------------------- #

PREFERRED_LOCATIONS: tuple[str, ...] = (
    "pune",
    "noida",
    "hyderabad",
    "mumbai",
    "delhi",
    "gurgaon",
    "bangalore",
    "bengaluru",
)

# --------------------------------------------------------------------------- #
# Free-text build-evidence phrases (intelligence report §6.2). Lower-cased.
# Used to rescue Tier-5 plain-language fits whose skills are understated.
# --------------------------------------------------------------------------- #

IR_TEXT_EVIDENCE_PHRASES: tuple[str, ...] = (
    "recommendation system",
    "recommender system",
    "semantic search",
    "vector search",
    "information retrieval",
    "learning to rank",
    "search engine",
    "search infrastructure",
    "ranking model",
    "ranking system",
    "relevance",
    "personalization",
    "personalisation",
    "embedding",
    "retrieval",
    "nearest neighbor",
    "nearest neighbour",
)

# The verbatim "dabbler / keyword-curious" summary template (appears ~63k times).
# Its presence is a strong keyword-stuffer signature (negative N5).
DABBLER_SUMMARY_MARKERS: tuple[str, ...] = (
    "curious how tools could augment",
    "experimented",
    "productivity",
)


def normalize(text: str | None) -> str:
    """Lower-case and collapse surrounding whitespace; ``None`` -> ``''``."""
    return (text or "").strip().lower()


def normalized_set(items: Iterable[str]) -> frozenset[str]:
    return frozenset(normalize(i) for i in items)
