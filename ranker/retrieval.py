"""V2 Stage B + C — JD semantic expansion and top-K retrieval.

Stage B builds a handcrafted semantic query (NO LLM) that expands the JD's hidden
intent into the concrete capabilities a fit candidate would describe. Stage C
embeds that query with the same backend and retrieves the top-K candidates by
cosine similarity. The returned similarity is used ONLY to decide which
candidates are eligible for the semantic-evidence boost — never as a rank.
"""

from __future__ import annotations

import logging

import numpy as np

from .embeddings import EmbeddingBackend, EmbeddingMatrix

logger = logging.getLogger(__name__)


# Handcrafted JD intent expansion (Stage B). Each line is a capability a genuine
# Senior-AI-Engineer fit would express in prose, derived from the JD's "what it
# means" section and the intelligence report — not from any LLM.
JD_QUERY_PHRASES: tuple[str, ...] = (
    "built recommendation systems at a product company",
    "owned search relevance and ranking quality",
    "designed retrieval systems and semantic search infrastructure",
    "learning to rank and ranking model optimization",
    "embeddings based retrieval and vector search",
    "hybrid search with elasticsearch and vector databases",
    "marketplace matching candidates to jobs",
    "personalization and relevance optimization in production",
    "production machine learning systems deployed to real users",
    "information retrieval and nearest neighbor search",
    "evaluation of ranking systems with ndcg mrr and ab testing",
    "shipped end to end search recommendation or matching at scale",
    "fine tuning large language models for retrieval and reranking",
    "feature pipelines and real time inference for ranking",
)


def build_jd_query_text() -> str:
    """Concatenate the expanded intent phrases into one query document."""
    return ". ".join(JD_QUERY_PHRASES) + "."


def embed_query(backend: EmbeddingBackend) -> np.ndarray:
    vec = backend.encode([build_jd_query_text()])[0]
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def retrieve_top_k(
    embeddings: EmbeddingMatrix, backend: EmbeddingBackend, top_k: int
) -> dict[str, float]:
    """Return ``{candidate_id: cosine_similarity}`` for the top-K matches."""
    query = embed_query(backend).astype(np.float32)
    sims = embeddings.matrix @ query  # rows are L2-normalized -> cosine
    k = min(top_k, sims.shape[0])
    top_idx = np.argpartition(-sims, k - 1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    result = {embeddings.ids[i]: float(sims[i]) for i in top_idx}
    logger.info(
        "Retrieved top-%d semantic matches (sim range %.3f..%.3f)",
        k, sims[top_idx[-1]], sims[top_idx[0]],
    )
    return result
