"""V2 Stage A — offline candidate embedding pipeline.

Embeddings are used ONLY as a recall gate (which candidates are eligible for a
small semantic boost). They never determine final rank — see ``semantic_evidence``
and ``pipeline_v2`` for how the boost is derived from explicit evidence snippets,
not from cosine similarity.

Backends (auto-selected):
  * ``SentenceTransformerBackend`` — ``sentence-transformers/all-MiniLM-L6-v2``.
    The production path. Used whenever ``sentence-transformers`` is importable.
  * ``HashingBackend`` — a deterministic, dependency-light TF-IDF/hashing
    vectorizer (numpy + scikit-learn). Offline, CPU-only, reproducible. Used as a
    fallback so the pipeline runs anywhere (incl. sandboxes without torch).

Both backends emit L2-normalized rows, so cosine similarity == dot product.
Embeddings are cached to disk keyed by backend + candidate count, so the
(potentially slow) generation is a one-time pre-computation; the ranking step
just loads the cache.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np

from . import vocab
from .loader import stream_candidates

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Candidate text representation
# --------------------------------------------------------------------------- #

def build_candidate_text(candidate: dict) -> str:
    """Concatenate the free-text fields that carry semantic fit signal.

    Per the V2 spec: headline, summary, current title, and every career-history
    title + description. This is where Tier-5 plain-language fits hide.
    """
    profile = candidate.get("profile") or {}
    parts: list[str] = [
        profile.get("headline") or "",
        profile.get("summary") or "",
        profile.get("current_title") or "",
    ]
    for role in candidate.get("career_history") or []:
        parts.append(role.get("title") or "")
        parts.append(role.get("description") or "")
    return "  ".join(p for p in parts if p).strip()


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #

class EmbeddingBackend(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray:  # (n, d) float32, L2-normalized
        ...


class SentenceTransformerBackend:
    """Production backend: sentence-transformers/all-MiniLM-L6-v2 (CPU)."""

    name = "all-MiniLM-L6-v2"

    def __init__(self, batch_size: int = 256) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self.batch_size = batch_size
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)


class HashingBackend:
    """Deterministic offline fallback: hashing + TF-IDF, L2-normalized rows.

    Not a neural embedding, but a reproducible CPU-only dense-ish representation
    that supports the same cosine-retrieval interface. Dimensionality is reduced
    so the matrix stays dense and small for fast top-k retrieval.
    """

    name = "hashing-tfidf"

    def __init__(self, n_features: int = 4096) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer

        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            stop_words="english",
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        sparse = self.vectorizer.transform(texts)  # already L2-normalized rows
        return np.asarray(sparse.todense(), dtype=np.float32)


def get_backend(prefer: str = "auto", batch_size: int = 256) -> EmbeddingBackend:
    """Return the requested backend, falling back to hashing when ST is absent."""
    if prefer in ("auto", "st", "sentence-transformers"):
        try:
            backend = SentenceTransformerBackend(batch_size=batch_size)
            logger.info("Embedding backend: %s (sentence-transformers)", backend.name)
            return backend
        except Exception as exc:  # noqa: BLE001 - any import/load failure -> fallback
            if prefer != "auto":
                raise
            logger.warning(
                "sentence-transformers unavailable (%s); falling back to hashing-tfidf. "
                "Install torch + sentence-transformers for the production embedding.",
                type(exc).__name__,
            )
    backend = HashingBackend()
    logger.info("Embedding backend: %s (offline fallback)", backend.name)
    return backend


# --------------------------------------------------------------------------- #
# Store / cache
# --------------------------------------------------------------------------- #

@dataclass
class EmbeddingMatrix:
    ids: list[str]
    matrix: np.ndarray  # (n, d) float32, L2-normalized
    backend_name: str

    def __len__(self) -> int:
        return len(self.ids)


def _batched(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


class EmbeddingStore:
    """Builds and caches candidate embeddings (Stage A)."""

    def __init__(self, cache_dir: str | Path, backend: EmbeddingBackend) -> None:
        self.cache_dir = Path(cache_dir)
        self.backend = backend
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _paths(self) -> tuple[Path, Path]:
        stem = self.backend.name.replace("/", "_")
        return (
            self.cache_dir / f"emb_{stem}.npy",
            self.cache_dir / f"meta_{stem}.json",
        )

    def build_or_load(self, candidates_path: str | Path, batch_size: int = 256) -> EmbeddingMatrix:
        emb_path, meta_path = self._paths()
        if emb_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            matrix = np.load(emb_path)
            if len(meta["ids"]) == matrix.shape[0]:
                logger.info("Loaded cached embeddings: %d x %d from %s",
                            matrix.shape[0], matrix.shape[1], emb_path)
                return EmbeddingMatrix(meta["ids"], matrix, self.backend.name)
            logger.warning("Cache size mismatch; regenerating embeddings.")

        ids: list[str] = []
        texts: list[str] = []
        for cand in stream_candidates(candidates_path):
            ids.append(cand.get("candidate_id", ""))
            texts.append(build_candidate_text(cand))
        logger.info("Encoding %d candidate documents with %s ...", len(texts), self.backend.name)

        chunks = [self.backend.encode(batch) for batch in _batched(texts, batch_size)]
        matrix = np.vstack(chunks).astype(np.float32)

        np.save(emb_path, matrix)
        meta_path.write_text(json.dumps({"ids": ids, "backend": self.backend.name}), encoding="utf-8")
        logger.info("Cached embeddings to %s (%d x %d)", emb_path, *matrix.shape)
        return EmbeddingMatrix(ids, matrix, self.backend.name)
