"""V2 orchestration — a retrieval-augmentation layer on top of V1.

V1 is consumed as-is: ``score_candidate`` is called unchanged, and the V1
weighted base is recomputed from its returned components. The semantic-evidence
score is then folded in as ONE new component at a small fixed weight, so V1
remains dominant:

    v2_base = ( Σ w_i·c_i(V1)  +  w_sem·semantic_evidence ) / ( Σ w_i + w_sem )
    v2_score = clamp( v2_base · stuffer_modifier )

With w_sem = 8 and Σw_i = 100, the semantic component contributes 8/108 ≈ 7.4%
of the final score — within the 5-10% target and never able to outvote V1.
Embeddings only gate *eligibility* for the boost (top-K retrieval); the boost
magnitude comes from explicit evidence snippets, never from cosine similarity.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from . import features as feat
from . import loader, scoring
from .config import Config
from .embeddings import EmbeddingStore, get_backend
from .explain import build_reasoning
from .honeypot import HoneypotDetector
from .pipeline import resolve_reference_date
from .retrieval import retrieve_top_k
from .semantic_evidence import (
    EvidenceResult,
    extract_evidence,
    semantic_evidence_score,
    top_snippets,
)
from .submission import ScoredCandidate, write_submission

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class V2Config:
    semantic_weight: float = 8.0       # absolute weight of the new component
    retrieval_top_k: int = 1000
    embedding_backend: str = "auto"    # auto | st | hashing
    cache_dir: str = ".cache_embeddings"
    batch_size: int = 256


def load_v2_config(path: str | Path | None) -> V2Config:
    if path is None:
        return V2Config()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    section = raw.get("v2", {}) or {}
    allowed = {f.name for f in V2Config.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return V2Config(**{k: v for k, v in section.items() if k in allowed})


@dataclass
class V2Record:
    candidate_id: str
    v1_score: float
    v2_score: float
    semantic_evidence: float
    semantic_similarity: float
    retrieved: bool
    title: str
    direct_ai_title: bool
    adjacent_title: bool
    snippets: list[str]
    reasoning: str


def _v1_weighted_base(components: dict[str, float], cfg: Config) -> float:
    weights = cfg.weights.as_dict()
    return sum(weights[k] * components[k] for k in components)


def _blend(v1_components: dict[str, float], cfg: Config, v2cfg: V2Config,
           sem: float, stuffer_modifier: float) -> float:
    numer = _v1_weighted_base(v1_components, cfg) + v2cfg.semantic_weight * sem
    denom = cfg.weights.total() + v2cfg.semantic_weight
    base = numer / denom if denom > 0 else 0.0
    return max(0.0, min(1.0, base * stuffer_modifier))


def _augment_reasoning(v1_reasoning: str, evidence: EvidenceResult, boosted: bool) -> str:
    """Append a grounded semantic-evidence clause only when real evidence exists."""
    if not boosted or not evidence.snippets:
        return v1_reasoning
    snips = top_snippets(evidence, limit=3)
    return f"{v1_reasoning} Semantic evidence: {'; '.join(snips)}."


def run_pipeline_v2(
    candidates_path: str | Path,
    out_path: str | Path,
    cfg: Config,
    v2cfg: V2Config,
) -> tuple[list[V2Record], dict]:
    t0 = time.perf_counter()
    reference_date: date = resolve_reference_date(candidates_path, cfg)
    detector = HoneypotDetector(reference_date, cfg.honeypot)

    # --- Stage A: embeddings (pre-computation; cached) ---
    backend = get_backend(v2cfg.embedding_backend, v2cfg.batch_size)
    store = EmbeddingStore(v2cfg.cache_dir, backend)
    embeddings = store.build_or_load(candidates_path, v2cfg.batch_size)
    t_embed = time.perf_counter()

    # --- Stage B + C: JD expansion + top-K retrieval ---
    retrieved = retrieve_top_k(embeddings, backend, v2cfg.retrieval_top_k)

    # --- ranking step (must stay < 5 min): screen, score, blend ---
    total = honeypots = boosted = 0
    records: list[V2Record] = []
    for candidate in loader.stream_candidates(candidates_path):
        total += 1
        if detector.screen(candidate).final_rank_excluded:
            honeypots += 1
            continue

        f = feat.extract(candidate, cfg.scoring, reference_date)
        v1 = scoring.score_candidate(f, cfg)

        cid = f.candidate_id
        sim = retrieved.get(cid)
        sem = 0.0
        snippets: list[str] = []
        evidence = EvidenceResult(candidate_id=cid)
        if sim is not None:  # Stage D + E only for retrieved candidates
            evidence = extract_evidence(candidate, sim)
            sem = semantic_evidence_score(evidence)
            snippets = top_snippets(evidence, limit=3)

        v2_score = _blend(v1.components, cfg, v2cfg, sem, v1.stuffer_modifier)
        is_boost = sem > 0.0 and v2_score > v1.score + 1e-9
        if is_boost:
            boosted += 1

        records.append(
            V2Record(
                candidate_id=cid,
                v1_score=v1.score,
                v2_score=v2_score,
                semantic_evidence=sem,
                semantic_similarity=sim or 0.0,
                retrieved=sim is not None,
                title=f.current_title,
                direct_ai_title=f.direct_ai_title,
                adjacent_title=f.adjacent_title,
                snippets=snippets,
                # Score-tier fallback reasoning; rank-aware text is filled in below
                # for the selected top-N (which is all that gets written).
                reasoning=_augment_reasoning(build_reasoning(f, v1), evidence, is_boost),
            )
        )

    # --- rank-aware reasoning for the written rows (text only; order is frozen) ---
    # Ordering and scores are unchanged: we sort by the same key used for output,
    # then regenerate ONLY the reasoning string for the selected top-N using each
    # candidate's final rank. A second short pass re-extracts features for just
    # those N candidates, so memory stays flat.
    top_n = cfg.output.top_n
    by_id = {r.candidate_id: r for r in records}
    ordered = sorted(records, key=lambda r: (-r.v2_score, r.candidate_id))
    selected_ranks = {r.candidate_id: i + 1 for i, r in enumerate(ordered[:top_n])}
    for candidate in loader.stream_candidates(candidates_path):
        cid = candidate.get("candidate_id", "")
        rank = selected_ranks.get(cid)
        if rank is None:
            continue
        f = feat.extract(candidate, cfg.scoring, reference_date)
        v1 = scoring.score_candidate(f, cfg)
        sim = retrieved.get(cid)
        evidence = EvidenceResult(candidate_id=cid)
        is_boost = False
        if sim is not None:
            evidence = extract_evidence(candidate, sim)
            sem = semantic_evidence_score(evidence)
            is_boost = sem > 0.0 and by_id[cid].v2_score > v1.score + 1e-9
        by_id[cid].reasoning = _augment_reasoning(
            build_reasoning(f, v1, rank=rank, total=top_n), evidence, is_boost
        )

    scored = [ScoredCandidate(r.candidate_id, r.v2_score, r.reasoning) for r in records]
    selected = write_submission(scored, out_path, top_n, cfg.output.score_decimals)

    elapsed = time.perf_counter() - t0
    stats = {
        "total_candidates": total,
        "honeypots_rejected": honeypots,
        "eligible": len(records),
        "retrieved_top_k": len(retrieved),
        "semantically_boosted": boosted,
        "selected": len(selected),
        "embedding_backend": backend.name,
        "embed_seconds": round(t_embed - t0, 2),
        "ranking_seconds": round(elapsed - (t_embed - t0), 2),
        "total_seconds": round(elapsed, 2),
        "top_v2_score": selected[0].score if selected else None,
        "cutoff_v2_score": selected[-1].score if selected else None,
    }
    logger.info("V2 pipeline stats: %s", stats)
    return records, stats
