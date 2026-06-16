"""V2 Stage F — compare V1 vs V2 top-100 from a single set of V2Records.

Both rankings are derived from the same records (V1 by ``v1_score``, V2 by
``v2_score``) so the comparison is apples-to-apples: the only difference is the
semantic component.
"""

from __future__ import annotations

from dataclasses import dataclass

from .pipeline_v2 import V2Record


def _rank(records: list[V2Record], key) -> list[V2Record]:
    return sorted(records, key=lambda r: (-key(r), r.candidate_id))


@dataclass
class V2Comparison:
    v1_top: list[str]
    v2_top: list[str]
    added: list[V2Record]            # entered top-100 in V2
    removed: list[V2Record]          # dropped out of top-100 in V2
    rescues: list[V2Record]          # added candidates rescued by description evidence
    potential_false_positives: list[V2Record]
    top_n: int


def compare(records: list[V2Record], top_n: int = 100) -> V2Comparison:
    v1_sorted = _rank(records, lambda r: r.v1_score)[:top_n]
    v2_sorted = _rank(records, lambda r: r.v2_score)[:top_n]
    v1_ids = {r.candidate_id for r in v1_sorted}
    v2_ids = {r.candidate_id for r in v2_sorted}

    by_id = {r.candidate_id: r for r in records}
    added = [by_id[i] for i in v2_ids - v1_ids]
    removed = [by_id[i] for i in v1_ids - v2_ids]

    # A "rescue" is an added candidate whose lift comes from description-level
    # semantic evidence rather than a direct AI title — exactly V2's purpose.
    rescues = sorted(
        [r for r in added if not r.direct_ai_title and r.semantic_evidence > 0],
        key=lambda r: (-r.semantic_evidence, -r.v2_score),
    )

    # Potential false positives: added but with weak career-grounded evidence
    # (no snippets sourced from real roles, or adjacent/irrelevant title with thin
    # diversity). Flagged for human review; not auto-removed.
    potential_fp = sorted(
        [
            r for r in added
            if (not any("(career)" in s for s in r.snippets))
            and r.semantic_evidence < 0.5
        ],
        key=lambda r: r.semantic_evidence,
    )

    return V2Comparison(
        v1_top=[r.candidate_id for r in v1_sorted],
        v2_top=[r.candidate_id for r in v2_sorted],
        added=sorted(added, key=lambda r: -r.v2_score),
        removed=sorted(removed, key=lambda r: -r.v1_score),
        rescues=rescues,
        potential_false_positives=potential_fp,
        top_n=top_n,
    )
