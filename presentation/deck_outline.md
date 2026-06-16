# Deck outline

Outline for the submission slides. The exported PDF/PPTX is submitted separately.

1. **Title** — project name, team, problem statement.

2. **The problem** — rank 100 of 100,000 candidates for one role; keyword filters
   miss good people; no labels and no leaderboard to tune against.

3. **The data** — mostly off-topic profiles, evenly sprinkled skills, some
   impossible profiles. The good candidates are a small slice.

4. **Approach** — a weighted rule engine plus an optional semantic matching step.
   Title and real skills over keywords; availability counts; drop invalid
   profiles.

5. **Pipeline** — load, filter, features, score, explain, rank. One diagram.

6. **Scoring** — the eight components and their weights, mapped to what the job
   description asks for.

7. **Semantic step** — embeddings find candidates whose fit is in their
   descriptions; the boost is capped so the rules stay in control.

8. **Explanations** — one fact-based reason per candidate, no made-up skills.

9. **Results** — top 100 are all relevant roles, no invalid profiles, runs in
   ~18s on CPU, passes the official checker, 34 tests.

10. **Limitations and next steps** — hand-set weights, partial filter, fallback
    embedding; what we'd do with more time or labels.

11. **Links** — GitHub repo, ranked output CSV, demo.

Figures go in `figures/` (pipeline diagram, scoring table, before/after title
mix).
