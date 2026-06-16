# Findings

Notes from exploring the 100,000-candidate dataset before building the ranker.

## Dataset observations

- **Most of the pool is off-topic.** About two-thirds of the profiles have titles
  like HR Manager, Accountant, Sales Executive, or Mechanical Engineer. Only
  around 1,200 have a direct AI/ML/Search/Data title. The good candidates are a
  small slice.

- **Skills are sprinkled almost evenly.** Roughly 80 common skills (HTML, AWS,
  Kafka, etc.) each show up about 12,000 times, so listing a skill barely says
  anything. The AI skills split into two groups: a "buzzword" group (Pinecone,
  RAG, Embeddings) at about 5,000 each, and a smaller, more meaningful group
  (PyTorch, BM25, Learning-to-Rank) at about 1,380 each.

- **Skill claims can be checked.** Each profile has assessment scores per skill.
  About 800 candidates claim "advanced" on a skill but score under 25 on the
  matching test, which is a useful signal that the claim is weak.

- **Summaries are templated.** A lot of summaries are generated from a few
  templates. One "casual AI dabbler" template appears about 63,000 times, which
  makes a handy negative signal.

- **There are impossible profiles.** A small number have contradictions: a job
  lasting more years than it has existed, or "expert" in a skill used for 0
  months. Using three simple checks we flag 43 of them and drop them before
  scoring.

- **Companies cluster.** A few real product/startup names (Swiggy, Razorpay, CRED,
  and AI firms like Sarvam, Haptik) show up as positive signals, while consulting
  firms (TCS, Infosys, Wipro) are a negative signal for this role.

## What worked

- Weighting title fit highly keeps the obvious keyword-stuffers out of the top
  100.
- Splitting AI skills into "buzzword" vs. "real" and checking assessment scores
  separates people who list skills from people who have them.
- Using behavioral signals as part of the score pushes down strong-on-paper
  profiles that aren't actually reachable.
- The impossible-profile filter is precise: it never wrongly dropped a real
  candidate in our checks.

## What did not work as well

- The semantic matching step helped less than expected with the offline fallback
  embedding. It mostly reshuffled candidates that were already strong rather than
  surfacing hidden ones. With a proper sentence-transformer model it would likely
  do more, but the gain on the top 100 is small either way.
- The invalid-profile filter only catches contradictions we can verify from the
  data. Some likely-bad profiles need information we don't have (like company
  founding dates), so we can't flag them.
