# Approach

## Problem

Rank the top 100 candidates from a 100,000-profile pool for one job description (a
Senior AI Engineer role). The scoring is based mostly on getting the top 10 right
(NDCG@10), so the very top of the list matters most. The dataset also contains
profiles that are deliberately misleading: people who list every AI keyword but
have an unrelated job, and a small number of profiles with impossible details.

The catch is there are no labels and no leaderboard. We can't train a model on
"correct" answers and we can't test submissions to see what scores well.

## Solution

Because there's nothing to train on, we went with a weighted rule engine instead
of a learned model. We read the job description, listed what the role actually
needs, and turned each requirement into a feature with a weight.

The main ideas:

- **Title matters more than keywords.** A "Marketing Manager" who lists nine AI
  skills is not a fit. A "Search Engineer" or "ML Engineer" usually is. Title fit
  is the biggest single weight.
- **Real skills over buzzwords.** In this dataset the AI skills split into a
  common set (Pinecone, RAG) that almost everyone lists, and a rarer set (PyTorch,
  BM25, Learning-to-Rank) that's a better signal of actual experience. We weight
  the rarer ones higher and check the candidate's assessment scores to see if the
  claims hold up.
- **Availability is part of fit.** A strong profile that hasn't logged in for
  months or never replies to recruiters isn't actually hireable. We use the
  behavioral signals (response rate, last active, notice period) as part of the
  score.
- **Drop the impossible profiles.** Some profiles have contradictions you can
  check directly, like a job lasting longer than it has existed. We filter those
  out before scoring.

On top of the rule engine, `rank_v2.py` adds a semantic matching step. It uses
sentence embeddings to find candidates whose descriptions match the role even if
their skill tags don't, and gives them a small, capped boost.

## Why these choices

- **No trained model** because there are no labels. A learned ranker would just be
  fitting our own guesses, which adds risk without a way to check it.
- **Rules in config, not code** so the weights are easy to read and adjust.
- **Bounded semantic step** so the embeddings help with recall but can't push a
  bad profile into the top 100.
- **Fact-based reasons** so the explanations can be trusted and don't make things
  up.
- **CPU-only and fast** because a real recruiting system can't call a large model
  once per candidate. The whole ranking runs in under a minute.
