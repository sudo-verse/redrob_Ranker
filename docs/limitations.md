# Limitations and future work

## Known limitations

- **Weights are hand-tuned.** There are no labels in the dataset, so the eight
  weights come from what we learned about the data, not from training. They're
  reasonable but we can't prove they're optimal.

- **Invalid-profile filter is partial.** We only drop profiles with
  contradictions we can check directly (43 of them). Other suspicious profiles
  would need data we don't have, so they stay in the pool. None of them reach the
  top 100 in practice, but that isn't guaranteed.

- **Semantic step uses a fallback embedding here.** The code prefers
  `sentence-transformers` (all-MiniLM-L6-v2) but falls back to a TF-IDF
  representation when it isn't installed. Our reported numbers use the fallback,
  so the semantic step is a bit weaker than it could be.

- **Vocabulary is specific to this dataset.** The title, skill, and company lists
  in `vocab.py` were built from this pool. They wouldn't transfer directly to a
  different role or a real production system without updating.

- **Some reasons read similarly.** The explanations are fact-based and vary by
  candidate, but the strong picks naturally share a similar structure because the
  top candidates are a fairly uniform group.

## Future improvements

- Install the sentence-transformer model and re-check whether the semantic step
  moves the top 100 more than the fallback does.
- If labels (or recruiter feedback) ever become available, replace the hand-set
  weights with a learning-to-rank model trained on them.
- Add more invalid-profile checks if richer data (company info, verified dates)
  becomes available.
- Make the vocabulary lists configurable so the same pipeline can serve other job
  descriptions.
