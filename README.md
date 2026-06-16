# Redrob Candidate Ranker

A candidate ranking pipeline for the Redrob "Intelligent Candidate Discovery &
Ranking" challenge. Given a pool of 100,000 candidate profiles and one job
description (a Senior AI Engineer role), it returns the top 100 candidates as a
ranked CSV, with a short reason for each pick.

The whole thing runs on CPU, offline, in well under a minute. It's deterministic,
so the same input always gives the same output.

## What it does

Recruiters miss good people because keyword filters only see exact word matches.
We tried to rank candidates closer to how a recruiter would actually read a
profile: look at the job title, the real skills behind the buzzwords, the career
history, and whether the person is actually reachable.

The scoring is a weighted rule engine (we call it V1). On top of that there's an
optional semantic matching step (V2) that uses sentence embeddings to give a
small boost to people whose fit shows up in their written descriptions rather than
their skill tags. The semantic step is capped so the rule engine always stays in
control.

## Pipeline

```
candidates.jsonl  (100,000 profiles)
        |
        v
 [load]  stream the file one record at a time
        |
        v
 [filter] drop invalid / impossible profiles
        |
        v
 [features] pull out title, skills, career, signals, location
        |
        v
 [score]  8 weighted components  ->  score in [0, 1]
        |
        +-- (optional) semantic matching boost, capped at ~7%
        |
        v
 [explain] one short, fact-based reason per candidate
        |
        v
 [rank]  sort by score, take the top 100  ->  submission CSV
```

## Setup

Python 3.11+ (works on 3.13 too).

```bash
pip install -r requirements.txt
```

`requirements.txt` only needs PyYAML, NumPy and scikit-learn. The semantic step
can optionally use `sentence-transformers` (all-MiniLM-L6-v2); if it isn't
installed, the code falls back to a TF-IDF based representation automatically.

## Usage

Rule engine only (no embeddings, fastest):

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

With the semantic matching step:

```bash
python rank_v2.py --candidates ./candidates.jsonl --out ./submission.csv
```

Run the tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

The candidate file (`candidates.jsonl`) is the dataset from the challenge bundle
and is not included in this repo. Point `--candidates` at your local copy.

## Results

On the full 100,000-candidate pool:

- The top 100 are all AI / ML / Search / Data roles. No invalid profiles make it
  into the shortlist.
- 43 profiles are filtered out before scoring as invalid (impossible dates,
  contradictory experience).
- Ranking runs in about 18 seconds with the semantic step (around 23 seconds for
  the rule engine alone) and stays under 5 GB of memory.
- Output passes the official `validate_submission.py` checker.
- 34 unit tests cover the filtering, feature, scoring, and output stages.

See `submission/sample_output.csv` for an example of the ranked output.

## Repository layout

```
rank.py, rank_v2.py     command-line entry points
config.yaml             scoring weights and thresholds
ranker/                 the pipeline (loader, filter, features, scoring, output)
tests/                  unit tests
docs/                   approach, findings, architecture, limitations
presentation/           slide outline
submission/             sample ranked output + metadata
```

## Limitations

The weights are set by hand from what we learned about the data, not trained,
because the challenge ships no ground-truth labels. The invalid-profile filter
only catches contradictions we can actually check in the data. More detail is in
`docs/limitations.md`.

## License

MIT. See `LICENSE`.
