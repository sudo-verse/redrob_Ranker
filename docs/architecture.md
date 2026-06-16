# Architecture

## Overview

The system is a straight pipeline. A profile comes in, passes through a few
stages, and either gets a score or gets dropped. At the end we sort everyone and
write the top 100. Every stage is a small module so it's easy to test and read.

There's no database and no service. It's a CLI program that reads a JSONL file and
writes a CSV.

## Pipeline stages

1. **Load** (`loader.py`)
   Reads `candidates.jsonl` one line at a time so we never hold the whole 465 MB
   file in memory.

2. **Filter** (`honeypot.py`)
   Drops profiles that are internally impossible: a job that lasted longer than
   the time since it started, "expert" in a skill used for 0 months, or total
   work history far longer than the stated years of experience. These are clear
   data problems, so we remove them before scoring.

3. **Features** (`features.py`, `vocab.py`)
   Turns the raw profile into about 30 simple values: is the title an AI/ML role,
   how many real IR skills vs. buzzwords, product vs. consulting employers,
   recruiter response rate, last active date, location, and so on. `vocab.py`
   holds the lookup lists (which titles count as AI roles, which companies are
   product companies, etc.).

4. **Score** (`scoring.py`, `config.yaml`)
   Combines the features into one number between 0 and 1. Each component is
   normalized to [0, 1] and multiplied by a weight from `config.yaml`:

   | component          | weight |
   |--------------------|-------:|
   | title fit          | 25 |
   | IR skill depth     | 20 |
   | product experience | 15 |
   | availability       | 15 |
   | assessment trust   | 10 |
   | location           | 5 |
   | experience range   | 5 |
   | github             | 5 |

5. **Semantic matching** (`embeddings.py`, `retrieval.py`, `semantic_evidence.py`)
   Optional step used by `rank_v2.py`. It embeds the job description and the
   candidate text, finds the closest 1,000 profiles, and adds a small boost for
   ones that mention relevant work (search, ranking, recommendations). The boost
   is capped at about 7% so it can shuffle borderline cases but can't override the
   rule engine.

6. **Explain** (`explain.py`)
   Writes a one or two sentence reason for each candidate using only facts from
   their profile, so it never invents a skill or employer they don't have.

7. **Output** (`submission.py`)
   Sorts by score (ties broken by candidate id), takes the top 100, and writes the
   CSV in the required format.

## Key modules

- `pipeline.py` / `pipeline_v2.py` — wire the stages together for the rule engine
  and the semantic version.
- `config.py` — loads weights and thresholds from `config.yaml`.
- `submission.py` — handles sorting and CSV writing, and enforces the format
  rules (100 rows, unique ranks, scores non-increasing).

## Design notes

- Everything is deterministic. No randomness, so results are reproducible.
- Weights live in `config.yaml`, not in code, so they're easy to inspect and
  change.
- The semantic step is optional and bounded on purpose: the embeddings help find
  candidates but don't get to decide the final order.
