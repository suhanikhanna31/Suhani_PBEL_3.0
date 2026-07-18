# Insider Threat NLP — Behavioral / Psycholinguistic Drift Detection

An internship project detecting insider-risk and account-compromise signals
from **how** people communicate internally — sentiment, urgency language, and
stylometric drift relative to each person's own baseline — rather than from
logs or network traffic. Built around IBM watsonx.ai / watsonx.governance,
supervised + unsupervised ML, and a set of DSA structures chosen for the
scale this kind of monitoring runs at.

Privacy is treated as a core design constraint, not an afterthought — see
[`docs/ETHICS_AND_PRIVACY.md`](docs/ETHICS_AND_PRIVACY.md) before reading
anything else.

## What this actually does

1. Ingests CERT-schema internal communication data (email metadata + content).
2. Gates every row on recorded user **consent**, then **pseudonymizes**
   usernames with a salted HMAC — no raw identifiers or raw message text
   are ever persisted or shown to an analyst.
3. Extracts linguistic features (sentiment, urgency-phrase matching via an
   Aho-Corasick automaton, readability, lexical diversity) and stylometric
   features (function-word ratios, punctuation habits, POS ratios).
4. Maintains a **per-user rolling baseline** for each feature (an O(1)
   amortized sliding-window mean/variance structure) and scores each new
   message's **drift** from that baseline as a weighted z-score.
5. Trains both a **supervised** classifier (RandomForest + XGBoost, against
   ground-truth labels) and **unsupervised** models (IsolationForest for
   message-level anomalies, DBSCAN for user-level structural outliers) —
   three independent signals, surfaced together rather than collapsed into
   one opaque score.
6. Calls **watsonx.ai** to turn a flagged user's statistics into a
   plain-language explanation, and exposes a **watsonx Assistant** webhook
   so an analyst can ask "show me users whose tone changed this week" in
   natural language.
7. Logs every scoring decision to a **hash-chained, tamper-evident audit
   log** (the watsonx.governance-style piece), gated behind a role-based
   access control layer.
8. Serves a minimalist analyst dashboard (FastAPI + static HTML/JS,
   IBM-Carbon-inspired design) and deploys as a single container to
   **IBM Cloud Code Engine**.

## Dataset

This uses the **CERT Insider Threat dataset (r4.2)**, the same dataset
repackaged by both Kaggle links you found:
- `kaggle.com/datasets/nitishabharathi/cert-insider-threat`
- `kaggle.com/datasets/andrihjonior/cert-insider-threat-dataset-r4-2`

**Kaggle isn't reachable from this build environment**, so the code ships
with `src/data/synthetic_cert_data.py`, which generates data in the exact
same schema (`id, date, user, pc, to, cc, bcc, from, activity, size,
attachments, content`) with realistic normal vs. risky message templates
and a simulated "drift before an incident" pattern for a subset of users —
this is what lets the entire pipeline run and be demoed today.

**To use the real dataset:** download `email.csv` from either Kaggle link
and drop it into `data/raw/`. `src/data/ingest.py` auto-detects it and uses
it instead of generating synthetic data — nothing else needs to change.
If you want to train the supervised model against real ground truth, add a
`data/raw/insider_labels.csv` with columns `user, is_malicious_insider`
(CERT ships separate "answer key" files identifying the synthetic insider
scenarios — map those to this schema).

## Project structure

See the tree in this repo — it mirrors the structure discussed with your
mentor almost exactly (Part 1 = data/consent/features/baseline DSA,
Part 2 = drift scoring/models/API/governance, Part 3 = frontend/tests/deployment).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or your preferred env tool
pip install -r requirements.txt --break-system-packages   # drop the flag if not needed on your system
cp .env.example .env   # then fill in your watsonx.ai API key + project ID
```

## Running the pipeline

```bash
python -m src.pipeline
```

This ingests data (real if present in `data/raw/`, else synthetic),
anonymizes it, extracts features, builds baselines, scores drift, runs the
unsupervised models, and writes results to `data/processed/`.

Train the supervised models separately (needs labels):

```bash
python -m src.models.supervised.train
```

## Running the API + dashboard

```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000` for the dashboard, or `http://localhost:8000/docs`
for the interactive API docs (FastAPI's auto-generated Swagger UI).

API requests need a dev role token header (replace with real IAM before
production use — see `docs/ETHICS_AND_PRIVACY.md`):
```
X-API-Role-Token: dev-analyst-token
```
or `dev-admin-token` for endpoints that also allow consent revocation /
audit-log access.

## Individual module demos

Every module under `src/` has a `if __name__ == "__main__":` block you can
run directly to see it in isolation, e.g.:
```bash
python -m src.dsa.sliding_window
python -m src.dsa.trie_phrase_matcher
python -m src.features.baseline_engine
python -m src.models.watsonx.client
```

## Notebooks

- `notebooks/01_eda.ipynb` — exploratory data analysis on the ingested
  dataset (distribution of message lengths, activity over time, per-user
  message counts) — this is where the NASSCOM EDA course material applies
  most directly.
- `notebooks/02_feature_exploration.ipynb` — visualizing the extracted
  linguistic/stylometric features and how baseline z-scores separate the
  simulated insiders from everyone else.

## Deploying

See [`deployment/ibm-cloud/DEPLOY.md`](deployment/ibm-cloud/DEPLOY.md) for
step-by-step IBM Cloud Code Engine deployment commands.

## Tests

```bash
pytest tests/
```

## Design docs

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — pipeline shape and the
  reasoning behind each structural choice, including why the DSA
  structures are there and what supervised vs. unsupervised are each for.
- [`docs/ETHICS_AND_PRIVACY.md`](docs/ETHICS_AND_PRIVACY.md) — consent,
  anonymization, minimization, governance, and what would still need to
  change before this could touch real employee data.
