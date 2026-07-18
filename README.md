#PROJECT 14 (AI-Based Cyber Threat detection Framework) Signal: Behavioral & Psycholinguistic Insider-Threat Detection

**An AI/ML system that flags insider-risk and account-compromise signals from *how* people communicate internally — not from logs or network traffic.**

Built for an IBM internship using watsonx.ai, watsonx Assistant, and watsonx.governance, with a full supervised + unsupervised ML pipeline, custom-optimized data structures, and a privacy-first design throughout.

---

## The idea

Most insider-threat tooling watches *what* people access — file transfers, login anomalies, badge swipes. This project watches *how* people write.

The premise, from the original project brief: subtle shifts in tone, urgency, or phrasing in internal communications (email, Slack, tickets) correlate with three distinct risk patterns —

- **Insider risk** — someone's communication style drifting before a harmful action (the CERT insider-threat research dataset this project is built on documents this pattern directly).
- **Social engineering susceptibility** — messages showing markers of manipulation (urgency, secrecy language, authority pressure).
- **Account compromise** — someone impersonating a colleague, whose writing style doesn't match the account's normal baseline.

The system doesn't try to read intent. It measures **statistical drift** — how far a person's current writing has moved from their own historical baseline — and surfaces that as a ranked, explainable signal for a human analyst to review. It never acts autonomously on a flag.

---

## How it actually works, end to end

```
Raw email data (CERT schema)
        │
        ▼
  Consent gate  →  drops any user without an active consent record
        │
        ▼
  Anonymization  →  salted HMAC pseudonymizes every username
        │              (emp_a1b2c3d4e5f6 — irreversible without the salt)
        ▼
  Feature extraction (per message)
        │
        ├─ Linguistic features: sentiment, subjectivity, urgency-phrase
        │  matching, readability, lexical diversity, surface stats
        │
        └─ Stylometric features: function-word ratios, punctuation habits,
           part-of-speech ratios — the "fingerprint" of how someone writes,
           independent of what they're writing about
        │
        ▼
  Per-user rolling baseline  →  a sliding window of each person's own
        │                        recent history, per feature
        ▼
  Drift scoring  →  weighted z-score: how far is *this* message from
        │            *this person's own* recent normal?
        ▼
  ┌─────────────────┬──────────────────────┬───────────────────────┐
  │  Aggregate       │  IsolationForest     │  DBSCAN               │
  │  per-user risk    │  (message-level,     │  (user-level,         │
  │  (drift trend)    │  unsupervised        │  unsupervised         │
  │                    │  anomaly detection)  │  structural outliers) │
  └─────────────────┴──────────────────────┴───────────────────────┘
        │
        ▼
  Supervised classifier  →  RandomForest + XGBoost, trained on
        │                    aggregated per-user drift features against
        │                    ground-truth insider labels
        ▼
  watsonx.ai  →  turns a flagged user's raw statistics into a
        │         plain-language explanation an analyst can read in
        │         one glance
        ▼
  watsonx Assistant webhook  →  "show me users whose communication tone
        │                        changed significantly this week"
        ▼
  Audit log (hash-chained, tamper-evident)  +  Role-based access control
        │
        ▼
  Analyst dashboard (FastAPI + minimalist IBM-style frontend)
```

Every one of those boxes is a real, working, independently-testable module — not a diagram of an idea. The project ships with 17 passing unit tests and a synthetic-data generator that exercises the entire pipeline end to end without needing the real dataset present.

---

## The data science / ML core

### Data analysis, visualization, and feature engineering

This is where the NASSCOM EDA training and the core Python data stack get put to direct use, not just imported and forgotten:

- **pandas** is the backbone of the entire pipeline — every stage from raw ingestion through final risk aggregation is a DataFrame transformation (`src/data/ingest.py`, `src/features/*.py`, `src/features/drift_scoring.py`). Groupby aggregations roll thousands of per-message rows up into per-user risk summaries; merges join ground-truth labels against pseudonymized feature tables for supervised training.
- **numpy** underpins the statistical core: the sliding-window baseline engine computes running mean/variance incrementally (see DSA section below), and every feature extractor — sentiment scores, z-scores, drift scores — is numeric array math under the hood.
- **matplotlib** and **seaborn** drive the two exploratory notebooks (`notebooks/01_eda.ipynb`, `notebooks/02_feature_exploration.ipynb`): message-volume time series, per-user activity distributions, message-length histograms, and — the key validation plot — a boxplot comparing average drift scores between normal and ground-truth-insider users, which is the empirical check on whether the entire linguistic-drift hypothesis actually holds on this data before trusting it operationally.
- **scikit-learn** provides `StandardScaler`, `train_test_split`, `RandomForestClassifier`, `IsolationForest`, `DBSCAN`, and the full classification-report/AUC/confusion-matrix evaluation stack.
- **NLTK** and **TextBlob** do the lightweight NLP (POS tagging, sentiment) — deliberately kept light per the project's own constraint, rather than reaching for a heavyweight transformer stack that would be harder to explain, audit, or run cheaply at message-level scale.

### Supervised learning

`src/models/supervised/train.py` trains two classifiers side by side on the same aggregated per-user features (`avg_drift_score`, `max_drift_score`, `n_messages`, `flagged_message_rate`):

- **RandomForest** (bagging, `class_weight="balanced"` to handle the inherent rarity of insiders in the data)
- **XGBoost** (boosting, `scale_pos_weight` tuned the same way)

Running both side by side isn't redundant — it's a deliberate comparison of two different ensemble philosophies on a class-imbalanced problem, evaluated with AUC, a full classification report, and a confusion matrix, not just accuracy (which is a misleading metric when positives are rare).

### Unsupervised learning

Ground-truth insider labels only exist here because CERT is a research dataset with synthetic scenarios baked in — a real deployment won't have that luxury. So the system also runs two label-free models, treated as independent signals rather than a fallback:

- **IsolationForest** — flags individual *messages* as statistical outliers in feature space, no labels required.
- **DBSCAN** — clusters *users* by their aggregate behavioral profile; anyone who doesn't fall into a dense peer cluster (`label == -1`) is structurally different from everyone else, which is a distinct signal from "drifted from their own baseline."

The dashboard and API surface all three signals — drift score, IsolationForest anomaly, DBSCAN outlier status — side by side, so an analyst can see where they agree and where they don't, instead of trusting one model's opinion as ground truth.

---

## DSA — and why each structure is there, not just what it does

This project treats data-structure choice as a real engineering decision tied to the scale insider-threat monitoring actually runs at (tens of thousands of employees, hundreds of messages each), not a checkbox:

| Structure | File | Problem it solves | Complexity win |
|---|---|---|---|
| **Sliding-window rolling stats** | `src/dsa/sliding_window.py` | Recomputing a user's baseline mean/variance from scratch on every new message | O(w) per update → **O(1) amortized**, via an incremental running-sum approach (a bounded-window variant of Welford's algorithm) |
| **Aho-Corasick automaton** | `src/dsa/trie_phrase_matcher.py` | Scanning every message against dozens of urgency/social-engineering phrases | O(phrases × text length) naive substring search → **O(text length + matches)**, independent of how many phrases are in the lexicon |
| **Bounded min-heap** | `src/dsa/top_k_heap.py` | Finding the Top-K riskiest users out of a full population for the dashboard | O(N log N) full sort → **O(N log K)**, since K (10–50) is tiny compared to N (thousands) |
| **LRU cache** | `src/dsa/lru_cache_baselines.py` | Keeping every user's baseline resident in memory forever in a long-running service | Unbounded memory growth → **O(capacity)** bounded memory, keyed to actually-recent activity |

Each one is unit-tested in isolation (`tests/test_core.py`) and has a runnable self-test (`python -m src.dsa.<module>`) demonstrating the exact behavior it's built for — including edge cases like zero-variance baselines and near-zero-variance z-score blowup (which the drift-scoring layer explicitly clips, see `MAX_ABS_Z` in `src/features/drift_scoring.py`, so one near-constant feature can't dominate an otherwise-meaningful risk score).

---

## The watsonx stack

- **watsonx.ai** (`src/models/watsonx/client.py`) — used narrowly, on top of the local ML, for the one task a foundation model is genuinely better suited to than hand-built features: turning a flagged user's raw statistics into a short, readable explanation (`explain_drift()`), and offering a second-opinion classification on individual messages (`classify_message_risk()`). All local scoring runs independently of watsonx — if it's unconfigured, the pipeline still runs end to end; watsonx is an enhancement layer, not a dependency.
- **watsonx Assistant** (`src/api/routes/assistant.py`) — a webhook endpoint (`POST /api/assistant/webhook`) built specifically so an analyst can ask natural-language questions like *"show me users whose communication tone changed significantly this week"* and have the Assistant call back into the same ranked risk data the dashboard uses.
- **watsonx.governance** — implemented as a hash-chained, tamper-evident audit log (`src/governance/audit_log.py`) that records every drift flag, model prediction, and watsonx.ai call against a pseudonymized user, with `verify_chain()` detecting any retroactive edit or deletion. This mirrors the tamper-evidence principle watsonx.governance provides at enterprise scale, implemented here at a scale that's fully inspectable in a single codebase.

---

## The privacy design (the "huge angle to address explicitly")

This is documented in full in [`docs/ETHICS_AND_PRIVACY.md`](docs/ETHICS_AND_PRIVACY.md) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), but the short version:

- **Consent-gated by default** — no user's data enters the feature pipeline without an active consent record (`src/governance/consent.py`), revocable at any time.
- **Pseudonymized, not just "anonymized"** — every username is replaced with a salted HMAC-SHA256 hash before any feature is computed or persisted. Analysts see `emp_a1b2c3d4e5f6`, never a real name. Re-identification requires deliberately re-running the hash with the salt — a step that should require authorization outside the system, not something the dashboard can do.
- **Minimized** — raw message content is read in memory during feature extraction and then discarded. It is never written to disk, never returned by the API, never sent to watsonx.ai. Only derived numeric features and scores persist.
- **Auditable** — every scoring decision is logged, tamper-evidently, with role-based access control gating who can see drift data versus who can revoke consent or inspect the audit log.
- **Honest about its limits** — the ethics doc explicitly states what pseudonymization does *not* protect against (small-organization re-identification via metadata), and what would need to change before this could touch real employee data (real consent system of record, real IAM, legal/HR review, bias auditing across non-native English speakers and different communication styles).

---

## Project structure

```
insider-threat-nlp/
├── src/
│   ├── config.py                  Central config (env-driven)
│   ├── data/                      Ingestion, anonymization, validation, synthetic data
│   ├── features/                  Linguistic + stylometric extraction, baseline engine, drift scoring
│   ├── dsa/                       Sliding window, Aho-Corasick, top-K heap, LRU cache
│   ├── models/
│   │   ├── supervised/            RandomForest + XGBoost training
│   │   ├── unsupervised/          IsolationForest + DBSCAN
│   │   └── watsonx/                watsonx.ai client
│   ├── governance/                Consent, anonymization, audit log, access control
│   ├── api/                       FastAPI app + routes (users, drift, assistant, governance)
│   └── pipeline.py                End-to-end orchestrator
├── frontend/                      Analyst dashboard (FastAPI-served static HTML/JS)
├── notebooks/                     EDA + feature exploration (pandas/matplotlib/seaborn)
├── deployment/                    Dockerfile + IBM Cloud Code Engine deploy guide
├── docs/                          Architecture + Ethics/Privacy design docs
└── tests/                         17 unit tests across DSA, drift scoring, anonymization
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in watsonx.ai credentials if available

python -m src.pipeline                                    # run the full pipeline
python -m src.models.supervised.train                     # train supervised models
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000  # dashboard at localhost:8000
pytest tests/                                              # 17 tests
```

No real dataset required to see it run — `src/data/synthetic_cert_data.py` generates CERT-schema data with a simulated pre-incident drift pattern, so the whole pipeline (features → baselines → drift scoring → unsupervised models → supervised training → dashboard) is demonstrable immediately. Drop the real [CERT Insider Threat dataset](https://kaggle.com/datasets/nitishabharathi/cert-insider-threat) into `data/raw/email.csv` to switch to real data with no code changes.

## Deployment status

The project is fully containerized (`deployment/Dockerfile`) and documented for IBM Cloud Code Engine (`deployment/ibm-cloud/DEPLOY.md`), including exact CLI commands for both a registry-based build and a build-from-source flow. Live deployment was blocked by IBM Cloud account-level billing verification requirements on Cloud Object Storage (a prerequisite for any watsonx.ai project) — not by anything in the application itself, which runs correctly locally end to end, dashboard included. The deployment path is a config/credentials step away from working once account verification is resolved.
