# Architecture

## Pipeline overview

```
data/raw/email.csv (CERT schema, real or synthetic)
        │
        ▼
  src/data/ingest.py          — load + parse, auto-generates synthetic
        │                        fallback if no real file present
        ▼
  src/data/validate.py        — schema/sanity checks, fail loudly
        │
        ▼
  src/governance/consent.py   — drop rows for non-consented users
        │
        ▼
  src/data/anonymize.py       — salted HMAC pseudonymization of user IDs
        │
        ▼
  src/features/
    linguistic_features.py    — sentiment, urgency (Aho-Corasick),
                                 readability, lexical diversity
    stylometry.py              — function words, punctuation, POS ratios
        │
        ▼
  src/features/baseline_engine.py
        — per-user, per-feature rolling baseline via SlidingWindowStats
          (src/dsa/sliding_window.py), cached per-user via an LRU cache
          (src/dsa/lru_cache_baselines.py)
        │
        ▼
  src/features/drift_scoring.py
        — weighted z-score aggregation -> drift_score + flagged_features
        │
        ├──────────────┬──────────────────────────┐
        ▼              ▼                          ▼
  aggregate_user_risk   IsolationForest       DBSCAN clustering
  (per-user rollup)     (message-level         (user-level structural
                         anomaly score)         outliers)
        │
        ▼
  src/models/supervised/train.py
        — RandomForest + XGBoost trained on aggregated per-user features
          against ground-truth labels (synthetic, or your own)
        │
        ▼
  data/processed/{scored_messages.csv, user_risk.csv}
        │
        ▼
  src/api/app.py (FastAPI)  ── src/api/routes/{users,drift,assistant,governance}.py
        │                          │
        │                          └── src/models/watsonx/client.py
        │                              (watsonx.ai: plain-language drift
        │                               explanations, second-opinion
        │                               message classification)
        ▼
  frontend/index.html  (analyst dashboard, served by the same FastAPI app)
```

Every stage that touches a specific user writes to
`src/governance/audit_log.py`'s hash-chained log.

## Why this shape

**Feature extraction is separated from baseline/drift scoring.**
`linguistic_features.py`/`stylometry.py` are pure functions of a single
message — they don't know about "baseline" or "drift" at all. That's
deliberate: it means they're independently testable, and the *definition*
of what counts as drift (which features, what weights, what threshold)
lives in exactly one place (`drift_scoring.py`) rather than being
scattered through feature code.

**Two DSA structures do the heavy lifting for scale, not correctness.**
`SlidingWindowStats` and `AhoCorasick` are both cases where a naive
implementation (recompute mean/std from a full slice; regex-search for
every phrase separately) would still produce *correct* output — the DSA
choice is purely about asymptotic cost at CERT-dataset scale (tens of
thousands of users × hundreds of messages each). `TopKRiskHeap` and
`LRUCache` are the same story at the serving layer: keeping the dashboard
responsive without holding every user's full state in memory or sorting
the entire population on every request.

**Supervised and unsupervised models are complementary, not redundant.**
The supervised classifier needs labels, which only exist here because
CERT is a synthetic-scenario research dataset. IsolationForest and DBSCAN
don't need labels at all, so they're what generalizes to a real
deployment where "is this actually an insider" ground truth doesn't
exist yet. The API and dashboard surface both, plus the raw drift score,
as three independent signals — an analyst sees where they agree and
where they don't, rather than one model's opinion presented as ground
truth.

**watsonx.ai sits at the edge, not the core.** All scoring is done locally
with classical ML/NLP (cheap, fast, auditable). watsonx.ai is called
narrowly for tasks a foundation model is actually better suited to:
turning a flagged user's statistics into a readable explanation, and
offering a second-opinion classification on an individual message. If
watsonx.ai isn't configured, the pipeline still runs end-to-end — it's an
enhancement layer, not a dependency.

**One deployable container.** `src/api/app.py` serves both the API and
the static frontend from the same FastAPI process (see the mount-ordering
comment in `app.py`), which is what let `deployment/Dockerfile` stay a
single-service build — matching IBM Code Engine's per-app, per-container
model without needing a reverse proxy or a second service for the
frontend.

## Data flow boundaries (privacy)

```
raw username ──┐
               │  (consent-gated, then HMAC-hashed)
               ▼
        pseudonym only  ──► everything from here on: features, models,
                             API responses, dashboard, audit log, watsonx.ai
                             prompts

raw message content ──► read in-memory during feature extraction ──► DISCARDED
                         (never written to data/interim/ or data/processed/,
                          never sent to watsonx.ai, never returned by the API)
```
