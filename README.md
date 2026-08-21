# PROJECT 14 (AI-Based Cyber Threat Detection Framework) Signal: Behavioral & Psycholinguistic Insider-Threat Detection

**An AI/ML system that flags insider-risk and account-compromise signals from *how* people communicate internally — not from logs or network traffic.**

Built for an IBM internship using watsonx.ai, watsonx Assistant, and watsonx.governance, with a full supervised + unsupervised ML pipeline, custom-optimized data structures, and a privacy-first design throughout.

**🔗 Live demo:** **[suhani-pbel-3-0-ffz4.vercel.app](https://suhani-pbel-3-0-ffz4.vercel.app)** — deployed on Vercel's free tier. First load after idle can take ~30–60s to spin back up; see [Deployment status](#deployment-status) for the one deliberate trade-off made to fit the NLP stack into 512MB of RAM.

---
> **Quick start:** `pip install -r requirements.txt && uvicorn src.api.app:app --reload` → open http://localhost:8000 (or just open the live demo above — no setup needed)
## The idea

Most insider-threat tooling watches *what* people access — file transfers, login anomalies, badge swipes. This project watches *how* people write.

The premise, from the original project brief: subtle shifts in tone, urgency, or phrasing in internal communications (email, Slack, tickets) correlate with three distinct risk patterns —

- **Insider risk** — someone's communication style drifting before a harmful action (the CERT insider-threat research dataset this project is built on documents this pattern directly).
- **Social engineering susceptibility** — messages showing markers of manipulation: urgency language, plus a categorized lexicon (authority spoofing, isolation/secrecy framing, artificial scarcity, trust exploitation, curiosity baiting) informed by attention-warfare / media-literacy manipulation-technique taxonomies rather than one flat "urgency" bucket — see the DSA table below.
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
  Agentic investigation  →  multi-step agent gathers evidence (risk
        │                    summary, message history, prior audit
        │                    events, governance docs), asks watsonx.ai to
        │                    reason over all of it, and proposes one
        │                    action — never executes it. Read-only tools
        │                    only; human review is mandatory, not a toggle.
        ▼
  Audit log (hash-chained, tamper-evident)  +  Role-based access control
        │
        ▼
  Analyst dashboard (FastAPI + minimalist IBM-style frontend)
```

Every one of those boxes is a real, working, independently-testable module — not a diagram of an idea. The project ships with 29 passing unit tests and a synthetic-data generator that exercises the entire pipeline end to end without needing the real dataset present.

---

## The data science / ML core

### Data analysis, visualization, and feature engineering

This is where the NASSCOM EDA training and the core Python data stack get put to direct use, not just imported and forgotten:

- **pandas** is the backbone of the entire pipeline — every stage from raw ingestion through final risk aggregation is a DataFrame transformation (`src/data/ingest.py`, `src/features/*.py`, `src/features/drift_scoring.py`). Groupby aggregations roll thousands of per-message rows up into per-user risk summaries; merges join ground-truth labels against pseudonymized feature tables for supervised training.
- **numpy** underpins the statistical core: the sliding-window baseline engine computes running mean/variance incrementally (see DSA section below), and every feature extractor — sentiment scores, z-scores, drift scores — is numeric array math under the hood.
- **matplotlib** and **seaborn** drive the two exploratory notebooks (`notebooks/01_eda.ipynb`, `notebooks/02_feature_exploration.ipynb`): message-volume time series, per-user activity distributions, message-length histograms, and — the key validation plot — a boxplot comparing average drift scores between normal and ground-truth-insider users, which is the empirical check on whether the entire linguistic-drift hypothesis actually holds on this data before trusting it operationally.
- **scikit-learn** provides `StandardScaler`, `train_test_split`, `RandomForestClassifier`, `IsolationForest`, `DBSCAN`, and the full classification-report/AUC/confusion-matrix evaluation stack.
- **Hugging Face `transformers`** (DistilBERT fine-tuned on SST-2) does sentiment scoring, and **spaCy** (`en_core_web_sm`) does POS tagging for stylometry — model-driven NLP rather than a hand-built lexicon, with NLTK/TextBlob kept in as automatic fallbacks if a model can't be downloaded (e.g. offline/CI, or the memory-constrained free-tier deploy — see [Deployment status](#deployment-status)).

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
| **Categorized Aho-Corasick lexicon** | `src/dsa/social_engineering_lexicon.py` | Distinguishing *which* manipulation technique a message resembles, not just that manipulative language is present | One `AhoCorasick` matcher per category (authority spoofing, isolation/secrecy, artificial scarcity, trust exploitation, curiosity baiting), each still **O(text length + matches)** — reuses the same automaton unchanged, at the cost of a small constant factor (5 scans instead of 1) in exchange for a per-category, analyst-explainable breakdown |
| **Bounded min-heap** | `src/dsa/top_k_heap.py` | Finding the Top-K riskiest users out of a full population for the dashboard | O(N log N) full sort → **O(N log K)**, since K (10–50) is tiny compared to N (thousands) |
| **LRU cache** | `src/dsa/lru_cache_baselines.py` | Keeping every user's baseline resident in memory forever in a long-running service | Unbounded memory growth → **O(capacity)** bounded memory, keyed to actually-recent activity |

Each one is unit-tested in isolation (`tests/test_core.py`) and has a runnable self-test (`python -m src.dsa.<module>`) demonstrating the exact behavior it's built for — including edge cases like zero-variance baselines and near-zero-variance z-score blowup (which the drift-scoring layer explicitly clips, see `MAX_ABS_Z` in `src/features/drift_scoring.py`, so one near-constant feature can't dominate an otherwise-meaningful risk score).

### Extending the phrase lexicon: attention-warfare-informed social engineering detection

The original urgency lexicon (`config.URGENCY_PHRASES`) treats all manipulative language as one undifferentiated "urgency" signal. `src/dsa/social_engineering_lexicon.py` extends this by drawing on the standard attention-warfare / media-literacy taxonomy of manipulation techniques — **authority spoofing**, **isolation/secrecy framing**, **artificial scarcity**, **trust exploitation**, and **curiosity baiting** — to categorize phrases by *which* manipulation technique they represent, not just that manipulation is present.

Each category runs through its own `AhoCorasick` automaton (`src/dsa/trie_phrase_matcher.py`, reused unchanged), so per-category scanning stays O(text length + matches) regardless of lexicon size. The categorized phrase lists live in `config.SOCIAL_ENGINEERING_LEXICON`, kept deliberately non-overlapping with `URGENCY_PHRASES` — the two lexicons run as independent features rather than one replacing the other.

The aggregate `social_engineering_score` (mean of the five category scores, for the same reason `MAX_ABS_Z` clips any single feature elsewhere: one strongly-matched category shouldn't alone drown out the rest) feeds into the same per-user baseline and drift-scoring pipeline as every other linguistic feature — tracked in `TRACKED_FEATURES` (`src/features/baseline_engine.py`) and weighted at 1.8 in `FEATURE_WEIGHTS` (`src/features/drift_scoring.py`), just under urgency's 2.0, since it's a targeted-manipulation signal rather than a general tone shift. The per-category breakdown (`SocialEngineeringLexicon.category_scores()` / `.scan()`) is kept available separately, for a future analyst-facing explanation of *why* a message resembles a specific manipulation pattern, rather than only a single opaque number.

This is rule-based phrase matching, not machine learning — no training, no labels, no model, same technique class as the existing urgency detector. It strengthens the ML side only indirectly: `social_engineering_score` becomes one more input feature to the supervised classifiers (RandomForest/XGBoost) and unsupervised models (IsolationForest/DBSCAN) already described above, without being learned itself — consistent with this project's existing preference for auditable, explainable scoring over black-box models wherever the two are substitutable (see [`docs/ETHICS_AND_PRIVACY.md`](docs/ETHICS_AND_PRIVACY.md)).

8 new unit tests cover this module (`tests/test_social_engineering_lexicon.py`), kept in its own file rather than appended to `tests/test_core.py` so each is independently reviewable.

---

## The watsonx stack

- **watsonx.ai** (`src/models/watsonx/client.py`) — used narrowly, on top of the local ML, for the one task a foundation model is genuinely better suited to than hand-built features: turning a flagged user's raw statistics into a short, readable explanation (`explain_drift()`), and offering a second-opinion classification on individual messages (`classify_message_risk()`). All local scoring runs independently of watsonx — if it's unconfigured, the pipeline still runs end to end; watsonx is an enhancement layer, not a dependency.
- **watsonx Assistant** (`src/api/routes/assistant.py`) — a webhook endpoint (`POST /api/assistant/webhook`) built specifically so an analyst can ask natural-language questions like *"show me users whose communication tone changed significantly this week"* and have the Assistant call back into the same ranked risk data the dashboard uses.
- **watsonx.governance** — implemented as a hash-chained, tamper-evident audit log (`src/governance/audit_log.py`) that records every drift flag, model prediction, and watsonx.ai call against a pseudonymized user, with `verify_chain()` detecting any retroactive edit or deletion. This mirrors the tamper-evidence principle watsonx.governance provides at enterprise scale, implemented here at a scale that's fully inspectable in a single codebase.

---

## Agentic AI — the investigation agent

Everything above produces a single number or a single sentence per user — a drift score, an anomaly flag, a one-shot watsonx.ai explanation. `src/agents/investigation_agent.py` adds an actual **agent**: given a flagged pseudonym, it plans and executes a short sequence of read-only tool calls against the project's own modules, accumulates what it finds, and — when watsonx.ai is configured — asks the model to reason over *all* of that accumulated evidence (recent message trend, prior audit history, related governance-doc passages) before proposing a next step, rather than judging a single row of stats in isolation.

```
investigate_user(pseudonym)
  → get_risk_summary            (user_risk.csv)
  → get_recent_messages         (scored_messages.csv)
  → get_prior_audit_events      (audit_log.jsonl)
  → get_watsonx_explanation     (existing explain_drift())
  → search_governance_docs      (rag.retrieve(), only if drift is already above threshold)
  → synthesize_recommendation   (watsonx.ai reasoning over all of the above,
                                  or a transparent rule-based fallback if
                                  watsonx isn't configured)
  → escalate_for_manual_review | continue_monitoring | no_action_needed
```

Exposed as `POST /api/agent/investigate/{pseudonym}` (analyst/admin only), as an `investigate_user` intent on the existing watsonx Assistant webhook, and as a "Run investigation" button on the dashboard's user-detail panel, which shows the recommendation, rationale, and the trace of tools the agent consulted.

This is deliberately additive, not a replacement for anything upstream, and it inherits the same "never acts autonomously" boundary the rest of the project holds to:

- Every tool available to the agent is **read-only** — it has no import path to consent revocation, the QRadar export, or any other write/enforcement code.
- Its output is always a **recommendation**, never an action. `requires_human_review: true` is hard-coded in the report structure, not something a prompt (or a hallucinating model) can turn off.
- Every investigation, including each intermediate tool call, is written to the same tamper-evident audit log as every other scoring decision, so a run is reviewable after the fact exactly like a model prediction is.
- If watsonx.ai isn't configured, the agent still runs end to end via the rule-based fallback — same graceful-degradation pattern as the rest of the watsonx integration.

See `tests/test_agent.py` for the fallback path under test (unknown user, low/high drift, and a guardrail test asserting `requires_human_review` can never be flipped).

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
│   ├── dsa/                       Sliding window, Aho-Corasick, categorized social-engineering
│   │                              lexicon, top-K heap, LRU cache
│   ├── models/
│   │   ├── supervised/            RandomForest + XGBoost training, metrics report, SHAP explainability
│   │   ├── unsupervised/          IsolationForest + DBSCAN
│   │   └── watsonx/                watsonx.ai client
│   ├── governance/                Consent, anonymization, audit log, access control,
│   │                              bias/fairness audit, OpenScale-style monitoring, RAG Q&A
│   ├── agents/                    Agentic investigation loop (src/agents/investigation_agent.py)
│   ├── integrations/              QRadar (SIEM) CEF export stub
│   ├── api/                       FastAPI app + routes (users, drift, assistant, governance, agent)
│   └── pipeline.py                End-to-end orchestrator
├── frontend/                      Analyst dashboard (FastAPI-served static HTML/JS)
├── notebooks/                     EDA + feature exploration (pandas/matplotlib/seaborn)
├── deployment/                    Dockerfile + IBM Cloud Code Engine deploy guide
├── docs/                          Architecture + Ethics/Privacy design docs
└── tests/                         29 unit tests across DSA, drift scoring, anonymization,
                                  agent fallback, social-engineering lexicon

(CI workflow lives at the repo root: .github/workflows/ci.yml)
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in watsonx.ai credentials if available

python -m src.pipeline                                    # run the full pipeline
python -m src.models.supervised.train                     # train supervised models
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000  # dashboard at localhost:8000
pytest tests/                                              # 29 tests
```

No real dataset required to see it run — `src/data/synthetic_cert_data.py` generates CERT-schema data with a simulated pre-incident drift pattern, so the whole pipeline (features → baselines → drift scoring → unsupervised models → supervised training → dashboard) is demonstrable immediately. Drop the real [CERT Insider Threat dataset](https://kaggle.com/datasets/nitishabharathi/cert-insider-threat) into `data/raw/email.csv` to switch to real data with no code changes.

## Model evaluation (real numbers, regenerated every run)

`python -m src.models.supervised.train` now writes actual metrics — not
just a qualitative description — to
`data/processed/reports/model_metrics.{json,md}` on every run: AUC,
precision/recall/F1 for the positive class, and the confusion matrix for
both RandomForest and XGBoost. A feature-attribution report (via SHAP) is
also written to `shap_importance_{model}.{json,png}`, showing which
aggregated per-user features (`avg_drift_score`, `flagged_message_rate`,
etc.) actually drove each classifier's prediction — a second, independent
explanation layer alongside the drift score's own per-feature breakdown.

**Read the numbers honestly, not optimistically.** On the default
synthetic run (60 users, ~5% simulated insiders → 3 positive users total,
15 in the test split), RandomForest scores a clean AUC of 1.0 and XGBoost
scores 0.5 — both numbers are close to meaningless with a single positive
example in the test fold; they say more about the tiny synthetic
population than about either algorithm. This is exactly the kind of
result a larger, real dataset is needed to make trustworthy, and the
report is deliberately not hand-edited to hide that.

**Real-data evaluation (unsupervised only — no labels available):** the
deployed live demo (see [Deployment status](#deployment-status)) runs on
a real, ~50,000-row slice of the actual CERT r4.2 `email.csv` rather
than the synthetic generator described above. That slice has no
accompanying `insider_labels.csv` — it's a real, unlabeled sample, not
CERT's synthetic scenario data — so `train.py` now detects this and
exits cleanly with an explanatory message rather than crashing (see
"Fixes found by running on real data" below), and no AUC/precision/
recall/F1 is reported for it. What *can* be reported, and was actually
computed by running the full pipeline against the committed real
sample:

- 10,000 real messages sampled (streaming reservoir sample, see below)
  across 942 real users, spanning Jan 2–12, 2010.
- At the default `BASELINE_WINDOW_SIZE` (30 messages), this 10-day slice
  is too short for *any* user to fill a baseline window, so every drift
  score reports as `0.0` — a real, correctly-computed null result, not
  a bug. Lowering `BASELINE_WINDOW_SIZE` to 5 (still a real, if thin,
  baseline) is what produces the numbers below.
- **Drift scoring:** 654/10,000 messages (6.5%) flagged with at least
  one drifted feature; mean drift score 0.121 (σ = 0.363) among flagged
  messages, max 3.83.
- **IsolationForest:** 500/10,000 messages (5.0%, matching the
  configured `contamination=0.05`) flagged as statistical outliers.
- **Agreement between signals:** 49 messages were flagged by both drift
  scoring and IsolationForest — ~7.5% of drift-flagged messages, ~9.8%
  of IsolationForest-flagged messages. Materially overlapping but not
  identical, the same pattern claimed for the synthetic run above, now
  shown on real data.
- **DBSCAN:** found 2 non-noise clusters (927 and 3 users) among the 942
  real users, with 12 users (1.3%) landing in no dense cluster and
  flagged as structural outliers.

Supply your own label file at `data/raw/insider_labels.csv` (see the
docstring in `src/models/supervised/train.py`) to get supervised
metrics on real data too.

## Fixes found by running on real data

Two edge cases the synthetic generator never exercised, both caught by
actually running the pipeline against the real sample above rather than
assumed away:

- **`score_drift_df()` crashed when zero users had a full baseline
  window.** With real data this sparse (10 days, ~10 messages/user),
  nobody reaches the default 30-message window, so `z_df` ends up with
  no `z_*` columns at all. `pandas.DataFrame.apply(axis=1)` over a
  zero-column frame doesn't reliably reproduce a per-row Series of
  dicts, which surfaced as a `KeyError` on `drift_score`/`n_flagged`
  deep in `aggregate_user_risk`'s named aggregation. Fixed by handling
  the zero-`z_cols` case explicitly: every message defaults to
  `drift_score=0.0` (not yet measurable), logged plainly rather than
  silently.
- **`train.py`'s smoke test crashed on missing labels.** The
  `__main__` block assumed `load_insider_labels()` always returns a
  DataFrame — true for the synthetic generator, false for a real,
  unlabeled sample — and crashed with a bare `TypeError` on the next
  line. Fixed to detect `labels is None`, print a clear explanation,
  and exit 0, matching how the rest of the codebase already degrades
  gracefully (watsonx, transformers, spaCy) instead of crashing.

## Bias / fairness audit and ongoing monitoring

Two governance additions close gaps the ethics doc named but never
actually checked:

- **`src/governance/bias_audit.py`** — since no real demographic data ever
  enters this privacy-minimized pipeline, this checks flagged-rate and
  drift-score disparity across writing-style *proxies* (avg word length,
  lexical diversity, readability) as an always-runnable first pass. It
  is explicitly documented as a proxy check, not a substitute for a real
  fairness review with actual protected-attribute data — see the module
  docstring and `docs/ETHICS_AND_PRIVACY.md`.
- **`src/governance/openscale_monitor.py`** — a lightweight,
  watsonx.OpenScale-style monitoring snapshot: org-wide feature-distribution
  drift and aggregate risk stats over time (distinct from the per-user
  drift the core pipeline already computes), appended to a JSONL history
  file after every pipeline run so successive runs are comparable.

Both run automatically as part of `python -m src.pipeline` and write to
`data/processed/reports/`.

## SIEM integration (IBM QRadar) — export stub

`src/integrations/qradar_export.py` converts flagged users into CEF
(Common Event Format) event strings — the format QRadar and most SIEMs
ingest over syslog — so this signal could sit alongside the log/network
alerts a SOC analyst already triages, instead of living only in a
second, easy-to-ignore dashboard. This is an honestly-labeled
**integration stub**: it produces valid CEF lines but does not open a
network connection to a live QRadar instance, since no real QRadar
credentials/log source are available in this environment. Wiring
`send_to_qradar` up to a real syslog listener is a small, well-scoped
next step once one exists.

## Continuous integration

`.github/workflows/ci.yml` runs on every push/PR: installs dependencies,
runs the 29-test suite, smoke-tests the full synthetic pipeline and
supervised training end-to-end, and uploads the generated evaluation/
governance reports as build artifacts — so "the tests pass" is verified
automatically rather than only asserted in this README.

## Retrieval-augmented question answering

The watsonx Assistant webhook above answers *structured* questions about risk data —
"show me the top risky users" — by calling back into ranked scores the pipeline
already computed. It has no way to answer a different, equally realistic class of
question: how the system itself behaves. An analyst asking "what happens if a user
revokes consent?" or "how does the audit log detect tampering?" needs an answer
grounded in this project's actual privacy design and audit trail, not a
plausible-sounding guess from a model's general training. `src/governance/rag.py`
adds exactly that, as a new `POST /api/assistant/ask` endpoint alongside the
existing webhook, without touching drift scoring, either classifier, or
`pipeline.py` — the same independently-callable-module pattern already used for
the bias audit and OpenScale-style monitor above.

Retrieval is deliberately **TF-IDF and cosine similarity** — both already available
through scikit-learn, an existing dependency — rather than an embedding model or
vector database. This follows directly from two things already true of this
project: the toolchain philosophy of favoring well-audited, explainable tools over
the newest available library, and the free-tier memory ceiling described in
[Deployment status](#deployment-status), where `transformers` and `torch` were
already stripped from the live deploy for exceeding 512MB before scoring a single
message — a second heavy model for retrieval would reopen exactly that problem.
TF-IDF also has a real, related advantage beyond fitting the memory budget: which
words in the question matched which words in the retrieved passage is directly
inspectable, the same transparency argument made elsewhere for keeping urgency and
readability scoring rule-based even after adopting a transformer for sentiment.

The retrieval corpus is built from two real sources: `docs/ETHICS_AND_PRIVACY.md`
and `docs/ARCHITECTURE.md`, chunked along their existing markdown section headers,
plus the most recent entries from the real hash-chained audit log, reformatted
into retrievable sentences via a new `audit_log.get_recent_entries()` function.
Folding the audit log into the corpus, not just the static docs, means a question
like "was anything unusual logged recently?" can be answered from what the system
actually did, not only from what its documentation says it should do. When
watsonx.ai is configured, a new `answer_with_context()` function
(`src/models/watsonx/client.py`, following the identical pattern as
`explain_drift()` and `classify_message_risk()`) asks it to phrase an answer using
only the retrieved passages, and to say so plainly if they don't actually answer
the question rather than filling the gap with outside knowledge. When watsonx.ai
isn't configured — the live deployment's current state — the retrieved passages
are returned directly, clearly labeled as unprocessed retrieval rather than a
generated summary, the same "never silently fabricate, state the stub plainly"
rule the dashboard already follows for `explain_drift()`.

Every question asked through this endpoint is itself written to the audit log as a
`rag_query` event — including how many sources were retrieved and whether
watsonx.ai generated the final answer — so this new capability is governed by the
same tamper-evident trail as every other automated decision in the system, not an
exception to it.

## Handling datasets far larger than a laptop can hold

Earlier development runs against the real CERT r4.2 `email.csv`
(~2.6M rows, several GB once parsed, since it includes full message
text) would freeze on a single `pd.read_csv()` call, because pandas
parses and allocates the *entire* file before anything downstream can
run. `src/data/ingest.py` now streams the file in chunks
(`pd.read_csv(..., chunksize=...)`) and keeps a fixed-size, uniformly
random sample via reservoir sampling (Algorithm R), so peak memory is
bounded by one chunk plus the sample — never the whole file. The default
cap is `MAX_INGEST_ROWS = 10000` (tunable via `.env` or
`load_email_data(max_rows=...)`), which is comfortably enough data for
per-user baselines and drift scoring without needing gigabytes of RAM.
On a synthetic 500k-row / 155MB test file, this sampled 10,000 rows in
~1.5s at ~120MB peak RSS instead of loading the whole file first.

This is no longer a purely defensive design for a hypothetical future
file: a real ~25MB, ~50,000-row slice of the actual CERT r4.2
`email.csv` is committed at `data/raw/email.csv` and is exactly what
this streaming/reservoir-sampling path processes on every deploy — see
"Model evaluation" above for what actually comes out of that real run.

## Deployment status

**Live now:** [suhani-pbel-3-0-ffz4.vercel.app](https://suhani-pbel-3-0-ffz4.vercel.app), deployed straight from `deployment/Dockerfile` on Vercel's free web-service tier (512MB RAM, spins down on idle — expect a slow first request after inactivity).

The project is also fully containerized and documented for IBM Cloud Code Engine (`deployment/ibm-cloud/DEPLOY.md`), including exact CLI commands for both a registry-based build and a build-from-source flow. Live deployment there was blocked by IBM Cloud account-level billing verification requirements on Cloud Object Storage (a prerequisite for any watsonx.ai project) — not by anything in the application itself. That path remains a config/credentials step away from working once account verification is resolved.

**One deliberate trade-off on the free-tier deployment:** `torch` + `transformers` alone reliably exceed 512MB before a single message is even scored, which OOM-killed the first deploy attempt. Rather than pay for a bigger instance, the live demo runs with `transformers`, `torch`, and `spacy` left out of `requirements.txt` entirely — code that was already written to support this (see `_get_sentiment_pipeline()` in `src/features/linguistic_features.py` and `_get_spacy_pipeline()` in `src/features/stylometry.py`, both wrapped in `try/except` with an automatic fallback), not a change made to accommodate hosting. So the hosted demo scores sentiment via TextBlob and POS via NLTK rather than DistilBERT/spaCy — slightly less robust on sarcasm and negation, as the code's own docstrings already document as the known trade-off — while every other module (drift scoring, the four DSA structures, both classifiers, unsupervised models, governance, the dashboard) runs identically to the full local setup, since all of them only ever see plain floats regardless of which sentiment backend produced them. Run locally with the full `requirements.txt` (or on a ≥2GB instance) to get the transformer/spaCy path instead.
