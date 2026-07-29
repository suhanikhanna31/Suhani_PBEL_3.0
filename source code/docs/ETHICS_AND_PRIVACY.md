# Ethics & Privacy Design

This document exists because the project itself is ethically loaded: it
analyzes *how people communicate at work* as a security signal. That's a
legitimate use case (insider threat and account-compromise detection are
real problems security teams face), but it's also exactly the kind of
system that becomes invasive surveillance if built carelessly. This
document is the argument for why the design choices below aren't
decoration — they're the actual technical answer to that risk, and it's
meant to be read alongside the code, not instead of it.

## Threat model: what this system is, and isn't, for

**In scope:** statistical drift in *aggregate linguistic patterns*
(sentiment, urgency language, stylometric markers) relative to a person's
own baseline, surfaced to a human analyst as one input among several.

**Explicitly out of scope:**
- Reading, storing, or displaying raw message content to analysts or in
  any persisted file (`data/processed/*` never contains the `content`
  column — see `src/api/routes/drift.py`'s `EXPOSED_COLUMNS` allowlist).
- Automated action against a flagged user. This system produces a ranked
  signal for human review, never a verdict, block, or termination trigger.
- Identifying who a pseudonym maps to, from within the pipeline itself.
  Re-identification requires the raw consent ledger and the
  `ANONYMIZATION_SALT`, held outside this codebase's normal execution
  path (see "Re-identification" below).

## Consent (`src/governance/consent.py`)

No user's messages enter the feature pipeline unless a consent record
marks them `consented=True` in the consent ledger
(`data/interim/consent.csv`). This is enforced in
`src/data/anonymize.py::anonymize_dataframe`, which is the *only* entry
point every downstream module (features, baselines, models, API) reads
from — there's no code path that reaches raw, non-consented data.

In this repo, the ledger auto-generates as "everyone consented" **only**
for local development against the synthetic dataset — that default is
explicitly logged as a warning every time it fires
(`ensure_consent_ledger`), and the code comments say plainly: replace this
with a real consent system of record (HR/legal-integrated) before running
against real employee data. Consent can be revoked at any time
(`revoke_consent`, exposed via `POST /api/governance/consent/revoke`,
admin-role only) — revocation takes effect on the next pipeline run.

## Anonymization (`src/data/anonymize.py`)

Usernames are replaced with a salted HMAC-SHA256 pseudonym
(`emp_<12 hex chars>`) before any feature is computed or persisted. The
salt (`ANONYMIZATION_SALT`) lives only in environment configuration
(`.env`, or an IBM Cloud Code Engine secret in deployment) — never
committed to the repo. This is a deliberate trade-off: analysts using the
dashboard cannot casually browse "what is Alice from Accounting saying"
by design. Re-identifying a pseudonym requires deliberately re-running the
same HMAC over a candidate username with the same salt — a step that
should itself require authorization outside this system (e.g. legal/HR
sign-off), not something the dashboard or API can do.

**What pseudonymization does *not* protect against:** if an organization
is small enough, or an analyst has independent knowledge (e.g. "only one
person sent 40 messages last Tuesday"), behavioral pattern + metadata can
sometimes narrow down identity even without the raw name. This is a
known, structural limitation of pseudonymization (as opposed to
differential privacy or k-anonymity guarantees), and is worth stating
plainly rather than implying stronger protection than HMAC hashing
actually provides.

## Minimization

Only derived, aggregate features are persisted to `data/processed/`
(sentiment scores, urgency scores, z-scores, drift scores) — never raw
message text. `data/interim/` holds the anonymized-but-still-per-message
feature table; it's gitignored, matching the project's `.gitignore`
convention of never committing anything under `data/`.

## Governance / audit trail (`src/governance/audit_log.py`)

Every drift flag, model prediction, and watsonx.ai call against a
pseudonymized user is written to a hash-chained, append-only audit log
(`data/processed/audit_log.jsonl`). Each entry embeds the hash of the
previous entry, so any retroactive edit or deletion is detectable via
`verify_chain()` (exposed at `GET /api/governance/audit-log/verify`).
This is the same tamper-evidence principle IBM's watsonx.governance
provides for model decisions in production — implemented here at small
scale so it's inspectable end-to-end in a student project, with the
comment in the code noting exactly where a real deployment would swap the
local file for a governed store (e.g. Cloud Object Storage with a
retention lock, or watsonx.governance's own audit trail).

## Access control (`src/governance/access_control.py`)

API routes that expose drift/risk data require a role
(`analyst` or `admin`) via a header token. This repo's implementation is
explicitly marked dev-only — a real deployment must replace it with actual
IBM Cloud IAM / App ID token validation — but the *shape* of the
permission model (who can view drift data vs. who can revoke consent or
read the audit log) is meant to reflect a real access policy, not just
demonstrate the concept.

## False positives and human review

Drift scoring produces a *statistical* signal, not a determination. A
person's communication style can shift for many benign reasons: a
stressful personal period, a new role, writing quickly on a phone,
translating from a non-native language under deadline pressure, or simply
noise in a small sample. The weighted z-score design in
`drift_scoring.py` and the watsonx.ai explanation in
`explain_drift()` are both built to show *which specific features*
drove a flag (not just an opaque number), and the explanation prompt
explicitly instructs the model to frame output as "a statistical signal
for human review, not a determination of wrongdoing" — because burying
that caveat defeats the point of including it.

## What would need to change before real deployment

This is written as an internship project against a public research
dataset, not a production system, and the gap matters:

1. Consent must come from an actual system of record, not an
   auto-generated CSV.
2. Access control must use real IAM tokens, not header strings.
3. Legal/HR/works-council review is a prerequisite in most
   jurisdictions before monitoring employee communications at all —
   this is a policy and legal question, not something a technical design
   doc can resolve on its own.
4. Bias auditing: linguistic features (readability, sentiment tools,
   POS taggers) are known to perform unevenly across non-native English
   speakers and different communication styles/cultures — this needs a
   fairness evaluation with real, diverse data before any flag is treated
   as reliable across an entire workforce, not just a note in a doc.
