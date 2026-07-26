"""
PART 3 (governance addendum) — Bias / fairness audit.

docs/ETHICS_AND_PRIVACY.md explicitly names a real risk: this system could
flag non-native-English speakers or people with atypical (but entirely
benign) communication styles at a higher rate than everyone else, simply
because their *normal* writing looks more "different from the crowd" even
though it isn't different from *their own* baseline in a risk-relevant
way. That claim was documented but never actually checked — this script
checks it.

Important honesty note: because the pipeline is deliberately
privacy-minimizing (no demographic/HR data ever enters it — see
docs/ETHICS_AND_PRIVACY.md), there is no real protected-attribute column
to audit against. This script instead uses two *observable writing-style
proxies* that are the closest analogue available without adding new PII
collection:

  - avg_word_length / lexical_diversity   → correlates with formality/
    vocabulary style, one of several proxies for non-native or ESL
    writing patterns in the sociolinguistics literature
  - readability_flesch                    → correlates with sentence
    complexity / writing register

It buckets users into quartiles on these proxies and compares each
group's *rate of being flagged* (n_flagged_messages / n_messages) and
mean drift score. A real deployment would still need a proper fairness
review with actual protected-attribute data and legal/HR sign-off (this
is stated as a hard requirement in ETHICS_AND_PRIVACY.md) — this script
is a first, always-runnable check, not a substitute for that review.
"""
import json
import logging

import pandas as pd

from src.config import DATA_PROCESSED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPORTS_DIR = DATA_PROCESSED / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PROXY_COLUMNS = ["avg_word_length", "lexical_diversity", "readability_flesch"]


def audit_drift_by_style_proxy(scored_messages: pd.DataFrame) -> dict:
    """
    scored_messages: the per-message DataFrame produced by
    src/features/drift_scoring.py (scored_messages.csv), which includes
    the raw linguistic features plus drift_score / n_flagged.

    Returns a dict report and also writes it to
    data/processed/reports/bias_audit.{json,md}.
    """
    missing = [c for c in PROXY_COLUMNS if c not in scored_messages.columns]
    if missing:
        raise ValueError(f"scored_messages is missing expected columns: {missing}")

    df = scored_messages.copy()
    group_reports = {}

    for proxy in PROXY_COLUMNS:
        valid = df[proxy].notna() & (df[proxy] != 0)
        if valid.sum() < 20:
            continue
        try:
            binned = pd.qcut(df.loc[valid, proxy], 4, duplicates="drop")
        except ValueError:
            continue
        # qcut can collapse to fewer than 4 bins if the proxy has many
        # repeated values (duplicates="drop") — label whatever bins result
        # rather than assuming exactly 4.
        n_bins = binned.cat.categories.size
        bin_labels = [f"Q{i + 1}" for i in range(n_bins)]
        if n_bins < 2:
            continue
        df[f"{proxy}_quartile"] = binned.cat.rename_categories(bin_labels)

        grouped = df.groupby(f"{proxy}_quartile", observed=True).agg(
            n_messages=("drift_score", "count"),
            mean_drift_score=("drift_score", "mean"),
            flagged_rate=("n_flagged", lambda s: float((s > 0).mean())),
        )
        group_reports[proxy] = grouped.reset_index().rename(columns={f"{proxy}_quartile": "quartile"}).to_dict("records")

    # Simple disparity flag: is the highest-group flagged_rate more than
    # 1.5x the lowest-group flagged_rate, for any proxy? A crude threshold,
    # deliberately conservative (over-flagging a real disparity is safer
    # than missing one), meant to prompt a closer manual look — not to be
    # the final word.
    disparities = {}
    for proxy, rows in group_reports.items():
        rates = [r["flagged_rate"] for r in rows if r["flagged_rate"] is not None]
        if not rates or min(rates) == 0:
            continue
        ratio = max(rates) / max(min(rates), 1e-9)
        disparities[proxy] = round(ratio, 3)

    flagged_proxies = [p for p, r in disparities.items() if r >= 1.5]

    report = {
        "groups": group_reports,
        "max_to_min_flagged_rate_ratio": disparities,
        "proxies_with_disparity_over_1_5x": flagged_proxies,
        "note": (
            "These are writing-style proxies, not real protected attributes. "
            "A ratio flag here means 'go look closer', not 'this is confirmed bias'."
        ),
    }

    (REPORTS_DIR / "bias_audit.json").write_text(json.dumps(report, indent=2, default=str))

    lines = ["# Bias / fairness audit (writing-style proxies)", ""]
    if flagged_proxies:
        lines.append(f"**Flagged for closer review:** {', '.join(flagged_proxies)} "
                      f"(flagged-rate ratio >= 1.5x between highest and lowest quartile).")
    else:
        lines.append("No proxy showed a >=1.5x disparity in flagged rate across quartiles on this run.")
    lines.append("")
    for proxy, rows in group_reports.items():
        lines.append(f"## {proxy}")
        lines.append("| Quartile | n messages | mean drift score | flagged rate |")
        lines.append("|---|---|---|---|")
        for r in rows:
            lines.append(f"| {r['quartile']} | {r['n_messages']} | {r['mean_drift_score']:.3f} | {r['flagged_rate']:.3f} |")
        lines.append("")
    (REPORTS_DIR / "bias_audit.md").write_text("\n".join(lines))

    logger.info(f"Wrote bias audit to {REPORTS_DIR}/bias_audit.{{json,md}}")
    return report


if __name__ == "__main__":
    path = DATA_PROCESSED / "scored_messages.csv"
    if not path.exists():
        raise SystemExit("Run `python -m src.pipeline` first to produce scored_messages.csv")
    scored = pd.read_csv(path)
    result = audit_drift_by_style_proxy(scored)
    print(json.dumps(result["max_to_min_flagged_rate_ratio"], indent=2))
