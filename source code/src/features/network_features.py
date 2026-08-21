"""
PART 1 (extension) — Communication-network drift features.

Turns raw To/Cc/Bcc recipient columns into two per-message numeric
features (contact_degree, interaction_concentration) via the
dict-of-dicts CommunicationGraph (src/dsa/communication_graph.py), the
same sliding-window replay pattern baseline_engine.py already uses for
linguistic features — this module just replays the *graph*, not the
text.

Deliberately kept as its own module rather than folded into
linguistic_features.py / stylometry.py: those two only ever look at
message *content*; this one only ever looks at message *headers*
(sender/recipients), never the text. Keeping the boundary clean means
the privacy story stays simple — this module never touches the
`content` column at all, so it can't leak raw message text no matter
what happens upstream.

Output feeds straight into the existing baseline/drift machinery: once
contact_degree and interaction_concentration are columns on the feature
DataFrame, TRACKED_FEATURES (baseline_engine.py) and FEATURE_WEIGHTS
(drift_scoring.py) treat them exactly like any linguistic feature — same
per-user sliding-window baseline, same z-score, same weighted drift
score, same MAX_ABS_Z clipping. No new scoring logic was needed, only a
new feature source.
"""
import logging
import re
from typing import List

import pandas as pd

from src.config import BASELINE_WINDOW_SIZE
from src.dsa.communication_graph import CommunicationGraph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_SPLIT_RE = re.compile(r"[;,]")

RECIPIENT_COLUMNS = ["to", "cc", "bcc"]


def _parse_recipients(row: pd.Series) -> List[str]:
    """CERT-schema recipient columns are semicolon-separated address lists; cc/bcc are frequently blank/NaN."""
    recipients: List[str] = []
    for col in RECIPIENT_COLUMNS:
        raw = row.get(col)
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        recipients.extend(addr.strip() for addr in _SPLIT_RE.split(str(raw)) if addr.strip())
    return recipients


def extract_network_features_df(
    df: pd.DataFrame,
    user_col: str = "user",
    date_col: str = "date",
    window_size: int = BASELINE_WINDOW_SIZE,
) -> pd.DataFrame:
    """
    Replays a chronologically-sorted message DataFrame through a single
    CommunicationGraph, appending contact_degree / interaction_concentration
    columns computed from each sender's *prior* sliding window (i.e. this
    message's own recipients aren't counted in its own feature values,
    mirroring how BaselineEngine.process_message scores against prior
    history before updating — see baseline_engine.py's docstring).
    """
    if date_col in df.columns:
        df = df.sort_values(date_col).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    logger.info(f"Building communication-network features for {len(df)} messages (window_size={window_size})...")

    graph = CommunicationGraph(window_size=window_size)
    degrees = []
    concentrations = []

    for _, row in df.iterrows():
        sender = row[user_col]
        recipients = _parse_recipients(row)

        feats = graph.network_features(sender)
        degrees.append(feats["contact_degree"])
        concentrations.append(feats["interaction_concentration"])

        graph.record_message(sender, recipients)

    out = df.copy()
    out["contact_degree"] = degrees
    out["interaction_concentration"] = concentrations
    return out


if __name__ == "__main__":
    sample = pd.DataFrame([
        {"user": "emp_1", "date": "01/01/2010 09:00:00", "to": "a@co.com;b@co.com", "cc": "", "bcc": ""},
        {"user": "emp_1", "date": "01/01/2010 09:05:00", "to": "c@co.com", "cc": "", "bcc": ""},
        {"user": "emp_1", "date": "01/01/2010 09:10:00", "to": "single@external.com", "cc": "", "bcc": ""},
    ])
    print(extract_network_features_df(sample, window_size=5)[["user", "contact_degree", "interaction_concentration"]])
