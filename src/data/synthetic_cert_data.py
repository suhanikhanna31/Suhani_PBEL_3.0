"""
Synthetic CERT-schema data generator.

The real CERT Insider Threat dataset (r4.2, the version both Kaggle mirrors
you linked repackage) ships an email.csv with columns:
    id, date, user, pc, to, cc, bcc, from, activity, size, attachments, content

We can't fetch the real file from this sandbox (no network route to
kaggle.com). This generator produces data in the *exact same schema* with
plausible content, so the rest of the pipeline (anonymize -> features ->
baseline -> drift -> models -> API -> frontend) runs and is testable today.

To use the real dataset instead: download email.csv (and psychometric.csv /
LDAP files if you want those features too) from either Kaggle link into
data/raw/, and skip this generator — ingest.py auto-detects which is
present.
"""
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

from src.config import DATA_RAW

random.seed(42)
np.random.seed(42)

NORMAL_TEMPLATES = [
    "Hi {name}, attaching the {doc} for your review. Let me know if you have questions.",
    "Thanks for the update on {doc}. I'll follow up next week.",
    "Can we schedule a call to discuss the {doc}?",
    "Here are my notes from today's meeting about {doc}.",
    "Please find the {doc} attached as requested.",
    "Following up on our conversation regarding {doc}.",
    "Quick question about the {doc} — do you have a minute today?",
    "Great work on the {doc}, really appreciate the effort.",
]

RISKY_TEMPLATES = [
    "This is urgent, I need you to wire transfer the funds ASAP, don't tell anyone yet.",
    "Please reset your credentials immediately and send me the new password to verify.",
    "Confidential request — bypass the approval process just this once, before end of day.",
    "Click here now to verify your account or it will be suspended: {doc}",
    "Keep this between us, I need the {doc} exported before I leave the company.",
    "Final notice: send the gift card codes right away, it's for an urgent client issue.",
]

DOCS = ["Q3 report", "budget spreadsheet", "client contract", "server credentials",
        "HR file", "source code repo", "database backup", "vendor invoice"]

NAMES = [f"user{i:03d}" for i in range(1, 61)]  # 60 synthetic employees


def _random_content(risky: bool) -> str:
    template = random.choice(RISKY_TEMPLATES if risky else NORMAL_TEMPLATES)
    return template.format(name=random.choice(NAMES), doc=random.choice(DOCS))


def generate_synthetic_email_csv(n_rows: int = 6000, insider_fraction: float = 0.05,
                                  out_path: Path = None) -> Path:
    """
    Generates a synthetic email.csv matching CERT r4.2 schema. A small
    subset of users (insider_fraction) get a rising rate of risky-template
    messages in the final third of their timeline, simulating the
    "communication drift before an incident" pattern this whole project
    is designed to detect.
    """
    out_path = out_path or (DATA_RAW / "email.csv")
    n_insiders = max(1, int(len(NAMES) * insider_fraction))
    insiders = set(random.sample(NAMES, n_insiders))

    start_date = datetime(2024, 1, 1)
    rows = []
    row_id = 0

    for user in NAMES:
        n_msgs = random.randint(60, 140)
        for i in range(n_msgs):
            ts = start_date + timedelta(days=int(i * random.uniform(0.8, 1.4)),
                                         hours=random.randint(8, 18),
                                         minutes=random.randint(0, 59))
            # insiders drift toward risky content in the last third of their timeline
            progress = i / max(n_msgs - 1, 1)
            is_risky = user in insiders and progress > 0.66 and random.random() < 0.35
            content = _random_content(is_risky)

            rows.append({
                "id": f"{{{row_id:08d}}}",
                "date": ts.strftime("%m/%d/%Y %H:%M:%S"),
                "user": user,
                "pc": f"PC-{random.randint(1000, 1099)}",
                "to": random.choice(NAMES),
                "cc": "",
                "bcc": "",
                "from": f"{user}@company.com",
                "activity": "Send",
                "size": len(content) * random.randint(2, 5),
                "attachments": random.choice([0, 0, 0, 1]),
                "content": content,
            })
            row_id += 1

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df.to_csv(out_path, index=False)

    # Also drop a ground-truth label file for supervised training/eval
    # (mirrors CERT's separate answer-key files, kept out of the feature set itself)
    labels = pd.DataFrame({
        "user": NAMES,
        "is_malicious_insider": [1 if u in insiders else 0 for u in NAMES],
    })
    labels.to_csv(DATA_RAW / "insider_labels.csv", index=False)

    return out_path


if __name__ == "__main__":
    path = generate_synthetic_email_csv()
    print(f"Wrote synthetic dataset to {path}")
