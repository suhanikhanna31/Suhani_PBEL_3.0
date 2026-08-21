"""
Context-aware alert suppression: "is this person's problem, or everyone's?"

openscale_monitor.py already computes an org-wide monitoring snapshot and
compares it to the previous run to detect *organizational* feature drift
(a whole team's baseline shifting together — layoff rumors, a crunch
period, a shared external stressor). Until now nothing downstream actually
used that comparison to change how individual users get escalated, which
is a real, common UEBA failure mode: alert fatigue from correlated,
non-individual events. If the whole org's tone shifts, every individual
rides that shift into looking anomalous relative to their *own* history,
even though nothing about them personally changed — that's what
drift_scoring.py's self-baseline is blind to by construction, and it's
exactly the failure mode this module exists to catch.

This directly operationalizes the psychosocial-research point already
cited in this project's ethics doc: organizational stressors, not just
individual traits, drive the trajectory of insider risk. Punishing an
individual for reacting the same way as their whole team just did is the
wrong response to that; gating on relative standing among an
*also-elevated* population is the right one.

Deliberately pure logic over data the pipeline already computes — no new
model, no new dependency. This sits in governance/, not features/, because
it's a policy decision about *whether to escalate*, not a new signal about
the person.
"""
import logging
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# How many standard deviations above the current run's own population mean
# a user's avg_drift_score needs to be, relative to their peers in *this*
# run, to warrant escalation on a normal (non-correlated) run.
BASE_COHORT_Z_THRESHOLD = 1.5

# During a run where openscale_monitor.compare_to_previous_snapshot()
# flagged an org-wide feature-distribution shift, the bar is raised rather
# than left as-is: an individual has to stand out *further* above their
# also-elevated peers to be treated as an individual signal instead of
# noise from the shared event. This is the actual suppression mechanism —
# it doesn't hide anyone, it requires more relative evidence during a
# correlated event.
CORRELATED_EVENT_Z_MULTIPLIER = 1.5


def apply_context_aware_suppression(
    user_risk: pd.DataFrame,
    org_wide_shift_detected: bool,
    score_col: str = "avg_drift_score",
    base_z_threshold: float = BASE_COHORT_Z_THRESHOLD,
    correlated_multiplier: float = CORRELATED_EVENT_Z_MULTIPLIER,
) -> pd.DataFrame:
    """
    Adds cohort_z_score (how far above/below *this run's own population* a
    user's drift score sits — a relative-to-peers measure, distinct from
    the self-baseline z-scores in drift_scoring.py) and
    escalation_recommended (the actual gate) to a per-user risk table.

    org_wide_shift_detected is meant to be passed in from
    openscale_monitor.compare_to_previous_snapshot(current)["alerts"] being
    non-empty — i.e. "did this run's aggregate feature distribution shift
    materially since the last run."
    """
    out = user_risk.copy()
    if out.empty:
        out["cohort_z_score"] = pd.Series(dtype=float)
        out["escalation_recommended"] = pd.Series(dtype=bool)
        out["suppressed_correlated_event"] = pd.Series(dtype=bool)
        return out

    mean = out[score_col].mean()
    std = out[score_col].std()
    std = std if std and std > 0 else 1e-9

    out["cohort_z_score"] = ((out[score_col] - mean) / std).round(4)

    required_z = base_z_threshold * (correlated_multiplier if org_wide_shift_detected else 1.0)
    out["escalation_recommended"] = out["cohort_z_score"] >= required_z

    # Someone who was individually noteworthy in absolute terms (their own
    # self-baseline already flagged them — n_flagged_messages > 0) but who
    # didn't clear the raised cohort bar during a correlated event is the
    # specific case this module suppresses from over-eager escalation,
    # kept visible (not hidden) via this flag rather than dropped silently.
    if "n_flagged_messages" in out.columns:
        out["suppressed_correlated_event"] = (
            org_wide_shift_detected
            & (out["n_flagged_messages"] > 0)
            & ~out["escalation_recommended"]
        )
    else:
        out["suppressed_correlated_event"] = False

    if org_wide_shift_detected:
        logger.info(
            f"Org-wide shift detected this run — raised cohort escalation bar to "
            f"{required_z:.2f}σ ({int(out['escalation_recommended'].sum())} users still "
            f"escalation_recommended, {int(out['suppressed_correlated_event'].sum())} suppressed "
            f"as likely correlated-event noise)."
        )
    return out


if __name__ == "__main__":
    df = pd.DataFrame({
        "user": ["emp_a", "emp_b", "emp_c", "emp_d"],
        "avg_drift_score": [0.9, 1.0, 0.95, 0.1],
        "n_flagged_messages": [3, 4, 2, 0],
    })
    print("normal run:\n", apply_context_aware_suppression(df, org_wide_shift_detected=False))
    print("\ncorrelated org-wide event:\n", apply_context_aware_suppression(df, org_wide_shift_detected=True))
