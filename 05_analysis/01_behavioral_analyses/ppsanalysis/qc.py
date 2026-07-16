"""Trial usability marking and data-quality checks.

mark_pps_usable is the single gate deciding which trials enter every hypothesis.
Change it here and it changes everywhere. That is the point of pulling it out.
"""
import numpy as np
import pandas as pd

from .config import RT_MIN_MS, RT_MAX_MS, DROP_POSITIONS
from ._compat import display

def repair_pps_rt_ms_absolute_timestamps(df):
    """
    Repairs old PPS files where rt_ms was accidentally stored as an absolute
    response timestamp instead of response_time_ms - vibrotactile_onset_ms.

    Keeps a backup column: rt_ms_before_repair.
    Visual-only trials are set to NaN because H1 uses tactile RTs only.
    """
    df = df.copy()

    # Convert likely timing columns to numeric.
    timing_cols = [
        "rt_ms",
        "reaction_time_ms",
        "response_time_ms",
        "vibrotactile_onset_ms",
        "stimulus_onset_ms",
    ]

    for c in timing_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "rt_ms" not in df.columns:
        df["rt_ms"] = np.nan

    df["rt_ms_before_repair"] = df["rt_ms"]

    tactile = df["sensory_condition"].isin(["T", "VT"])

    # A true RT should be in the hundreds of ms, not hundreds of thousands.
    bad_or_missing_rt = (
        df["rt_ms"].isna()
        | (df["rt_ms"].abs() > 10000)
    )

    # Best reconstruction: response time minus vibrotactile onset.
    if {"response_time_ms", "vibrotactile_onset_ms"}.issubset(df.columns):
        rt_from_vibration = df["response_time_ms"] - df["vibrotactile_onset_ms"]

        plausible = rt_from_vibration.between(-5000, 5000)

        repair_mask = tactile & bad_or_missing_rt & plausible

        df.loc[repair_mask, "rt_ms"] = rt_from_vibration.loc[repair_mask]

        print(f"Repaired RTs using response_time_ms - vibrotactile_onset_ms: {repair_mask.sum()} rows")

    else:
        print("Could not repair from response_time_ms - vibrotactile_onset_ms.")
        print("Available timing columns:")
        print([c for c in df.columns if "time" in c.lower() or "rt" in c.lower() or "onset" in c.lower()])

    # Fallback: if reaction_time_ms exists and is already plausible, use it.
    if "reaction_time_ms" in df.columns:
        plausible_reaction = df["reaction_time_ms"].between(-5000, 5000)

        repair_mask = tactile & df["rt_ms"].isna() & plausible_reaction

        df.loc[repair_mask, "rt_ms"] = df.loc[repair_mask, "reaction_time_ms"]

        print(f"Repaired RTs using reaction_time_ms fallback: {repair_mask.sum()} rows")

    # Visual-only trials should not enter tactile RT analyses.
    df.loc[df["sensory_condition"].eq("V"), "rt_ms"] = np.nan

    return df


def mark_pps_usable(pps, rt_min_ms=100, rt_max_ms=1000, drop_positions=None):
    pps = pps.copy()

    drop_positions = [] if drop_positions is None else list(drop_positions)

    valid_rt = pps["rt_ms"].between(rt_min_ms, rt_max_ms, inclusive="neither")
    dropped_pos = pps["position_rank"].isin(drop_positions)

    pps["usable"] = valid_rt & ~dropped_pos

    pps["exclusion_reason"] = "usable"
    pps.loc[pps["sensory_condition"].eq("V"), "exclusion_reason"] = "visual_only"
    pps.loc[pps["rt_ms"].isna(), "exclusion_reason"] = "missing_rt"
    pps.loc[pps["rt_ms"] <= rt_min_ms, "exclusion_reason"] = "too_fast"
    pps.loc[pps["rt_ms"] >= rt_max_ms, "exclusion_reason"] = "too_slow"
    pps.loc[dropped_pos, "exclusion_reason"] = "dropped_position"

    return pps


def summarize_trials(name, df):
    n = len(df)
    usable = int(df["usable"].sum()) if n and "usable" in df.columns else 0

    return {
        "task": name,
        "rows": n,
        "usable": usable,
        "excluded": n - usable,
        "usable_%": round(100 * usable / n, 1) if n else 0,
    }


def pps_qc(pps):
    if pps.empty:
        print("No PPS trials.")
        return

    print("PPS trial counts by condition:")
    display(
        pps.pivot_table(
            index=["subject", "session"],
            columns="sensory_condition",
            values="trial",
            aggfunc="count",
            fill_value=0,
        )
    )

    print("PPS usable trials by distance:")
    display(
        pps[pps["sensory_condition"].isin(["T", "VT"])]
        .groupby(["subject", "position", "position_rank"], dropna=False)
        .agg(
            total=("trial", "size"),
            usable=("usable", "sum"),
        )
        .reset_index()
        .assign(
            excluded=lambda d: d["total"] - d["usable"],
            usable_percent=lambda d: round(100 * d["usable"] / d["total"], 1),
        )
        .sort_values(["subject", "position_rank"])
    )

    print("PPS exclusion reasons:")
    display(
        pps["exclusion_reason"]
        .value_counts(dropna=False)
        .rename_axis("reason")
        .reset_index(name="n")
    )


def quick_qc(pps, collision=None):
    rows = []

    if pps is not None and len(pps):
        rows.append({
            "dataset": "PPS",
            "rows": len(pps),
            "usable": int(pps["usable"].sum()),
            "excluded": int((~pps["usable"]).sum()),
            "usable_%": round(100 * pps["usable"].mean(), 1),
        })

    if collision is not None and len(collision):
        rows.append({
            "dataset": "Hit/Miss",
            "rows": len(collision),
            "usable": int(collision["usable"].sum()),
            "excluded": int((~collision["usable"]).sum()),
            "usable_%": round(100 * collision["usable"].mean(), 1),
        })

    display(pd.DataFrame(rows))
