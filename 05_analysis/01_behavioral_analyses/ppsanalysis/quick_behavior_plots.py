"""
Quick behavioral QC plots for PPS and Hit/Miss tasks.

Use this script from the notebook after building `t`:

    from ppsanalysis import quick_behavior_plots as qbp

    qbp.run(
        t,
        rt_min=config.RT_MIN_MS,
        rt_max=config.RT_MAX_MS,
        save_dir="figures/quick_qc",
        show=True,
    )

The goal is descriptive QC before exclusions:
- PPS trial counts by V / VT / T, per subject and speed
- PPS response rate by modality, per subject and speed
- PPS RT boxplots before exclusion, per modality and speed
- Hit/Miss accuracy by trial type, per subject and speed
- Hit/Miss RT boxplots before exclusion, per trial type and speed
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
# Small helpers
# -------------------------------------------------------------------------

def _ensure_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Return a copy with missing columns added as NaN."""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    return df


def _sorted_nonmissing(series: pd.Series) -> list:
    """Stable sorted list of non-missing values."""
    return sorted(series.dropna().unique().tolist())


def _savefig(save_dir: str, name: str, show: bool = True) -> None:
    os.makedirs(save_dir, exist_ok=True)

    png_path = os.path.join(save_dir, f"{name}.png")
    pdf_path = os.path.join(save_dir, f"{name}.pdf")

    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.savefig(pdf_path)

    print(f"saved: {png_path}")

    if show:
        plt.show()
    else:
        plt.close()


def _boxplot_by_subject(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
    save_dir: str,
    filename: str,
    rt_min: Optional[float] = None,
    rt_max: Optional[float] = None,
    show: bool = True,
) -> None:
    """One boxplot with subjects on x-axis."""
    if df.empty:
        print(f"Skipping empty plot: {title}")
        return

    subjects = _sorted_nonmissing(df["subject"])
    if not subjects:
        print(f"Skipping plot with no subjects: {title}")
        return

    data = [
        pd.to_numeric(
            df.loc[df["subject"].eq(subject), value_col],
            errors="coerce",
        ).dropna().to_numpy()
        for subject in subjects
    ]

    # Skip if every subject has no data.
    if not any(len(x) for x in data):
        print(f"Skipping plot with no values: {title}")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(data, labels=subjects, showfliers=True)

    if rt_min is not None:
        ax.axhline(rt_min, linestyle="--", linewidth=1)
    if rt_max is not None:
        ax.axhline(rt_max, linestyle="--", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Subject")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)

    _savefig(save_dir, filename, show=show)


# -------------------------------------------------------------------------
# PPS plots
# -------------------------------------------------------------------------

def plot_pps_trial_counts(
    pps: pd.DataFrame,
    save_dir: str = "figures/quick_qc",
    show: bool = True,
) -> pd.DataFrame:
    """
    PPS trial counts by subject, sensory condition and speed.

    Uses all trials, before exclusions.
    """
    df = _ensure_cols(pps, ["subject", "sensory_condition", "speed"])

    counts = (
        df.groupby(["subject", "speed", "sensory_condition"])
        .size()
        .reset_index(name="n")
        .sort_values(["subject", "speed", "sensory_condition"])
    )

    subjects = _sorted_nonmissing(df["subject"])
    speeds = _sorted_nonmissing(df["speed"])
    conditions = ["T", "VT", "V"]

    for speed in speeds:
        sub = counts[counts["speed"].eq(speed)]

        table = (
            sub.pivot_table(
                index="subject",
                columns="sensory_condition",
                values="n",
                fill_value=0,
            )
            .reindex(index=subjects, columns=conditions, fill_value=0)
        )

        ax = table.plot(kind="bar", figsize=(8, 4))
        ax.set_title(f"PPS trial counts by modality — {speed}")
        ax.set_xlabel("Subject")
        ax.set_ylabel("Number of trials")
        ax.legend(title="Condition")

        _savefig(
            save_dir,
            f"pps_trial_counts_by_condition_{speed}",
            show=show,
        )

    return counts


def plot_pps_response_rate(
    pps: pd.DataFrame,
    save_dir: str = "figures/quick_qc",
    show: bool = True,
) -> pd.DataFrame:
    """
    PPS response rate by subject, condition and speed.

    PPS does not have accuracy in the same way as Hit/Miss. For PPS we plot
    response rate instead.
    """
    df = _ensure_cols(
        pps,
        ["subject", "speed", "sensory_condition", "response_made"],
    )

    df["response_made_num"] = df["response_made"].astype(float)

    summary = (
        df.groupby(["subject", "speed", "sensory_condition"])
        .agg(
            n=("response_made_num", "size"),
            response_rate=("response_made_num", "mean"),
        )
        .reset_index()
    )

    summary["response_rate_percent"] = 100 * summary["response_rate"]

    for condition in ["T", "VT", "V"]:
        sub = summary[summary["sensory_condition"].eq(condition)]
        if sub.empty:
            continue

        table = sub.pivot_table(
            index="subject",
            columns="speed",
            values="response_rate_percent",
            fill_value=np.nan,
        )

        ax = table.plot(kind="bar", figsize=(8, 4))
        ax.set_title(f"PPS response rate — {condition}")
        ax.set_xlabel("Subject")
        ax.set_ylabel("Response rate (%)")
        ax.set_ylim(0, 105)
        ax.legend(title="Speed")

        _savefig(
            save_dir,
            f"pps_response_rate_{condition}",
            show=show,
        )

    return summary.sort_values(["subject", "speed", "sensory_condition"])


def plot_pps_rt_boxplots(
    pps: pd.DataFrame,
    rt_col: str = "rt_ms",
    rt_min: Optional[float] = 100,
    rt_max: Optional[float] = 1000,
    save_dir: str = "figures/quick_qc",
    show: bool = True,
) -> pd.DataFrame:
    """
    PPS RT boxplots before exclusion.

    Uses all tactile-present trials with non-missing RT. It does not require
    usable == True. Visual-only trials are summarized separately if they contain
    RT-like values.
    """
    df = _ensure_cols(
        pps,
        ["subject", "speed", "sensory_condition", rt_col],
    )
    df[rt_col] = pd.to_numeric(df[rt_col], errors="coerce")

    tactile = df[
        df["sensory_condition"].isin(["T", "VT"])
        & df[rt_col].notna()
    ].copy()

    summary = (
        tactile.groupby(["subject", "speed", "sensory_condition"])
        .agg(
            n=(rt_col, "size"),
            median_rt=(rt_col, "median"),
            mean_rt=(rt_col, "mean"),
            min_rt=(rt_col, "min"),
            max_rt=(rt_col, "max"),
        )
        .reset_index()
        .sort_values(["subject", "speed", "sensory_condition"])
    )

    speeds = _sorted_nonmissing(tactile["speed"])

    for condition in ["T", "VT"]:
        for speed in speeds:
            sub = tactile[
                tactile["sensory_condition"].eq(condition)
                & tactile["speed"].eq(speed)
            ]

            _boxplot_by_subject(
                sub,
                value_col=rt_col,
                title=f"PPS RT boxplot before exclusion — {condition}, {speed}",
                ylabel="RT from tactile event (ms)",
                save_dir=save_dir,
                filename=f"pps_rt_boxplot_{condition}_{speed}",
                rt_min=rt_min,
                rt_max=rt_max,
                show=show,
            )

    visual = df[
        df["sensory_condition"].eq("V")
        & df[rt_col].notna()
    ].copy()

    if len(visual) > 0:
        print("Visual-only PPS trials with RT-like values were found:")
        print(
            visual.groupby(["subject", "speed"])
            .agg(
                n=(rt_col, "size"),
                median_rt=(rt_col, "median"),
                min_rt=(rt_col, "min"),
                max_rt=(rt_col, "max"),
            )
            .round(2)
            .to_string()
        )

    return summary


# -------------------------------------------------------------------------
# Hit/Miss plots
# -------------------------------------------------------------------------

def plot_collision_trial_counts(
    collision: pd.DataFrame,
    save_dir: str = "figures/quick_qc",
    show: bool = True,
) -> pd.DataFrame:
    """Hit/Miss trial counts by subject, trial type and speed."""
    df = _ensure_cols(collision, ["subject", "speed", "trial_type"])

    counts = (
        df.groupby(["subject", "speed", "trial_type"])
        .size()
        .reset_index(name="n")
        .sort_values(["subject", "speed", "trial_type"])
    )

    subjects = _sorted_nonmissing(df["subject"])
    speeds = _sorted_nonmissing(df["speed"])

    for speed in speeds:
        sub = counts[counts["speed"].eq(speed)]

        table = sub.pivot_table(
            index="subject",
            columns="trial_type",
            values="n",
            fill_value=0,
        ).reindex(index=subjects, fill_value=0)

        ax = table.plot(kind="bar", figsize=(9, 4))
        ax.set_title(f"Hit/Miss trial counts by trial type — {speed}")
        ax.set_xlabel("Subject")
        ax.set_ylabel("Number of trials")
        ax.legend(title="Trial type")

        _savefig(
            save_dir,
            f"collision_trial_counts_by_type_{speed}",
            show=show,
        )

    return counts


def plot_collision_accuracy(
    collision: pd.DataFrame,
    save_dir: str = "figures/quick_qc",
    show: bool = True,
) -> pd.DataFrame:
    """Hit/Miss accuracy by subject, trial type and speed."""
    df = _ensure_cols(collision, ["subject", "speed", "trial_type", "accuracy"])
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")

    summary = (
        df.groupby(["subject", "speed", "trial_type"])
        .agg(
            n=("accuracy", "size"),
            accuracy=("accuracy", "mean"),
        )
        .reset_index()
    )

    summary["accuracy_percent"] = 100 * summary["accuracy"]

    speeds = _sorted_nonmissing(summary["speed"])

    for speed in speeds:
        sub = summary[summary["speed"].eq(speed)]

        table = sub.pivot_table(
            index="subject",
            columns="trial_type",
            values="accuracy_percent",
            fill_value=np.nan,
        )

        ax = table.plot(kind="bar", figsize=(9, 4))
        ax.set_title(f"Hit/Miss accuracy by trial type — {speed}")
        ax.set_xlabel("Subject")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 105)
        ax.legend(title="Trial type")

        _savefig(
            save_dir,
            f"collision_accuracy_by_trial_type_{speed}",
            show=show,
        )

    return summary.sort_values(["subject", "speed", "trial_type"])


def plot_collision_rt_boxplots(
    collision: pd.DataFrame,
    rt_col: str = "rt_ms",
    save_dir: str = "figures/quick_qc",
    show: bool = True,
) -> pd.DataFrame:
    """
    Hit/Miss RT boxplots before exclusion.

    Uses all non-missing RTs. It does not require usable == True.
    """
    df = _ensure_cols(collision, ["subject", "speed", "trial_type", rt_col])
    df[rt_col] = pd.to_numeric(df[rt_col], errors="coerce")

    d = df[df[rt_col].notna()].copy()

    summary = (
        d.groupby(["subject", "speed", "trial_type"])
        .agg(
            n=(rt_col, "size"),
            median_rt=(rt_col, "median"),
            mean_rt=(rt_col, "mean"),
            min_rt=(rt_col, "min"),
            max_rt=(rt_col, "max"),
        )
        .reset_index()
        .sort_values(["subject", "speed", "trial_type"])
    )

    speeds = _sorted_nonmissing(d["speed"])
    trial_types = _sorted_nonmissing(d["trial_type"])

    for speed in speeds:
        for trial_type in trial_types:
            sub = d[
                d["speed"].eq(speed)
                & d["trial_type"].eq(trial_type)
            ]

            _boxplot_by_subject(
                sub,
                value_col=rt_col,
                title=f"Hit/Miss RT boxplot before exclusion — {trial_type}, {speed}",
                ylabel="RT (ms)",
                save_dir=save_dir,
                filename=f"collision_rt_boxplot_{trial_type}_{speed}",
                show=show,
            )

    return summary


# -------------------------------------------------------------------------
# Master call
# -------------------------------------------------------------------------

def run(
    t,
    rt_min: Optional[float] = 100,
    rt_max: Optional[float] = 1000,
    save_dir: str = "figures/quick_qc",
    show: bool = True,
) -> dict:
    """
    Run all quick QC plots from a Tables object.

    Returns a dictionary of summary DataFrames so the notebook can display or
    export them.
    """
    pps = t.pps_trials.copy()
    collision = t.collision_trials.copy()

    results = {}

    print("=== PPS trial counts ===")
    results["pps_trial_counts"] = plot_pps_trial_counts(
        pps,
        save_dir=save_dir,
        show=show,
    )
    print(results["pps_trial_counts"].to_string(index=False))

    print("\n=== PPS response rate ===")
    results["pps_response_rate"] = plot_pps_response_rate(
        pps,
        save_dir=save_dir,
        show=show,
    )
    print(results["pps_response_rate"].round(2).to_string(index=False))

    print("\n=== PPS RT boxplots before exclusion ===")
    results["pps_rt_summary"] = plot_pps_rt_boxplots(
        pps,
        rt_col="rt_ms",
        rt_min=rt_min,
        rt_max=rt_max,
        save_dir=save_dir,
        show=show,
    )
    print(results["pps_rt_summary"].round(2).to_string(index=False))

    print("\n=== Hit/Miss trial counts ===")
    results["collision_trial_counts"] = plot_collision_trial_counts(
        collision,
        save_dir=save_dir,
        show=show,
    )
    print(results["collision_trial_counts"].to_string(index=False))

    print("\n=== Hit/Miss accuracy ===")
    results["collision_accuracy"] = plot_collision_accuracy(
        collision,
        save_dir=save_dir,
        show=show,
    )
    print(results["collision_accuracy"].round(2).to_string(index=False))

    print("\n=== Hit/Miss RT boxplots before exclusion ===")
    results["collision_rt_summary"] = plot_collision_rt_boxplots(
        collision,
        rt_col="rt_ms",
        save_dir=save_dir,
        show=show,
    )
    print(results["collision_rt_summary"].round(2).to_string(index=False))

    return results
