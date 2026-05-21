#!/usr/bin/env python3
"""
dream_batch.py  ―  run Part A (server) or Part B (local) over multiple nights
==============================================================================

SERVER  (Part A — extract DREAM data for each night, save npz + video):
------------------------------------------------------------------------
    import dream_batch as db
    db.extract_nights(
        dates    = ["2025-07-15", "2025-07-20", "2025-07-23"],
        csv_path = "feb5_data.csv",
        out_dir  = "./npz_outputs",
    )

LOCAL  (Part B — run greedy comparison for each npz, aggregate plots):
-----------------------------------------------------------------------
    import dream_batch as db
    db.compare_nights(
        npz_dir    = "./npz_outputs",   # directory of dream_night_*.npz
        output_dir = "./outputs",
        band       = "r",
    )

    # or point at specific files:
    db.compare_nights(
        npz_files  = ["dream_night_20250715.npz",
                      "dream_night_20250720.npz",
                      "dream_night_20250723.npz"],
        output_dir = "./outputs",
        band       = "r",
    )
"""

from __future__ import annotations
import os
import glob
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ─────────────────────────────────────────────────────────────────────────────
# PART A  —  batch extraction (server)
# ─────────────────────────────────────────────────────────────────────────────

def extract_nights(
    dates:      list[str],
    csv_path:   str,
    out_dir:    str  = ".",
    max_frames: int | None = None,
    no_video:   bool = False,
) -> list[str]:
    """
    Run dream_extract_server.extract_night() for each date in `dates`.
    Skips dates whose npz already exists (re-run safe).

    Returns list of npz paths produced.
    """
    import dream_extract_server as pa
    os.makedirs(out_dir, exist_ok=True)
    npz_paths = []

    for date in dates:
        tag      = date.replace("-", "")
        npz_path = os.path.join(out_dir, f"dream_night_{tag}.npz")

        if os.path.exists(npz_path):
            print(f"\n  [{date}]  already exists — skipping extraction")
            npz_paths.append(npz_path)
            continue

        print(f"\n{'='*60}")
        print(f"  EXTRACTING  {date}")
        print(f"{'='*60}")
        try:
            result = pa.extract_night(date, csv_path, max_frames, out_dir)
            (saved_path, mjds, ext_grids,
             abs_altaz, abs_radec, mot_altaz, mot_radec) = result

            if not no_video:
                vid_out = os.path.abspath(out_dir)
                try:
                    pa.make_video(mjds, ext_grids, abs_altaz, abs_radec,
                                  mot_altaz, mot_radec, tag, out_dir=vid_out)
                except Exception as e:
                    print(f"  Video failed ({e}) — continuing")

            npz_paths.append(saved_path)
        except Exception as e:
            print(f"  FAILED for {date}: {e}")
            traceback.print_exc()

    print(f"\nExtraction complete — {len(npz_paths)} npz files")
    return npz_paths


# ─────────────────────────────────────────────────────────────────────────────
# PART B  —  batch comparison (local)
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "DREAM Absolute": "#2ca02c",
    "DREAM Motion":   "#1f77b4",
    "Greedy Sched":   "#d62728",
}
STRATEGIES = list(COLORS.keys())


def compare_nights(
    npz_files:  list[str] | None = None,
    npz_dir:    str | None       = None,
    output_dir: str              = "outputs",
    band:       str              = "r",
    verbose:    bool             = False,
) -> pd.DataFrame:
    """
    Run dream_compare_local.compare() for each npz, collect per-night
    summaries, and produce aggregate plots.

    Parameters
    ----------
    npz_files  : explicit list of npz paths (takes priority over npz_dir)
    npz_dir    : directory to glob for dream_night_*.npz
    output_dir : where to write per-night outputs + aggregate plots
    band       : photometric band
    verbose    : passed to sim_runner

    Returns
    -------
    DataFrame with one row per (night, strategy)
    """
    import dream_compare_local as pb

    # Collect npz files
    if npz_files is None:
        if npz_dir is None:
            raise ValueError("Provide either npz_files or npz_dir")
        npz_files = sorted(glob.glob(os.path.join(npz_dir,
                                                   "dream_night_*.npz")))
    if not npz_files:
        raise FileNotFoundError("No npz files found")

    print(f"\nBATCH COMPARISON — {len(npz_files)} nights, band={band}")
    for f in npz_files:
        print(f"  {f}")

    os.makedirs(output_dir, exist_ok=True)

    all_summaries = []   # list of dicts, one per (night × strategy)
    failed        = []

    for npz_path in npz_files:
        night_tag = _tag_from_path(npz_path)
        night_dir = os.path.join(output_dir, f"night_{night_tag}")
        os.makedirs(night_dir, exist_ok=True)

        print(f"\n{'─'*60}")
        print(f"  Processing {npz_path}  →  {night_dir}/")
        print(f"{'─'*60}")

        try:
            all_m, summary = pb.compare(
                npz_file   = npz_path,
                output_dir = night_dir,
                band       = band,
                verbose    = verbose,
            )
            for row in summary:
                row["night"]     = night_tag
                row["npz_path"]  = npz_path
            all_summaries.extend(summary)
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed.append(npz_path)

    if not all_summaries:
        print("\nNo successful nights — nothing to aggregate")
        return pd.DataFrame()

    df = pd.DataFrame(all_summaries)
    df["date"] = pd.to_datetime(df["night"], format="%Y%m%d", errors="coerce")
    df = df.sort_values(["date", "Strategy"]).reset_index(drop=True)

    # Save combined CSV
    csv_path = os.path.join(output_dir, "summary_all_nights.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nCombined CSV → {csv_path}")

    if failed:
        print(f"\nFailed nights ({len(failed)}):")
        for f in failed:
            print(f"  {f}")

    # Aggregate plots
    _plot_aggregate(df, output_dir, band)

    print(f"\nBATCH DONE — {len(npz_files) - len(failed)} nights processed")
    return df


def _tag_from_path(npz_path: str) -> str:
    base = os.path.basename(npz_path)
    # dream_night_20250715.npz → 20250715
    import re
    m = re.search(r"(\d{8})", base)
    return m.group(1) if m else base.replace(".npz", "")


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def _plot_aggregate(df: pd.DataFrame, output_dir: str, band: str):
    """
    Six-panel aggregate figure + per-metric trend plots across all nights.
    """
    nights   = sorted(df["night"].unique())
    n_nights = len(nights)
    if n_nights < 1:
        return

    x       = np.arange(n_nights)
    x_labs  = [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in nights]
    metrics = [
        ("Total_photons",       "Total photons\n(mag-20 ref, slew-gated)",  False, "sci"),
        ("N_slots",             "Observation slots per night",               False, "int"),
        ("Shutter_eff_pct",     "Shutter efficiency (%)",                    False, "f"),
        ("Mean_slew_s",         "Mean slew time (s)",                        False, "f"),
        ("Mean_extinction_mag", "Mean cloud extinction (mag)",               False, "f"),
        ("Median_5sig_depth",   "Median 5σ depth (mag)",                     True,  "f"),
        ("Unique_pixels",       "Unique HEALPix pixels visited",             False, "int"),
    ]

    # ── Fig 1: 2×3 grid of key metrics across nights ─────────────────────────
    plot_metrics = [m for m in metrics if m[0] != "N_slots"][:6]
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(
        f"Multi-Night Strategy Comparison  [{band}-band]\n"
        f"DREAM Absolute  ·  DREAM Motion  ·  Greedy Scheduler",
        fontsize=13, weight="bold")

    for ax, (col, ylabel, invert, fmt) in zip(axes.flat, plot_metrics):
        width = 0.25
        for i, (strat, color) in enumerate(COLORS.items()):
            vals = []
            for night in nights:
                sub = df[(df["night"] == night) & (df["Strategy"] == strat)]
                vals.append(float(sub[col].iloc[0]) if len(sub) else np.nan)
            bars = ax.bar(x + (i - 1) * width, vals, width,
                          color=color, alpha=0.82,
                          edgecolor="black", lw=0.8, label=strat)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labs, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=7)
        if invert:
            ax.invert_yaxis()
        if fmt == "sci":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    plt.tight_layout()
    out1 = os.path.join(output_dir, f"aggregate_metrics_{band}.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out1}")

    # ── Fig 2: Photon advantage over greedy, per night ────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, n_nights * 1.4), 5))
    width = 0.35
    for i, strat in enumerate(["DREAM Absolute", "DREAM Motion"]):
        pcts = []
        for night in nights:
            s_dream  = df[(df["night"] == night) & (df["Strategy"] == strat)]
            s_greedy = df[(df["night"] == night) & (df["Strategy"] == "Greedy Sched")]
            if len(s_dream) and len(s_greedy):
                base = float(s_greedy["Total_photons"].iloc[0])
                val  = float(s_dream["Total_photons"].iloc[0])
                pcts.append((val - base) / max(abs(base), 1) * 100)
            else:
                pcts.append(np.nan)
        bars = ax.bar(x + (i - 0.5) * width, pcts, width,
                      color=COLORS[strat], alpha=0.82,
                      edgecolor="black", lw=0.8, label=strat)
        for bar, v in zip(bars, pcts):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + (1 if v >= 0 else -3),
                        f"{v:+.0f}%", ha="center", va="bottom",
                        fontsize=8, weight="bold",
                        color=COLORS[strat])

    ax.axhline(0, color="k", lw=1.5, ls="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labs, rotation=30, ha="right")
    ax.set_ylabel("Photon advantage vs Greedy Scheduler (%)")
    ax.set_title(
        f"DREAM Photon Advantage Over Greedy Scheduler — All Nights  [{band}-band]",
        weight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out2 = os.path.join(output_dir, f"aggregate_advantage_{band}.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out2}")

    # ── Fig 3: Slot counts — sanity check that DREAM ≈ Greedy ────────────────
    fig, ax = plt.subplots(figsize=(max(8, n_nights * 1.4), 5))
    width = 0.25
    for i, (strat, color) in enumerate(COLORS.items()):
        vals = []
        for night in nights:
            sub = df[(df["night"] == night) & (df["Strategy"] == strat)]
            vals.append(float(sub["N_slots"].iloc[0]) if len(sub) else np.nan)
        ax.bar(x + (i - 1) * width, vals, width,
               color=color, alpha=0.82, edgecolor="black", lw=0.8, label=strat)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labs, rotation=30, ha="right")
    ax.set_ylabel("Observation slots")
    ax.set_title("Slot Count per Night per Strategy\n"
                 "(DREAM should be within ~20% of Greedy)",
                 weight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out3 = os.path.join(output_dir, f"aggregate_slots_{band}.png")
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out3}")

    # ── Fig 4: Extinction distribution across all nights ─────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for strat, color in COLORS.items():
        all_ext = df[df["Strategy"] == strat]["Mean_extinction_mag"].dropna().values
        if len(all_ext):
            ax.scatter(range(len(nights)),
                       [df[(df["night"]==n) & (df["Strategy"]==strat)
                           ]["Mean_extinction_mag"].values[0]
                        if len(df[(df["night"]==n) & (df["Strategy"]==strat)]) else np.nan
                        for n in nights],
                       color=color, s=80, label=strat, zorder=4)
            # connect with line
            vals = [df[(df["night"]==n) & (df["Strategy"]==strat)
                       ]["Mean_extinction_mag"].values[0]
                    if len(df[(df["night"]==n) & (df["Strategy"]==strat)]) else np.nan
                    for n in nights]
            ax.plot(range(len(nights)), vals, color=color, alpha=0.5, lw=1.5)

    ax.axhline(0, color="k", ls="--", lw=1, alpha=0.5)
    ax.set_xticks(range(len(nights)))
    ax.set_xticklabels(x_labs, rotation=30, ha="right")
    ax.set_ylabel("Mean cloud extinction (mag)")
    ax.set_title("Cloud Extinction per Night per Strategy\n"
                 "(DREAM Absolute should track toward 0 = clearest sky)",
                 weight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out4 = os.path.join(output_dir, f"aggregate_extinction_{band}.png")
    plt.savefig(out4, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out4}")

    # ── Fig 5: Depth vs extinction scatter (all nights combined) ─────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle(
        f"5σ Depth vs Cloud Extinction — All Nights Combined  [{band}-band]",
        fontsize=12, weight="bold")
    # We need per-slot data — load from per-night CSVs if available
    # Fall back to per-night summary scatter using mean values
    for ax, strat, color in zip(axes, STRATEGIES, COLORS.values()):
        sub = df[df["Strategy"] == strat]
        ax.scatter(sub["Mean_extinction_mag"], sub["Median_5sig_depth"],
                   c=[list(nights).index(n) for n in sub["night"]],
                   cmap="viridis", s=100, edgecolors="k", lw=0.8, zorder=4)
        ax.set_xlabel("Mean extinction (mag)", fontsize=10)
        ax.set_ylabel("Median 5σ depth (mag)", fontsize=10)
        ax.set_title(strat, weight="bold", color=color)
        ax.invert_yaxis()
        ax.grid(alpha=0.3)
        # Annotate night labels
        for _, row in sub.iterrows():
            d = str(row["night"])
            label = f"{d[4:6]}/{d[6:]}"
            ax.annotate(label,
                        (row["Mean_extinction_mag"], row["Median_5sig_depth"]),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=7, alpha=0.8)
    plt.tight_layout()
    out5 = os.path.join(output_dir, f"aggregate_depth_ext_{band}.png")
    plt.savefig(out5, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out5}")

    # ── Fig 6: Summary table as figure ───────────────────────────────────────
    pivot = df.pivot_table(
        index=["night"], columns="Strategy",
        values=["Total_photons", "N_slots", "Shutter_eff_pct",
                "Mean_extinction_mag", "Median_5sig_depth", "Unique_pixels"],
        aggfunc="first",
    )
    # Compute advantage columns
    for strat in ["DREAM Absolute", "DREAM Motion"]:
        key = ("Total_photons", strat)
        base_key = ("Total_photons", "Greedy Sched")
        if key in pivot.columns and base_key in pivot.columns:
            pivot[("Photon_adv_%", strat)] = (
                (pivot[key] - pivot[base_key]) / pivot[base_key].abs() * 100
            ).round(1)

    tbl_path = os.path.join(output_dir, "summary_all_nights.csv")
    df.to_csv(tbl_path, index=False)

    print(f"\n  Aggregate figures saved to {output_dir}/")
    print(f"  Files: aggregate_metrics, aggregate_advantage, aggregate_slots,")
    print(f"         aggregate_extinction, aggregate_depth_ext  (all _{band}.png)")


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: run everything from one cell
# ─────────────────────────────────────────────────────────────────────────────

def run_all(
    dates:      list[str],
    csv_path:   str        = "feb5_data.csv",
    npz_dir:    str        = "./npz",
    output_dir: str        = "./outputs",
    band:       str        = "r",
    max_frames: int | None = None,
    no_video:   bool       = False,
    skip_extract: bool     = False,
) -> pd.DataFrame:
    """
    One-call convenience wrapper for running on the server:
      1. Extract all nights (Part A)
      2. Compare all nights (Part B)
      3. Save aggregate plots

    On a local machine (no DREAM access), set skip_extract=True and
    point npz_dir at your downloaded npz files.

    Example (server notebook cell):
    --------------------------------
        import dream_batch as db
        df = db.run_all(
            dates      = ["2025-07-15", "2025-07-20", "2025-07-23"],
            csv_path   = "feb5_data.csv",
            npz_dir    = "./npz",
            output_dir = "./outputs",
            band       = "r",
        )
    """
    if not skip_extract:
        extract_nights(dates, csv_path, npz_dir, max_frames, no_video)

    return compare_nights(
        npz_dir    = npz_dir,
        output_dir = output_dir,
        band       = band,
        verbose    = False,
    )
