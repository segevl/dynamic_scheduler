#!/usr/bin/env python3
"""
DREAM Night Sky Condition Classifier & Analyser
================================================

Reads the DREAM cloud_sys CSV, fetches real HDF5 files for ~10 probe frames
per night, computes real patchiness metrics, then classifies and clusters.

What this script does
---------------------
1. CLEARLY EXPLAINS the three cloud metrics and what they mean
2. Classifies every night into:
      CLEAR            - valid_frac < 0.10  (almost no clouds to map)
      PARTIALLY CLOUDY - patchy with high spatial variance
      OVERCAST         - valid_frac > 0.90, uniform, mean_ext > 0.3
3. Runs K-Means + hierarchical clustering WITHIN partially-cloudy nights
4. Prints example nights for each class and each sub-type
5. Reports total nights, % partially cloudy, monthly distributions

Outputs
-------
  dream_night_classification.png   - 8-panel analysis figure
  dream_night_summary.csv          - per-night metrics + labels

Usage
-----
  python dream_night_classifier.py [path/to/feb5_data.csv]
"""

import io
import os
import sys
import warnings
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# Try LSST resource loader, fall back to urllib
try:
    from lsst.resources import ResourcePath
    HAS_LSST = True
except ImportError:
    HAS_LSST = False

URL_COL  = "lsst.sal.DREAM.logevent_largeFileObjectAvailable.url"
TIME_COL = "time"

# Pixel quality thresholds
MAX_SIGMA_MAG  = 0.3
MAX_FLAG_VALUE = 0

# Night classification thresholds
PATCHINESS_LOW_FRAC   = 0.10
PATCHINESS_HIGH_FRAC  = 0.90
PATCHINESS_VAR_THRESH = 0.05   # mag^2

# Number of probe frames fetched per night
PROBE_N = 10

# Output paths: same directory as the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PNG    = os.path.join(SCRIPT_DIR, "dream_night_classification.png")
OUT_CSV    = os.path.join(SCRIPT_DIR, "dream_night_summary.csv")


# =============================================================================
# METRIC EXPLAINER
# =============================================================================

METRIC_EXPLAINER = """
+==============================================================================+
|          DREAM CLOUD CONDITION METRICS -- WHAT THEY MEAN                    |
+==============================================================================+

The DREAM system produces HEALPix sky maps every ~30 s.
Each pixel carries three values:

  clouds   - Systematic cloud extinction in MAGNITUDES at that sky position.
             0.0  = perfectly clear  (no extra dimming).
             0.3  = mild cloud       (~25% of photons lost).
             0.5  = moderate cloud   (~40% of photons lost).
             2.0  = very thick cloud (~85% of photons lost; near-opaque).
             NaN  = pixel masked out (no ref-stars / too uncertain).

  sigma    - 1-sigma uncertainty on the extinction (mag).
             Pixels with sigma > 0.3 mag are masked before analysis.

  flags    - Integer quality flag. 0 = good; >0 = bad measurement.

From those raw maps we compute three NIGHT-LEVEL summary metrics:

+------------------------------------------------------------------------------+
| METRIC 1 -- valid_frac                                                       |
|   = (pixels with good extinction measurement) / (all horizon-visible pixels) |
|                                                                              |
|  Near 0.0  -> almost no clouds detectable; sky mostly clear OR instrument   |
|              issue. DREAM can't map what it can't see.                       |
|  0.10-0.90 -> genuine partial cloud cover; some sky clear, some blocked.    |
|  Near 1.0  -> ref stars seen everywhere. Either covered by thin/translucent |
|              cloud (high extinction) OR uniformly clear (low extinction).    |
|              These two cases are separated by mean_ext.                      |
|                                                                              |
|  Scheduling implication:                                                     |
|    < 0.10  -> skip; not enough cloud structure to navigate around.           |
|    > 0.90  -> skip; uniform conditions (photometric or uniform overcast).    |
|    0.10-0.90 -> ACT: cloud-aware pointing is worth doing.                   |
+------------------------------------------------------------------------------+

+------------------------------------------------------------------------------+
| METRIC 2 -- spatial_var  (mag^2)                                             |
|   = Variance of cloud extinction across all valid pixels in a frame,         |
|     averaged over ~10 probe frames through the night.                        |
|                                                                              |
|  ~0.00  -> extinction same everywhere: uniform sky condition.                |
|  > 0.05 -> extinction varies significantly: genuine patchwork structure.    |
|  > 0.20 -> dramatic patchwork: clear windows next to thick cloud banks.    |
|                                                                              |
|  Clear night    -> spatial_var ~0, valid_frac ~0                            |
|  Overcast night -> spatial_var ~0, valid_frac ~1                            |
|  Patchy night   -> spatial_var > 0.05, valid_frac in [0.10, 0.90]          |
+------------------------------------------------------------------------------+

+------------------------------------------------------------------------------+
| METRIC 3 -- mean_ext  (mag)                                                  |
|   = Mean cloud extinction across all valid pixels.                           |
|                                                                              |
|  0.00-0.10 -> Photometric or nearly so (<10% extra dimming).                |
|  0.10-0.30 -> Mildly cloudy: science still productive.                      |
|  0.30-0.70 -> Significantly cloudy: 5-sigma depth reduced by 0.3-0.7 mag.  |
|  0.70-1.00 -> Heavily cloudy: major losses.                                 |
|  > 1.00    -> Very thick: most programs abort.                              |
|                                                                              |
|  A 0.30 mag extinction hit ~= 25% loss in effective survey area at fixed    |
|  depth over Rubin's 10-year programme.                                       |
+------------------------------------------------------------------------------+

CLASSIFICATION RULES:
  CLEAR          : valid_frac < 0.10
                   OR (valid_frac > 0.90 AND spatial_var < 0.05 AND mean_ext < 0.10)
  OVERCAST       : valid_frac > 0.90 AND spatial_var < 0.05 AND mean_ext > 0.30
  PARTIALLY CLOUDY: 0.10 <= valid_frac <= 0.90 AND spatial_var >= 0.05

  CLEAR nights are the BEST nights for science (deep, photometric).
  PARTIALLY CLOUDY nights are where adaptive cloud-aware scheduling pays off.
  OVERCAST nights are largely lost -- the cloud is too uniform to dodge.
"""

print(METRIC_EXPLAINER)


# =============================================================================
# URL HELPER
# =============================================================================

def transform_url(url):
    url = str(url).strip()
    if url.startswith("https://s3.cp.lsst.org/"):
        return url.replace("https://s3.cp.lsst.org/", "s3://lfa@")
    return url


# =============================================================================
# HDF5 FETCH
# =============================================================================

def _read_bytes(url):
    if HAS_LSST:
        rp = ResourcePath(url)
        with rp.open("rb") as fd:
            return fd.read()
    else:
        import urllib.request
        with urllib.request.urlopen(url) as resp:
            return resp.read()


def fetch_cloud_map(url):
    """
    Load a cloud_sys HDF5 file.
    Returns (clouds, sigma) HEALPix arrays with bad pixels set to NaN.
    """
    data = _read_bytes(url)
    with h5py.File(io.BytesIO(data), "r") as f:
        clouds = np.array(f["clouds"],       dtype=float).ravel()
        sigma  = np.array(f["sigma"],        dtype=float).ravel()
        flags  = np.array(f["flags"],        dtype=int  ).ravel()
        vis    = np.array(f["mask_visible"], dtype=bool ).ravel()
        nobs   = np.array(f["nobs"],         dtype=int  ).ravel()

    bad = (~vis | (nobs == 0) | (flags > MAX_FLAG_VALUE)
           | (sigma > MAX_SIGMA_MAG) | ~np.isfinite(clouds))
    clouds[bad] = np.nan
    sigma[bad]  = np.nan
    return clouds, sigma


# =============================================================================
# METRICS FROM ONE HEALPIX MAP
# =============================================================================

def map_metrics(clouds):
    """
    Compute valid_frac, spatial_var, mean_ext, std_ext from one HEALPix frame.
    Returns None if fewer than 50 valid pixels.
    """
    valid   = ~np.isnan(clouds)
    n_valid = int(valid.sum())
    n_total = len(clouds)
    if n_valid < 50:
        return None
    vals = clouds[valid]
    return dict(
        valid_frac  = float(n_valid / n_total),
        spatial_var = float(np.var(vals)),
        mean_ext    = float(np.mean(vals)),
        std_ext     = float(np.std(vals)),
    )


# =============================================================================
# PER-NIGHT METRIC COMPUTATION  (fetches real HDF5 files)
# =============================================================================

def fetch_night_metrics(df_night, night_key):
    """
    Fetch ~PROBE_N evenly-spaced HDF5 frames for one night, compute metrics,
    return a per-night summary dict.  Returns None if too few frames load.
    """
    n = len(df_night)
    if n < 2:
        return None

    probe_idx = np.linspace(0, n - 1, min(PROBE_N, n), dtype=int)

    valid_fracs, spatial_vars, mean_exts, std_exts = [], [], [], []
    n_failed = 0

    for i in probe_idx:
        row = df_night.iloc[i]
        url = transform_url(row[URL_COL])
        try:
            clouds, _ = fetch_cloud_map(url)
            m = map_metrics(clouds)
            if m:
                valid_fracs.append(m["valid_frac"])
                spatial_vars.append(m["spatial_var"])
                mean_exts.append(m["mean_ext"])
                std_exts.append(m["std_ext"])
        except Exception:
            n_failed += 1

    if len(valid_fracs) < 2:
        return None

    times = df_night[TIME_COL].sort_values()
    gaps  = np.diff(times.values).astype("int64") * 1e-9   # seconds
    gap_cv = float(np.std(gaps) / (np.mean(gaps) + 1e-3))
    dur_h  = float((times.iloc[-1] - times.iloc[0]).total_seconds() / 3600.0)

    return {
        "night_key":   str(night_key),
        "n_frames":    n,
        "n_probed":    len(valid_fracs),
        "n_failed":    n_failed,
        "dur_h":       dur_h,
        "valid_frac":  float(np.mean(valid_fracs)),
        "spatial_var": float(np.mean(spatial_vars)),
        "mean_ext":    float(np.mean(mean_exts)),
        "std_ext":     float(np.mean(std_exts)),
        "gap_cv":      gap_cv,
        "month":       str(df_night["month"].iloc[0]),
        "year_month":  df_night[TIME_COL].iloc[0].strftime("%Y-%m"),
    }


# =============================================================================
# CSV LOADING
# =============================================================================

def load_dream_csv(csv_file):
    print(f"\nLoading CSV: {csv_file}")
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.replace('"', "").str.strip()
    df = df.dropna(subset=[URL_COL]).copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
    df = df.dropna(subset=[TIME_COL])
    df = df[df[URL_COL].str.contains(r"\.hdf5",    case=False, na=False, regex=True)]
    df = df[df[URL_COL].str.contains("cloud_sys",  case=False, na=False)]
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    shifted         = df[TIME_COL] - pd.Timedelta(hours=12)
    df["night_key"] = shifted.dt.date
    df["month"]     = df[TIME_COL].dt.to_period("M")

    nights = sorted(df["night_key"].unique())
    print(f"  {len(df):,} cloud_sys frames across {len(nights)} nights")
    print(f"  Date range: {df[TIME_COL].min()} -> {df[TIME_COL].max()}")
    return df


def compute_all_night_metrics(all_sys_df):
    nights  = sorted(all_sys_df["night_key"].unique())
    records = []
    print(f"\nComputing metrics for {len(nights)} nights "
          f"(fetching up to {PROBE_N} HDF5 frames per night) ...")

    for i, nk in enumerate(nights):
        df_n = all_sys_df[all_sys_df["night_key"] == nk]
        rec  = fetch_night_metrics(df_n, nk)
        if rec:
            records.append(rec)
        if (i + 1) % 10 == 0 or (i + 1) == len(nights):
            print(f"  {i+1:3d}/{len(nights)}  ({len(records)} nights with valid data)")

    df = pd.DataFrame(records)
    print(f"\n  -> {len(df)} nights with valid HDF5 metrics")
    return df


# =============================================================================
# CLASSIFICATION
# =============================================================================

def classify_night(row):
    vf  = row["valid_frac"]
    sv  = row["spatial_var"]
    ext = row["mean_ext"]

    if vf < PATCHINESS_LOW_FRAC:
        return "CLEAR"
    if vf > PATCHINESS_HIGH_FRAC:
        if sv < PATCHINESS_VAR_THRESH:
            return "CLEAR" if ext < 0.10 else "OVERCAST"
        else:
            return "PARTIALLY CLOUDY"
    # 0.10 <= vf <= 0.90
    return "PARTIALLY CLOUDY" if sv >= PATCHINESS_VAR_THRESH else "OVERCAST"


def apply_classification(df):
    df = df.copy()
    df["sky_class"] = df.apply(classify_night, axis=1)
    return df


# =============================================================================
# ML CLUSTERING  (within partially-cloudy nights)
# =============================================================================

CLUSTER_DESCRIPTIONS = {
    "Thin Scattered":
        ("Low mean extinction (<0.3 mag), moderate spatial variance.\n"
         "   Small isolated patches; most sky accessible.\n"
         "   Best case for scheduling: patches easily dodged."),
    "Heavy Patchy":
        ("High mean extinction (>0.5 mag), high spatial variance.\n"
         "   Dense cloud banks with brief clear windows.\n"
         "   Large scheduling gain but windows are short."),
    "Broken Overcast":
        ("High valid_frac (>0.75), moderate-high extinction.\n"
         "   Thick sheet with occasional holes; limited scheduling gain.\n"
         "   Consider abort unless clear windows are persistent."),
    "Isolated Cells":
        ("Moderate extinction, very high spatial variance.\n"
         "   Compact convective cells moving fast across the sky.\n"
         "   Motion tracking critical; windows narrow and mobile."),
    "Thin Uniform Haze":
        ("Near-uniform low extinction across the whole sky.\n"
         "   Probably a thin cirrus layer; science still viable.\n"
         "   Low scheduling gain because the whole sky is similar."),
    "Severe Storm":
        ("Very high extinction (>1.0 mag), large valid_frac.\n"
         "   Major weather event; most programmes would abort.\n"
         "   Cloud-aware scheduling cannot recover enough throughput."),
}

CLUSTER_COLORS = {
    "Thin Scattered":   "#3498db",
    "Heavy Patchy":     "#e74c3c",
    "Broken Overcast":  "#9b59b6",
    "Isolated Cells":   "#e67e22",
    "Thin Uniform Haze":"#2ecc71",
    "Severe Storm":     "#c0392b",
}
FALLBACK_COLORS = ["#1abc9c","#f39c12","#d35400","#7f8c8d","#2980b9","#8e44ad"]


def _name_cluster(mean_ext, spatial_var, valid_frac):
    if mean_ext > 1.0:
        return "Severe Storm"
    if spatial_var < 0.04 and mean_ext < 0.25:
        return "Thin Uniform Haze"
    if valid_frac > 0.75 and mean_ext > 0.4:
        return "Broken Overcast"
    if spatial_var > 0.20 and mean_ext > 0.4:
        return "Heavy Patchy"
    if spatial_var > 0.12 and mean_ext < 0.35:
        return "Isolated Cells"
    if mean_ext < 0.35:
        return "Thin Scattered"
    return "Heavy Patchy"


def cluster_partial_nights(df_partial):
    features = ["valid_frac", "spatial_var", "mean_ext", "std_ext", "gap_cv"]
    X  = df_partial[features].fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    best_k, best_score = 2, -1
    max_k = min(7, len(df_partial) - 1)
    for k in range(2, max_k + 1):
        km   = KMeans(n_clusters=k, random_state=42, n_init=10)
        labs = km.fit_predict(Xs)
        if len(np.unique(labs)) > 1:
            sc = silhouette_score(Xs, labs)
            if sc > best_score:
                best_score, best_k = sc, k

    print(f"\n  K-Means optimal k={best_k}, silhouette={best_score:.3f}")
    km     = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    centers_orig = scaler.inverse_transform(km.cluster_centers_)

    used_names, name_map = set(), {}
    for cid in range(best_k):
        c   = centers_orig[cid]
        raw = _name_cluster(c[2], c[1], c[0])
        if raw in used_names:
            raw = raw + f" ({cid})"
        name_map[cid] = raw
        used_names.add(raw)

    df_out = df_partial.copy()
    df_out["cluster"]      = labels
    df_out["cluster_name"] = [name_map[l] for l in labels]

    pca  = PCA(n_components=2)
    Xpca = pca.fit_transform(Xs)
    df_out["pca1"] = Xpca[:, 0]
    df_out["pca2"] = Xpca[:, 1]

    return df_out, best_k, best_score


# =============================================================================
# PRINTING HELPERS
# =============================================================================

SKY_EMOJIS = {
    "CLEAR":            "CLEAR",
    "PARTIALLY CLOUDY": "PARTIALLY CLOUDY",
    "OVERCAST":         "OVERCAST",
}

CLASS_COLORS = {
    "CLEAR":            "#2ecc71",
    "PARTIALLY CLOUDY": "#f39c12",
    "OVERCAST":         "#7f8c8d",
}


def print_example_nights(df, sky_class, n=3):
    sub = df[df["sky_class"] == sky_class].copy()
    if sub.empty:
        print(f"  No {sky_class} nights found.")
        return
    idx    = np.linspace(0, len(sub) - 1, min(n, len(sub)), dtype=int)
    sample = sub.iloc[idx]
    print(f"\n  [{sky_class}] -- {len(sub)} total nights of this type:")
    print(f"  {'Night':<14} {'Frames':>7} {'Probed':>7} "
          f"{'valid_frac':>11} {'spatial_var':>12} {'mean_ext':>10} {'dur_h':>7}")
    print(f"  {'-'*72}")
    for _, r in sample.iterrows():
        print(f"  {str(r['night_key']):<14} {int(r['n_frames']):>7} "
              f"{int(r.get('n_probed', 0)):>7} "
              f"{r['valid_frac']:>11.3f} {r['spatial_var']:>12.4f} "
              f"{r['mean_ext']:>10.3f} {r['dur_h']:>7.1f}")


def print_partial_cloud_clusters(df_partial, n=2):
    if "cluster_name" not in df_partial.columns:
        return
    n_partial = len(df_partial)
    for cn in sorted(df_partial["cluster_name"].unique()):
        sub  = df_partial[df_partial["cluster_name"] == cn]
        desc = CLUSTER_DESCRIPTIONS.get(cn.split(" (")[0], "No description.")
        print(f"\n  +-- Sub-type: {cn}  ({len(sub)} nights, "
              f"{100*len(sub)/n_partial:.1f}% of patchy nights)")
        for line in desc.split("\n"):
            print(f"  |   {line}")
        print(f"  |")
        print(f"  |   {'Night':<14} {'valid_frac':>11} {'spatial_var':>12} "
              f"{'mean_ext':>10} {'std_ext':>9}")
        print(f"  |   {'-'*58}")
        idx = np.linspace(0, len(sub) - 1, min(n, len(sub)), dtype=int)
        for _, r in sub.iloc[idx].iterrows():
            print(f"  |   {str(r['night_key']):<14} {r['valid_frac']:>11.3f} "
                  f"{r['spatial_var']:>12.4f} {r['mean_ext']:>10.3f} "
                  f"{r['std_ext']:>9.3f}")
        print(f"  |")
        print(f"  |   Centroid: valid_frac={sub['valid_frac'].mean():.3f}  "
              f"spatial_var={sub['spatial_var'].mean():.4f}  "
              f"mean_ext={sub['mean_ext'].mean():.3f}  "
              f"std_ext={sub['std_ext'].mean():.3f}")
        print(f"  +{'-'*59}")


def print_summary_statistics(df):
    n_total   = len(df)
    counts    = df["sky_class"].value_counts()
    n_partial = int(counts.get("PARTIALLY CLOUDY", 0))
    n_clear   = int(counts.get("CLEAR",            0))
    n_over    = int(counts.get("OVERCAST",         0))
    pct_p     = 100 * n_partial / n_total
    pct_c     = 100 * n_clear   / n_total
    pct_o     = 100 * n_over    / n_total

    print("\n" + "="*62)
    print("  FULL DREAM HISTORY -- NIGHT CLASSIFICATION SUMMARY")
    print("="*62)
    print(f"  Total nights analysed  : {n_total:4d}")
    print(f"  Date range             : {df['night_key'].min()}  ->  {df['night_key'].max()}")
    print()
    print(f"  CLEAR                  : {n_clear:4d}  ({pct_c:.1f}%)")
    print(f"  PARTIALLY CLOUDY       : {n_partial:4d}  ({pct_p:.1f}%)")
    print(f"  OVERCAST               : {n_over:4d}  ({pct_o:.1f}%)")
    print()
    pc = df[df["sky_class"] == "PARTIALLY CLOUDY"]
    if not pc.empty:
        print(f"  Among partially-cloudy nights:")
        print(f"    mean cloud extinction : "
              f"{pc['mean_ext'].mean():.3f} +/- {pc['mean_ext'].std():.3f} mag")
        print(f"    spatial variance      : "
              f"{pc['spatial_var'].mean():.4f} +/- {pc['spatial_var'].std():.4f} mag^2")
        print(f"    median valid_frac     : {pc['valid_frac'].median():.3f}")
        best_idx = pc['mean_ext'].idxmin()
        worst_idx = pc['mean_ext'].idxmax()
        print(f"    best night (min ext)  : {pc.loc[best_idx, 'mean_ext']:.3f} mag  "
              f"({pc.loc[best_idx, 'night_key']})")
        print(f"    worst night (max ext) : {pc.loc[worst_idx, 'mean_ext']:.3f} mag  "
              f"({pc.loc[worst_idx, 'night_key']})")
    print("="*62)
    return n_total, n_partial, n_clear, n_over, pct_p


# =============================================================================
# FIGURE
# =============================================================================

def make_main_figure(df, df_partial, n_total, n_partial, n_clear, n_over, pct_partial):

    fig = plt.figure(figsize=(24, 20))
    fig.patch.set_facecolor("#0d1117")

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.44, wspace=0.38,
                           left=0.07, right=0.97, top=0.93, bottom=0.06)

    DARK  = "#1a1f2e"
    GRID  = "#2a3040"
    TEXT  = "#e0e6f0"
    WHITE = "#ffffff"

    def _style(ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor(DARK)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.set_title(title, color=WHITE, fontsize=11, weight="bold", pad=8)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.2, color=GRID, lw=0.7)

    # Panel 1: Pie chart
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor(DARK)
    sizes = [n_clear, n_partial, n_over]
    lbl   = [f"CLEAR\n{n_clear} ({100*n_clear/n_total:.1f}%)",
             f"PARTIALLY CLOUDY\n{n_partial} ({pct_partial:.1f}%)",
             f"OVERCAST\n{n_over} ({100*n_over/n_total:.1f}%)"]
    cols  = [CLASS_COLORS["CLEAR"], CLASS_COLORS["PARTIALLY CLOUDY"], CLASS_COLORS["OVERCAST"]]
    ax0.pie(sizes, labels=lbl, colors=cols, startangle=90,
            wedgeprops={"edgecolor": "#0d1117", "linewidth": 2},
            textprops={"color": TEXT, "fontsize": 8.5})
    ax0.set_title(f"Night Classification\n({n_total} total nights)",
                  color=WHITE, fontsize=11, weight="bold")

    # Panel 2: valid_frac histogram
    ax1 = fig.add_subplot(gs[0, 1])
    _style(ax1, "Cloudiness Distribution\n(valid_frac -- all nights)",
           "valid_frac  (fraction of sky with measurable clouds)",
           "Number of nights")
    bins = np.linspace(0, 1, 31)
    for cls in ["CLEAR", "PARTIALLY CLOUDY", "OVERCAST"]:
        sub = df[df["sky_class"] == cls]["valid_frac"]
        if not sub.empty:
            ax1.hist(sub, bins=bins, color=CLASS_COLORS[cls], alpha=0.78,
                     label=cls, edgecolor="#0d1117", lw=0.5)
    ax1.axvline(PATCHINESS_LOW_FRAC,  color="#e74c3c", lw=1.5, ls="--",
                label=f"Clear boundary ({PATCHINESS_LOW_FRAC})")
    ax1.axvline(PATCHINESS_HIGH_FRAC, color="#e74c3c", lw=1.5, ls=":",
                label=f"Overcast boundary ({PATCHINESS_HIGH_FRAC})")
    ax1.legend(fontsize=7.5, facecolor=DARK, labelcolor=TEXT, edgecolor=GRID)

    # Panel 3: mean_ext histogram (partial nights)
    ax2 = fig.add_subplot(gs[0, 2])
    _style(ax2, "Cloud Extinction Distribution\n(partially-cloudy nights only)",
           "Mean cloud extinction (mag)", "Number of nights")
    if not df_partial.empty:
        ext_vals = df_partial["mean_ext"].dropna()
        n_bins   = min(25, max(5, len(ext_vals) // 3))
        ax2.hist(ext_vals, bins=n_bins, color=CLASS_COLORS["PARTIALLY CLOUDY"],
                 edgecolor="#0d1117", lw=0.5, alpha=0.85)
        for lo, hi, lbl_, col in [
            (0.0, 0.1, "Photometric",  "#2ecc71"),
            (0.1, 0.3, "Mild",         "#f1c40f"),
            (0.3, 0.7, "Moderate",     "#e67e22"),
            (0.7, 1.0, "Heavy",        "#e74c3c"),
            (1.0, 9.9, "Extreme",      "#c0392b"),
        ]:
            clip_hi = min(hi, ext_vals.max() + 0.05)
            if lo < clip_hi:
                ax2.axvspan(lo, clip_hi, alpha=0.09, color=col)
        med = ext_vals.median()
        ax2.axvline(med, color=WHITE, lw=1.5, ls="--", label=f"Median {med:.2f} mag")
        ax2.legend(fontsize=8, facecolor=DARK, labelcolor=TEXT, edgecolor=GRID)
    ax2.set_xlim(left=0)

    # Panel 4: Monthly bar chart (full width)
    ax3 = fig.add_subplot(gs[1, :2])
    _style(ax3, "Monthly Night Counts by Sky Condition",
           "Month", "Number of nights")
    monthly = (df.groupby(["year_month", "sky_class"])
                 .size()
                 .unstack(fill_value=0))
    for cls in ["CLEAR", "PARTIALLY CLOUDY", "OVERCAST"]:
        if cls not in monthly.columns:
            monthly[cls] = 0
    months = monthly.index.tolist()
    x = np.arange(len(months))
    w = 0.28
    ax3.bar(x - w, monthly["CLEAR"].values,            w,
            color=CLASS_COLORS["CLEAR"],            label="CLEAR",
            alpha=0.82, edgecolor="#0d1117", lw=0.4)
    ax3.bar(x,     monthly["PARTIALLY CLOUDY"].values, w,
            color=CLASS_COLORS["PARTIALLY CLOUDY"], label="PARTIALLY CLOUDY",
            alpha=0.82, edgecolor="#0d1117", lw=0.4)
    ax3.bar(x + w, monthly["OVERCAST"].values,         w,
            color=CLASS_COLORS["OVERCAST"],         label="OVERCAST",
            alpha=0.82, edgecolor="#0d1117", lw=0.4)
    step = max(1, len(months) // 18)
    ax3.set_xticks(x[::step])
    ax3.set_xticklabels(months[::step], rotation=45, ha="right", fontsize=7.5)
    ax3.legend(fontsize=8, facecolor=DARK, labelcolor=TEXT, edgecolor=GRID)

    # Panel 5: Cloud structure scatter
    ax4 = fig.add_subplot(gs[1, 2])
    _style(ax4, "Cloud Structure Space\n(all nights)",
           "Mean cloud extinction (mag)", "Spatial variance (mag^2)")
    for cls in ["CLEAR", "PARTIALLY CLOUDY", "OVERCAST"]:
        sub = df[df["sky_class"] == cls]
        if not sub.empty:
            ax4.scatter(sub["mean_ext"], sub["spatial_var"],
                        c=CLASS_COLORS[cls], s=20, alpha=0.6, label=cls, lw=0)
    ax4.axhline(PATCHINESS_VAR_THRESH, color="#e74c3c", lw=1.2, ls="--", alpha=0.7,
                label=f"var threshold ({PATCHINESS_VAR_THRESH})")
    ax4.set_ylim(bottom=0)
    ax4.legend(fontsize=7.5, facecolor=DARK, labelcolor=TEXT, edgecolor=GRID)

    # Panel 6: PCA of sub-types
    ax5 = fig.add_subplot(gs[2, 0])
    _style(ax5, "Partially-Cloudy Sub-types\n(K-Means, PCA 2D projection)",
           "PC1", "PC2")
    if not df_partial.empty and "pca1" in df_partial.columns:
        for i, cn in enumerate(sorted(df_partial["cluster_name"].unique())):
            sub = df_partial[df_partial["cluster_name"] == cn]
            cc  = CLUSTER_COLORS.get(cn.split(" (")[0], FALLBACK_COLORS[i % 6])
            ax5.scatter(sub["pca1"], sub["pca2"], c=cc, s=25, alpha=0.78,
                        label=f"{cn} ({len(sub)})", lw=0)
        ax5.legend(fontsize=7, facecolor=DARK, labelcolor=TEXT, edgecolor=GRID)

    # Panel 7: Sub-type mean extinction bars
    ax6 = fig.add_subplot(gs[2, 1])
    _style(ax6, "Sub-type Mean Extinction\n(partially-cloudy nights)",
           "Sub-type", "Mean extinction +/- std (mag)")
    if not df_partial.empty and "cluster_name" in df_partial.columns:
        cnames = sorted(df_partial["cluster_name"].unique())
        means  = [df_partial[df_partial["cluster_name"]==c]["mean_ext"].mean() for c in cnames]
        stds   = [df_partial[df_partial["cluster_name"]==c]["mean_ext"].std()  for c in cnames]
        bcols  = [CLUSTER_COLORS.get(c.split(" (")[0], FALLBACK_COLORS[i % 6])
                  for i, c in enumerate(cnames)]
        bars   = ax6.bar(cnames, means, color=bcols, alpha=0.82,
                         edgecolor="#0d1117", lw=0.8,
                         yerr=stds, error_kw={"ecolor": TEXT, "capsize": 4})
        ax6.tick_params(axis="x", labelsize=7, rotation=20)
        for bar, v in zip(bars, means):
            ax6.text(bar.get_x() + bar.get_width()/2, v,
                     f"{v:.2f}", ha="center", va="bottom",
                     fontsize=8.5, color=WHITE, weight="bold")

    # Panel 8: Monthly partially-cloudy fraction
    ax7 = fig.add_subplot(gs[2, 2])
    _style(ax7, "Monthly Partially-Cloudy Fraction\n(% of nights that month)",
           "Month", "Fraction partially cloudy (%)")
    month_all     = df.groupby("year_month").size()
    month_partial = (df[df["sky_class"] == "PARTIALLY CLOUDY"]
                     .groupby("year_month").size())
    frac_partial  = (month_partial / month_all * 100).fillna(0)
    months2 = frac_partial.index.tolist()
    x2      = np.arange(len(months2))
    ax7.fill_between(x2, frac_partial.values, alpha=0.30,
                     color=CLASS_COLORS["PARTIALLY CLOUDY"])
    ax7.plot(x2, frac_partial.values, "o-",
             color=CLASS_COLORS["PARTIALLY CLOUDY"], lw=2, ms=5)
    mean_frac = frac_partial.mean()
    ax7.axhline(mean_frac, color="#e74c3c", lw=1.5, ls="--",
                label=f"Mean {mean_frac:.1f}%")
    step3 = max(1, len(months2) // 12)
    ax7.set_xticks(x2[::step3])
    ax7.set_xticklabels(months2[::step3], rotation=45, ha="right", fontsize=7.5)
    ax7.set_ylim(0, 100)
    ax7.legend(fontsize=8.5, facecolor=DARK, labelcolor=TEXT, edgecolor=GRID)

    fig.suptitle("DREAM Observatory -- Full Sky Condition History Analysis",
                 fontsize=16, weight="bold", color=WHITE, y=0.97)

    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  -> Figure saved: {OUT_PNG}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main(csv_file="feb5_data.csv"):

    all_sys = load_dream_csv(csv_file)
    df      = compute_all_night_metrics(all_sys)

    if df.empty:
        print("ERROR: No night metrics computed. "
              "Check network access to the S3 store and CSV contents.")
        return

    df["night_key"]  = pd.to_datetime(df["night_key"])
    df["year_month"] = df["night_key"].dt.strftime("%Y-%m")
    df["night_key"]  = df["night_key"].dt.date.astype(str)

    df = apply_classification(df)

    n_total, n_partial, n_clear, n_over, pct_p = print_summary_statistics(df)

    print("\n" + "-"*72)
    print("  EXAMPLE NIGHTS BY CATEGORY")
    print("-"*72)
    for cls in ["CLEAR", "PARTIALLY CLOUDY", "OVERCAST"]:
        print_example_nights(df, cls, n=3)

    df_partial = (df[df["sky_class"] == "PARTIALLY CLOUDY"]
                  .copy()
                  .reset_index(drop=True))

    print("\n" + "-"*72)
    print("  ML CLUSTERING -- PARTIALLY CLOUDY NIGHT SUB-TYPES")
    print("-"*72)

    if len(df_partial) >= 6:
        df_partial, best_k, best_score = cluster_partial_nights(df_partial)
        print(f"\n  {len(df_partial)} partially-cloudy nights -> "
              f"{best_k} sub-types  (silhouette = {best_score:.3f})")
        print_partial_cloud_clusters(df_partial, n=2)
        df = df.merge(
            df_partial[["night_key", "cluster", "cluster_name"]],
            on="night_key", how="left")
    else:
        print(f"  Only {len(df_partial)} partially-cloudy nights -- "
              "too few for clustering.")
        df["cluster"]      = np.nan
        df["cluster_name"] = np.nan

    df.to_csv(OUT_CSV, index=False)
    print(f"\n  -> Per-night CSV saved: {OUT_CSV}")

    print("  Generating analysis figure ...")
    make_main_figure(df,
                     df_partial if len(df_partial) >= 6 else pd.DataFrame(),
                     n_total, n_partial, n_clear, n_over, pct_p)

    print("\n" + "="*62)
    print("  FINAL NUMBERS")
    print("="*62)
    print(f"  Total nights in DREAM history : {n_total}")
    print(f"  Partially cloudy              : {n_partial}  ({pct_p:.1f}%)")
    print(f"  Clear                         : {n_clear}  ({100*n_clear/n_total:.1f}%)")
    print(f"  Overcast                      : {n_over}  ({100*n_over/n_total:.1f}%)")
    if not df_partial.empty and "cluster_name" in df_partial.columns:
        print(f"\n  Partially-cloudy sub-type breakdown:")
        vc = df_partial["cluster_name"].value_counts()
        for cn, cnt in vc.items():
            print(f"    {cn:<26s}: {cnt:3d} nights  ({100*cnt/n_partial:.1f}%)")
    print("="*62)
    print("\nDone.")
    return df


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "feb5_data.csv"
    main(csv_path)
