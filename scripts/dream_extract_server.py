#!/usr/bin/env python3
"""
dream_extract_server.py  ―  PART A  (run on server with DREAM access)
======================================================================
For a given night, reads the DREAM cloud_sys HDF5 files, projects each
HEALPix map onto a Cartesian alt/az grid, computes two pointing strategies
(Absolute minimum, Motion-predicted minimum), and saves everything to a
.npz for local analysis (Part B).

Also saves a short MP4 video of the cloud maps with both strategy
pointings overlaid, evolving through the night.

Usage
-----
    python dream_extract_server.py --date 2025-07-15 --csv feb5_data.csv
    python dream_extract_server.py --date 2025-07-15 --csv feb5_data.csv --max-frames 200

Outputs
-------
    dream_night_20250715.npz   — data for Part B
    dream_night_20250715.mp4   — video of cloud maps + pointings
"""

from __future__ import annotations
import argparse
import io
import os
import re
import warnings
import numpy as np
import pandas as pd
import h5py
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
from lsst.resources import ResourcePath

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# SITE / HEALPIX
# ─────────────────────────────────────────────────────────────────────────────
RUBIN_LAT      = -30.244639
RUBIN_LON      = -70.749417
RUBIN_HEIGHT_M = 2663.0
NSIDE          = 32
NEST           = True          # cloud_sys files use nested ordering
NPIX           = hp.nside2npix(NSIDE)

RUBIN_LOC = EarthLocation(lat=RUBIN_LAT * u.deg,
                           lon=RUBIN_LON * u.deg,
                           height=RUBIN_HEIGHT_M * u.m)

# ─────────────────────────────────────────────────────────────────────────────
# CARTESIAN GRID  (alt/az projected, same as Part B expects)
# ─────────────────────────────────────────────────────────────────────────────
MIN_ALT_DEG  = 15.0
BIN_SIZE_KM  = 1000          # grid cell size
R_PROJ       = 10_000.0      # projection radius [km]
GRID_RANGE   = 15            # ±15 bins  →  30×30 grid
GRID_N       = 2 * GRID_RANGE   # 30 cells

# Pre-build edge arrays (shared with Part B)
_edges = (np.arange(-GRID_RANGE, GRID_RANGE + 1) * BIN_SIZE_KM).astype(float)
X_EDGES = _edges.copy()
Y_EDGES = _edges.copy()

# ─────────────────────────────────────────────────────────────────────────────
# MOTION TRACKER PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
MOTION_HISTORY  = 5      # frames to use for velocity estimate
MOTION_LEAD     = 3      # frames to extrapolate ahead

# ─────────────────────────────────────────────────────────────────────────────
# VIDEO
# ─────────────────────────────────────────────────────────────────────────────
VIDEO_FPS        = 10
VIDEO_MAX_FRAMES = 600   # subsample to at most this many frames in the video
                         # (the npz always gets every frame)

# ─────────────────────────────────────────────────────────────────────────────
# URL / S3 HELPERS
# ─────────────────────────────────────────────────────────────────────────────
URL_COL  = "lsst.sal.DREAM.logevent_largeFileObjectAvailable.url"
TIME_COL = "time"


def transform_url(url: str) -> str:
    url = str(url).strip()
    if url.startswith("https://s3.cp.lsst.org/"):
        return url.replace("https://s3.cp.lsst.org/", "s3://lfa@")
    return url


def is_cloud_sys(url: str) -> bool:
    return "cloud_sys" in str(url)


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_cloud_sys(url: str) -> np.ndarray | None:
    """
    Fetch a cloud_sys HDF5 from S3 and return the masked clouds array
    (HEALPix nside=32, nested).  Returns None on any error.
    """
    try:
        rp = ResourcePath(transform_url(url))
        with rp.open("rb") as fd:
            raw = fd.read()
        if raw[:4] != b'\x89HDF':
            return None
        with h5py.File(io.BytesIO(raw), "r") as f:
            clouds = np.array(f["clouds"], dtype=float).ravel()
            if len(clouds) != NPIX:
                # Try to re-read at actual nside
                nside_actual = hp.npix2nside(len(clouds))
                clouds = hp.ud_grade(clouds, NSIDE, order_in="NESTED",
                                     order_out="NESTED")
            # Mask bad pixels
            if all(k in f for k in ("sigma", "flags", "mask_visible", "nobs")):
                sigma        = np.array(f["sigma"],        dtype=float).ravel()
                flags        = np.array(f["flags"],        dtype=int  ).ravel()
                mask_visible = np.array(f["mask_visible"], dtype=bool ).ravel()
                nobs         = np.array(f["nobs"],         dtype=int  ).ravel()
                bad = (~mask_visible | (nobs == 0) | (flags > 0)
                       | (sigma > 0.3) | ~np.isfinite(clouds))
                clouds[bad] = np.nan
            else:
                clouds[~np.isfinite(clouds)] = np.nan
        return clouds
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# HEALPIX → CARTESIAN GRID
# ─────────────────────────────────────────────────────────────────────────────

def _build_healpix_altaz_lut(mjd: float) -> tuple[np.ndarray, np.ndarray]:
    """
    For each HEALPix pixel (nside=32 nested), compute (alt_deg, az_deg)
    at the given MJD from Rubin's location.
    Returns (alt_array, az_array) both shape (NPIX,).
    Pixels below MIN_ALT_DEG get alt=-99.
    """
    theta, phi = hp.pix2ang(NSIDE, np.arange(NPIX), nest=NEST)
    ra_rad  = phi
    dec_rad = np.pi / 2 - theta
    ra_deg  = np.degrees(ra_rad)
    dec_deg = np.degrees(dec_rad)

    t   = Time(mjd, format="mjd", scale="utc")
    sky = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    aa  = sky.transform_to(AltAz(obstime=t, location=RUBIN_LOC))
    alt = aa.alt.deg
    az  = aa.az.deg % 360.0
    alt[alt < MIN_ALT_DEG] = -99.0
    return alt, az


def healpix_to_cartesian(clouds: np.ndarray, mjd: float) -> np.ndarray:
    """
    Project a HEALPix cloud_sys map onto a 30×30 Cartesian extinction grid.

    Projection:
        x =  -cos(alt) * sin(az) * R / sin(alt)   [km East]
        y =   cos(alt) * cos(az) * R / sin(alt)   [km North]

    Returns grid[GRID_N, GRID_N] in mag extinction (NaN where no data).
    """
    alt, az = _build_healpix_altaz_lut(mjd)
    visible = alt >= MIN_ALT_DEG

    alt_r = np.radians(alt[visible])
    az_r  = np.radians(az[visible])
    sin_a = np.sin(alt_r)
    s     = R_PROJ / sin_a
    xkm   = -np.cos(alt_r) * np.sin(az_r) * s
    ykm   =  np.cos(alt_r) * np.cos(az_r) * s

    xi = np.floor(xkm / BIN_SIZE_KM + GRID_RANGE).astype(int)
    yi = np.floor(ykm / BIN_SIZE_KM + GRID_RANGE).astype(int)

    grid = np.full((GRID_N, GRID_N), np.nan)
    cnt  = np.zeros((GRID_N, GRID_N), dtype=int)

    pix_visible = np.where(visible)[0]
    for k, (ix, iy, pix) in enumerate(zip(xi, yi, pix_visible)):
        if 0 <= ix < GRID_N and 0 <= iy < GRID_N:
            val = clouds[pix]
            if np.isfinite(val):
                if np.isnan(grid[iy, ix]):
                    grid[iy, ix] = val
                    cnt[iy, ix]  = 1
                else:
                    grid[iy, ix] += val
                    cnt[iy, ix]  += 1

    mask = cnt > 0
    grid[mask] /= cnt[mask]   # average where multiple pixels map to same cell
    return grid


def grid_to_altaz(xi: int, yi: int) -> tuple[float, float]:
    """Convert grid cell (xi, yi) back to alt/az degrees."""
    xkm = (xi - GRID_RANGE + 0.5) * BIN_SIZE_KM
    ykm = (yi - GRID_RANGE + 0.5) * BIN_SIZE_KM
    r   = np.sqrt(xkm**2 + ykm**2)
    if r < 1e-3:
        return 90.0, 0.0
    sin_a = R_PROJ / np.sqrt(r**2 + R_PROJ**2)
    alt   = np.degrees(np.arcsin(sin_a))
    az    = np.degrees(np.arctan2(-xkm, ykm)) % 360.0
    return float(alt), float(az)


def altaz_to_radec(alt_deg: float, az_deg: float, mjd: float) -> tuple[float, float]:
    t   = Time(mjd, format="mjd", scale="utc")
    aa  = AltAz(alt=alt_deg * u.deg, az=az_deg * u.deg,
                obstime=t, location=RUBIN_LOC)
    sky = SkyCoord(aa).icrs
    return float(sky.ra.deg), float(sky.dec.deg)


# ─────────────────────────────────────────────────────────────────────────────
# POINTING STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────

def strategy_absolute(grid: np.ndarray) -> tuple[int, int] | None:
    """
    Absolute minimum: point to the grid cell with lowest extinction.
    Returns (xi, yi) or None if grid is all NaN.
    """
    if np.all(np.isnan(grid)):
        return None
    idx = np.nanargmin(grid)
    yi, xi = np.unravel_index(idx, grid.shape)
    return int(xi), int(yi)


def strategy_motion(grids_history: list[np.ndarray],
                    lead: int = MOTION_LEAD) -> tuple[int, int] | None:
    """
    Motion-predicted minimum: find the clear-patch centroid in each recent
    frame, fit a linear velocity, extrapolate `lead` frames ahead, return
    the predicted minimum location.

    Falls back to absolute minimum if history is too short or fit fails.
    """
    if len(grids_history) < 2:
        return strategy_absolute(grids_history[-1])

    use = grids_history[-min(MOTION_HISTORY, len(grids_history)):]

    centroids = []
    for g in use:
        if np.all(np.isnan(g)):
            centroids.append(None)
            continue
        # Weight by inverse extinction (clearer = higher weight)
        w = np.where(np.isfinite(g), np.exp(-g), 0.0)
        total = w.sum()
        if total < 1e-9:
            centroids.append(None)
            continue
        ys, xs = np.meshgrid(np.arange(GRID_N), np.arange(GRID_N), indexing="ij")
        cx = float((w * xs).sum() / total)
        cy = float((w * ys).sum() / total)
        centroids.append((cx, cy))

    valid = [(i, c) for i, c in enumerate(centroids) if c is not None]
    if len(valid) < 2:
        return strategy_absolute(grids_history[-1])

    times = np.array([v[0] for v in valid], dtype=float)
    cxs   = np.array([v[1][0] for v in valid])
    cys   = np.array([v[1][1] for v in valid])

    # Linear velocity fit
    try:
        vx = np.polyfit(times, cxs, 1)[0]
        vy = np.polyfit(times, cys, 1)[0]
    except Exception:
        return strategy_absolute(grids_history[-1])

    # Predicted centroid `lead` steps ahead
    t_pred = times[-1] + lead
    px = cxs[-1] + vx * lead
    py = cys[-1] + vy * lead
    px = int(round(np.clip(px, 0, GRID_N - 1)))
    py = int(round(np.clip(py, 0, GRID_N - 1)))

    # Find actual minimum near predicted location (search radius = 3 cells)
    rad = 3
    x0, x1 = max(0, px - rad), min(GRID_N, px + rad + 1)
    y0, y1 = max(0, py - rad), min(GRID_N, py + rad + 1)
    subgrid = grids_history[-1][y0:y1, x0:x1]
    if np.all(np.isnan(subgrid)):
        return strategy_absolute(grids_history[-1])
    local_idx = np.nanargmin(subgrid)
    ly, lx    = np.unravel_index(local_idx, subgrid.shape)
    return int(x0 + lx), int(y0 + ly)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_night(date_str: str, csv_path: str,
                  max_frames: int | None = None,
                  out_dir: str = ".") -> str:
    """
    Process all DREAM cloud_sys frames for `date_str` (YYYY-MM-DD).
    Returns path to the saved .npz file.
    """
    night_tag = date_str.replace("-", "")
    print(f"\n{'='*60}")
    print(f"PART A  ―  SERVER EXTRACTION  —  {date_str}")
    print(f"{'='*60}")

    # ── Load and filter CSV ──────────────────────────────────────────────────
    print(f"\nLoading DREAM index: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.replace('"', '').str.strip()
    df = df.dropna(subset=[URL_COL]).copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
    df = df.dropna(subset=[TIME_COL])
    df = df[df[URL_COL].apply(is_cloud_sys)].copy()

    # Filter to the target night (local noon → next noon, UTC)
    # Cerro Pachón is UTC-4; astronomical night ~= UTC date + 1 noon transition
    night_start = pd.Timestamp(date_str, tz="UTC") + pd.Timedelta(hours=20)  # 20:00 UTC = 16:00 local
    night_end   = night_start + pd.Timedelta(hours=12)
    df = df[(df[TIME_COL] >= night_start) & (df[TIME_COL] < night_end)].copy()
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    print(f"  cloud_sys frames on {date_str}: {len(df)}")
    if len(df) == 0:
        raise RuntimeError(
            f"No cloud_sys frames found for {date_str}. "
            f"Check CSV path and date.")

    if max_frames and len(df) > max_frames:
        step = len(df) / max_frames
        idx  = [int(round(i * step)) for i in range(max_frames)]
        df   = df.iloc[idx].reset_index(drop=True)
        print(f"  Subsampled to {len(df)} frames (--max-frames {max_frames})")

    N = len(df)

    # Survey start = noon before the night (UTC)
    survey_start_mjd = Time(night_start - pd.Timedelta(hours=8)).mjd  # UTC noon

    # ── Allocate output arrays ───────────────────────────────────────────────
    mjds       = np.zeros(N)
    ext_grids  = np.full((N, GRID_N, GRID_N), np.nan)
    abs_altaz  = np.full((N, 2), np.nan)
    abs_radec  = np.full((N, 2), np.nan)
    abs_ext    = np.full(N, np.nan)
    mot_altaz  = np.full((N, 2), np.nan)
    mot_radec  = np.full((N, 2), np.nan)
    mot_ext    = np.full(N, np.nan)

    grids_history: list[np.ndarray] = []
    n_corrupt = 0
    n_ok      = 0

    print(f"\nProcessing {N} frames …")
    for fi, (_, row) in enumerate(df.iterrows()):
        url = str(row[URL_COL])
        mjd = Time(row[TIME_COL]).mjd
        mjds[fi] = mjd

        if (fi + 1) % 100 == 0 or fi == N - 1:
            print(f"  frame {fi+1:5d}/{N}  MJD={mjd:.5f}  "
                  f"ok={n_ok}  corrupt={n_corrupt}")

        clouds = load_cloud_sys(url)
        if clouds is None:
            n_corrupt += 1
            # Forward-fill last good grid
            if grids_history:
                ext_grids[fi] = ext_grids[fi - 1]
                grids_history.append(grids_history[-1])
            continue

        grid = healpix_to_cartesian(clouds, mjd)
        ext_grids[fi] = grid
        grids_history.append(grid)
        n_ok += 1

        # ── Absolute strategy ────────────────────────────────────────────────
        pt_a = strategy_absolute(grid)
        if pt_a is not None:
            xi_a, yi_a   = pt_a
            alt_a, az_a  = grid_to_altaz(xi_a, yi_a)
            ra_a,  dec_a = altaz_to_radec(alt_a, az_a, mjd)
            abs_altaz[fi] = [alt_a, az_a]
            abs_radec[fi] = [ra_a, dec_a]
            abs_ext[fi]   = float(grid[yi_a, xi_a])

        # ── Motion strategy ──────────────────────────────────────────────────
        pt_m = strategy_motion(grids_history)
        if pt_m is not None:
            xi_m, yi_m   = pt_m
            alt_m, az_m  = grid_to_altaz(xi_m, yi_m)
            ra_m,  dec_m = altaz_to_radec(alt_m, az_m, mjd)
            mot_altaz[fi] = [alt_m, az_m]
            mot_radec[fi] = [ra_m, dec_m]
            mot_ext[fi]   = float(grid[yi_m, xi_m])

    print(f"\n  Done — {n_ok} good frames, {n_corrupt} corrupt/missing")

    # ── Save .npz ────────────────────────────────────────────────────────────
    npz_path = os.path.join(out_dir, f"dream_night_{night_tag}.npz")
    np.savez_compressed(
        npz_path,
        night_key        = date_str,
        mjds             = mjds,
        ext_grids        = ext_grids.astype(np.float32),  # save space
        x_edges          = X_EDGES,
        y_edges          = Y_EDGES,
        abs_altaz        = abs_altaz,
        abs_radec        = abs_radec,
        abs_ext          = abs_ext,
        mot_altaz        = mot_altaz,
        mot_radec        = mot_radec,
        mot_ext          = mot_ext,
        rubin_lat        = RUBIN_LAT,
        rubin_lon        = RUBIN_LON,
        rubin_height_m   = RUBIN_HEIGHT_M,
        nside            = NSIDE,
        survey_start_mjd = survey_start_mjd,
    )
    print(f"\n  Saved: {npz_path}")
    return npz_path, mjds, ext_grids, abs_altaz, abs_radec, mot_altaz, mot_radec


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO
# ─────────────────────────────────────────────────────────────────────────────

def make_video(mjds, ext_grids, abs_altaz, abs_radec,
               mot_altaz, mot_radec, night_tag: str,
               out_dir: str = ".") -> str:
    """
    Render an MP4 showing the Cartesian cloud extinction map evolving
    through the night, with DREAM Absolute (green) and Motion (blue)
    pointings overlaid.
    """
    N = len(mjds)
    # Subsample for video
    if N > VIDEO_MAX_FRAMES:
        step   = N / VIDEO_MAX_FRAMES
        frames = [int(round(i * step)) for i in range(VIDEO_MAX_FRAMES)]
    else:
        frames = list(range(N))

    print(f"\nRendering video ({len(frames)} frames @ {VIDEO_FPS} fps) …")

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xlim(X_EDGES[0], X_EDGES[-1])
    ax.set_ylim(Y_EDGES[0], Y_EDGES[-1])
    ax.set_xlabel("East  [km]", fontsize=11)
    ax.set_ylabel("North  [km]", fontsize=11)
    ax.set_aspect("equal")

    # Horizon circle
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(R_PROJ * np.cos(th), R_PROJ * np.sin(th),
            "w-", lw=1, alpha=0.3, zorder=1)
    ax.plot(0, 0, "r+", ms=8, mew=2, zorder=5, label="Zenith")

    # Initial mesh
    xe = X_EDGES
    ye = Y_EDGES
    fi0 = frames[0]
    g0  = ext_grids[fi0]
    mesh = ax.pcolormesh(xe, ye, g0, cmap="viridis",
                          vmin=-0.2, vmax=2.0, shading="flat", zorder=0)
    cbar = fig.colorbar(mesh, ax=ax, label="Extinction (mag)")

    # Strategy markers
    pt_abs, = ax.plot([], [], "o", color="#2ca02c", ms=12, zorder=6,
                       label="DREAM Absolute", mew=1.5, mec="white")
    pt_mot, = ax.plot([], [], "D", color="#1f77b4", ms=10, zorder=6,
                       label="DREAM Motion",   mew=1.5, mec="white")
    # Trails
    trail_abs_x, trail_abs_y = [], []
    trail_mot_x, trail_mot_y = [], []
    TRAIL = 10
    line_abs, = ax.plot([], [], "-", color="#2ca02c", alpha=0.4, lw=1.5, zorder=5)
    line_mot, = ax.plot([], [], "-", color="#1f77b4", alpha=0.4, lw=1.5, zorder=5)

    title = ax.set_title("", fontsize=11, weight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)

    def _xy_from_altaz(alt_deg, az_deg):
        if np.isnan(alt_deg) or alt_deg < MIN_ALT_DEG:
            return np.nan, np.nan
        alt_r = np.radians(alt_deg)
        az_r  = np.radians(az_deg)
        s     = R_PROJ / np.sin(alt_r)
        x     = -np.cos(alt_r) * np.sin(az_r) * s
        y     =  np.cos(alt_r) * np.cos(az_r) * s
        return float(x), float(y)

    def update(frame_number):
        fi = frames[frame_number]
        g  = ext_grids[fi]
        mesh.set_array(g.ravel())

        t_str = Time(mjds[fi], format="mjd").strftime("%Y-%m-%d %H:%M UTC")
        title.set_text(f"DREAM cloud_sys  —  {t_str}  (frame {fi})")

        # Absolute
        alt_a, az_a = abs_altaz[fi]
        xa, ya = _xy_from_altaz(alt_a, az_a)
        if np.isfinite(xa):
            pt_abs.set_data([xa], [ya])
            trail_abs_x.append(xa); trail_abs_y.append(ya)
        trail_abs_x[:] = trail_abs_x[-TRAIL:]
        trail_abs_y[:] = trail_abs_y[-TRAIL:]
        line_abs.set_data(trail_abs_x, trail_abs_y)

        # Motion
        alt_m, az_m = mot_altaz[fi]
        xm, ym = _xy_from_altaz(alt_m, az_m)
        if np.isfinite(xm):
            pt_mot.set_data([xm], [ym])
            trail_mot_x.append(xm); trail_mot_y.append(ym)
        trail_mot_x[:] = trail_mot_x[-TRAIL:]
        trail_mot_y[:] = trail_mot_y[-TRAIL:]
        line_mot.set_data(trail_mot_x, trail_mot_y)

        return mesh, pt_abs, pt_mot, line_abs, line_mot, title

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames),
        interval=1000 / VIDEO_FPS, blit=True)

    out_dir  = os.path.abspath(out_dir) if out_dir else os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    vid_path = os.path.join(out_dir, f"dream_night_{night_tag}.mp4")
    writer   = animation.FFMpegWriter(fps=VIDEO_FPS, bitrate=1800,
                                       extra_args=["-vcodec", "libx264",
                                                   "-pix_fmt", "yuv420p"])
    ani.save(vid_path, writer=writer)
    plt.close(fig)
    print(f"  Saved: {vid_path}")
    return vid_path


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Extract DREAM cloud_sys frames for one night (server-side)")
    ap.add_argument("--date",       required=True,
                    help="Night date YYYY-MM-DD  (e.g. 2025-07-15)")
    ap.add_argument("--csv",        required=True,
                    help="Path to DREAM index CSV  (e.g. feb5_data.csv)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Cap total frames loaded (useful for quick tests)")
    ap.add_argument("--outdir",     default=".",
                    help="Directory for .npz and .mp4 outputs")
    ap.add_argument("--no-video",   action="store_true",
                    help="Skip video rendering")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)

    night_tag = a.date.replace("-", "")
    npz_path, mjds, ext_grids, abs_altaz, abs_radec, mot_altaz, mot_radec = \
        extract_night(a.date, a.csv, a.max_frames, a.outdir)

    if not a.no_video:
        try:
            make_video(mjds, ext_grids, abs_altaz, abs_radec,
                       mot_altaz, mot_radec, night_tag, a.outdir)
        except Exception as e:
            print(f"\n  Video failed (ffmpeg missing?): {e}")
            print("  Re-run with --no-video to skip, or install ffmpeg.")

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  NPZ  : {npz_path}")
    if not a.no_video:
        print(f"  MP4  : {a.outdir}/dream_night_{night_tag}.mp4")
    print(f"  → Copy NPZ to local machine and run dream_compare_local.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    import sys
    _jupyter = ("ipykernel" in sys.modules or
                any("ipykernel" in a for a in sys.argv))

    if _jupyter:
        # ── Edit these for your run ──────────────────────────────────────
        _DATE       = "2025-07-15"
        _CSV        = "feb5_data.csv"
        _OUTDIR     = "./"
        _MAX_FRAMES = None        # set e.g. 50 for a quick test
        _NO_VIDEO   = False
        # ────────────────────────────────────────────────────────────────

        os.makedirs(_OUTDIR, exist_ok=True)
        night_tag = _DATE.replace("-", "")

        npz_path, mjds, ext_grids, abs_altaz, abs_radec, mot_altaz, mot_radec = \
            extract_night(_DATE, _CSV, _MAX_FRAMES, _OUTDIR)

        if not _NO_VIDEO:
            try:
                make_video(mjds, ext_grids, abs_altaz, abs_radec,
                           mot_altaz, mot_radec, night_tag, _OUTDIR)
            except Exception as e:
                print(f"\n  Video failed (ffmpeg missing?): {e}")
                print("  Set _NO_VIDEO = True to skip.")
    else:
        main()
