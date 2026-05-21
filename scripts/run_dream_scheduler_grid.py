#!/usr/bin/env python3
"""
Run a full grid comparison of all DREAM nights against all scheduler nights.
Results are saved to a CSV with resume capability.
Usage:
  python run_dream_scheduler_grid.py --csv feb5_data.csv --db baseline_v5.1.0_10yrs.db --output results.csv --max-workers 8
  # test mode (small subset):
  python run_dream_scheduler_grid.py --test
"""

import os
import sys
import time
import argparse
import concurrent.futures
import numpy as np
import pandas as pd
import io
import sqlite3
import warnings
import h5py
import healpy as hp
from scipy.ndimage import gaussian_filter, shift
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_body
from astropy.time import Time
import astropy.units as u
from lsst.resources import ResourcePath
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =============================================================================
# Constants (from original pipeline)
# =============================================================================
NSIDE_EXPECTED = 32
NEST           = True
UNSEEN         = hp.UNSEEN if hasattr(hp, "UNSEEN") else np.nan

RUBIN_LAT      = -30.244639
RUBIN_LON      = -70.749417
RUBIN_HEIGHT_M = 2663.0

BIN_SIZE_KM  = 1000
R_PROJECTION = 10000.0

MIN_ALT_DEG = 15.0
MOON_EXCLUSION_DEG = 30.0

# Photometry
RUBIN_EFFECTIVE_AREA     = np.pi * (6.4 / 2) ** 2
RUBIN_QUANTUM_EFFICIENCY = 0.8
EXPOSURE_TIME            = 30.0
READOUT_TIME             = 2.0
SLOT_TIME                = EXPOSURE_TIME + READOUT_TIME
PHOTON_FLUX_MAG20_ZENITH = 100.0

# Slew
MAX_SLEW_SPEED_ALT = 3.5
MAX_SLEW_SPEED_AZ  = 7.0
SLEW_SETTLE_TIME   = 2.0

# Pixel quality
MAX_SIGMA_MAG  = 0.3
MAX_FLAG_VALUE = 0

# Scoring weights
W_CLOUD  = 0.50
W_SLEW   = 0.25
W_ZENITH = 0.25
MAX_EXTINCTION_NORM = 2.0
MAX_SLEW_NORM       = 60.0

# URLs
URL_COL  = "lsst.sal.DREAM.logevent_largeFileObjectAvailable.url"
TIME_COL = "time"

# =============================================================================
# Helper functions
# =============================================================================
def transform_url(url: str) -> str:
    url = str(url).strip()
    if url.startswith("https://s3.cp.lsst.org/"):
        return url.replace("https://s3.cp.lsst.org/", "s3://lfa@")
    return url

def _make_location():
    return EarthLocation(lat=RUBIN_LAT*u.deg, lon=RUBIN_LON*u.deg, height=RUBIN_HEIGHT_M*u.m)

def _ensure_time(t):
    if not isinstance(t, Time):
        t = Time(t)
    return t.utc

def fetch_sys_map(url: str):
    rp = ResourcePath(url)
    with rp.open("rb") as fd:
        data = fd.read()
    with h5py.File(io.BytesIO(data), "r") as f:
        clouds = np.array(f["clouds"], dtype=float).ravel()
        sigma  = np.array(f["sigma"], dtype=float).ravel()
        flags  = np.array(f["flags"], dtype=int).ravel()
        mask_visible = np.array(f["mask_visible"], dtype=bool).ravel()
        nobs   = np.array(f["nobs"], dtype=int).ravel()
    bad = (~mask_visible | (nobs == 0) | (flags > MAX_FLAG_VALUE) | (sigma > MAX_SIGMA_MAG) | ~np.isfinite(clouds))
    clouds[bad] = np.nan
    sigma[bad] = np.nan
    return clouds, sigma

def radec_to_altaz(ra_deg, dec_deg, obstime):
    t = _ensure_time(obstime)
    loc = _make_location()
    sky = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame="icrs")
    aa = sky.transform_to(AltAz(obstime=t, location=loc))
    return float(aa.alt.deg), float(aa.az.deg % 360.0)

def altaz_to_xy(alt_deg, az_deg):
    if alt_deg < MIN_ALT_DEG:
        return None, None
    alt_r = np.radians(alt_deg)
    az_r  = np.radians(az_deg)
    scale = R_PROJECTION / np.sin(alt_r)
    x_km = -np.cos(alt_r) * np.sin(az_r) * scale
    y_km =  np.cos(alt_r) * np.cos(az_r) * scale
    return x_km, y_km

def xy_to_altaz(x_km, y_km):
    r = np.sqrt(x_km**2 + y_km**2)
    alt_deg = float(np.degrees(np.arctan2(R_PROJECTION, r)))
    az_deg  = float(np.degrees(np.arctan2(-x_km, y_km)) % 360.0)
    return alt_deg, az_deg

def xy_to_zenith_angle(x_km, y_km):
    alt_deg, _ = xy_to_altaz(x_km, y_km)
    return 90.0 - alt_deg

def radec_to_xy(ra_deg, dec_deg, obstime):
    alt, az = radec_to_altaz(ra_deg, dec_deg, obstime)
    return altaz_to_xy(alt, az) + (alt, az)

def healpix_to_altaz_vals(mp, nside, obstime):
    npix = hp.nside2npix(nside)
    pix = np.arange(npix)
    theta, phi = hp.pix2ang(nside, pix, nest=NEST)
    ra  = np.degrees(phi)
    dec = 90.0 - np.degrees(theta)
    t   = _ensure_time(obstime)
    loc = _make_location()
    sky = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs")
    aa  = sky.transform_to(AltAz(obstime=t, location=loc))
    vals = np.asarray(mp, dtype=float)
    vals = np.where(vals == UNSEEN, np.nan, vals)
    return aa.az.deg % 360.0, aa.alt.deg, vals

def _healpix_to_grid(mp, obstime, bin_size=BIN_SIZE_KM):
    az, alt, vals = healpix_to_altaz_vals(mp, NSIDE_EXPECTED, obstime)
    alt_r = np.radians(alt)
    az_r  = np.radians(az)
    above = alt > MIN_ALT_DEG
    scale = np.where(above, R_PROJECTION / np.sin(alt_r), np.nan)
    xf = -np.cos(alt_r) * np.sin(az_r) * scale
    yf =  np.cos(alt_r) * np.cos(az_r) * scale
    vf = np.where(above, vals, np.nan)

    r = np.sqrt(xf**2 + yf**2)
    c = r <= 15000.0
    ok = ~np.isnan(vf[c])

    x_edges = np.arange(-15000, 15001, bin_size)
    y_edges = np.arange(-15000, 15001, bin_size)
    Hs, _, _ = np.histogram2d(xf[c][ok], yf[c][ok], bins=[x_edges, y_edges], weights=vf[c][ok])
    Hc, _, _ = np.histogram2d(xf[c][ok], yf[c][ok], bins=[x_edges, y_edges])
    with np.errstate(divide="ignore", invalid="ignore"):
        H = Hs / Hc
    H[Hc == 0] = np.nan
    H = H.T
    xc = (x_edges[:-1] + x_edges[1:]) / 2
    yc = (y_edges[:-1] + y_edges[1:]) / 2
    Xg, Yg = np.meshgrid(xc, yc)
    H[np.sqrt(Xg**2 + Yg**2) > 15000] = np.nan
    return H, Xg, Yg, x_edges, y_edges

def process_to_grid(clouds, sigma, obstime, bin_size=BIN_SIZE_KM):
    ext_grid, Xg, Yg, xe, ye = _healpix_to_grid(clouds, obstime, bin_size)
    return ext_grid, Xg, Yg, xe, ye

def get_value_at_position(grid, x_km, y_km):
    xi = int(round((x_km / BIN_SIZE_KM) + 15))
    yi = int(round((y_km / BIN_SIZE_KM) + 15))
    if 0 <= xi < grid.shape[1] and 0 <= yi < grid.shape[0]:
        return grid[yi, xi]
    return np.nan

def get_moon_altaz(obstime):
    t = _ensure_time(obstime)
    loc = _make_location()
    moon = get_body("moon", t, loc)
    aa = moon.transform_to(AltAz(obstime=t, location=loc))
    return float(aa.alt.deg), float(aa.az.deg % 360.0)

def angular_separation_altaz(alt1, az1, alt2, az2):
    a1, a2 = np.radians(alt1), np.radians(alt2)
    z1, z2 = np.radians(az1), np.radians(az2)
    cos_sep = np.sin(a1)*np.sin(a2) + np.cos(a1)*np.cos(a2)*np.cos(z1 - z2)
    cos_sep = np.clip(cos_sep, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_sep)))

def is_moon_safe(alt_deg, az_deg, moon_alt, moon_az, exclusion_deg=MOON_EXCLUSION_DEG):
    if alt_deg < MIN_ALT_DEG:
        return False
    sep = angular_separation_altaz(alt_deg, az_deg, moon_alt, moon_az)
    return sep >= exclusion_deg

def xy_moon_safe(x_km, y_km, moon_alt, moon_az, exclusion_deg=MOON_EXCLUSION_DEG):
    alt, az = xy_to_altaz(x_km, y_km)
    return is_moon_safe(alt, az, moon_alt, moon_az, exclusion_deg)

def find_absolute_minimum(grid, moon_alt=None, moon_az=None):
    work = grid.copy()
    if moon_alt is not None and moon_az is not None:
        ny, nx = work.shape
        for yi in range(ny):
            for xi in range(nx):
                if np.isnan(work[yi, xi]):
                    continue
                x_km = (xi - 15) * BIN_SIZE_KM
                y_km = (yi - 15) * BIN_SIZE_KM
                if not xy_moon_safe(x_km, y_km, moon_alt, moon_az):
                    work[yi, xi] = np.nan
    if not np.any(~np.isnan(work)):
        work = grid.copy()
    if not np.any(~np.isnan(work)):
        return 0, 0, np.nan
    idx = np.nanargmin(work)
    yi, xi = np.unravel_index(idx, work.shape)
    return (xi - 15) * BIN_SIZE_KM, (yi - 15) * BIN_SIZE_KM, grid[yi, xi]

def compute_motion_with_correlation(grid1, grid2, sigma=5.0, search_range=10):
    g1 = np.nan_to_num(grid1, nan=0)
    g2 = np.nan_to_num(grid2, nan=0)
    g1s = gaussian_filter(g1, sigma=sigma)
    g2s = gaussian_filter(g2, sigma=sigma)
    m1 = ~np.isnan(grid1) & (grid1 != 0)
    m2 = ~np.isnan(grid2) & (grid2 != 0)
    best_corr, best_dx, best_dy = -np.inf, 0, 0
    for dy in range(-search_range, search_range + 1):
        for dx in range(-search_range, search_range + 1):
            sh = shift(g2s, (dy, dx), order=1, mode="constant", cval=0)
            sm = shift(m2.astype(float), (dy, dx), order=0, mode="constant", cval=0) > 0.5
            val = m1 & sm
            if np.sum(val) < 100:
                continue
            v1, v2 = g1s[val], sh[val]
            if np.std(v1) > 0 and np.std(v2) > 0:
                corr = np.corrcoef(v1, v2)[0, 1]
                if corr > best_corr:
                    best_corr, best_dx, best_dy = corr, dx, dy
    return best_dx, best_dy, best_corr

def _score_candidate(x_km, y_km, ext_val, prev_alt, prev_az, moon_alt=None, moon_az=None):
    if np.isnan(ext_val):
        return np.inf
    alt, az = xy_to_altaz(x_km, y_km)
    if alt < MIN_ALT_DEG:
        return np.inf
    if moon_alt is not None and moon_az is not None:
        if not is_moon_safe(alt, az, moon_alt, moon_az):
            return np.inf
    cloud_norm = np.clip(ext_val / MAX_EXTINCTION_NORM, 0.0, 1.0)
    if prev_alt is not None:
        slew_s = calculate_slew_time(prev_alt, prev_az, alt, az)
    else:
        slew_s = 0.0
    slew_norm = np.clip(slew_s / MAX_SLEW_NORM, 0.0, 1.0)
    zenith_norm = (90.0 - alt) / 90.0
    return W_CLOUD * cloud_norm + W_SLEW * slew_norm + W_ZENITH * zenith_norm

def predict_future_position(cx, cy, dx_pix, dy_pix, current_grid,
                             prev_alt=None, prev_az=None,
                             moon_alt=None, moon_az=None,
                             fallback_threshold=0.5, n_candidates=9):
    nx = cx - dx_pix * BIN_SIZE_KM
    ny = cy - dy_pix * BIN_SIZE_KM
    r = np.sqrt(nx**2 + ny**2)
    if r > 14000:
        nx *= 14000 / r
        ny *= 14000 / r
    offsets = [0] + list(range(-2, 3))
    candidates = []
    for dyi in offsets:
        for dxi in offsets:
            cx_ = nx + dxi * BIN_SIZE_KM
            cy_ = ny + dyi * BIN_SIZE_KM
            ext = get_value_at_position(current_grid, cx_, cy_)
            candidates.append((cx_, cy_, ext))
    best_score = np.inf
    best_x, best_y = nx, ny
    for (cx_, cy_, ext) in candidates:
        s = _score_candidate(cx_, cy_, ext, prev_alt, prev_az, moon_alt, moon_az)
        if s < best_score:
            best_score = s
            best_x, best_y = cx_, cy_
    if best_score == np.inf:
        best_x, best_y, _ = find_absolute_minimum(current_grid, moon_alt, moon_az)
    return best_x, best_y

def calculate_slew_time(alt1, az1, alt2, az2):
    da = abs(alt2 - alt1)
    daz = abs(az2 - az1)
    if daz > 180:
        daz = 360 - daz
    return max(da / MAX_SLEW_SPEED_ALT, daz / MAX_SLEW_SPEED_AZ) + SLEW_SETTLE_TIME

def compute_photon_collection(ext_mag, zenith_angle_deg=None):
    if np.isnan(ext_mag):
        return np.nan
    flux = 10 ** (-0.4 * ext_mag)
    rate = PHOTON_FLUX_MAG20_ZENITH * RUBIN_EFFECTIVE_AREA * RUBIN_QUANTUM_EFFICIENCY * flux
    return rate * EXPOSURE_TIME

def photons_to_magnitude(photons):
    if photons is None or np.isnan(photons) or photons <= 0:
        return np.nan
    snr = photons / (5 * np.sqrt(photons))
    return 20.0 - 2.5 * np.log10(snr) if snr > 0 else np.nan

# =============================================================================
# Data loading and preprocessing
# =============================================================================
def load_all_sys_frames(csv_file):
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.replace('"', "").str.strip()
    df = df.dropna(subset=[URL_COL]).copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
    df = df.dropna(subset=[TIME_COL]).copy()
    df = df[df[URL_COL].str.contains(".hdf5", case=False, na=False)].copy()
    df = df[df[URL_COL].str.contains("cloud_sys", case=False, na=False)].copy()
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    shifted = df[TIME_COL] - pd.Timedelta(hours=12)
    df["night_key"] = shifted.dt.date
    return df

def get_night_df(all_df, night_key):
    return all_df[all_df["night_key"] == night_key].copy().reset_index(drop=True)

def preprocess_dream_night(df_night):
    """Load all frames for a night, compute grids, absolute and motion pointings."""
    all_grids = []
    all_metas = []
    for _, row in df_night.iterrows():
        url = transform_url(row[URL_COL])
        try:
            clouds, sigma = fetch_sys_map(url)
            meta = {"time": row[TIME_COL].to_pydatetime()}
            ext_g, _, _, _, _ = process_to_grid(clouds, sigma, meta["time"])
            all_grids.append(ext_g)
            all_metas.append(meta)
        except Exception as e:
            print(f"  Frame load failed: {e}")
            continue
    if len(all_grids) < 2:
        return None

    # Moon positions
    moon_pos = [get_moon_altaz(m["time"]) for m in all_metas]

    # Absolute minimum positions
    abs_pos = []
    abs_ext = []
    for i, grid in enumerate(all_grids):
        x, y, ext = find_absolute_minimum(grid, moon_pos[i][0], moon_pos[i][1])
        abs_pos.append((x, y))
        abs_ext.append(ext)

    # Motion tracking positions
    motion_pos = []
    motion_ext = []
    xp, yp = abs_pos[0]
    prev_alt, prev_az = xy_to_altaz(xp, yp)
    motion_pos.append((xp, yp))
    motion_ext.append(abs_ext[0])
    for i in range(1, len(all_grids)):
        if i >= 3:
            dxs, dys = [], []
            for j in range(1, 4):
                dx, dy, conf = compute_motion_with_correlation(all_grids[i-j], all_grids[i-j+1])
                if conf > 0.5:
                    dxs.append(dx); dys.append(dy)
            dx_avg = float(np.mean(dxs)) if dxs else 0.0
            dy_avg = float(np.mean(dys)) if dys else 0.0
        else:
            dx_avg, dy_avg, _ = compute_motion_with_correlation(all_grids[i-1], all_grids[i])
        xp, yp = predict_future_position(xp, yp, dx_avg, dy_avg, all_grids[i],
                                          prev_alt, prev_az,
                                          moon_pos[i][0], moon_pos[i][1])
        motion_pos.append((xp, yp))
        ext = get_value_at_position(all_grids[i], xp, yp)
        motion_ext.append(ext)
        prev_alt, prev_az = xy_to_altaz(xp, yp)

    # Build per-frame data
    frames = []
    for i in range(len(all_grids)):
        t = all_metas[i]["time"]
        alt_abs, az_abs = xy_to_altaz(abs_pos[i][0], abs_pos[i][1])
        alt_mot, az_mot = xy_to_altaz(motion_pos[i][0], motion_pos[i][1])
        frames.append({
            "frame_idx": i,
            "time": t,
            "grid": all_grids[i],
            "abs_x": abs_pos[i][0], "abs_y": abs_pos[i][1], "abs_ext": abs_ext[i],
            "abs_alt": alt_abs, "abs_az": az_abs,
            "mot_x": motion_pos[i][0], "mot_y": motion_pos[i][1], "mot_ext": motion_ext[i],
            "mot_alt": alt_mot, "mot_az": az_mot,
            "moon_alt": moon_pos[i][0], "moon_az": moon_pos[i][1],
        })
    return frames

def load_scheduler_nights(db_file):
    """Load all scheduler nights from SQLite DB."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]
    table_name = next((n for n in ["observations", "SummaryAllProps", "Summary", "obs"] if n in tables), tables[0] if tables else None)
    if table_name is None:
        raise RuntimeError("No tables found in scheduler DB")
    obs_df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()

    night_col = next((c for c in obs_df.columns if "night" in c.lower()), None)
    if night_col is None:
        raise RuntimeError("No night column found")
    obs_df["night"] = obs_df[night_col].astype(int)

    # Ensure required columns
    for req in ["observationStartMJD", "fieldRA", "fieldDec"]:
        found = False
        for col in obs_df.columns:
            if req.lower() in col.lower():
                if col != req:
                    obs_df[req] = obs_df[col]
                found = True
                break
        if not found:
            raise RuntimeError(f"Required column {req} not found")

    nights = {}
    for night, group in obs_df.groupby("night"):
        obs_list = []
        for _, row in group.iterrows():
            mjd = row["observationStartMJD"]
            t = Time(mjd, format="mjd").datetime
            obs_list.append({
                "time": t,
                "mjd": mjd,
                "ra": float(row["fieldRA"]),
                "dec": float(row["fieldDec"]),
                "band": row.get("band", "?")
            })
        nights[night] = obs_list
    return nights

def match_scheduler_to_frames(scheduler_obs, frame_times):
    """Return for each frame the closest scheduler observation by hour-of-day."""
    if not scheduler_obs:
        return [None] * len(frame_times)
    frame_frac = [t.hour + t.minute/60.0 + t.second/3600.0 for t in frame_times]
    obs_frac = [o["time"].hour + o["time"].minute/60.0 + o["time"].second/3600.0 for o in scheduler_obs]
    matched = []
    for ff in frame_frac:
        diffs = [abs(ff - of) for of in obs_frac]
        best_idx = np.argmin(diffs)
        matched.append(scheduler_obs[best_idx])
    return matched

# =============================================================================
# Metrics computation for one combination
# =============================================================================
def compute_combination_metrics(dream_frames, scheduler_obs):
    n_frames = len(dream_frames)
    if n_frames == 0:
        return None

    # Match scheduler to frames
    frame_times = [f["time"] for f in dream_frames]
    matched = match_scheduler_to_frames(scheduler_obs, frame_times)

    # Compute scheduler pointing extinction and alt/az for each frame
    sched_x = []
    sched_y = []
    sched_ext = []
    sched_alt = []
    sched_az = []
    for i, obs in enumerate(matched):
        if obs is None:
            sched_ext.append(np.nan)
            sched_x.append(np.nan)
            sched_y.append(np.nan)
            sched_alt.append(np.nan)
            sched_az.append(np.nan)
            continue
        x, y, alt, az = radec_to_xy(obs["ra"], obs["dec"], frame_times[i])
        if x is None:
            sched_ext.append(np.nan)
            sched_x.append(np.nan)
            sched_y.append(np.nan)
            sched_alt.append(np.nan)
            sched_az.append(np.nan)
            continue
        ext = get_value_at_position(dream_frames[i]["grid"], x, y)
        sched_ext.append(ext)
        sched_x.append(x)
        sched_y.append(y)
        sched_alt.append(alt)
        sched_az.append(az)

    # Helper to compute slew‑gated photons for a sequence
    def compute_photons_for_sequence(pos_x, pos_y, ext_list):
        n = len(pos_x)
        if n == 0:
            return 0.0, []
        prev_alt, prev_az = xy_to_altaz(pos_x[0], pos_y[0])
        total_photons = 0.0
        per_frame = []
        for i in range(n):
            if np.isnan(ext_list[i]) or np.isnan(pos_x[i]):
                per_frame.append(np.nan)
                continue
            alt, az = xy_to_altaz(pos_x[i], pos_y[i])
            if alt < MIN_ALT_DEG:
                per_frame.append(np.nan)
                continue
            if i == 0:
                slew = 0.0
            else:
                slew = calculate_slew_time(prev_alt, prev_az, alt, az)
            expose_t = max(0.0, SLOT_TIME - slew)
            photons_full = compute_photon_collection(ext_list[i])
            collected = photons_full * (expose_t / EXPOSURE_TIME)
            total_photons += collected
            per_frame.append(collected)
            prev_alt, prev_az = alt, az
        return total_photons, per_frame

    # Absolute
    abs_x = [f["abs_x"] for f in dream_frames]
    abs_y = [f["abs_y"] for f in dream_frames]
    abs_ext = [f["abs_ext"] for f in dream_frames]
    total_abs, _ = compute_photons_for_sequence(abs_x, abs_y, abs_ext)

    # Motion
    mot_x = [f["mot_x"] for f in dream_frames]
    mot_y = [f["mot_y"] for f in dream_frames]
    mot_ext = [f["mot_ext"] for f in dream_frames]
    total_mot, _ = compute_photons_for_sequence(mot_x, mot_y, mot_ext)

    # Scheduler
    total_sched, _ = compute_photons_for_sequence(sched_x, sched_y, sched_ext)

    # Aggregate statistics
    result = {
        "total_photons_abs": total_abs,
        "total_photons_motion": total_mot,
        "total_photons_scheduler": total_sched,
        "mean_abs_ext": np.nanmean(abs_ext),
        "mean_mot_ext": np.nanmean(mot_ext),
        "mean_sched_ext": np.nanmean(sched_ext),
        "mean_abs_zenith": np.nanmean([xy_to_zenith_angle(f["abs_x"], f["abs_y"]) for f in dream_frames]),
        "mean_mot_zenith": np.nanmean([xy_to_zenith_angle(f["mot_x"], f["mot_y"]) for f in dream_frames]),
        "mean_sched_zenith": np.nanmean([xy_to_zenith_angle(sched_x[i], sched_y[i]) for i in range(n_frames) if not np.isnan(sched_x[i])]),
        "n_frames": n_frames,
        "n_valid_sched": np.sum(~np.isnan(sched_ext)),
    }
    return result

# =============================================================================
# Main driver
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Grid search over DREAM nights × scheduler nights.")
    parser.add_argument("--csv", default="feb5_data.csv", help="CSV file with DREAM data")
    parser.add_argument("--db", default="baseline_v5.1.0_10yrs.db", help="SQLite scheduler database")
    parser.add_argument("--output", default="full_combination_metrics.csv", help="Output CSV file")
    parser.add_argument("--max-workers", type=int, default=8, help="Number of parallel processes")
    parser.add_argument("--test", action="store_true", help="Run test mode with a few nights only")
    args = parser.parse_args()

    print("="*80)
    print("DREAM vs Scheduler Grid Analysis")
    print("="*80)
    print(f"CSV:          {args.csv}")
    print(f"Scheduler DB: {args.db}")
    print(f"Output:       {args.output}")
    print(f"Max workers:  {args.max_workers}")
    print(f"Test mode:    {args.test}")

    # Load DREAM data
    print("\nLoading DREAM data...")
    all_dream_df = load_all_sys_frames(args.csv)
    all_dream_nights = sorted(all_dream_df["night_key"].unique())
    print(f"Found {len(all_dream_nights)} DREAM nights")

    # Preprocess DREAM nights
    print("\nPreprocessing DREAM nights...")
    dream_data = {}
    for night in tqdm(all_dream_nights, desc="Preprocessing"):
        df_night = get_night_df(all_dream_df, night)
        frames = preprocess_dream_night(df_night)
        if frames is not None and len(frames) >= 2:
            dream_data[night] = frames
    print(f"Loaded {len(dream_data)} usable DREAM nights")

    # Load scheduler nights
    print("\nLoading scheduler nights...")
    all_sched_nights = load_scheduler_nights(args.db)
    sched_night_list = sorted(all_sched_nights.keys())
    print(f"Found {len(sched_night_list)} scheduler nights")

    # Build list of all combinations
    combinations = []
    for dream_night, frames in dream_data.items():
        for sched_night in sched_night_list:
            sched_obs = all_sched_nights.get(sched_night, [])
            if sched_obs:
                combinations.append((dream_night, frames, sched_night, sched_obs))

    # If test mode, reduce the number of combinations
    if args.test:
        # Use first 2 dream nights and first 2 scheduler nights
        test_dream_nights = list(dream_data.keys())[:2]
        test_sched_nights = sched_night_list[:2]
        combinations = [(dn, dream_data[dn], sn, all_sched_nights[sn])
                        for dn in test_dream_nights
                        for sn in test_sched_nights
                        if sn in all_sched_nights]
        print(f"\nTEST MODE: using {len(test_dream_nights)} dream nights and {len(test_sched_nights)} scheduler nights -> {len(combinations)} combos")

    print(f"\nTotal combinations to process: {len(combinations)}")
    if len(combinations) == 0:
        print("No valid combinations. Exiting.")
        return

    # Resume: load existing output, skip already processed combos
    processed = set()
    if os.path.exists(args.output):
        try:
            existing = pd.read_csv(args.output)
            for _, row in existing.iterrows():
                processed.add((row["dream_night"], row["scheduler_night"]))
            print(f"Loaded existing output: {len(processed)} combinations already done.")
        except Exception as e:
            print(f"Could not read existing output: {e}")

    # Filter combinations to those not processed
    remaining = []
    for dn, frames, sn, sched_obs in combinations:
        if (str(dn), sn) not in processed:
            remaining.append((dn, frames, sn, sched_obs))
    print(f"Remaining to process: {len(remaining)}")
    if not remaining:
        print("All combinations already processed. Exiting.")
        return

    # Process remaining combinations in parallel
    start_time = time.time()
    new_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_combo = {executor.submit(compute_combination_metrics, frames, sched_obs): (dn, sn)
                           for dn, frames, sn, sched_obs in remaining}
        for future in tqdm(concurrent.futures.as_completed(future_to_combo), total=len(remaining), desc="Processing combos"):
            dn, sn = future_to_combo[future]
            try:
                metrics = future.result()
                if metrics is not None:
                    new_results.append({
                        "dream_night": str(dn),
                        "scheduler_night": sn,
                        **metrics
                    })
            except Exception as e:
                print(f"Error processing ({dn}, {sn}): {e}")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

    # Append new results to output CSV
    if new_results:
        df_new = pd.DataFrame(new_results)
        # Reorder columns
        cols = ["dream_night", "scheduler_night"] + [c for c in df_new.columns if c not in ["dream_night", "scheduler_night"]]
        df_new = df_new[cols]
        # Append or create file
        if os.path.exists(args.output):
            df_old = pd.read_csv(args.output)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            df_combined.to_csv(args.output, index=False)
        else:
            df_new.to_csv(args.output, index=False)
        print(f"Saved {len(new_results)} new rows to {args.output}")
    else:
        print("No new results to save.")

if __name__ == "__main__":
    main()