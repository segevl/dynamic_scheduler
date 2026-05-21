"""
dream_compare_local.py  ―  PART B  (local machine with rubin_scheduler)
========================================================================
Loads the .npz produced by dream_extract_server.py (Part A), runs the
rubin_scheduler greedy simulation for the same night, then overlays
those scheduler pointings onto the saved DREAM cloud grids and computes
a full metric comparison.

Three strategies compared
--------------------------
  DREAM Absolute  – always slew to current extinction minimum
  DREAM Motion    – follow the motion-predicted clear patch
  Greedy Sched    – what rubin_scheduler would have pointed at
                    (simulated under ideal/clear sky, then photons
                     evaluated against the real DREAM cloud data)

Key fixes in this version
--------------------------
  1. SLOT COUNTING  — DREAM strategies subsample cloud frames to one
     decision per SLOT_TIME (32 s), matching the telescope cadence.
     Raw cloud data arrives at ~1 Hz; we pick one frame per slot, not
     one slot per frame.  Previously DREAM had ~14 900 "slots" vs
     ~877 for the scheduler because it was iterating over raw frames.

  2. NIGHT ALIGNMENT  — both DREAM and Greedy strategies are evaluated
     over exactly the same MJD window (the scheduler's own obs span),
     so slot budgets are directly comparable.

  3. DEPTH CALIBRATION  — anchored to the Rubin SRD r-band 5σ depth
     of 23.4 mag for a clear 30-s exposure (matching rubin_scheduler's
     own fivesigmadepth).  Previously the formula
       snr = ph / (5*sqrt(ph))  →  depth = 20 - 2.5*log10(snr)
     was self-referential and gave ~15.7 mag, ~7-8 mag too shallow.
     New formula:
       m5 = ZP_band - ext_mag - 1.25*log10(t_expose / 30)
     where ZP_band is the clear-sky 30-s 5σ depth from the SRD.
     For the greedy strategy, rubin_scheduler's own fivesigmadepth
     field is used directly, then degraded by the real cloud extinction.

  4. SLEW GATING  — mount slews during readout (2 s) for free; only
     the excess beyond readout cuts into the 30-s exposure window.
     Previously slew time was subtracted from a combined slot budget
     rather than from the exposure time alone.

  5. GREEDY SLEW TIME  — scheduler's own 'slewtime' field is used
     where available instead of recomputing from alt/az deltas.

Requirements (local only)
--------------------------
    pip install rubin-scheduler healpy astropy numpy scipy matplotlib pandas

Quick start
-----------
    python dream_compare_local.py --npz dream_night.npz [--band r]

    # or from a Jupyter cell:
    from dream_compare_local import compare
    compare("dream_night.npz")

Outputs  (written to ./outputs/ by default)
-------------------------------------------
    comparison_<NIGHT>.png
    sky_coverage_<NIGHT>.png
    healpix_<NIGHT>.png
    cloud_maps_<NIGHT>.png
    cumulative_<NIGHT>.png
    depth_dist_<NIGHT>.png
    summary_<NIGHT>.csv
"""

from __future__ import annotations
import argparse
import os
import warnings
import numpy as np
import pandas as pd
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u

warnings.filterwarnings("ignore")

import rubin_scheduler.scheduler.basis_functions as bf
import rubin_scheduler.scheduler.detailers as detailers
from rubin_scheduler.scheduler.surveys import GreedySurvey
from rubin_scheduler.scheduler.utils import CurrentAreaMap, make_rolling_footprints
from rubin_scheduler.site_models import Almanac, CloudMap
from rubin_scheduler.scheduler.schedulers import CoreScheduler
from rubin_scheduler.scheduler.model_observatory import ModelObservatory
from rubin_scheduler.scheduler import sim_runner

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

EXPOSURE_TIME = 30.0          # s
READOUT_TIME  = 2.0           # s
SLOT_TIME     = EXPOSURE_TIME + READOUT_TIME   # 32 s

# Rubin SRD 5σ depth for a single 30-s clear-sky visit, per band.
# These match rubin_scheduler's typical fivesigmadepth output at zenith
# with median seeing.  Source: LSST SRD Table 1 / OpSim documentation.
BAND_ZP = {
    "u": 22.7,
    "g": 23.4,
    "r": 23.4,
    "i": 23.1,
    "z": 22.7,
    "y": 21.9,
}
DEFAULT_BAND = "r"

# Photon count for a mag-20 AB source in r-band through Rubin in 30 s.
# Used for the *relative* photon metric (all strategies share the same
# reference source so the comparison is fair).
# N = F0 * area * QE * t
#   F0(r, m=20, AB) ≈ 5.52e9 * 10^(-0.4*20) photons/s/m²
#                   ≈ 5.52e9 * 1e-8 = 55.2 photons/s/m²
#   area  = 35.04 m²  (Rubin effective aperture from SRD)
#   QE    = 0.9
#   t     = 30 s
RUBIN_AREA_M2   = 35.04
QE              = 0.9
F0_R_PHOTONS    = 5.52e9       # ph/s/m² for AB mag-0 in r-band
N_MAG20_30S     = (F0_R_PHOTONS * 10**(-0.4 * 20)
                   * RUBIN_AREA_M2 * QE * EXPOSURE_TIME)   # ≈ 5.97e4 ph

# Slew performance
MAX_SLEW_SPEED_ALT = 3.5   # deg/s
MAX_SLEW_SPEED_AZ  = 7.0   # deg/s
SLEW_SETTLE_TIME   = 2.0   # s

MIN_ALT_DEG  = 15.0
BIN_SIZE_KM  = 1000
R_PROJECTION = 10_000.0

CAMERA_ROT_LIMITS = (-80.0, 80.0)
SCIENCE_PROGRAM   = "BLOCK-407"

COLORS = {
    "DREAM Absolute": "#2ca02c",
    "DREAM Motion":   "#1f77b4",
    "Greedy Sched":   "#d62728",
}


# ─────────────────────────────────────────────────────────────────────────────
# LOAD .NPZ
# ─────────────────────────────────────────────────────────────────────────────

def load_npz(path: str) -> dict:
    raw = np.load(path, allow_pickle=True)
    d   = {k: raw[k] for k in raw.files}
    for k in ("night_key", "rubin_lat", "rubin_lon", "rubin_height_m",
              "nside", "survey_start_mjd"):
        if k in d and d[k].ndim == 0:
            d[k] = d[k].item()
    if "rubin_height_m" not in d:
        d["rubin_height_m"] = 2663.0
    print(f"  Loaded: {path}")
    print(f"    night          : {d['night_key']}")
    print(f"    frames         : {len(d['mjds'])}")
    print(f"    survey_start   : MJD {d['survey_start_mjd']:.4f}")
    print(f"    grid shape     : {d['ext_grids'].shape}")
    return d


# ─────────────────────────────────────────────────────────────────────────────
# COORDINATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _loc(d: dict) -> EarthLocation:
    return EarthLocation(lat=d["rubin_lat"] * u.deg,
                         lon=d["rubin_lon"] * u.deg,
                         height=d["rubin_height_m"] * u.m)


def radec_to_altaz(ra_deg, dec_deg, mjd, loc) -> tuple[float, float]:
    t   = Time(mjd, format="mjd", scale="utc")
    sky = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    aa  = sky.transform_to(AltAz(obstime=t, location=loc))
    return float(aa.alt.deg), float(aa.az.deg % 360.0)


def altaz_to_xy(alt_deg, az_deg):
    if alt_deg < MIN_ALT_DEG:
        return None, None
    alt_r = np.radians(alt_deg)
    az_r  = np.radians(az_deg)
    s     = R_PROJECTION / np.sin(alt_r)
    return -np.cos(alt_r) * np.sin(az_r) * s, np.cos(alt_r) * np.cos(az_r) * s


def get_val(grid, x_km, y_km) -> float:
    xi = int(round(x_km / BIN_SIZE_KM + 15))
    yi = int(round(y_km / BIN_SIZE_KM + 15))
    if 0 <= xi < grid.shape[1] and 0 <= yi < grid.shape[0]:
        return float(grid[yi, xi])
    return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# PHOTOMETRY
# ─────────────────────────────────────────────────────────────────────────────

def photons_mag20(ext_mag: float, t_expose: float = EXPOSURE_TIME) -> float:
    """
    Detected photons from a fiducial mag-20 AB source in r-band given
    cloud extinction ext_mag and open-shutter time t_expose.
    Used for relative comparisons only — the absolute scale is consistent
    across all three strategies.
    """
    if np.isnan(ext_mag):
        return np.nan
    return N_MAG20_30S * 10**(-0.4 * max(0.0, ext_mag)) * (t_expose / EXPOSURE_TIME)


def depth_from_ext(ext_mag: float, t_expose: float = EXPOSURE_TIME,
                   band: str = DEFAULT_BAND) -> float:
    """
    5σ limiting magnitude anchored to the Rubin SRD ZP.

      m5 = ZP_band - ext_mag - 1.25 * log10(t_expose / 30)

    The ZP_band term sets the clear-sky 30-s depth (e.g. 23.4 in r).
    Extinction dims sources by ext_mag uniformly.
    Reduced shutter time scales SNR as sqrt(t/30), so depth changes by
    -1.25 * log10(t/30).
    """
    if np.isnan(ext_mag) or t_expose <= 0:
        return np.nan
    zp = BAND_ZP.get(band, BAND_ZP[DEFAULT_BAND])
    return zp - max(0.0, ext_mag) - 1.25 * np.log10(t_expose / EXPOSURE_TIME)


def slew_time(alt1, az1, alt2, az2) -> float:
    """Kinematic slew time between two alt/az positions (seconds)."""
    if alt1 is None:
        return 0.0
    da  = abs(alt2 - alt1)
    daz = abs(az2 - az1)
    if daz > 180:
        daz = 360 - daz
    return max(da / MAX_SLEW_SPEED_ALT, daz / MAX_SLEW_SPEED_AZ) + SLEW_SETTLE_TIME


def effective_expose(slew_s: float) -> float:
    """
    Open-shutter time after accounting for slew overhead.
    The mount slews during the 2-s readout window for free;
    only excess slew time beyond readout cuts into the exposure.
    """
    overhead = max(0.0, slew_s - READOUT_TIME)
    return max(0.0, EXPOSURE_TIME - overhead)


# ─────────────────────────────────────────────────────────────────────────────
# RUBIN SCHEDULER SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def _footprint(mjd_start: float, nside: int = 32):
    sky = CurrentAreaMap(nside=nside)
    fp_arr, labels = sky.return_maps()
    roll_indx = np.where((labels == "lowdust") | (labels == "virgo"))[0]
    fp_hp = {k: fp_arr[k] for k in fp_arr.dtype.names}
    alm    = Almanac(mjd_start=mjd_start)
    sun_ra = alm.get_sun_moon_positions(mjd_start)["sun_RA"].copy()
    return make_rolling_footprints(
        fp_hp=fp_hp, mjd_start=mjd_start, sun_ra_start=sun_ra,
        nslice=2, scale=0.9, nside=nside, wfd_indx=roll_indx,
        order_roll=1, n_cycles=3, uniform=True)


def _build_surveys(footprints, nside=32, band="r"):
    bfs = [
        (bf.M5DiffBasisFunction(bandname=band, nside=nside),           3.0),
        (bf.FootprintBasisFunction(bandname=band, footprint=footprints,
                                    out_of_bounds_val=np.nan,
                                    nside=nside),                      0.75),
        (bf.SlewtimeBasisFunction(bandname=band, nside=nside),         3.0),
        (bf.BandChangeBasisFunction(bandname=band),                  100.0),
        (bf.BandLoadedBasisFunction(bandnames=[band]),                  0.0),
        (bf.VisitRepeatBasisFunction(gap_min=0, gap_max=120.0,
                                      bandname=None, nside=nside,
                                      npairs=20),                      -1.0),
        (bf.AltAzShadowMaskBasisFunction(nside=nside, shadow_minutes=15.0,
                                          max_alt=76.0, pad=3.0),       0.0),
        (bf.MoonAvoidanceBasisFunction(nside=nside, moon_distance=30),  0.0),
        (bf.PlanetMaskBasisFunction(nside=nside),                       0.0),
    ]
    dl = [
        detailers.CameraRotDetailer(min_rot=min(CAMERA_ROT_LIMITS),
                                     max_rot=max(CAMERA_ROT_LIMITS)),
        detailers.LabelRegionsAndDDFs(),
    ]
    return [GreedySurvey(
        [v[0] for v in bfs], [v[1] for v in bfs],
        exptime=EXPOSURE_TIME, bandname=band, nside=nside, nexp=1,
        detailers=dl,
        survey_name=f"greedy_{band}",
        science_program=SCIENCE_PROGRAM,
        observation_reason=f"singles_{band}",
        ignore_obs=["DD", "twilight_near_sun", "ToO"],
        block_size=1, smoothing_kernel=None,
        seed=42, camera="LSST", dither="night",
    )]


def _sky_data_mjd_range() -> tuple[float, float]:
    try:
        from rubin_scheduler.skybrightness_pre import SkyModelPre
        sm = SkyModelPre()
        return float(sm.mjd_left.min()), float(sm.mjd_right.max())
    except Exception:
        try:
            from rubin_scheduler.data import get_data_dir
            import glob, re
            data_dir = get_data_dir()
            files = glob.glob(os.path.join(data_dir, "skybrightness_pre", "*.h5"))
            if not files:
                files = glob.glob(os.path.join(data_dir, "**", "*.h5"),
                                  recursive=True)
            mjds = []
            for fn in files:
                for n in re.findall(r"(\d{5,6})", os.path.basename(fn)):
                    mjds.append(float(n))
            if mjds:
                return min(mjds), max(mjds)
        except Exception:
            pass
    return None, None


def _valid_survey_start(desired_mjd: float) -> float:
    lo, hi = _sky_data_mjd_range()
    if lo is None:
        return desired_mjd
    if lo <= desired_mjd <= hi:
        return desired_mjd
    shifted = desired_mjd
    for _ in range(30):
        shifted += 365.25
        if lo <= shifted <= hi:
            return shifted
    return lo


def run_greedy_sim(dream_survey_start: float, dream_duration_days: float,
                   band: str = DEFAULT_BAND,
                   verbose: bool = True) -> np.ndarray:
    sim_start = _valid_survey_start(dream_survey_start)
    print(f"\n{'='*60}")
    print("GREEDY SCHEDULER SIMULATION  (ideal / clear sky)")
    print(f"{'='*60}")
    lo, hi = _sky_data_mjd_range()
    if lo:
        print(f"  Sky data range : MJD {lo:.0f} – {hi:.0f}")
    print(f"  DREAM night    : MJD {dream_survey_start:.4f}")
    if abs(sim_start - dream_survey_start) > 0.5:
        shift = sim_start - dream_survey_start
        print(f"  Sim start      : MJD {sim_start:.4f}  "
              f"(shifted +{shift:.1f} d to match sky-data range)")
    else:
        print(f"  Sim start      : MJD {sim_start:.4f}")
    print(f"  Duration       : {dream_duration_days*24:.2f} h")

    fp    = _footprint(sim_start)
    svys  = _build_surveys(fp, band=band)
    cm    = CloudMap()
    cm.add_frame(np.zeros(hp.nside2npix(32)), sim_start)
    sched = CoreScheduler(svys, nside=32)
    mo    = ModelObservatory(nside=32, mjd_start=sim_start,
                              cloud_data="ideal", downtimes="ideal",
                              cloud_maps=cm)
    mo, sched, obs = sim_runner(mo, sched,
                                 sim_duration=dream_duration_days,
                                 verbose=verbose)
    print(f"  {len(obs)} scheduler observations produced")
    return obs


# ─────────────────────────────────────────────────────────────────────────────
# OBS FIELD ACCESSORS
# ─────────────────────────────────────────────────────────────────────────────

def _obs_mjd(row) -> float:
    for f in ("observationStartMJD", "start_mjd", "mjd", "MJD"):
        try:
            return float(row[f])
        except (KeyError, ValueError, IndexError):
            pass
    raise KeyError(f"No MJD field.  Available: {row.dtype.names}")


def _obs_radec(row) -> tuple[float, float]:
    """Returns (RA_deg, Dec_deg). rubin_scheduler stores angles in radians."""
    for rf, df_ in [("RA", "dec"), ("fieldRA", "fieldDec"), ("ra", "dec")]:
        try:
            ra  = float(row[rf])
            dec = float(row[df_])
            if abs(ra) <= 2 * np.pi + 0.01:
                ra  = np.degrees(ra) % 360.0
                dec = np.degrees(dec)
            return ra, dec
        except (KeyError, ValueError, IndexError):
            pass
    raise KeyError(f"No RA/Dec fields.  Available: {row.dtype.names}")


def _obs_depth(row) -> float | None:
    """Read fivesigmadepth directly from obs record if plausible."""
    for f in ("fivesigmadepth", "fiveSigmaDepth", "m5"):
        try:
            v = float(row[f])
            if 15.0 < v < 30.0:
                return v
        except (KeyError, ValueError, IndexError):
            pass
    return None


def _obs_slewtime(row) -> float | None:
    """Read slewtime from obs record if present and plausible."""
    try:
        v = float(row["slewtime"])
        if 0.0 < v < 300.0:
            return v
    except (KeyError, ValueError, IndexError):
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# METRICS CONTAINER
# ─────────────────────────────────────────────────────────────────────────────

def _empty():
    return {"photons": [], "expose_t": [], "slew_t": [], "ext": [],
            "depth": [], "frame_idx": [], "ra": [], "dec": [], "pix": set()}


def _record(m, t_expose, ext_val, sl_s, fi, alt, az, ra, dec,
            pa, paa, nside=32, band=DEFAULT_BAND, sched_depth=None):
    """Append one slot's worth of metrics."""
    ph = photons_mag20(ext_val, t_expose)

    # Depth: use scheduler's ideal-sky value degraded by cloud if available;
    # otherwise derive from the SRD zero-point anchor.
    if sched_depth is not None and not np.isnan(sched_depth):
        d5 = sched_depth - max(0.0, ext_val)
    else:
        d5 = depth_from_ext(ext_val, t_expose, band)

    m["photons"].append(ph)
    m["expose_t"].append(t_expose)
    m["slew_t"].append(sl_s)
    m["ext"].append(ext_val)
    m["depth"].append(d5)
    m["frame_idx"].append(fi)
    m["ra"].append(ra)
    m["dec"].append(dec)
    m["pix"].add(hp.ang2pix(nside, np.radians(90 - alt),
                             np.radians(az), nest=False))
    pa[0]  = alt
    paa[0] = az


# ─────────────────────────────────────────────────────────────────────────────
# DREAM METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_dream_metrics(data: dict,
                          obs_mjd_start: float,
                          obs_mjd_end: float,
                          band: str = DEFAULT_BAND) -> dict[str, dict]:
    """
    Re-derive slot metrics from the pre-saved DREAM pointings, restricted
    to [obs_mjd_start, obs_mjd_end] (the window the scheduler actually used).

    Cloud frames arrive at ~1 Hz; we subsample to one decision per
    SLOT_TIME (32 s) to match telescope cadence.
    """
    print(f"\n{'='*60}")
    print("DREAM METRICS  (from saved pointings)")
    print(f"{'='*60}")
    print(f"  Night window   : MJD {obs_mjd_start:.4f} → {obs_mjd_end:.4f}")
    print(f"  ({(obs_mjd_end - obs_mjd_start)*24:.2f} h, "
          f"subsampled to 1 slot per {SLOT_TIME:.0f} s)")

    grids = data["ext_grids"]
    mjds  = data["mjds"].astype(float)
    nside = int(data["nside"])
    N     = len(mjds)

    # Frame index range that overlaps the scheduler window
    fi_start = int(np.searchsorted(mjds, obs_mjd_start, side="left"))
    fi_end   = int(np.searchsorted(mjds, obs_mjd_end,   side="right")) - 1
    fi_start = max(0, min(fi_start, N - 1))
    fi_end   = max(fi_start, min(fi_end, N - 1))

    # Build list of frame indices at telescope cadence (one per 32-s slot)
    slot_days = SLOT_TIME / 86400.0
    slot_fis  = []
    t_cursor  = mjds[fi_start]
    while t_cursor <= mjds[fi_end]:
        fi = int(np.searchsorted(mjds, t_cursor, side="left"))
        fi = min(fi, fi_end)
        slot_fis.append(fi)
        t_cursor += slot_days

    print(f"  Frame range    : [{fi_start}, {fi_end}]  "
          f"({fi_end - fi_start + 1} raw frames)")
    print(f"  Slots to eval  : {len(slot_fis)}")

    strats = {k: _empty() for k in ("absolute", "motion")}
    pa_a, paa_a = [None], [None]
    pa_m, paa_m = [None], [None]

    for fi in slot_fis:
        grid = grids[fi]  # noqa: F841  (kept for future per-slot cloud lookup)

        # ── Absolute strategy ──────────────────────────────────────────────
        alt_a, az_a  = data["abs_altaz"][fi]
        ra_a,  dec_a = data["abs_radec"][fi]
        ext_a = float(data["abs_ext"][fi])
        if alt_a >= MIN_ALT_DEG and not np.isnan(ext_a):
            sl   = slew_time(pa_a[0], paa_a[0], alt_a, az_a)
            texp = effective_expose(sl)
            _record(strats["absolute"], texp, ext_a, sl, fi,
                    alt_a, az_a, ra_a, dec_a, pa_a, paa_a, nside, band)

        # ── Motion strategy ────────────────────────────────────────────────
        alt_m, az_m  = data["mot_altaz"][fi]
        ra_m,  dec_m = data["mot_radec"][fi]
        ext_m = float(data["mot_ext"][fi])
        if alt_m >= MIN_ALT_DEG and not np.isnan(ext_m):
            sl   = slew_time(pa_m[0], paa_m[0], alt_m, az_m)
            texp = effective_expose(sl)
            _record(strats["motion"], texp, ext_m, sl, fi,
                    alt_m, az_m, ra_m, dec_m, pa_m, paa_m, nside, band)

    for k, m in strats.items():
        print(f"  {k}: {len(m['photons'])} slots  |  "
              f"{len(m['pix'])} unique pixels")
    return strats


# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULER METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_scheduler_metrics(obs: np.ndarray, data: dict,
                               band: str = DEFAULT_BAND
                               ) -> tuple[dict, float, float]:
    """
    Map scheduler pointings onto DREAM cloud grids by fractional night
    position.

    Returns (metrics_dict, sim_mjd_start, sim_mjd_end) so the DREAM
    strategies can be evaluated over the same window.
    """
    print(f"\n{'='*60}")
    print("SCHEDULER METRICS  (vs real DREAM clouds)")
    print(f"{'='*60}")

    if obs is None or len(obs) == 0:
        print("  No observations — skipping")
        return _empty(), 0.0, 0.0

    dream_mjds = data["mjds"].astype(float)
    grids      = data["ext_grids"]
    loc        = _loc(data)
    nside      = int(data["nside"])
    N_dream    = len(dream_mjds)

    sim_mjds = []
    for row in obs:
        try:
            sim_mjds.append(_obs_mjd(row))
        except KeyError:
            pass
    if not sim_mjds:
        print("  Could not extract MJDs — skipping")
        return _empty(), 0.0, 0.0

    sim_t0  = min(sim_mjds)
    sim_t1  = max(sim_mjds)
    sim_dur = sim_t1 - sim_t0 if sim_t1 > sim_t0 else 1.0

    dream_t0 = float(dream_mjds[0])
    dream_t1 = float(dream_mjds[-1])

    print(f"  Sim night span   : MJD {sim_t0:.4f} → {sim_t1:.4f}  "
          f"({sim_dur*24:.2f} h,  {len(obs)} obs)")
    print(f"  DREAM window     : MJD {dream_t0:.4f} → {dream_t1:.4f}  "
          f"({(dream_t1 - dream_t0)*24:.2f} h,  {N_dream} frames)")
    print(f"  Matching by fractional night position …")

    m        = _empty()
    pa, paa  = [None], [None]
    n_low = n_outside = 0

    for row in obs:
        try:
            obs_mjd         = _obs_mjd(row)
            ra_deg, dec_deg = _obs_radec(row)
            sched_d5        = _obs_depth(row)
        except KeyError as e:
            print(f"  Schema error: {e}"); break

        # Fractional position within sim night → DREAM frame index
        frac = (obs_mjd - sim_t0) / sim_dur
        frac = max(0.0, min(1.0, frac))
        fi   = int(round(frac * (N_dream - 1)))
        grid = grids[fi]

        # Alt/Az at the DREAM frame's actual time
        alt_deg, az_deg = radec_to_altaz(ra_deg, dec_deg, dream_mjds[fi], loc)
        x, y = altaz_to_xy(alt_deg, az_deg)
        if x is None:
            n_low += 1
            continue

        ext = get_val(grid, x, y)
        if np.isnan(ext):
            ext = 0.0
            n_outside += 1

        # Slew time: prefer scheduler's own field; fall back to kinematics
        sl_raw = _obs_slewtime(row)
        if sl_raw is not None:
            sl = sl_raw + SLEW_SETTLE_TIME
        else:
            sl = slew_time(pa[0], paa[0], alt_deg, az_deg)

        texp = effective_expose(sl)

        _record(m, texp, ext, sl, fi,
                alt_deg, az_deg, ra_deg, dec_deg,
                pa, paa, nside, band, sched_depth=sched_d5)

    print(f"  {len(m['photons'])} obs processed  "
          f"(below horizon: {n_low}  |  outside cloud map: {n_outside})")
    return m, sim_t0, sim_t1


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def _summarize(label: str, m: dict) -> dict:
    ph  = np.array(m["photons"])
    et  = np.array(m["expose_t"])
    sl  = np.array(m["slew_t"])
    ex  = np.array(m["ext"])
    dp  = np.array([v for v in m["depth"] if not np.isnan(v)])
    n   = len(ph)
    slots_s = n * SLOT_TIME

    row = dict(
        Strategy            = label,
        N_slots             = n,
        Total_photons       = float(np.sum(ph)),
        Shutter_eff_pct     = (float(np.sum(et) / slots_s * 100)
                               if slots_s > 0 else 0.0),
        Mean_slew_s         = float(np.mean(sl[1:])) if len(sl) > 1 else 0.0,
        Mean_extinction_mag = float(np.nanmean(ex)),
        Median_5sig_depth   = float(np.median(dp)) if len(dp) else np.nan,
        Unique_pixels       = len(m["pix"]),
    )
    print(f"\n  ── {label} ──")
    for k, v in row.items():
        if k == "Strategy":
            continue
        print(f"    {k:<26}  {v:.5g}")
    return row


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def _bar(ax, vals, labels, ylabel, invert=False, pct_vs=None, fmt=".3g"):
    bars = ax.bar(labels, vals,
                  color=[COLORS[l] for l in labels],
                  alpha=0.82, edgecolor="black", lw=1.2)
    for i, (bar, v) in enumerate(zip(bars, vals)):
        txt = format(v, fmt)
        if pct_vs is not None and i != pct_vs:
            base = vals[pct_vs]
            txt += f"\n({(v - base) / max(abs(base), 1) * 100:+.1f}%)"
        ax.text(bar.get_x() + bar.get_width() / 2,
                v * 1.01 if not invert else v - abs(v) * 0.03,
                txt, ha="center", va="bottom", fontsize=9, weight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, axis="y")
    if invert:
        ax.invert_yaxis()


def plot_all(all_m, summary, data, night_tag, output_dir, band=DEFAULT_BAND):
    os.makedirs(output_dir, exist_ok=True)
    labels = list(all_m.keys())

    # ── Figure 1: main metrics ───────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 15))
    gs  = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    _bar(ax, [r["Total_photons"] for r in summary], labels,
         "Total photons  (mag-20 ref, slew-gated)", pct_vs=2)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_title("Total Night Photon Collection", weight="bold")

    ax = fig.add_subplot(gs[0, 1])
    _bar(ax, [r["Shutter_eff_pct"] for r in summary], labels,
         "Shutter-open efficiency (%)", fmt=".1f")
    ax.set_ylim(0, 110)
    ax.set_title("Observation Efficiency", weight="bold")

    ax = fig.add_subplot(gs[0, 2])
    _bar(ax, [r["Median_5sig_depth"] for r in summary], labels,
         "Median 5σ depth (mag)", invert=True, fmt=".2f")
    zp_band = BAND_ZP.get(band, BAND_ZP[DEFAULT_BAND])
    ax.axhline(zp_band, color="k", ls="--", lw=1, alpha=0.6,
               label=f"SRD ZP {zp_band:.1f}")
    ax.legend(fontsize=8)
    ax.set_title("Survey Depth (SRD-anchored)", weight="bold")

    ax = fig.add_subplot(gs[1, :])
    for lbl, c in COLORS.items():
        fi = np.array(all_m[lbl]["frame_idx"])
        ph = np.array(all_m[lbl]["photons"])
        if len(fi):
            order = np.argsort(fi)
            ax.plot(fi[order], ph[order], color=c, alpha=0.8, lw=1.5, label=lbl)
    ax.set_xlabel("Cloud frame index (time →)")
    ax.set_ylabel("Photons / slot  (mag-20 ref, slew-gated)")
    ax.set_title(f"Photon Collection Over Night {night_tag} "
                 "(real DREAM cloud data)", weight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    ax = fig.add_subplot(gs[2, 0])
    for lbl, c in COLORS.items():
        sl = all_m[lbl]["slew_t"][1:]
        if sl:
            ax.hist(sl, bins=30, alpha=0.6, color=c, label=lbl,
                    edgecolor="black", lw=0.5)
    ax.set_xlabel("Slew time (s)")
    ax.set_ylabel("Count")
    ax.set_title("Slew Time Distribution", weight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    for lbl, c in COLORS.items():
        ex = all_m[lbl]["ext"]
        if ex:
            ax.hist(ex, bins=30, alpha=0.6, color=c, label=lbl,
                    edgecolor="black", lw=0.5)
    ax.set_xlabel("Cloud extinction (mag)")
    ax.set_ylabel("Count")
    ax.set_title("Extinction Distribution per Strategy", weight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, 2])
    _bar(ax, [r["Unique_pixels"] for r in summary], labels,
         "Unique HEALPix pixels visited")
    ax.set_title("Sky Coverage", weight="bold")

    plt.suptitle(
        f"Pointing Strategy Comparison — Night {night_tag}  [{band}-band]\n"
        "DREAM Absolute Min  ·  DREAM Motion Tracking  ·  Greedy Scheduler\n"
        "real DREAM cloud data · slew-gated · SRD ZP · same night window",
        fontsize=12, weight="bold", y=1.005)
    out1 = os.path.join(output_dir, f"comparison_{night_tag}.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out1}")

    # ── Figure 2: RA/Dec Aitoff ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(21, 6),
                              subplot_kw={"projection": "aitoff"})
    for ax, lbl, c in zip(axes, labels, COLORS.values()):
        ra  = np.array(all_m[lbl]["ra"])
        dec = np.array(all_m[lbl]["dec"])
        if len(ra):
            ax.scatter(np.radians(ra - 180), np.radians(dec),
                       s=4, alpha=0.5, color=c)
        ax.set_title(f"{lbl}\n({len(ra)} pointings)", weight="bold", pad=14)
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"Sky Coverage (RA/Dec) — Night {night_tag}",
                 fontsize=13, weight="bold")
    plt.tight_layout()
    out2 = os.path.join(output_dir, f"sky_coverage_{night_tag}.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out2}")

    # ── Figure 3: HEALPix Mollweide ─────────────────────────────────────────
    nside   = int(data["nside"])
    npix_hp = hp.nside2npix(nside)
    fig = plt.figure(figsize=(22, 7))
    for idx, (lbl, c) in enumerate(COLORS.items()):
        cov = np.zeros(npix_hp)
        for p in all_m[lbl]["pix"]:
            cov[p] = 1.0
        hp.mollview(cov, fig=fig.number, sub=(1, 3, idx + 1),
                    title=f"{lbl}\n({len(all_m[lbl]['pix'])} pixels)",
                    cmap="YlOrRd", min=0, max=1, hold=True)
    plt.suptitle(f"HEALPix Coverage — Night {night_tag}",
                 fontsize=13, weight="bold")
    out3 = os.path.join(output_dir, f"healpix_{night_tag}.png")
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out3}")

    # ── Figure 4: cloud map snapshots ────────────────────────────────────────
    grids = data["ext_grids"]
    xe, ye = data["x_edges"], data["y_edges"]
    N   = len(grids)
    fis = [0, N // 4, N // 2, 3 * N // 4, N - 1]
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for ax, fi, ttl in zip(axes, fis, ["Start", "25%", "Mid", "75%", "End"]):
        im = ax.pcolormesh(xe, ye, grids[fi],
                           cmap="viridis", vmin=-0.2, vmax=2.0, shading="flat")
        plt.colorbar(im, ax=ax, label="Extinction (mag)")
        th = np.linspace(0, 2 * np.pi, 300)
        ax.plot(15_000 * np.cos(th), 15_000 * np.sin(th),
                "w-", lw=1, alpha=0.3)
        ax.plot(0, 0, "r+", ms=10, mew=2)
        ax.set_title(f"{ttl}  (frame {fi})", weight="bold")
        ax.set_aspect("equal")
        ax.set_xlabel("X km")
        ax.set_ylabel("Y km")
    plt.suptitle(f"DREAM Cloud Maps — Night {night_tag}",
                 fontsize=13, weight="bold")
    out4 = os.path.join(output_dir, f"cloud_maps_{night_tag}.png")
    plt.savefig(out4, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out4}")

    # ── Figure 5: cumulative photons ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    for lbl, c in COLORS.items():
        fi = np.array(all_m[lbl]["frame_idx"])
        ph = np.array(all_m[lbl]["photons"])
        if len(fi):
            order = np.argsort(fi)
            ax.plot(fi[order], np.cumsum(ph[order]), color=c, lw=2, label=lbl)
    ax.set_xlabel("Cloud frame index (time →)")
    ax.set_ylabel("Cumulative photons  (mag-20 ref, slew-gated)")
    ax.set_title(f"Cumulative Photon Collection — Night {night_tag}",
                 weight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    out5 = os.path.join(output_dir, f"cumulative_{night_tag}.png")
    plt.savefig(out5, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out5}")

    # ── Figure 6: 5σ depth distribution ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for lbl, c in COLORS.items():
        dp = [v for v in all_m[lbl]["depth"] if not np.isnan(v)]
        if dp:
            ax.hist(dp, bins=40, alpha=0.6, color=c, label=lbl,
                    edgecolor="black", lw=0.5)
    zp_band = BAND_ZP.get(band, BAND_ZP[DEFAULT_BAND])
    ax.axvline(zp_band, color="k", ls="--", lw=1.5,
               label=f"SRD ZP {zp_band:.1f} (clear sky)")
    ax.set_xlabel("5σ limiting magnitude")
    ax.set_ylabel("Count")
    ax.set_title(f"5σ Depth Distribution — Night {night_tag}  [{band}-band]",
                 weight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    out6 = os.path.join(output_dir, f"depth_dist_{night_tag}.png")
    plt.savefig(out6, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out6}")

    return out1, out2, out3, out4, out5, out6


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def compare(
    npz_file:   str  = "dream_night.npz",
    output_dir: str  = "outputs",
    band:       str  = DEFAULT_BAND,
    verbose:    bool = True,
) -> tuple[dict, list[dict]]:
    """
    Load Part A .npz, run greedy scheduler, compute and plot all metrics.

    All three strategies (DREAM Absolute, DREAM Motion, Greedy Sched)
    are evaluated over the same MJD window — the scheduler's actual
    observing span within the night.  DREAM strategies subsample their
    1-Hz cloud frames to one decision per 32-s slot.

    Parameters
    ----------
    npz_file   : output of dream_extract_server.py
    output_dir : where to write plots + CSV
    band       : photometric band for ZP calibration (default "r")
    verbose    : pass to sim_runner

    Returns
    -------
    (all_metrics dict, summary list-of-dicts)
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("PART B  ―  LOCAL COMPARISON")
    print(f"{'='*60}")
    print(f"  Band           : {band}")
    print(f"  ZP (clear 30s) : {BAND_ZP.get(band, BAND_ZP[DEFAULT_BAND]):.1f} mag")
    print(f"  N_mag20_30s    : {N_MAG20_30S:.3e} photons  (reference source)")

    data      = load_npz(npz_file)
    night_tag = str(data["night_key"]).replace("-", "")

    dream_mjd_start = float(data["mjds"][0])
    dream_mjd_end   = float(data["mjds"][-1])
    survey_start    = float(data["survey_start_mjd"])
    dream_night_dur = dream_mjd_end - survey_start

    print(f"\n  DREAM window  : MJD {dream_mjd_start:.6f} → {dream_mjd_end:.6f}")
    print(f"  ({(dream_mjd_end - dream_mjd_start)*24:.2f} h of cloud frames)")

    # ── 1. Greedy sim ────────────────────────────────────────────────────────
    sched_obs = run_greedy_sim(survey_start, dream_night_dur, band, verbose)

    # ── 2. Scheduler metrics — establishes common night window ───────────────
    sched_m, sim_t0, sim_t1 = compute_scheduler_metrics(sched_obs, data, band)

    # Fall back to DREAM window if scheduler produced nothing useful
    if sim_t0 == 0.0 and sim_t1 == 0.0:
        sim_t0, sim_t1 = dream_mjd_start, dream_mjd_end

    # ── 3. DREAM metrics — same window, same cadence ─────────────────────────
    dream_m = compute_dream_metrics(data, sim_t0, sim_t1, band)

    all_m = {
        "DREAM Absolute": dream_m["absolute"],
        "DREAM Motion":   dream_m["motion"],
        "Greedy Sched":   sched_m,
    }

    # ── 4. Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("NIGHT PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    summary = [_summarize(lbl, m) for lbl, m in all_m.items()]

    baseline = summary[2]["Total_photons"]
    print(f"\n  Δ total photons vs Greedy Sched:")
    for r in summary[:2]:
        pct = (r["Total_photons"] - baseline) / max(abs(baseline), 1) * 100
        print(f"    {r['Strategy']:<22}  {pct:+.1f}%")

    csv_out = os.path.join(output_dir, f"summary_{night_tag}.csv")
    pd.DataFrame(summary).to_csv(csv_out, index=False)
    print(f"\n  Summary CSV → {csv_out}")

    # ── 5. Plots ─────────────────────────────────────────────────────────────
    print(f"\n  Saving plots → {output_dir}/")
    plot_all(all_m, summary, data, night_tag, output_dir, band)

    print(f"\n{'='*60}")
    print(f"DONE — outputs in {output_dir}/")
    print(f"{'='*60}")
    return all_m, summary


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    _jupyter = ("ipykernel" in sys.modules or
                any("ipykernel" in a for a in sys.argv))

    if _jupyter:
        compare(
            npz_file   = "dream_night.npz",   # ← edit path
            output_dir = "outputs",
            band       = "r",
            verbose    = True,
        )
    else:
        ap = argparse.ArgumentParser(
            description="Compare DREAM pointings vs greedy scheduler locally")
        ap.add_argument("--npz",    default="dream_night.npz")
        ap.add_argument("--outdir", default="outputs")
        ap.add_argument("--band",   default="r",
                        help="Photometric band (u/g/r/i/z/y, default: r)")
        ap.add_argument("--quiet",  action="store_true")
        a = ap.parse_args()
        compare(a.npz, a.outdir, a.band, not a.quiet)
