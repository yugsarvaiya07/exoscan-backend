"""
================================================================================
ISRO Hackathon — AI-enabled Detection of Exoplanets from Noisy Light Curves
================================================================================
Backend Service (FastAPI)

Pipeline:
    1. Ingestion   -> lightkurve search/download (Kepler/TESS) with a
                       deterministic synthetic-transit fallback generator
                       (used automatically if the network/MAST archive is
                       unreachable, so the demo never breaks on stage).
    2. Denoising    -> outlier clipping (cosmic rays / instrument artifacts)
                       + long-term trend flattening (stellar variability).
    3. Detection    -> Box Least Squares (BLS) periodogram to recover the
                       orbital period, transit duration, depth and a
                       confidence score derived from the BLS power spectrum.
    4. API          -> /api/analyze-star returns a clean JSON payload ready
                       to be charted by the frontend dashboard.

Author: Principal Architect (generated for hackathon use)
================================================================================
"""

from __future__ import annotations

import logging
import math
import re
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from scipy import signal as scipy_signal
from scipy import stats as scipy_stats

# --------------------------------------------------------------------------
# Optional heavy astronomy dependency. We import it lazily / defensively so
# that the API still boots (and the mock pipeline still works) even on a
# machine where `lightkurve` or its dependencies fail to install/import,
# which is a real risk under hackathon time pressure.
# --------------------------------------------------------------------------
try:
    import lightkurve as lk  # type: ignore

    LIGHTKURVE_AVAILABLE = True
except Exception:  # pragma: no cover - defensive import
    LIGHTKURVE_AVAILABLE = False

try:
    from astropy.timeseries import BoxLeastSquares  # type: ignore

    ASTROPY_BLS_AVAILABLE = True
except Exception:  # pragma: no cover - defensive import
    ASTROPY_BLS_AVAILABLE = False


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("exoplanet-api")


# ==========================================================================
# FastAPI app & CORS
# ==========================================================================
app = FastAPI(
    title="Exoplanet Transit Detection API",
    description=(
        "AI-enabled detection of exoplanets from noisy astronomical "
        "light curves (Kepler / TESS) — ISRO Hackathon prototype."
    ),
    version="1.0.0",
)

# Wide-open CORS for hackathon demo purposes. Tighten origins for prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================================
# Response models
# ==========================================================================
class TransitMetadata(BaseModel):
    detected: bool = Field(..., description="Whether a transit signal was detected")
    confidence_pct: float = Field(..., description="Detection confidence (0-100)")
    orbital_period_days: Optional[float] = Field(None, description="Best-fit orbital period in days")
    transit_duration_hours: Optional[float] = Field(None, description="Transit duration in hours")
    planet_radius_estimate_earth: Optional[float] = Field(
        None, description="Rough planet radius estimate relative to Earth, derived from transit depth"
    )
    transit_depth_ppm: Optional[float] = Field(None, description="Transit depth in parts-per-million")
    snr: Optional[float] = Field(None, description="Estimated signal-to-noise ratio of the transit")


class AnalyzeResponse(BaseModel):
    star_id: str
    data_source: str  # "lightkurve" | "synthetic_fallback"
    mission: str
    n_points: int
    time: list[float]
    flux_raw: list[float]
    flux_clean: list[float]
    transit_mask: list[bool]
    metadata: TransitMetadata
    notes: str


# ==========================================================================
# STAGE 1 — DATA INGESTION
# ==========================================================================
def _looks_like_valid_star_id(star_id: str) -> bool:
    """Loose sanity check so we don't ship junk strings to MAST."""
    pattern = re.compile(r"^(KIC|TIC|EPIC)?\s*\d{3,10}$", re.IGNORECASE)
    return bool(pattern.match(star_id.strip()))


def generate_synthetic_light_curve(
    star_id: str,
    n_points: int = 2000,
    period_days: float = 3.2,
    duration_hours: float = 2.8,
    depth_ppm: float = 4500.0,
    noise_ppm: float = 800.0,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Deterministic synthetic light-curve generator used as a demo-safe
    fallback when the network / MAST archive is unavailable.

    Produces:
      - Gaussian instrument/photon noise
      - A slow stellar-variability trend (sinusoid + drift) to be flattened
      - Sparse outlier spikes simulating cosmic ray hits
      - A clean, repeating U-shaped transit dip at the requested period

    The seed is derived deterministically from `star_id` so the same star
    always reproduces the same mock curve (useful for demo repeatability).
    """
    if seed is None:
        seed = abs(hash(star_id)) % (2**32)
    rng = np.random.default_rng(seed)

    baseline_days = 27.0  # ~ one TESS sector
    time = np.linspace(0, baseline_days, n_points)

    # 1. Base flux = 1.0 (normalized)
    flux = np.ones_like(time)

    # 2. Long-term stellar variability / instrumental drift (to be flattened later)
    flux += 0.004 * np.sin(2 * np.pi * time / 9.5)
    flux += 0.0015 * (time / baseline_days)

    # 3. Photon/instrument noise
    flux += rng.normal(0, noise_ppm * 1e-6, size=n_points)

    # 4. Cosmic-ray / artifact outliers (sparse positive & negative spikes)
    n_outliers = max(3, n_points // 250)
    outlier_idx = rng.choice(n_points, size=n_outliers, replace=False)
    flux[outlier_idx] += rng.choice([-1, 1], size=n_outliers) * rng.uniform(0.01, 0.03, size=n_outliers)

    # 5. Inject repeating U-shaped transits
    duration_days = duration_hours / 24.0
    depth = depth_ppm * 1e-6
    t0 = period_days * 0.3  # phase offset of first transit

    phase = ((time - t0 + period_days / 2) % period_days) - period_days / 2
    half_dur = duration_days / 2.0

    in_transit = np.abs(phase) <= half_dur
    # Smooth U-shape (not a hard box) via a cosine taper for realism
    taper = np.zeros_like(time)
    taper[in_transit] = depth * (0.5 * (1 + np.cos(np.pi * phase[in_transit] / half_dur)))
    flux -= taper

    source = "synthetic_fallback"
    return time, flux, source


def fetch_light_curve(star_id: str) -> tuple[np.ndarray, np.ndarray, str, str]:
    """
    Attempt to download a real light curve via `lightkurve` from the
    Kepler/TESS public archives. Falls back to synthetic data on *any*
    failure (network down, no internet at venue, target not found,
    lightkurve not installed, archive timeout, etc.) so the live demo
    never crashes.

    Returns: (time, flux, source, mission)
    """
    if not LIGHTKURVE_AVAILABLE:
        logger.warning("lightkurve not available — using synthetic fallback for %s", star_id)
        t, f, src = generate_synthetic_light_curve(star_id)
        return t, f, src, "Synthetic (lightkurve unavailable)"

    try:
        logger.info("Searching MAST archive for target: %s", star_id)
        search_result = lk.search_lightcurve(star_id, mission=["Kepler", "TESS"])

        if len(search_result) == 0:
            raise ValueError(f"No light curves found on MAST for '{star_id}'")

        # Pick the first available product (could be refined to choose
        # longest baseline / best quality product for production use).
        lc_collection = search_result[:1].download_all()
        if lc_collection is None or len(lc_collection) == 0:
            raise ValueError("Download returned no data")

        lc = lc_collection[0]
        lc = lc.remove_nans()

        time = np.asarray(lc.time.value, dtype=float)
        flux = np.asarray(lc.flux.value, dtype=float)

        # Normalize to relative flux around 1.0 if not already
        median_flux = np.nanmedian(flux)
        if median_flux != 0 and not (0.9 < median_flux < 1.1):
            flux = flux / median_flux

        mission = getattr(lc, "mission", "Kepler/TESS")
        logger.info("Downloaded %d data points for %s from %s", len(time), star_id, mission)
        return time, flux, "lightkurve", str(mission)

    except Exception as exc:  # noqa: BLE001 - intentional broad catch for demo robustness
        logger.warning(
            "lightkurve ingestion failed for '%s' (%s). Falling back to synthetic data.",
            star_id, exc,
        )
        t, f, src = generate_synthetic_light_curve(star_id)
        return t, f, src, "Synthetic (network/archive fallback)"


# ==========================================================================
# STAGE 2 — DENOISING PIPELINE
# ==========================================================================
def clip_outliers(flux: np.ndarray, sigma: float = 4.5) -> np.ndarray:
    """
    Remove extreme outliers (cosmic ray hits, instrument artifacts) via
    iterative sigma clipping. Outliers are replaced with the local median
    so the array length / time alignment is preserved for charting.
    """
    cleaned = flux.copy()
    for _ in range(3):  # iterative clipping converges quickly
        median = np.nanmedian(cleaned)
        std = np.nanstd(cleaned)
        if std == 0 or np.isnan(std):
            break
        mask = np.abs(cleaned - median) > sigma * std
        if not np.any(mask):
            break
        cleaned[mask] = median
    return cleaned


def flatten_light_curve(time: np.ndarray, flux: np.ndarray, window_days: float = 1.5) -> np.ndarray:
    """
    Remove long-term stellar variability / instrumental trends by
    estimating a smooth low-frequency baseline (Savitzky-Golay filter)
    and dividing it out, leaving a flat, transit-only signal centered
    near 1.0 — analogous to `lightkurve`'s `.flatten()` method.
    """
    n = len(flux)
    if n < 5:
        return flux

    # Convert desired smoothing window (in days) to an odd number of samples
    cadence = np.median(np.diff(time)) if n > 1 else 1.0
    window_pts = int(window_days / cadence) if cadence > 0 else 101
    window_pts = max(5, window_pts)
    if window_pts % 2 == 0:
        window_pts += 1
    window_pts = min(window_pts, n - 1 if (n - 1) % 2 == 1 else n - 2)
    window_pts = max(5, window_pts)

    polyorder = 2 if window_pts > 3 else 1

    try:
        trend = scipy_signal.savgol_filter(flux, window_length=window_pts, polyorder=polyorder)
        trend[trend == 0] = np.nanmedian(flux)
        flattened = flux / trend
    except Exception as exc:  # noqa: BLE001
        logger.warning("Savitzky-Golay flattening failed (%s); returning raw flux", exc)
        flattened = flux

    return flattened


def denoise_pipeline(time: np.ndarray, flux: np.ndarray) -> np.ndarray:
    """Full denoise chain: outlier clipping -> trend flattening."""
    despiked = clip_outliers(flux)
    flattened = flatten_light_curve(time, despiked)
    return flattened


# ==========================================================================
# STAGE 3 — TRANSIT DETECTION (Box Least Squares)
# ==========================================================================
def run_bls_detection(time: np.ndarray, flux: np.ndarray) -> dict:
    """
    Run a Box Least Squares periodogram to search for periodic, box-shaped
    transit signals. Uses `astropy.timeseries.BoxLeastSquares` when
    available; otherwise falls back to a lightweight custom BLS-style
    grid search implemented with numpy, so the endpoint always returns
    a result regardless of the installed environment.

    Returns a dict with detection status, confidence, period, duration,
    depth, and SNR.
    """
    # Remove any NaNs that may have leaked through
    valid = ~np.isnan(flux) & ~np.isnan(time)
    t, f = time[valid], flux[valid]

    if len(t) < 20:
        return _empty_detection_result()

    # Trial periods. We start a little above the shortest plausible
    # detectable period given the time baseline / cadence so that
    # implausibly-short periods (which lead to nonsensical duty cycles)
    # are not even offered to the search.
    period_grid = np.linspace(0.8, 15.0, 4000)        # days
    duration_grid = np.array([1, 2, 3, 4, 6, 8]) / 24  # hours -> days

    # Physical sanity cap: a real transit duration is always a SMALL
    # fraction of the orbital period (duty cycle). For a circular orbit
    # around a Sun-like star this is typically well under ~12%. We use a
    # generous 15% ceiling so we never report a "transit" that swallows
    # most of the phase-folded light curve (which is what a duration/period
    # mismatch looks like, and is the bug this cap fixes).
    MAX_DUTY_CYCLE = 0.15

    if ASTROPY_BLS_AVAILABLE:
        try:
            bls = BoxLeastSquares(t, f)
            result = bls.power(period_grid, duration_grid)

            power = np.asarray(result.power)
            periods = np.asarray(result.period)
            durations = np.asarray(result.duration)
            depths = np.asarray(result.depth)
            t0s = np.asarray(result.transit_time)

            # Mask out any period/duration pairs with an unrealistic duty
            # cycle BEFORE picking the best candidate, instead of
            # filtering only the winner.
            duty_cycle = durations / periods
            valid = np.isfinite(power) & (duty_cycle <= MAX_DUTY_CYCLE) & (depths > 0)

            if not np.any(valid):
                # Nothing physically plausible found — report no detection
                # rather than forcing a misleading best-of-the-rest pick.
                return _empty_detection_result()

            masked_power = np.where(valid, power, -np.inf)
            best_idx = int(np.argmax(masked_power))

            best_period = float(periods[best_idx])
            best_duration_days = float(durations[best_idx])
            best_t0 = float(t0s[best_idx])
            depth = float(depths[best_idx])

            power_clean = power[np.isfinite(power)]
            mean_p, std_p = np.mean(power_clean), np.std(power_clean)
            best_power = float(power[best_idx])
            snr = (best_power - mean_p) / std_p if std_p > 0 else 0.0

            confidence = _power_to_confidence(snr)
            detected = confidence >= 55.0 and depth > 0

            phase = ((t - best_t0 + best_period / 2) % best_period) - best_period / 2
            transit_mask = np.abs(phase) <= (best_duration_days / 2)

            radius_estimate = _depth_to_earth_radii(depth)

            return {
                "detected": bool(detected),
                "confidence_pct": round(float(confidence), 2),
                "orbital_period_days": round(best_period, 4),
                "transit_duration_hours": round(best_duration_days * 24, 3),
                "transit_depth_ppm": round(depth * 1e6, 2),
                "planet_radius_estimate_earth": round(radius_estimate, 3),
                "snr": round(float(snr), 2),
                "transit_mask": transit_mask,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("astropy BLS failed (%s); using fallback BLS grid search", exc)

    # ---- Lightweight numpy fallback BLS-style grid search -------------
    return _fallback_bls(t, f, period_grid, duration_grid, max_duty_cycle=MAX_DUTY_CYCLE)


def _fallback_bls(
    t: np.ndarray,
    f: np.ndarray,
    period_grid: np.ndarray,
    duration_grid: np.ndarray,
    max_duty_cycle: float = 0.15,
) -> dict:
    """
    Minimal custom periodogram: for each trial period & duration, fold the
    light curve and measure the depth/significance of the in-transit mean
    vs out-of-transit mean. Picks the configuration maximizing significance.
    Coarser & slower than astropy's BLS but dependency-free.

    `max_duty_cycle` enforces the same physical sanity constraint as the
    astropy path: a transit duration must be a small fraction of the
    orbital period, so we never flag the majority of the phase-folded
    curve as "in transit".
    """
    best = {"score": -np.inf}
    f_mean = np.mean(f)
    f_std = np.std(f) or 1e-6

    # Subsample period grid for speed in the pure-python fallback path
    coarse_periods = period_grid[::8]

    for period in coarse_periods:
        phase = (t % period) / period
        for dur_days in duration_grid:
            # Skip any duration/period combo with an unrealistic duty cycle
            if (dur_days / period) > max_duty_cycle:
                continue
            dur_phase = dur_days / period
            for t0_phase in np.linspace(0, 1, 12, endpoint=False):
                ph = (phase - t0_phase + 0.5) % 1.0 - 0.5
                in_tr = np.abs(ph) <= dur_phase / 2
                n_in = np.sum(in_tr)
                if n_in < 5 or n_in > len(f) * 0.5:
                    continue
                in_mean = np.mean(f[in_tr])
                out_mean = np.mean(f[~in_tr])
                depth = out_mean - in_mean
                if depth <= 0:
                    continue
                score = (depth / f_std) * math.sqrt(n_in)
                if score > best["score"]:
                    best = {
                        "score": score,
                        "period": period,
                        "duration_days": dur_days,
                        "t0_phase": t0_phase,
                        "depth": depth,
                        "in_tr_mask": in_tr,
                    }

    if best.get("score", -np.inf) <= 0 or "period" not in best:
        return _empty_detection_result()

    confidence = _power_to_confidence(best["score"])
    detected = confidence >= 55.0

    radius_estimate = _depth_to_earth_radii(best["depth"])

    return {
        "detected": bool(detected),
        "confidence_pct": round(float(confidence), 2),
        "orbital_period_days": round(float(best["period"]), 4),
        "transit_duration_hours": round(float(best["duration_days"]) * 24, 3),
        "transit_depth_ppm": round(float(best["depth"]) * 1e6, 2),
        "planet_radius_estimate_earth": round(radius_estimate, 3),
        "snr": round(float(best["score"]), 2),
        "transit_mask": best["in_tr_mask"],
    }


def _power_to_confidence(snr_like: float) -> float:
    """Map an SNR-like statistic to an intuitive 0-100% confidence score
    using a logistic squashing function."""
    if snr_like is None or np.isnan(snr_like):
        return 0.0
    confidence = 100 / (1 + math.exp(-0.55 * (snr_like - 4)))
    return float(np.clip(confidence, 0, 99.9))


def _depth_to_earth_radii(depth_fraction: float) -> float:
    """
    Rough planet-radius estimate from transit depth:
        depth ≈ (Rp / Rstar)^2
    Assuming a Sun-like host star (Rstar ≈ 1 Rsun ≈ 109 Rearth) for a
    quick order-of-magnitude estimate suitable for a dashboard display.
    """
    if depth_fraction <= 0:
        return 0.0
    r_star_earth_radii = 109.0
    radius_ratio = math.sqrt(depth_fraction)
    return radius_ratio * r_star_earth_radii


def _empty_detection_result() -> dict:
    return {
        "detected": False,
        "confidence_pct": 0.0,
        "orbital_period_days": None,
        "transit_duration_hours": None,
        "transit_depth_ppm": None,
        "planet_radius_estimate_earth": None,
        "snr": 0.0,
        "transit_mask": None,
    }


# ==========================================================================
# STAGE 4 — API ROUTES
# ==========================================================================
@app.get("/", tags=["health"])
def root():
    return {
        "service": "Exoplanet Transit Detection API",
        "status": "online",
        "lightkurve_available": LIGHTKURVE_AVAILABLE,
        "astropy_bls_available": ASTROPY_BLS_AVAILABLE,
        "docs": "/docs",
    }


@app.get("/api/health", tags=["health"])
def health_check():
    return {"status": "ok"}


@app.get("/api/analyze-star", response_model=AnalyzeResponse, tags=["analysis"])
def analyze_star(
    star_id: str = Query(
        ...,
        min_length=2,
        description="Kepler/TESS/K2 target identifier, e.g. 'KIC 10593626' or 'TIC 261136665'",
    ),
):
    """
    Full pipeline endpoint: ingest -> denoise -> BLS detect -> return JSON.

    Always returns a 200 with a usable payload — if the real archive is
    unreachable, `data_source` will indicate the synthetic fallback was
    used, but the dashboard keeps working uninterrupted.
    """
    star_id = star_id.strip()
    if len(star_id) == 0:
        raise HTTPException(status_code=400, detail="star_id must not be empty")

    logger.info("Received analysis request for star_id='%s'", star_id)

    try:
        # ---- Stage 1: Ingestion ----
        time, flux_raw, source, mission = fetch_light_curve(star_id)

        # ---- Stage 2: Denoising ----
        flux_clean = denoise_pipeline(time, flux_raw)

        # ---- Stage 3: Detection ----
        bls_result = run_bls_detection(time, flux_clean)
        transit_mask = bls_result.pop("transit_mask")
        if transit_mask is None:
            transit_mask = np.zeros_like(flux_clean, dtype=bool)

        metadata = TransitMetadata(**bls_result)

        notes = (
            "Live data retrieved from MAST via lightkurve."
            if source == "lightkurve"
            else "Demo-safe synthetic light curve used (network/archive unavailable or target not found)."
        )

        response = AnalyzeResponse(
            star_id=star_id,
            data_source=source,
            mission=mission,
            n_points=len(time),
            time=np.round(time, 6).tolist(),
            flux_raw=np.round(flux_raw, 6).tolist(),
            flux_clean=np.round(flux_clean, 6).tolist(),
            transit_mask=[bool(x) for x in transit_mask],
            metadata=metadata,
            notes=notes,
        )
        return response

    except Exception as exc:  # noqa: BLE001 - last-resort guard for live demo stability
        logger.exception("Unexpected error analyzing star_id='%s'", star_id)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc


@app.exception_handler(404)
async def not_found_handler(request, exc):  # noqa: ANN001
    return JSONResponse(status_code=404, content={"detail": "Route not found. See /docs for available endpoints."})


# ==========================================================================
# Entrypoint (for `python app.py` direct execution)
# ==========================================================================
if __name__ == "__main__":
    import os
    import uvicorn

    # Render (and most cloud hosts) assign a port dynamically via the PORT
    # environment variable. Locally, if PORT isn't set, we fall back to
    # 8000 exactly as before, so this still works unchanged on your PC.
    port = int(os.environ.get("PORT", 8000))
    is_local = "PORT" not in os.environ

    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=is_local)
