#src/analytics/funding_alpha.py

import numpy as np
import polars as pl
from scipy.optimize import curve_fit
from typing import Optional, Dict


def _saturation(h, A, lam):
    return A * (1 - np.exp(-lam * h))


def run_carry_analysis(
    df:                pl.DataFrame,
    funding_threshold: float = 0.00008,
    notional:          float = 100_000,
    max_horizon:       int   = 2688,
    n_bootstrap:       int   = 300,
    min_entries:       int   = 10,
    seed:              int   = 42,
) -> Dict:
    """
    Args:
        df                : output of build_dataset()
        funding_threshold : entry threshold in rate terms
        notional          : dollar notional per trade
        max_horizon       : hours to track carry after entry
        n_bootstrap       : bootstrap samples for half_saturation CI
        min_entries       : minimum valid entry curves required
        seed              : random seed

    Returns dict with:
        summary    -- two numbers, printable
        A          -- max carry scalar
        half_sat   -- half saturation time in hours
        ci_lo/hi   -- 90% CI on half_sat
        curve_df   -- average cumulative curve DataFrame
        n_entries  -- number of entry events used
    """
    df = df.sort("timestamp")

    # ── derive position from funding rate ─────────────────
    df = df.with_columns([
        pl.when(pl.col("funding_rate") >  funding_threshold).then(pl.lit(-1.0))
          .when(pl.col("funding_rate") < -funding_threshold).then(pl.lit(1.0))
          .otherwise(pl.lit(0.0))
          .alias("_position")
    ])

    # ── per-bar funding PnL ────────────────────────────────
    pos  = df["_position"].to_numpy().astype(float)
    fr   = np.nan_to_num(
               df["funding_rate_event"].to_numpy().astype(float),
               nan=0.0
           )
    fpnl = -pos * fr * notional

    df = df.with_columns([
        pl.Series("_fpnl", fpnl.tolist())
    ])

    # ── average cumulative curve ───────────────────────────
    signal = df["_position"].to_numpy().astype(float)
    n      = len(fpnl)
    curves = []

    for i in np.where(signal != 0)[0]:
        if i + max_horizon >= n:
            continue
        cum, curve = 0.0, []
        for h in range(1, max_horizon + 1):
            cum += fpnl[i + h]
            curve.append(cum)
        curves.append(curve)

    if len(curves) < min_entries:
        return {
            "error": (
                f"Only {len(curves)} complete entries found, "
                f"need {min_entries}. "
                f"Lower funding_threshold or use more data."
            )
        }

    arr      = np.array(curves)
    mean_c   = np.mean(arr, axis=0)
    std_c    = np.std(arr, axis=0, ddof=1)
    h_vals   = np.arange(1, max_horizon + 1, dtype=float)

    curve_df = pl.DataFrame({
        "horizon":   h_vals.tolist(),
        "cum_pnl":   mean_c.tolist(),
        "std_pnl":   std_c.tolist(),
        "n_entries": [len(curves)] * max_horizon,
    })

    # ── fit saturation model ───────────────────────────────
    try:
        (A, lam), _ = curve_fit(
            _saturation, h_vals, mean_c,
            p0=(max(float(np.max(mean_c)), 1.0), 0.1),
            bounds=([0, 1e-6], [1e9, 10.0]),
            maxfev=5000,
        )
    except RuntimeError:
        return {"error": "Saturation model failed to converge."}

    half_sat = float(np.log(2) / lam) if lam > 0 else None

    # ── bootstrap CI on half_sat ───────────────────────────
    rng     = np.random.default_rng(seed)
    samples = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(h_vals), size=len(h_vals))
        try:
            (_, l2), _ = curve_fit(
                _saturation, h_vals[idx], mean_c[idx],
                p0=(A, lam),
                bounds=([0, 1e-6], [1e9, 10.0]),
                maxfev=2000,
            )
            hs = np.log(2) / l2 if l2 > 0 else None
            if hs and np.isfinite(hs):
                samples.append(float(hs))
        except RuntimeError:
            continue

    ci_lo = float(np.percentile(samples, 5))  if len(samples) >= 30 else None
    ci_hi = float(np.percentile(samples, 95)) if len(samples) >= 30 else None

    ci_str = (
        f"[{ci_lo:.1f}h, {ci_hi:.1f}h]"
        if ci_lo else "n/a"
    )

    # ── summary ────────────────────────────────────────────
    summary = {
        "Max carry (A)":           f"${A:.2f}",
        "Half-saturation":         f"{half_sat:.1f}h" if half_sat else "n/a",
        "Half-saturation 90% CI":  ci_str,
        "Entries used":            len(curves),
        "Threshold":               f"{funding_threshold*10000:.1f}bps",
        "Notional":                f"${notional:,.0f}",
    }

    return {
        "summary":  summary,
        "A":        float(A),
        "half_sat": half_sat,
        "ci_lo":    ci_lo,
        "ci_hi":    ci_hi,
        "curve_df": curve_df,
        "n_entries": len(curves),
    }