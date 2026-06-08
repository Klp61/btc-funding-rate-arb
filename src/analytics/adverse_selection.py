#src/analytics/adverse_selection.py


import numpy as np
import polars as pl
from scipy import stats

def _kyle_lambda(
    prices:  np.ndarray,
    volumes: np.ndarray,
    window:  int,
    min_obs: int,
) -> tuple:

    n          = len(prices)
    lambdas    = np.full(n, np.nan)
    tstats     = np.full(n, np.nan)

    delta_p    = np.diff(prices, prepend=np.nan)
    signed_vol = np.sign(delta_p) * volumes

    for i in range(window, n):
        x    = signed_vol[i - window : i]
        y    = delta_p[i - window : i]
        mask = np.isfinite(x) & np.isfinite(y)

        if mask.sum() < min_obs:
            continue

        slope, _, _, _, se = stats.linregress(x[mask], y[mask])
        lambdas[i] = slope
        tstats[i]  = slope / se if se > 0 else np.nan

    return lambdas, tstats


def estimate_kyle_lambda_perp(
    df:         pl.DataFrame,
    price_col:  str = "perp_open",
    volume_col: str = "perp_volume",
    window:     int = 168,
    min_obs:    int = 30,
) -> pl.DataFrame:

    df = df.sort("timestamp")

    lambdas, tstats = _kyle_lambda(
        df[price_col].to_numpy().astype(float),
        df[volume_col].to_numpy().astype(float),
        window, min_obs,
    )

    return df.with_columns([
        pl.Series("kyle_lambda_perp",       lambdas.tolist()),
        pl.Series("kyle_lambda_perp_tstat", tstats.tolist()),
    ])


def calibrate_impact_beta(
    df:       pl.DataFrame,
    notional: float = 100_000,
) -> float:

    required = ["kyle_lambda_perp", "perp_mark", "perp_volume", "realized_vol_ewma"]
    for col in required:
        if col not in df.columns:
            return 0.08

    lam_vals = df["kyle_lambda_perp"].drop_nulls()
    if lam_vals.is_empty():
        return 0.08

    median_lambda   = float(lam_vals.median())
    median_price    = float(df["perp_mark"].median())
    median_vol_btc  = float(df["perp_volume"].median())   
    median_ewma_vol = float(df["realized_vol_ewma"].drop_nulls().median())

    if median_price <= 0 or median_ewma_vol <= 0:
        return 0.08

    trade_size_btc    = notional / median_price
    hourly_volume_btc = max(median_vol_btc, 1e-8)         

    dollar_vol = median_ewma_vol * median_price # scaled to dollar terms

    impact_beta = (
        median_lambda
        * np.sqrt(trade_size_btc * hourly_volume_btc)
        / dollar_vol
    )

    return float(np.clip(impact_beta, 0.001, 0.50))