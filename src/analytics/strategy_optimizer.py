#src/analytics/strategy_optimizer.py

import numpy as np
import polars as pl
from dataclasses import dataclass, field
from scipy import stats
import math
from datetime import timedelta, datetime
from datetime import date

def _fast_backtest(
    df: pl.DataFrame,
    funding_threshold: float,
    hold_days: int,
    spot_fee: float        = 0.001,
    perp_fee: float        = 0.0004,
    trade_notional: float  = 100_000,
    spread_alpha: float    = 0.5,  #crossing half the spread each time
    impact_beta: float     = 0.0058,
) -> dict:

    df = df.sort("timestamp")

    spot    = df["spot_open"].to_numpy().astype(float)
    perp    = df["perp_open"].to_numpy().astype(float)
    fund    = df["funding_rate"].to_numpy().astype(float)
    vol     = df["realized_vol_ewma"].to_numpy().astype(float)
    pvol    = df["perp_volume"].to_numpy().astype(float)
    spread  = df["basis_premium"].to_numpy().astype(float)
    fund_ev = df["funding_rate_event"].to_numpy().astype(float)
    times   = df["timestamp"].to_list()
    n       = len(spot)

    # build a date -> daily_pnl dict for annualised Sharpe
    daily_pnl: dict = {}
    trade_pnls = []
    i = 0

    while i < n:
        fr = fund[i]
        if not np.isfinite(fr):
            i += 1
            continue

        if fr > funding_threshold:
            position = -1
        elif fr < -funding_threshold:
            position = 1
        else:
            i += 1
            continue

        entry_idx = i + 1
        if entry_idx >= n:
            break

        exit_target = times[entry_idx] + timedelta(days=hold_days)
        exit_idx = None
        for j in range(entry_idx + 1, n):
            if times[j] >= exit_target:
                exit_idx = j
                break
        if exit_idx is None:
            break
        
        entry_spot = spot[entry_idx]
        entry_perp = perp[entry_idx]
        exit_spot  = spot[exit_idx]
        exit_perp  = perp[exit_idx]
        
        trade_size_btc_spot = trade_notional / entry_spot
        trade_size_btc_perp = trade_notional / entry_perp
        
        # spot pnl

        spot_pnl = -position * (exit_spot - entry_spot) * trade_size_btc_spot

        # perp price pnl

        perp_pnl = position * (exit_perp - entry_perp) * trade_size_btc_perp
        
        # funding pnl

        funding_pnl = 0.0
        
        for j in range(entry_idx, exit_idx):
            fr_ev = fund_ev[j]
            if np.isfinite(fr_ev) and fr_ev != 0.0:
                funding_pnl += -position * fr_ev * perp[j] * trade_size_btc_perp
        
        # fees

        fees = -(spot_fee + perp_fee) * trade_notional * 2

        # slippage (same model as full backtest)
        trade_size_btc = trade_notional / entry_perp
        hourly_vol_btc = max(pvol[entry_idx] / entry_perp, 1e-8)
        ev               = vol[entry_idx] if np.isfinite(vol[entry_idx]) else 0.0
        sp               = spread[entry_idx] if np.isfinite(spread[entry_idx]) else 0.0
        spread_cost  = spread_alpha * sp * trade_notional
        impact_cost  = impact_beta * ev * math.sqrt(trade_size_btc / hourly_vol_btc) * trade_notional
        slippage_pnl = -(spread_cost + impact_cost)

        total = spot_pnl + perp_pnl + funding_pnl + fees + slippage_pnl
        trade_pnls.append(total)

        # assign to entry date for daily series
        entry_date = times[entry_idx].date() if hasattr(times[entry_idx], 'date') else times[entry_idx]
        daily_pnl[entry_date] = daily_pnl.get(entry_date, 0.0) + total

        i = exit_idx + 1

    if len(trade_pnls) < 3:
        return {"sharpe": np.nan, "total_pnl": np.nan,
                "win_rate": np.nan, "n_trades": 0}

    arr = np.array(trade_pnls)
    
    if len(arr) < 3:
        return {
            "sharpe": np.nan,
            "total_pnl": np.nan,
            "win_rate": np.nan,
            "n_trades": 0
            }
    
    # build full daily series with zeros on non-trading days
    start_date = times[0].date() if hasattr(times[0], 'date') else times[0]
    end_date   = times[-1].date() if hasattr(times[-1], 'date') else times[-1]

    all_dates  = [start_date + timedelta(days=k) 
              for k in range((end_date - start_date).days + 1)]
    daily_vals = np.array([daily_pnl.get(d, 0.0) for d in all_dates])
    mu  = np.mean(daily_vals)
    sd  = np.std(daily_vals, ddof=1)
    sharpe = (mu / sd * math.sqrt(252)) if sd > 0 else np.nan


    return {
    "sharpe": float(sharpe),
    "total_pnl": float(np.sum(arr)),
    "win_rate": float(np.mean(arr > 0)),
    "n_trades": len(arr),
    "avg_slippage": float(-(spread_cost + impact_cost)) if len(trade_pnls) > 0 else np.nan,
    }


@dataclass
class OptimiserConfig:
    thresholds:      list  = field(default_factory=lambda: [
                                0.00004, 0.00006, 0.00008,
                                0.00010, 0.00015, 0.00020, 0.00030])
    hold_days:       list  = field(default_factory=lambda: [14, 21, 28, 35, 42, 56, 63, 70, 77])
    trade_notional:  float = 100_000
    spot_fee:        float = 0.001
    perp_fee:        float = 0.0004
    spread_alpha:    float = 0.5
    impact_beta:     float = 0.0058


def run_grid_search(
    df: pl.DataFrame,
    config: OptimiserConfig = None,
) -> pl.DataFrame:

    if config is None:
        config = OptimiserConfig()

    rows = []
    for thresh in config.thresholds:
        for hold in config.hold_days:
            
            result = _fast_backtest(
            df,
            thresh,
            hold,
            config.spot_fee,
            config.perp_fee,
            config.trade_notional,
            config.spread_alpha,
            config.impact_beta,
        )
            
            rows.append({
                "threshold": thresh,
                "hold_days": hold,
                **result,
            })

    return pl.DataFrame(rows)


def find_optimal_params(
    grid_df: pl.DataFrame,
    metric: str = "sharpe",
    min_trades: int = 1,
) -> dict:
    valid = grid_df.filter(pl.col("n_trades") >= min_trades).drop_nulls(metric)
    if valid.is_empty():
        return {}

    best_row = valid.sort(metric, descending=True).row(0, named=True)

    return {
        "best_threshold": best_row["threshold"],
        "best_hold_days": best_row["hold_days"],
        f"best_{metric}": round(float(best_row[metric]), 3),
        "n_trades":       best_row["n_trades"],
        "total_pnl":      round(float(best_row["total_pnl"]), 2),
        "win_rate":       round(float(best_row["win_rate"]), 3),
    }