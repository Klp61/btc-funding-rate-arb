#src/backtesting/engine.py

import polars as pl 
from .config import BacktestConfig
import numpy as np
from datetime import timedelta
import math


def run_backtest(
    df: pl.DataFrame,
    config: BacktestConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Returns:
        trades_df
        daily_pnl_df
    """

    df = df.sort("timestamp")

    spot = df["spot_open"].to_numpy().astype(float)
    perp = df["perp_open"].to_numpy().astype(float)

    fund = df["funding_rate"].to_numpy().astype(float)
    fund_ev = df["funding_rate_event"].to_numpy().astype(float)

    vol = df["realized_vol_ewma"].to_numpy().astype(float)
    pvol = df["perp_volume"].to_numpy().astype(float)
    spread = df["basis_premium"].to_numpy().astype(float)

    times = df["timestamp"].to_list()

    n = len(df)

    trades = []

    i = 0
    last_exit_time = None

    while i < n:

        fr = fund[i]

        if not np.isfinite(fr):
            i += 1
            continue

        #signal

        if fr > config.funding_threshold:
            position = -1

        elif fr < -config.funding_threshold:
            position = 1

        else:
            i += 1
            continue

        entry_idx = i + 1

        if entry_idx >= n:
            break

        entry_time = times[entry_idx]

        #optional trade cooldown

        if last_exit_time is not None:

            delta_hours = (
                entry_time - last_exit_time
            ).total_seconds() / 3600

            if delta_hours < config.min_trade_gap_hours:
                i += 1
                continue

        # exit search

        exit_target = entry_time + timedelta(days=config.hold_days)

        exit_idx = None

        for j in range(entry_idx + 1, n):

            if times[j] >= exit_target:
                exit_idx = j
                break

        if exit_idx is None:
            break

        exit_time = times[exit_idx]

        # prices

        entry_spot = spot[entry_idx]
        exit_spot = spot[exit_idx]

        entry_perp = perp[entry_idx]
        exit_perp = perp[exit_idx]

        # position_size

        trade_size_spot = (
            config.trade_notional / entry_spot
        )

        trade_size_perp = (
            config.trade_notional / entry_perp
        )

        # price_pnls

        spot_pnl = (
            -position
            * (exit_spot - entry_spot)
            * trade_size_spot
        )

        perp_pnl = (
            position
            * (exit_perp - entry_perp)
            * trade_size_perp
        )

        # funding_pnl

        funding_pnl = 0.0

        for j in range(entry_idx, exit_idx):

            fr_ev = fund_ev[j]

            if np.isfinite(fr_ev) and fr_ev != 0.0:

                funding_pnl += (
                    -position
                    * fr_ev
                    * perp[j]
                    * trade_size_perp
                )

        # fees

        fees = (
            -(config.spot_fee + config.perp_fee)
            * config.trade_notional
            * 2
        )

        # slippage

        trade_size_btc = (
            config.trade_notional / entry_perp
        )

        hourly_vol_btc = max(
            pvol[entry_idx] / entry_perp,
            1e-8,
        )

        ev = (
            vol[entry_idx]
            if np.isfinite(vol[entry_idx])
            else 0.0
        )

        sp = (
            spread[entry_idx]
            if np.isfinite(spread[entry_idx])
            else 0.0
        )

        spread_cost = (
            config.spread_alpha
            * sp
            * config.trade_notional
        )

        impact_cost = (
            config.impact_beta
            * ev
            * math.sqrt(
                trade_size_btc / hourly_vol_btc
            )
            * config.trade_notional
        )

        slippage = -(
            spread_cost + impact_cost
        )

        # total

        total_pnl = (
            spot_pnl
            + perp_pnl
            + funding_pnl
            + fees
            + slippage
        )

        trades.append({
            "entry_time": entry_time,
            "exit_time": exit_time,

            "position": position,

            "entry_spot": entry_spot,
            "exit_spot": exit_spot,

            "entry_perp": entry_perp,
            "exit_perp": exit_perp,

            "spot_pnl": spot_pnl,
            "perp_pnl": perp_pnl,
            "funding_pnl": funding_pnl,

            "fees": fees,
            "slippage": slippage,

            "total_pnl": total_pnl,
        })

        last_exit_time = exit_time

        i = exit_idx + 1

    trades_df = pl.DataFrame(trades)

    # =====================================================
    # daily_pnl_series
    # =====================================================

    if trades_df.is_empty():

        daily_df = pl.DataFrame({
            "date": [],
            "daily_pnl": [],
        })

        return trades_df, daily_df
    
    # full_equity_curve

    if trades_df.is_empty():
        daily_df = pl.DataFrame({"date": [], "daily_pnl": []})
        return trades_df, daily_df
    
    min_date = trades_df["entry_time"].min().date()
    max_date = trades_df["exit_time"].max().date()
    
    all_dates = pl.date_range(
        min_date,
        max_date,
        interval="1d",
        eager=True
    ).to_list()
    
    daily_df = (
        trades_df
        .with_columns(
            pl.col("entry_time").dt.date().alias("date")
        )
        .group_by("date")
        .agg(pl.col("total_pnl").sum().alias("daily_pnl"))
    )
    
    # convert to dict for alignment
    pnl_map = dict(zip(daily_df["date"], daily_df["daily_pnl"]))
    
    # fill missing days with 0
    daily_df = pl.DataFrame({
        "date": all_dates,
        "daily_pnl": [
            pnl_map.get(d, 0.0) for d in all_dates
        ]
    })

    return trades_df, daily_df
