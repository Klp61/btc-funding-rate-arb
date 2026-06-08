# performance metrics

#to calculate statistics to infer results from backtest engine

import polars as pl
import numpy as np 
import math

def compute_statistics(
    trades_df: pl.DataFrame,
    daily_df: pl.DataFrame,
) -> dict:

    if trades_df.is_empty():
        return {}

    trade_pnls = (
        trades_df["total_pnl"]
        .to_numpy()
        .astype(float)
    )

    daily_pnls = (
        daily_df["daily_pnl"]
        .to_numpy()
        .astype(float)
    )

    # sharpe

    mu = np.mean(daily_pnls)

    sd = np.std(daily_pnls, ddof=1)

    sharpe = (
        mu / sd * math.sqrt(252)
        if sd > 0
        else np.nan
    )

    # drawdown

    equity = np.cumsum(daily_pnls)

    running_max = np.maximum.accumulate(equity)

    drawdown = equity - running_max

    max_drawdown = float(np.min(drawdown))

    return {

        "n_trades":
            int(len(trade_pnls)),

        "total_pnl":
            float(np.sum(trade_pnls)),
        
        "avg_trade_pnl":
            float(np.mean(trade_pnls)),

        "win_rate":
            float(np.mean(trade_pnls > 0)),

        "sharpe":
            float(sharpe),

        "max_drawdown":
            float(max_drawdown),
    }