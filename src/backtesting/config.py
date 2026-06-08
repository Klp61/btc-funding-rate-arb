#src/backtesting/config.py

from dataclasses import dataclass

@dataclass
class BacktestConfig:
    funding_threshold: float
    hold_days: int

    trade_notional: float = 100_000
    spot_fee: float = 0.001
    perp_fee: float = 0.0004

    spread_alpha: float = 0.5
    impact_beta: float = 0.0058 #0.0057716419485706475

    min_trade_gap_hours: int = 8