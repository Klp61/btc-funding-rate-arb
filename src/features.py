# src/features.py

import polars as pl
import math


def add_basis(df):

    return df.with_columns([
        (
            pl.col("perp_open") - pl.col("spot_open")
        ).alias("basis"),

        (
            (pl.col("perp_open") - pl.col("spot_open"))
            / pl.col("spot_open")
        ).alias("basis_pct"),
    ])


def add_basis_features(df):

    return df.with_columns([
        pl.col("basis")
        .rolling_mean(8)
        .alias("basis_twap_8h"),

        pl.col("basis_pct")
        .rolling_std(8)
        .alias("basis_vol_8h"),
    ])


def add_funding_features(df: pl.DataFrame):

    df = df.with_columns([
        pl.col("timestamp").dt.hour().alias("hour"),
    ])

    df = df.with_columns([
        pl.col("hour")
        .is_in([0, 8, 16])
        .alias("is_funding_time"),
    ])

    df = df.with_columns([
        pl.when(pl.col("is_funding_time"))
        .then(pl.col("funding_rate"))
        .otherwise(None)
        .alias("funding_rate_event")
    ])

    df = df.with_columns([

        pl.col("funding_rate")
        .rolling_mean(window_size=72) #using past 3 funding rates
        .alias("funding_mean"),

        pl.col("funding_rate")
        .rolling_std(window_size=72) #using past 3 funding rates
        .alias("funding_std"),
    ])

    df = df.with_columns([
        pl.when(pl.col("funding_std") > 0)
        .then(
            (
                pl.col("funding_rate")
                - pl.col("funding_mean")
            )
            / pl.col("funding_std")
        )
        .otherwise(None)
        .alias("funding_z")
    ])

    return df.drop(["hour", "is_funding_time"])


def add_volatility_features(
    df: pl.DataFrame,
    ewma_half_life: int = 24
):

    # =========================
    # LOG RETURNS
    # =========================

    df = df.with_columns([
        (
            pl.col("perp_mark")
            .log()
            .diff()
        ).alias("log_return")
    ])

    # =========================
    # ANNUALIZED REALIZED VOL
    # =========================

    annualization_factor = math.sqrt(24 * 365)

    df = df.with_columns([
        (
            pl.col("log_return")
            .rolling_std(window_size=24)
            * annualization_factor
        ).alias("realized_vol_24h_ann")
    ])

    # =========================
    # EWMA VOLATILITY
    # =========================

    # decay factor
    lam = math.exp(-math.log(2) / ewma_half_life)

    var = (
        (pl.col("log_return") ** 2)
        .ewm_mean(alpha=(1 - lam), adjust=False)
        )
    
    df = df.with_columns([
        var.alias("ewma_variance"),
        var.sqrt().alias("realized_vol_ewma")
    ])

    return df

def add_basis_premium(df: pl.DataFrame):
    df = df.with_columns([
        (
            (
                pl.col("perp_mark")
                - pl.col("index_price")
            )
            / pl.col("index_price")
        ).alias("basis_premium_signed"),
        (
            (
                pl.col("perp_mark")
                - pl.col("index_price")
            ).abs()
            / pl.col("index_price")
        ).alias("basis_premium")
    ])
    return df