# src/dataset.py

import polars as pl

from .data_fetch import (
    get_spot_open,
    get_perp_open,
    get_perp_mark,
    get_index_price,
    get_funding
)

from .features import (
    add_basis,
    add_basis_features,
    add_funding_features,
    add_volatility_features,
    add_basis_premium
)

def build_dataset(symbol):
    spot = get_spot_open(symbol)
    perp = get_perp_open(symbol)
    perp_mark = get_perp_mark(symbol)
    index = get_index_price(symbol)
    funding = get_funding(symbol)

    # timestamp conversion
    spot = spot.with_columns(pl.from_epoch("timestamp", time_unit="ms"))
    perp = perp.with_columns(pl.from_epoch("timestamp", time_unit="ms"))
    perp_mark = perp_mark.with_columns(pl.from_epoch("timestamp", time_unit="ms"))
    index = index.with_columns(pl.from_epoch("timestamp", time_unit="ms"))
    funding = funding.with_columns(pl.from_epoch("timestamp", time_unit="ms"))

    # merge core market data
    base = spot.select("timestamp").sort("timestamp")
    df = base.join(spot, on="timestamp", how="left")
    df = df.join(perp, on="timestamp", how="left")
    df = df.join(index, on="timestamp", how="left")
    df = df.join(perp_mark, on="timestamp", how="left")

    # funding (as-of join)
    df = df.join_asof(
        funding.sort("timestamp"),
        on="timestamp",
        strategy="backward"
    )

    df = df.with_columns(pl.lit(symbol).alias("asset"))

    df = add_basis(df)
    df = add_basis_features(df)
    df = add_funding_features(df)
    df = add_basis_premium(df)
    df = add_volatility_features(df)

    cols = df.columns
    cols.remove("funding_rate")
    idx = cols.index("basis_vol_8h") + 1
    cols.insert(idx, "funding_rate")

    df = df.select(cols)

    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)

    return df.sort("timestamp")