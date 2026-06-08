# src/data_fetch.py

import requests
import time
import polars as pl
from .config import BASE_SPOT, BASE_FUTURES, INTERVAL, MS_IN_DAY, LOOKBACK_DAYS

START_TIME = int(time.time() * 1000) - (LOOKBACK_DAYS * MS_IN_DAY)


def safe_get(url, params):
    r = requests.get(url, params=params)
    data = r.json()

    if isinstance(data, dict):
        raise ValueError(f"Binance API error: {data}")

    return data


def get_historical_klines(url, params, start_time, limit=1000):
    all_data = []

    while True:
        params["startTime"] = start_time
        params["limit"] = limit

        data = safe_get(url, params)

        if not data:
            break

        all_data.extend(data)
        start_time = data[-1][0] + 1

        if len(data) < limit:
            break

        time.sleep(0.2)

    return all_data


def get_spot_open(symbol):
    url = f"{BASE_SPOT}/api/v3/klines"
    params = {"symbol": symbol, "interval": INTERVAL}

    data = get_historical_klines(url, params, START_TIME)

    return pl.DataFrame({
        "timestamp": [int(r[0]) for r in data],
        "spot_open": [ float(r[1]) for r in data],  # OPEN PRICE
        "spot_volume":  [float(r[5]) for r in data],
    })


def get_perp_mark(symbol):
    url = f"{BASE_FUTURES}/fapi/v1/markPriceKlines"
    params = {"symbol": symbol, "interval": INTERVAL}

    data = get_historical_klines(url, params, START_TIME)

    return pl.DataFrame({
        "timestamp": [int(r[0]) for r in data],
        "perp_mark": [float(r[1]) for r in data],
    })

def get_perp_open(symbol):
    url = f"{BASE_FUTURES}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": INTERVAL}

    data = get_historical_klines(url, params, START_TIME)

    return pl.DataFrame({
    "timestamp": [int(r[0]) for r in data],

    "perp_open": [
        float(r[1])
        for r in data
    ],

    "perp_volume": [
        float(r[5]) #base asset volume
        for r in data
    ],
})


def get_index_price(symbol):
    url = f"{BASE_FUTURES}/fapi/v1/indexPriceKlines"
    params = {"pair": symbol, "interval": INTERVAL}

    data = get_historical_klines(url, params, START_TIME)

    return pl.DataFrame({
        "timestamp": [int(r[0]) for r in data],
        "index_price": [float(r[1]) for r in data],
    })


def get_funding(symbol):
    url = f"{BASE_FUTURES}/fapi/v1/fundingRate"
    params = {"symbol": symbol, "limit": 1000}

    all_data = []
    start_time = START_TIME

    while True:
        params["startTime"] = start_time
        data = safe_get(url, params)

        if not data:
            break

        all_data.extend(data)
        start_time = data[-1]["fundingTime"] + 1

        if len(data) < 1000:
            break

        time.sleep(0.2)

    return pl.DataFrame({
        "timestamp": [int(r["fundingTime"]) for r in all_data],
        "funding_rate": [float(r["fundingRate"]) for r in all_data],
    })