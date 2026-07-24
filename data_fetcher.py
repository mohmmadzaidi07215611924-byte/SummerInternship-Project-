# data_fetcher.py
# Downloads stock prices using yfinance and saves them to a csv file
# so we don't have to keep hitting Yahoo Finance every time we run the app.
#
# Run this file directly to build/refresh the cache:
#   python data_fetcher.py

import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

import config


def fetch_price_data(tickers, period=config.DEFAULT_PERIOD, interval=config.DEFAULT_INTERVAL):
    # download data for all tickers at once
    print(f"Downloading {len(tickers)} tickers from Yahoo Finance...")
    data = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    if data.empty:
        raise RuntimeError("yfinance returned no data. Check internet connection or ticker list.")

    # when you download multiple tickers, yfinance gives back columns like
    # (ticker, "Close"), (ticker, "Open") etc. so we just grab the Close price
    # for each ticker and put it all into one simple dataframe.
    close_prices = {}
    if isinstance(data.columns, pd.MultiIndex):
        for ticker in tickers:
            if ticker in data.columns.get_level_values(0):
                col = data[ticker]["Close"]
                if col.notna().sum() > 0:
                    close_prices[ticker] = col
                else:
                    print(f"  skipping {ticker}, no data")
            else:
                print(f"  skipping {ticker}, not found")
        prices = pd.DataFrame(close_prices)
    else:
        # only one ticker was requested
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})

    prices.index.name = "Date"
    prices = prices.sort_index()
    return prices


def clean_price_data(prices, max_missing_fraction=config.MAX_MISSING_FRACTION):
    # drop tickers that have too many missing days (delisted, IPO'd late, etc)
    missing = prices.isna().mean()
    good_tickers = missing[missing <= max_missing_fraction].index.tolist()

    dropped = set(prices.columns) - set(good_tickers)
    if dropped:
        print(f"  dropping tickers with too much missing data: {sorted(dropped)}")

    prices = prices[good_tickers]
    prices = prices.ffill().bfill()   # fill small gaps
    prices = prices.dropna(axis=0)    # drop any row that still has a gap
    return prices


def cache_is_old(cache_path, max_age_hours):
    if not os.path.exists(cache_path):
        return True
    last_modified = datetime.fromtimestamp(os.path.getmtime(cache_path))
    return datetime.now() - last_modified > timedelta(hours=max_age_hours)


def get_data(tickers=None, period=config.DEFAULT_PERIOD, cache_path=config.CACHE_PATH,
             force_refresh=False, max_age_hours=config.CACHE_MAX_AGE_HOURS):
    # use the cached csv if it exists and isn't too old, otherwise download fresh data
    tickers = tickers or config.DEFAULT_TICKERS

    if not force_refresh and not cache_is_old(cache_path, max_age_hours):
        print(f"Loading cached data from {cache_path}")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    prices = fetch_price_data(tickers, period=period)
    prices = clean_price_data(prices)

    if prices.shape[1] < 2:
        raise RuntimeError("Fewer than 2 tickers survived cleaning, can't build pairs.")

    prices.to_csv(cache_path)
    print(f"Saved {prices.shape[0]} rows x {prices.shape[1]} tickers -> {cache_path}")
    return prices


def get_ticker_country(ticker):
    # check our lookup table first (fast, no internet needed)
    if ticker in config.TICKER_COUNTRY_MAP:
        return config.TICKER_COUNTRY_MAP[ticker]

    # otherwise try to ask yfinance directly
    try:
        info = yf.Ticker(ticker).info
        return info.get("country") or "Unknown"
    except Exception:
        return "Unknown"


def get_countries_for_tickers(tickers):
    # just build a ticker -> country dictionary for a list of tickers
    countries = {}
    for t in tickers:
        countries[t] = get_ticker_country(t)
    return countries


def main():
    prices = get_data(force_refresh=True)
    print(prices.tail())


if __name__ == "__main__":
    main()
