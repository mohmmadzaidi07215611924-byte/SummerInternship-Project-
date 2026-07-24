# dataprocessing.py
# Small helper functions to load the price csv and turn prices into returns.

import pandas as pd


def load_data(filepath):
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df.index.name = "Date"
    df = df.sort_index()
    return df


def calculate_returns(df):
    # turn prices into daily % change, e.g. 100 -> 101 becomes 0.01
    returns = df.pct_change()
    returns = returns.dropna()   # first row is always NaN since there's nothing before it
    return returns
