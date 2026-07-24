# parameter_calculation.py
# Tests pairs of stocks for cointegration, and calculates the spread and
# z-score used for the trading signal.
#
# Correlation vs cointegration: two stocks can go up and down together
# (correlated) but still drift apart forever. Cointegration means there is
# some combination of the two prices (the "spread") that stays roughly
# constant over time - that's what we actually need for pairs trading,
# since we're betting the spread comes back to its average.

from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def find_cointegrated_pairs(df, cluster_dict, p_value_threshold=0.05, country_map=None):
    country_map = country_map or {}
    results = []

    for cluster_id, tickers in cluster_dict.items():
        if len(tickers) < 2:
            continue  # need at least 2 stocks to make a pair

        # test every possible pair inside this cluster
        for s1, s2 in combinations(tickers, 2):
            pair_prices = df[[s1, s2]].dropna()
            if pair_prices.shape[0] < 20:
                continue  # not enough overlapping data to trust the test

            t_stat, p_value, _ = coint(pair_prices[s1], pair_prices[s2])

            if p_value < p_value_threshold:
                results.append({
                    "Cluster": cluster_id,
                    "Stock 1": s1,
                    "Country 1": country_map.get(s1, "Unknown"),
                    "Stock 2": s2,
                    "Country 2": country_map.get(s2, "Unknown"),
                    "P-Value": p_value,
                    "T-Statistic": t_stat,
                })

    columns = ["Cluster", "Stock 1", "Country 1", "Stock 2", "Country 2", "P-Value", "T-Statistic"]

    if not results:
        return pd.DataFrame(columns=columns)

    pairs_df = pd.DataFrame(results, columns=columns)
    pairs_df = pairs_df.sort_values("P-Value").reset_index(drop=True)
    return pairs_df


def top_n_pairs(pairs_df, n=5):
    # pairs_df is already sorted by p-value (best pairs first), so just take the top rows
    return pairs_df.head(n).reset_index(drop=True)


def calculate_hedge_ratio(df, s1, s2, method="ols"):
    pair_prices = df[[s1, s2]].dropna()
    price1 = pair_prices[s1]
    price2 = pair_prices[s2]

    if method == "covariance":
        cov_matrix = np.cov(price1, price2)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    else:
        # classic Engle-Granger approach: regress price1 on price2
        x = add_constant(price2)
        model = OLS(price1, x).fit()
        beta = model.params[s2]

    return float(beta)


def calculate_spread_and_zscore(df, s1, s2, method="ols"):
    pair_prices = df[[s1, s2]].dropna()
    beta = calculate_hedge_ratio(df, s1, s2, method=method)

    spread = pair_prices[s1] - beta * pair_prices[s2]
    zscore = (spread - spread.mean()) / spread.std()

    return spread, zscore, beta


def calculate_rolling_zscore(spread, window=30):
    # rolling z-score adapts to changing volatility instead of assuming the
    # whole history has one fixed mean/std - more realistic for live trading
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    rolling_z = (spread - rolling_mean) / rolling_std
    return rolling_z
