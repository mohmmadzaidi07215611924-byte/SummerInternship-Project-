
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def extract_features(returns, n_components):
    # we want one row per STOCK, not per day, so transpose the returns table
    returns_t = returns.T.values

    # because their numbers are bigger
    scaler = StandardScaler()
    returns_scaled = scaler.fit_transform(returns_t)

    n_components = min(n_components, returns_scaled.shape[0], returns_scaled.shape[1])

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(returns_scaled)
    explained_variance = pca.explained_variance_ratio_

    return components, explained_variance


def find_clusters(components, stock_tickers, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(components)

    clusters = {}
    for ticker, label in zip(stock_tickers, labels):
        if label == -1:
            continue  # -1 means DBSCAN thinks this stock doesn't belong anywhere
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(ticker)

    return clusters
