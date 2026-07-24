import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
import data_fetcher
import dataprocessing
import model_calculation
import parameter_calculation


st.set_page_config(page_title="ML Pair Trading Simulator", layout="wide")
st.title("Machine Learning Pair Trading Simulator")
st.caption(
    "Real market data (yfinance) -> PCA -> DBSCAN clustering -> "
    "Engle-Granger cointegration test -> spread / z-score signal."
)


with st.expander("Settings: Universe & Data", expanded=False):
    st.subheader("Universe & Data")
    ticker_input = st.text_area(
        "Tickers (comma-separated, minimum 20 recommended)",
        value=", ".join(config.DEFAULT_TICKERS),
        height=100,
    )
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    period = st.selectbox(
        "Lookback period",
        options=["1y", "2y", "3y", "5y"],
        index=["1y", "2y", "3y", "5y"].index(config.DEFAULT_PERIOD),
    )

    force_refresh = st.button("Fetch fresh data from Yahoo Finance")

n_components = config.DEFAULT_PCA_COMPONENTS
eps = config.DEFAULT_DBSCAN_EPS
min_samples = config.DEFAULT_DBSCAN_MIN_SAMPLES
p_value_threshold = config.DEFAULT_PVALUE_THRESHOLD
rolling_window = config.DEFAULT_ROLLING_WINDOW
entry_z = config.DEFAULT_ENTRY_Z
exit_z = config.DEFAULT_EXIT_Z


# cache so we don't re-download every time streamlit reruns the script
@st.cache_data(show_spinner="Fetching / loading price data...")
def load_prices(tickers, period, force_refresh):
    data = data_fetcher.get_data(tickers=tickers, period=period, force_refresh=force_refresh)
    return data


@st.cache_data(show_spinner="Looking up stock countries...")
def load_countries(tickers, refresh_token):
    # refresh_token just forces this to re-run whenever we fetch fresh data
    return data_fetcher.get_countries_for_tickers(tickers)


try:
    prices = load_prices(tickers, period, force_refresh)
except Exception as e:
    st.error(
        f"Failed to load price data: {e}\n\n"
        "If running in a restricted/offline sandbox, Yahoo Finance may not be "
        "reachable -- run this locally with internet access, or use the "
        "cached 'market_data.csv' file."
    )
    st.stop()

if prices.shape[1] < 20:
    st.warning(f"Only {prices.shape[1]} tickers survived data cleaning. 20+ is recommended.")

# figure out which country each stock belongs to (used to label pairs later)
country_map = load_countries(list(prices.columns), force_refresh)



returns = dataprocessing.calculate_returns(prices)

components, explained_variance = model_calculation.extract_features(returns, n_components)

stock_tickers = list(returns.columns)
cluster_dict = model_calculation.find_clusters(components, stock_tickers, eps, min_samples)

pairs_df = parameter_calculation.find_cointegrated_pairs(
    prices, cluster_dict, p_value_threshold=p_value_threshold, country_map=country_map
)
top_pairs_df = parameter_calculation.top_n_pairs(pairs_df, n=5)


tab_overview, tab_clusters, tab_pairs, tab_detail = st.tabs(
    ["Overview", "Clusters (DBSCAN)", "Top Cointegrated Pairs", "Pair Detail"]
)


with tab_overview:
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Tickers loaded", prices.shape[1])
    col_b.metric("Trading days", prices.shape[0])
    col_c.metric("Clusters found", len(cluster_dict))
    col_d.metric("Cointegrated pairs", len(pairs_df))

    variance_df = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(len(explained_variance))],
        "Explained Variance Ratio": explained_variance,
        "Cumulative": np.cumsum(explained_variance),
    })
    st.subheader("PCA Explained Variance")
    st.dataframe(
        variance_df.style.format({"Explained Variance Ratio": "{:.2%}", "Cumulative": "{:.2%}"}),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        f"The top {n_components} components explain {explained_variance.sum():.1%} "
        f"of the total variance in standardized returns."
    )

    st.subheader("Raw Price History")
    fig_all = go.Figure()
    for ticker in stock_tickers:
        fig_all.add_trace(go.Scatter(x=prices.index, y=prices[ticker], mode="lines", name=ticker))
    fig_all.update_layout(
        title="All Tickers - Raw Close Prices", xaxis_title="Date", yaxis_title="Price",
        margin=dict(l=40, r=20, t=60, b=40), height=450,
    )
    st.plotly_chart(fig_all, use_container_width=True)


with tab_clusters:
    st.subheader("DBSCAN Clusters")
    if cluster_dict:
        cluster_rows = []
        for cluster_id, tickers_in_cluster in sorted(cluster_dict.items()):
            cluster_rows.append({
                "Cluster": cluster_id,
                "Stocks": ", ".join(tickers_in_cluster),
                "Count": len(tickers_in_cluster),
            })
        st.dataframe(pd.DataFrame(cluster_rows), use_container_width=True, hide_index=True)
    else:
        st.warning(
            "No clusters found with the current DBSCAN parameters. Try increasing "
            "`eps` or decreasing `min_samples` in Settings above."
        )

with tab_pairs:
    st.subheader("All Cointegrated Pairs (Engle-Granger Test)")
    if not pairs_df.empty:
        st.dataframe(
            pairs_df.style.format({"P-Value": "{:.6f}", "T-Statistic": "{:.4f}"}),
            use_container_width=True, hide_index=True,
        )
        st.subheader("Top 5 Most Tradable Pairs")
        st.dataframe(
            top_pairs_df.style.format({"P-Value": "{:.6f}", "T-Statistic": "{:.4f}"}),
            use_container_width=True, hide_index=True,
        )
    else:
        st.warning(
            "No cointegrated pairs found below the current p-value threshold. "
            "Try widening `eps`, lowering `min_samples`, or raising the p-value "
            "threshold in Settings above."
        )

with tab_detail:
    if pairs_df.empty:
        st.info("No cointegrated pairs available yet -- adjust parameters in Settings above.")
        st.stop()

    pair_options = []
    for _, row in pairs_df.iterrows():
        label = (
            f"{row['Stock 1']} ({row['Country 1']}) / {row['Stock 2']} ({row['Country 2']})  "
            f"(p={row['P-Value']:.6f})"
        )
        pair_options.append(label)

    selected_option = st.selectbox("Select a cointegrated pair to analyze:", pair_options)
    selected_idx = pair_options.index(selected_option)
    selected_row = pairs_df.iloc[selected_idx]

    s1, s2 = selected_row["Stock 1"], selected_row["Stock 2"]
    country1, country2 = selected_row["Country 1"], selected_row["Country 2"]

    spread, zscore_full, beta = parameter_calculation.calculate_spread_and_zscore(prices, s1, s2)
    rolling_z = parameter_calculation.calculate_rolling_zscore(spread, window=rolling_window)

    st.caption(
        f"**{s1}** ({country1})  vs  **{s2}** ({country2})  --  "
        f"Hedge ratio (OLS beta) for spread = {s1} - (beta x {s2}):  **{beta:.4f}**"
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # normalize both prices to start at 1.0 so they're easy to compare
        normalized_p1 = prices[s1] / prices[s1].iloc[0]
        normalized_p2 = prices[s2] / prices[s2].iloc[0]

        fig_prices = go.Figure()
        fig_prices.add_trace(go.Scatter(
            x=normalized_p1.index, y=normalized_p1.values, mode="lines",
            name=f"{s1} ({country1})", line=dict(color="royalblue"),
        ))
        fig_prices.add_trace(go.Scatter(
            x=normalized_p2.index, y=normalized_p2.values, mode="lines",
            name=f"{s2} ({country2})", line=dict(color="darkorange"),
        ))
        fig_prices.update_layout(
            title=f"Normalized Prices: {s1} vs {s2}",
            xaxis_title="Date", yaxis_title="Normalized Price (start = 1.0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=20, t=60, b=40),
        )
        st.plotly_chart(fig_prices, use_container_width=True)

    with chart_col2:
        fig_zscore = go.Figure()
        fig_zscore.add_trace(go.Scatter(
            x=rolling_z.index, y=rolling_z.values, mode="lines",
            name=f"Rolling Z-Score ({rolling_window}d)", line=dict(color="purple"),
        ))
        fig_zscore.add_hline(
            y=entry_z, line_dash="dash", line_color="red",
            annotation_text="Short Spread", annotation_position="top left",
        )
        fig_zscore.add_hline(
            y=-entry_z, line_dash="dash", line_color="green",
            annotation_text="Long Spread", annotation_position="bottom left",
        )
        fig_zscore.add_hline(y=0.0, line_dash="solid", line_color="black")
        fig_zscore.update_layout(
            title=f"Rolling Spread Z-Score: {s1} - ({beta:.3f} x {s2})",
            xaxis_title="Date", yaxis_title="Z-Score",
            margin=dict(l=40, r=20, t=60, b=40),
        )
        st.plotly_chart(fig_zscore, use_container_width=True)

    st.markdown(
        """
**Signal interpretation:**
- Z-score **at/above the red line** -> spread is unusually wide -> *short the spread* (short Stock 1, long beta x Stock 2).
- Z-score **at/below the green line** -> spread is unusually narrow -> *long the spread* (long Stock 1, short beta x Stock 2).
- Z-score back near **0** -> close the position.
"""
    )