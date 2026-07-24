
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",     # Tech
    "JPM", "BAC", "WFC", "GS", "C",              # Banks
    "XOM", "CVX", "COP", "SLB", "OXY",           # Energy
    "JNJ", "PFE", "UNH", "MRK", "ABBV",          # Healthcare
    "PG", "KO", "PEP", "WMT", "COST",            # Consumer staples
]

DEFAULT_PERIOD = "3y"      # how much history to download
DEFAULT_INTERVAL = "1d"    # daily prices

CACHE_PATH = "market_data.csv"       # where we save downloaded data
CACHE_MAX_AGE_HOURS = 20             # re-download if the file is older than this

# default values for the sliders in the app
DEFAULT_PCA_COMPONENTS = 3
DEFAULT_DBSCAN_EPS = 2.0
DEFAULT_DBSCAN_MIN_SAMPLES = 2
DEFAULT_PVALUE_THRESHOLD = 0.05
DEFAULT_ROLLING_WINDOW = 30
DEFAULT_ENTRY_Z = 2.0
DEFAULT_EXIT_Z = 0.5


MAX_MISSING_FRACTION = 0.05


TICKER_COUNTRY_MAP = {
    "AAPL": "United States", "MSFT": "United States", "GOOGL": "United States",
    "AMZN": "United States", "META": "United States",
    "JPM": "United States", "BAC": "United States", "WFC": "United States",
    "GS": "United States", "C": "United States",
    "XOM": "United States", "CVX": "United States", "COP": "United States",
    "SLB": "United States", "OXY": "United States",
    "JNJ": "United States", "PFE": "United States", "UNH": "United States",
    "MRK": "United States", "ABBV": "United States",
    "PG": "United States", "KO": "United States", "PEP": "United States",
    "WMT": "United States", "COST": "United States",
}
