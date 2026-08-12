"""Download daily price data via Massive API (ETFs) + yfinance fallback (indices).

Ticker mapping (Bloomberg paper → proxy):
    MXWOHEUR → EFA   (MSCI EAFE ETF, inception 2001)
    NDUEEGF  → EEM   (MSCI Emerging Markets ETF, inception 2003)
    GOBC     → LQD   (iShares Investment Grade Corporate Bond, inception 2002)
    W3G1     → TLT   (iShares 20+ Year Treasury Bond ETF, inception 2002)
    HY Spread→ HYG   (iShares High Yield Corporate Bond ETF, inception 2007)
    VIX      → ^VIX  (CBOE Volatility Index, via yfinance; not in Massive equities)
    MOVE     → ^TNX  (10-Year Treasury Yield proxy, via yfinance)

Limitations documented in AGENTS.md:
  - Bloomberg institutional index data unavailable in public APIs
  - ETF inception dates limit Phase 1 training history to 2003+ for EM/LQD/TLT
  - SPY/IEF used as early gap-fill proxies where ETFs did not yet exist

Data flow:
  1. Massive RESTClient list_aggs (daily, adjusted close) for ETF tickers
  2. yfinance fallback for any Massive gaps and for ^VIX / ^TNX
  3. Forward-fill ≤ 5 days to handle market holidays; drop remaining NaN rows
  4. Save to data/raw_prices.csv with Date index
"""

import os
import time
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Ticker configuration ──────────────────────────────────────────────────────
ETF_TICKERS_MASSIVE = ["EFA", "EEM", "LQD", "TLT", "HYG"]   # available on Massive
INDEX_TICKERS_YFINANCE = ["^VIX", "^TNX"]                    # fetched from yfinance
START_DATE = "1995-01-01"
END_DATE = "2024-06-01"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "raw_prices.csv")

# ── Massive API helpers ───────────────────────────────────────────────────────

def _fetch_massive_ticker(
    ticker: str,
    start: str,
    end: str,
    api_key: str,
) -> pd.Series:
    """Fetch daily close prices for one ticker via Massive RESTClient.

    Args:
        ticker: Equity or ETF ticker (e.g. 'EFA').
        start: Inclusive start date 'YYYY-MM-DD'.
        end: Exclusive end date 'YYYY-MM-DD'.
        api_key: Massive API key.

    Returns:
        DatetimeIndex Series of daily close prices named *ticker*.
        Empty Series if the ticker has no data.
    """
    # Using Massive RESTClient — Context7 confirmed
    from massive import RESTClient  # noqa: PLC0415

    client = RESTClient(api_key=api_key)
    records: list[dict] = []

    try:
        for agg in client.list_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=start,
            to=end,
            adjusted=True,
            limit=50_000,
        ):
            # Massive returns timestamp in milliseconds UTC
            records.append({
                "date": pd.Timestamp(agg.timestamp, unit="ms", tz="UTC").normalize().tz_localize(None),
                "close": float(agg.close),
            })
    except Exception as exc:  # noqa: BLE001
        logger.warning("Massive fetch failed for %s: %s", ticker, exc)
        return pd.Series([], dtype=float, name=ticker)

    if not records:
        return pd.Series([], dtype=float, name=ticker)

    df = pd.DataFrame(records).set_index("date")
    df.index.name = "Date"
    series = df["close"].rename(ticker)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series


def _fetch_massive_all(
    tickers: list[str],
    start: str,
    end: str,
    api_key: str,
) -> pd.DataFrame:
    """Fetch multiple tickers from Massive and align on a common DatetimeIndex."""
    frames: dict[str, pd.Series] = {}
    for tkr in tickers:
        logger.info("Fetching %s from Massive …", tkr)
        series = _fetch_massive_ticker(tkr, start, end, api_key)
        if not series.empty:
            frames[tkr] = series
        time.sleep(0.1)   # respect rate limits

    if not frames:
        return pd.DataFrame()

    return pd.DataFrame(frames)


# ── yfinance fallback ─────────────────────────────────────────────────────────

def _fetch_yfinance(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download adjusted-close prices via yfinance.

    Args:
        tickers: List of Yahoo Finance ticker symbols.
        start: Inclusive start date.
        end: Exclusive end date.

    Returns:
        DataFrame with Date index and one column per ticker.
    """
    import yfinance as yf  # noqa: PLC0415

    if not tickers:
        return pd.DataFrame()

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw.copy()

    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"
    prices = prices[[t for t in tickers if t in prices.columns]]
    return prices


# ── Main download logic ───────────────────────────────────────────────────────

def download_prices(
    start: str = START_DATE,
    end: str = END_DATE,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Download all required price series and return a combined DataFrame.

    Priority:
      1. Massive RESTClient for ETF tickers (ETF_TICKERS_MASSIVE).
      2. yfinance fallback for any column that Massive could not populate.
      3. yfinance always used for index tickers (INDEX_TICKERS_YFINANCE).

    Args:
        start: Inclusive start date 'YYYY-MM-DD'.
        end: Exclusive end date 'YYYY-MM-DD'.
        api_key: Massive API key. Reads MASSIVE_TOKEN env var if None.

    Returns:
        DataFrame with Date index, columns = ETF_TICKERS_MASSIVE + INDEX_TICKERS_YFINANCE.
        Forward-filled ≤ 5 days; rows with any remaining NaN dropped.
    """
    if api_key is None:
        api_key = os.getenv("MASSIVE_TOKEN") or os.getenv("MASSIVE_API_KEY") or ""

    all_frames: list[pd.DataFrame] = []

    # ── Step 1: Massive for ETFs ──────────────────────────────────────────────
    if api_key:
        logger.info("Fetching ETFs from Massive API …")
        massive_df = _fetch_massive_all(ETF_TICKERS_MASSIVE, start, end, api_key)
        if not massive_df.empty:
            all_frames.append(massive_df)
            missing_etfs = [t for t in ETF_TICKERS_MASSIVE if t not in massive_df.columns]
        else:
            missing_etfs = list(ETF_TICKERS_MASSIVE)
            logger.warning("Massive returned no data; falling back fully to yfinance.")
    else:
        logger.info("No Massive API key found; using yfinance for all ETFs.")
        missing_etfs = list(ETF_TICKERS_MASSIVE)

    # ── Step 2: yfinance fallback for missing ETFs ────────────────────────────
    if missing_etfs:
        logger.info("Fetching %s from yfinance …", missing_etfs)
        yf_etf_df = _fetch_yfinance(missing_etfs, start, end)
        if not yf_etf_df.empty:
            all_frames.append(yf_etf_df)

    # ── Step 3: yfinance for index tickers ────────────────────────────────────
    logger.info("Fetching index tickers from yfinance: %s …", INDEX_TICKERS_YFINANCE)
    idx_df = _fetch_yfinance(INDEX_TICKERS_YFINANCE, start, end)
    if not idx_df.empty:
        all_frames.append(idx_df)

    if not all_frames:
        raise RuntimeError("No data downloaded from either Massive or yfinance.")

    # ── Merge on common business-day index ───────────────────────────────────
    combined = all_frames[0]
    for frame in all_frames[1:]:
        combined = combined.join(frame, how="outer")

    combined.sort_index(inplace=True)
    combined = combined.ffill(limit=5)
    combined = combined.dropna()

    # Validate expected columns are present
    required = ETF_TICKERS_MASSIVE + INDEX_TICKERS_YFINANCE
    missing_final = [c for c in required if c not in combined.columns]
    if missing_final:
        logger.warning("Missing columns after download: %s", missing_final)

    logger.info(
        "Download complete: shape=%s, date range=%s → %s",
        combined.shape,
        combined.index[0].date(),
        combined.index[-1].date(),
    )
    return combined


def main() -> None:
    """Download price data and save to data/raw_prices.csv."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print(f"Downloading price data: {START_DATE} → {END_DATE}")
    prices = download_prices()
    prices.to_csv(OUTPUT_PATH)
    print(
        f"Saved {OUTPUT_PATH}  |  shape={prices.shape}  |  "
        f"date range: {prices.index[0].date()} → {prices.index[-1].date()}"
    )
    print(f"Columns: {list(prices.columns)}")


if __name__ == "__main__":
    main()
