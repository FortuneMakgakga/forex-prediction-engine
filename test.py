"""
FX DATA COLLECTION PIPELINE (TEMPORARY NEWS DROPPED, ENHANCED FEATURES)
----------------------------------------------------------------------
- OHLCV Data (Yahoo Finance)
- RSI(14), MACD(12,26,9), MACD histogram, Bollinger Bands
- 1H Volatility, ATR
- Momentum / % change features
- Hour of day, Day of week
- Interest Rate Differential (daily from FRED, forward-filled, configurable history)
- Order Imbalance (placeholder 0)
- News Sentiment removed temporarily
"""

import yfinance as yf
import pandas as pd
import numpy as np
from fredapi import Fred
from datetime import datetime, timedelta

# ==========================================
# CONFIG / API KEYS
# ==========================================
FRED_API_KEY = "d1f21a31e35ad078c4d57b696e559139"
fred = Fred(api_key=FRED_API_KEY)

RATE_CODES = {
    "USD": "DFF",
    "EUR": "ECBMAIN",
    "GBP": "BOERAI",
    "JPY": "INTDSRJPM193N",
    "AUD": "IRLTAA01AUM156N",
    "CAD": "IRLTLT01CAM156N",
    "ZAR": None
}

# ==========================================
# PRICE DATA
# ==========================================
def fetch_price_data(pair="EURUSD=X", interval="1h", period="30d"):
    df = yf.download(pair, interval=interval, period=period)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df = df.dropna()
    df.index = df.index.tz_convert(None)
    return df

# ==========================================
# TECHNICAL INDICATORS
# ==========================================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def compute_bollinger(series, period=20, num_std=2):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    width = (upper - lower) / sma
    return upper, lower, width

def compute_volatility(df, window=24):
    returns = df["Close"].pct_change()
    return returns.rolling(window).std()

def compute_atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

# ==========================================
# INTEREST RATE DIFFERENTIAL (CONFIGURABLE RANGE)
# ==========================================
def split_pair(pair):
    if "=X" in pair:
        pair = pair.replace("=X", "")
    return pair[:3], pair[3:]

def get_fred_rate(code, start, end):
    try:
        series = fred.get_series(code, observation_start=start, observation_end=end)
        series = series.dropna()
        return series
    except:
        return pd.Series(dtype=float)

def get_interest_rate_differential(pair, start, end, max_history_days=365):
    base, quote = split_pair(pair)
    base_code = RATE_CODES.get(base)
    quote_code = RATE_CODES.get(quote)

    if base_code is None or quote_code is None:
        return pd.DataFrame({"interest_rate_differential": np.nan},
                            index=pd.date_range(start=start, end=end))

    start_dt = max(pd.to_datetime(start), pd.Timestamp(datetime.today() - timedelta(days=max_history_days)))
    base_series = get_fred_rate(base_code, start_dt, end)
    quote_series = get_fred_rate(quote_code, start_dt, end)

    if base_series.empty or quote_series.empty:
        return pd.DataFrame({"interest_rate_differential": np.nan},
                            index=pd.date_range(start=start, end=end))

    ird_daily = (base_series - quote_series).to_frame("interest_rate_differential")
    ird_daily.index = pd.to_datetime(ird_daily.index)
    ird_daily.index = ird_daily.index.tz_localize('UTC', ambiguous='NaT').tz_convert(None)

    return ird_daily

# ==========================================
# ORDER IMBALANCE (placeholder)
# ==========================================
def get_order_imbalance(df):
    df["order_imbalance"] = 0
    return df

# ==========================================
# ADDITIONAL FEATURES
# ==========================================
def add_time_features(df):
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    return df

def add_momentum_features(df):
    df["momentum_1h"] = df["Close"] - df["Close"].shift(1)
    df["momentum_3h"] = df["Close"] - df["Close"].shift(3)
    df["pct_change_1h"] = df["Close"].pct_change(1)
    df["pct_change_3h"] = df["Close"].pct_change(3)
    df["trend_direction"] = np.sign(df["Close"].diff())
    return df

# ==========================================
# BUILD DATASET
# ==========================================
def build_dataset(pair="EURUSD=X", period="30d", interval="1h", ird_history_days=365):
    df = fetch_price_data(pair, interval=interval, period=period)

    # Technical indicators
    df["rsi_14"] = compute_rsi(df["Close"])
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(df["Close"])
    df["bollinger_upper"], df["bollinger_lower"], df["bollinger_width"] = compute_bollinger(df["Close"])
    df["volatility_1h"] = compute_volatility(df)
    df["atr_14"] = compute_atr(df)

    # Momentum & time features
    df = add_momentum_features(df)
    df = add_time_features(df)

    # Interest rate differential
    start_date = df.index.min().date()
    end_date = df.index.max().date()
    ird_daily = get_interest_rate_differential(pair, start_date, end_date, max_history_days=ird_history_days)
    if not ird_daily.empty:
        df = df.merge(ird_daily, how="left", left_index=True, right_index=True)
        df["interest_rate_differential"] = df["interest_rate_differential"].ffill()

    # Order imbalance
    df = get_order_imbalance(df)

    # Fill remaining NaNs
    df = df.ffill().bfill()

    return df

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    pair = "EURUSD=X"
    print(f"Collecting dataset for {pair}...")
    df = build_dataset(pair, period="120d", interval="1h", ird_history_days=365)  # fetch IRD last 90 days
    df.to_csv(f"{pair}_feature_data.csv")
    print(f"Saved file: {pair}_feature_data.csv")
    print("\nPreview:")
    print(df.head())
