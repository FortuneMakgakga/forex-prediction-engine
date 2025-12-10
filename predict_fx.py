"""
FX DATA COLLECTION + LSTM TRAINING PIPELINE
--------------------------------------------
- Predicts next 12 candle % returns
- Converts returns back to predicted prices
- Features: OHLCV + technical indicators + momentum + IRD + synthetic features
- LSTM-ready sequences
- Chart of predicted vs actual prices
"""

import yfinance as yf
import pandas as pd
import numpy as np
from fredapi import Fred
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# ==========================
# CONFIG / API KEYS
# ==========================
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

# ==========================
# PRICE + INDICATOR PIPELINE
# ==========================
def fetch_price_data(pair="EURUSD=X", interval="1h", period="30d"):
    df = yf.download(pair, interval=interval, period=period)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df = df.dropna()
    df.index = df.index.tz_convert(None)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
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

def get_order_imbalance(df):
    df["order_imbalance"] = 0
    return df

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

# ==========================
# BUILD DATASET
# ==========================
def build_dataset(pair="EURUSD=X", period="30d", interval="1h", ird_history_days=365):
    df = fetch_price_data(pair, interval=interval, period=period)
    # Indicators
    df["rsi_14"] = compute_rsi(df["Close"])
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(df["Close"])
    df["bollinger_upper"], df["bollinger_lower"], df["bollinger_width"] = compute_bollinger(df["Close"])
    df["volatility_1h"] = compute_volatility(df)
    df["atr_14"] = compute_atr(df)
    # Synthetic
    df = add_momentum_features(df)
    df = add_time_features(df)
    # IRD
    start_date = df.index.min().date()
    end_date = df.index.max().date()
    ird_daily = get_interest_rate_differential(pair, start_date, end_date, max_history_days=ird_history_days)
    if not ird_daily.empty:
        df = df.merge(ird_daily, how="left", left_index=True, right_index=True)
        df["interest_rate_differential"] = df["interest_rate_differential"].ffill().bfill()
    # Order imbalance
    df = get_order_imbalance(df)
    df = df.ffill().bfill()
    # Target: next 12 returns
    df["return_1h"] = df["Close"].pct_change().shift(-1)
    for i in range(2, 13):
        df[f"return_{i}h"] = df["Close"].pct_change(i).shift(-i)
    df = df.dropna()
    return df

# ==========================
# CREATE SEQUENCES
# ==========================
def create_sequences(df, feature_cols, target_cols, input_steps=24, output_steps=12):
    X, y = [], []
    data = df[feature_cols].values
    targets = df[target_cols].values
    for i in range(len(df) - input_steps - output_steps + 1):
        X.append(data[i:i+input_steps])
        y.append(targets[i+input_steps:i+input_steps+output_steps])
    return np.array(X), np.array(y)

# ==========================
# TRAIN MODEL
# ==========================
def train_lstm(X_train, y_train, X_val, y_val, units=64, dropout=0.2, epochs=50, batch_size=32):
    model = Sequential()
    model.add(LSTM(units, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(y_train.shape[1]))
    model.compile(loss="mse", optimizer="adam")
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size, callbacks=[es])
    return model, history

# ==========================
# PREDICT PRICES
# ==========================
def returns_to_prices(last_close, returns):
    """
    Convert predicted returns to prices
    last_close: last known closing price
    returns: array of predicted % returns
    """
    prices = [last_close]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    return prices[1:]

# ==========================
# MAIN EXECUTION
# ==========================
if __name__ == "__main__":
    pair = "EURUSD=X"
    df = build_dataset(pair, period="60d", interval="1h", ird_history_days=90)

    feature_cols = [c for c in df.columns if not c.startswith("return_")]
    target_cols = [f"return_{i}h" for i in range(1,13)]

    # Scale features
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Create sequences
    X, y = create_sequences(df, feature_cols, target_cols, input_steps=24, output_steps=12)
    split = int(len(X)*0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Train LSTM
    model, history = train_lstm(X_train, y_train, X_val, y_val, units=64, dropout=0.2, epochs=50)

    # Predict
    y_pred = model.predict(X_val)

    # Convert returns to prices
    last_closes = df["Close"].values[split+24-1: -12]
    predicted_prices = []
    for i in range(len(y_pred)):
        predicted_prices.append(returns_to_prices(last_closes[i], y_pred[i]))
    predicted_prices = np.array(predicted_prices)
    actual_prices = []
    for i in range(len(y_val)):
        actual_prices.append(df["Close"].values[split+24+i:split+24+i+12])
    actual_prices = np.array(actual_prices)

    # Plot predicted vs actual for first sample
    plt.figure(figsize=(12,6))
    plt.plot(range(1,13), actual_prices[0], label="Actual Price")
    plt.plot(range(1,13), predicted_prices[0], label="Predicted Price")
    plt.xlabel("Hour Ahead")
    plt.ylabel("Price")
    plt.title(f"{pair} Predicted vs Actual Prices (Next 12 Hours)")
    plt.legend()
    plt.show()

    # Compute MSE on returns
    mse = np.mean((y_val - y_pred)**2)
    print(f"Test MSE on returns: {mse}")
# -------------------------