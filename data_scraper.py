"""
FULL DATA COLLECTION PIPELINE FOR FX PREDICTION
------------------------------------------------
Collects:
- OHLCV Data (Yahoo Finance)
- RSI(14)
- MACD (12,26,9)
- Bollinger Bands
- 1H Volatility
- News Sentiment Score (FinBERT)
- Interest Rate Differential (FRED)
- Order Imbalance (placeholder or Binance for crypto)

Save as: data_pipeline.py
"""

# ==========================================
# IMPORTS
# ==========================================
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
import torch

from newsapi import NewsApiClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fredapi import Fred

# ==========================================
# CONFIG / API KEYS
# ==========================================
NEWS_API_KEY = "22bd5cc21ee84a74a509bfa7292f5cf8"
FRED_API_KEY = "d1f21a31e35ad078c4d57b696e559139"

news_api = NewsApiClient(api_key=NEWS_API_KEY)
fred = Fred(api_key=FRED_API_KEY)

# Load FinBERT
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")


# ==========================================
# PRICE DATA
# ==========================================
def fetch_price_data(pair="EURUSD=X", interval="1h", period="1mo"):
    df = yf.download(pair, interval=interval, period=period)

    # Flatten multi-index columns from Yahoo Finance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.dropna()
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
    return macd, signal_line


def compute_bollinger(series, period=20, num_std=2):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, lower


def compute_volatility(df):
    returns = df["Close"].pct_change()
    return returns.rolling(24).std()


# ==========================================
# NEWS SENTIMENT SCORE (FinBERT)
# ==========================================
def fetch_news(query="EUR USD", date=None):
    if date is None:
        date = datetime.date.today().isoformat()

    result = news_api.get_everything(
        q=query,
        from_param=date,
        to=date,
        language="en",
        sort_by="relevancy",
        page_size=20
    )
    articles = result.get("articles", [])
    return [a["title"] for a in articles]


def finbert_sentiment(text_list):
    if not text_list:
        return 0.0

    scores = []
    for text in text_list:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        outputs = model(**inputs)
        logits = outputs.logits[0].detach().numpy()

        prob = torch.softmax(torch.tensor(logits), dim=0).numpy()
        score = prob[2] - prob[0]  # positive - negative
        scores.append(score)

    return float(np.mean(scores))


def get_news_sentiment(pair="EURUSD"):
    query = pair.replace("=X", "").replace("X", "")
    today = datetime.date.today()

    headlines = fetch_news(query=query, date=today)
    return finbert_sentiment(headlines)


# ==========================================
# INTEREST RATE DIFFERENTIAL
# ==========================================
RATE_CODES = {
    "USD": "DFF",          # Federal funds rate
    "EUR": "ECBMAIN",      # ECB main refinancing op rate
    "GBP": "BOERAI",       # Bank of England base rate
    "JPY": "INTDSRJPM193N",
    "AUD": "IRLTAA01AUM156N",
    "CAD": "IRLTLT01CAM156N",
    "ZAR": None            # SARB not available on FRED — manual needed
}

def get_fred_rate(code):
    try:
        data = fred.get_series_latest_release(code)
        return float(data.dropna().iloc[-1])
    except:
        return None


def split_pair(pair):
    if "=X" in pair:
        pair = pair.replace("=X", "")
    return pair[:3], pair[3:]


def get_interest_rate_differential(pair="EURUSD=X"):
    base, quote = split_pair(pair)

    base_code = RATE_CODES.get(base)
    quote_code = RATE_CODES.get(quote)

    if base_code is None or quote_code is None:
        return np.nan

    base_rate = get_fred_rate(base_code)
    quote_rate = get_fred_rate(quote_code)

    if base_rate is None or quote_rate is None:
        return np.nan

    return float(base_rate - quote_rate)


# ==========================================
# ORDER IMBALANCE (Placeholder)
# ==========================================
def get_order_imbalance(pair="EURUSD"):
    """
    Forex order book data is not freely available.
    For crypto (BTCUSDT), I can plug Binance here.
    """
    return np.nan


# ==========================================
# BUILD FULL FEATURE SET
# ==========================================
def build_dataset(pair="EURUSD=X"):
    df = fetch_price_data(pair)

    # technical indicators
    df["rsi_14"] = compute_rsi(df["Close"])
    df["macd"], df["macd_signal"] = compute_macd(df["Close"])
    df["bollinger_upper"], df["bollinger_lower"] = compute_bollinger(df["Close"])
    df["volatility_1h"] = compute_volatility(df)

    # external
    sentiment = get_news_sentiment(pair.replace("=X", ""))
    df["news_sentiment_score"] = sentiment

    df["interest_rate_differential"] = get_interest_rate_differential(pair)

    # TEMP: remove NaN imbalance
    df["order_imbalance"] = 0  

    # instead of dropna → forward fill
    df = df.fillna(method="ffill").fillna(method="bfill")

    return df

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    pair = "EURUSD=X"

    print(f"Collecting full dataset for {pair}...")
    df = build_dataset(pair)

    df.to_csv(f"{pair}_feature_data.csv")
    print("\nSaved file:", f"{pair}_feature_data.csv")
    print("\nPreview:")
    print(df.head())
