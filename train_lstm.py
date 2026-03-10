# train_lstm.py
# -----------------------------------
# Robust LSTM training script
# Handles dirty Yahoo Finance CSV data
# Includes Algorithm Watermarking for IP Protection
# -----------------------------------

import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -------------------------------
# WATERMARK CONFIG (IP PROTECTION)
# -------------------------------

WATERMARK_TEXT = "STOCK_LSTM_IP_MAYUKHMALA_2026"

def embed_watermark(model, watermark_text):
    """
    Embed a hidden watermark into the model weights.
    This does NOT affect prediction accuracy.
    """
    watermark_value = sum(ord(c) for c in watermark_text) % 1000
    watermark_value = watermark_value / 1e6  # extremely small change

    # Embed watermark into final Dense layer
    weights = model.layers[-1].get_weights()
    weights[0][0][0] += watermark_value
    model.layers[-1].set_weights(weights)

# -------------------------------
# CONFIGURATION
# -------------------------------

STOCKS = ["AAPL", "AMZN", "GOOGL", "MSFT", "TSLA"]
WINDOW_SIZE = 60
EPOCHS = 10
BATCH_SIZE = 32

CACHE_DIR = "cache"
MODEL_DIR = "lstm_model"

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------
# TRAIN MODEL FOR EACH STOCK
# -------------------------------

for stock in STOCKS:
    print(f"\n📈 Training LSTM model for {stock}")

    file_path = os.path.join(CACHE_DIR, f"{stock}.csv")

    # Load CSV safely
    df = pd.read_csv(file_path)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Select correct price column
    if "close" in df.columns:
        price_series = df["close"]
    elif "adj close" in df.columns:
        price_series = df["adj close"]
    else:
        raise ValueError(f"No Close price column found in {stock}.csv")

    # Convert to numeric & drop invalid rows
    price_series = pd.to_numeric(price_series, errors="coerce").dropna()

    if len(price_series) < WINDOW_SIZE + 1:
        raise ValueError(f"Not enough data points for {stock}")

    prices = price_series.values.reshape(-1, 1)

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)

    # Create training sequences
    X, y = [], []
    for i in range(WINDOW_SIZE, len(scaled_prices)):
        X.append(scaled_prices[i - WINDOW_SIZE:i, 0])
        y.append(scaled_prices[i, 0])

    X = np.array(X)
    y = np.array(y)

    # Reshape for LSTM [samples, time_steps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # -------------------------------
    # BUILD LSTM MODEL
    # -------------------------------

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(WINDOW_SIZE, 1)),
        LSTM(50),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    # -------------------------------
    # TRAIN MODEL
    # -------------------------------

    model.fit(
        X,
        y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # -------------------------------
    # EMBED WATERMARK (IP PROTECTION)
    # -------------------------------

    embed_watermark(model, WATERMARK_TEXT)

    # -------------------------------
    # SAVE WATERMARKED MODEL
    # -------------------------------

    model_path = os.path.join(MODEL_DIR, f"{stock}.h5")
    model.save(model_path)

    print(f"✅ Watermarked model saved: {model_path}")

print("\n🎉 All LSTM models trained and watermarked successfully!")
