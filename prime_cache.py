import yfinance as yf
import os

os.makedirs("cache", exist_ok=True)

stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

for stock in stocks:
    print(f"Fetching {stock}...")
    df = yf.download(stock, period="max", progress=False)
    df.reset_index(inplace=True)
    df.to_csv(f"cache/{stock}.csv", index=False)

print("Cache created successfully.")
