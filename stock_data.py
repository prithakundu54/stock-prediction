import yfinance as yf

# Get stock data
stock = yf.download("AAPL", period="6mo")

# Clean the data
stock = stock.dropna()
stock = stock[['Close']]

# Prices
current_price = stock['Close'].iloc[-1]
predicted_price = stock['Close'].tail(20).mean()

# Profit calculation
profit_percent = float(((predicted_price - current_price) / current_price) * 100)

# Decision
if profit_percent >= 10:
    decision = "Long-Term Investment"
elif profit_percent >= 3:
    decision = "Short-Term Investment"
else:
    decision = "Not Recommended"

print("Current Price:", round(current_price, 2))
print("Predicted Price:", round(predicted_price, 2))
print("Expected Profit (%):", round(profit_percent, 2))
print("Decision:", decision)
