import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Function to fetch stock data
def get_stock_data(ticker, period="5y"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if df.empty:
            st.error(f"⚠️ No data found for '{ticker}'. Please check the stock symbol.")
            return None
        
        df['Return'] = df['Close'].pct_change()
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"❌ Error fetching stock data: {e}")
        return None

# Function to predict the next day's stock price
def predict_next_day_price(ticker):
    df = get_stock_data(ticker)
    
    if df is None:
        return None, None, None

    # Define features and target variable
    X = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Close']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    # Predict next day's price
    next_day_features = pd.DataFrame([X.iloc[-1]], columns=X.columns)
    predicted_price = model.predict(next_day_features)[0]
    
    return predicted_price, df, mae

st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Stock Market Prediction App")

# User Input
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):", "AAPL").upper()

if st.button("Predict Stock Price"):
    predicted_price, stock_df, model_mae = predict_next_day_price(ticker)

    if predicted_price is not None:
        st.success(f"📈 **Predicted Stock Price for {ticker}:** ${predicted_price:.2f}")
        st.info(f"📊 **Model Accuracy (MAE):** ${model_mae:.2f}")

        # Streamlit Line Chart
        st.subheader(f"{ticker} Stock Price Trend")
        st.line_chart(stock_df[['Close']])
