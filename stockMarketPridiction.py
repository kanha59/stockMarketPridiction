import streamlit as st
import pandas as pd
import numpy as np
import ta
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.offline as pyo
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt

# Function to calculate technical indicators
def calculate_indicators(df):
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()
    df["rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    macd = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    df["returns"] = df["close"].pct_change()
    df["volatility_5"] = df["returns"].rolling(window=5).std()
    df["roc_5"] = df["close"].pct_change(periods=5)
    df["relative_strength"] = df["close"] / df["close"].rolling(window=5).mean()
    return df

# Function to train model
def train_model(df):
    features = ["open", "high", "low", "close", "volume", "sma_20", "sma_50", "rsi_14", "macd", "macd_signal", "macd_hist"]
    df["target_price"] = df["close"].shift(-1)
    df["trend"] = (df["target_price"] > df["close"]).astype(int)
    df.dropna(inplace=True)
    X = df[features]
    y = df["trend"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_test, y_test, y_pred

# Function to make prediction
def make_prediction(model, df):
    features = ["open", "high", "low", "close", "volume", "sma_20", "sma_50", "rsi_14", "macd", "macd_signal", "macd_hist"]
    last_row = df[features].iloc[[-1]]
    pred = model.predict(last_row)[0]
    return "Up" if pred == 1 else "Down"

# Function to make forecast
def make_forecast(df):
    prophet_df = df[["date", "close"]].rename(columns={"date": "ds", "close": "y"})
    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast

# Streamlit app
st.title("Stock Price Analysis and Prediction")

# Upload data
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Calculate technical indicators
    df = calculate_indicators(df)

    # Train model
    model, X_test, y_test, y_pred = train_model(df)

    # Make prediction
    prediction = make_prediction(model, df)

    # Make forecast
    forecast = make_forecast(df)

    # Layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Data Preview")
        st.write(df.head())

        st.subheader("Model Evaluation")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1_score = f1_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1 Score: {f1_score:.2f}")

        st.subheader("Prediction")
        st.write(f"Prediction: {prediction}")

    with col2:
        st.subheader("Close Price Chart")
        fig = plt.figure(figsize=(10, 6))
        plt.plot(df["date"], df["close"])
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.title("Close Price Chart")
        st.pyplot(fig)

        st.subheader("Forecast")
        fig = plot_plotly(Prophet(daily_seasonality=True).fit(df[["date", "close"]].rename(columns={"date": "ds", "close": "y"})).make_future_dataframe(periods=30))
        st.plotly_chart(fig)

    st.subheader("Technical Indicators")
    col3, col4 = st.columns(2)
    with col3:
        fig = plt.figure(figsize=(10, 6))
        plt.plot(df["date"], df["sma_20"], label="SMA 20")
        plt.plot(df["date"], df["sma_50"], label="SMA 50")
        plt.xlabel("Date")
        plt.ylabel("SMA")
        plt.title("SMA Chart")
        plt.legend()
        st.pyplot(fig)

    with col4:
        fig = plt.figure(figsize=(10, 6))
        plt.plot(df["date"], df["rsi_14"], label="RSI 14")
        plt.axhline(y=30, color='r', linestyle='--')
        plt.axhline(y=70, color='g', linestyle='--')
        plt.xlabel("Date")
        plt.ylabel("RSI")
        plt.title("RSI Chart")
        plt.legend()
        st.pyplot(fig)