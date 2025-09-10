import streamlit as st
import pandas as pd
import numpy as np
import ta
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.offline as pyo
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.markdown("""
<style>
body {
    color: #fff;
    background-color: #111;
}
</style>
""", unsafe_allow_html=True)

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
    prob = model.predict_proba(last_row)[0][pred]
    
    return "Up" if pred == 1 else "Down", round(prob * 100, 2)

# Function to make forecast
def make_forecast(df):
    prophet_df = df[["date", "close"]].rename(columns={"date": "ds", "close": "y"})
    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return model, forecast, prophet_df

# Streamlit app
st.title("Stock Price Analysis and Prediction")

# Upload data
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Data cleaning and preprocessing
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["date"] = pd.to_datetime(df["date"])
    numeric_cols = ["open", "high", "low", "close", "volume", "value", "no_of_trades"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "").astype(float)

    # Calculate technical indicators
    df = calculate_indicators(df)

    # Train model
    model, X_test, y_test, y_pred = train_model(df)

    # Make prediction
    trend, confidence = make_prediction(model, df)
    

    # Make forecast
    forecast_model, forecast, prophet_df = make_forecast(df)

    # Display results
    st.subheader("Next Day Trend Prediction")
    with st.container(border=True):
        st.success(f"Prediction: {trend} with {confidence}% confidence")

    # Display forecast
    st.subheader("Daily Forecast")
    with st.container(border=True):
        fig = plot_plotly(forecast_model, forecast)
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig)

    # Display evaluation metrics
    st.subheader("Model Evaluation Metrics")
    with st.container(border=True):
        comparisons = forecast.merge(prophet_df, on='ds', how='inner')
        y_true = comparisons['y']
        y_pred = comparisons['yhat']
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        rmse_pct = (rmse / y_true.mean()) * 100
        mae_pct = (mae / y_true.mean()) * 100
        st.markdown(f"""
        <div style="padding: 10px;">
            <h4>Model Evaluation Metrics</h4>
            <p><strong>RMSE = {rmse:.2f}</strong> → On average, your predictions are about ₹{rmse:.2f} away from actual prices.</p>
            <p><strong>MAE = {mae:.2f}</strong> → On average, error is ₹{mae:.2f} per prediction.</p>
            <p><strong>RMSE% ≈ {rmse_pct:.1f}%</strong> → Model’s average prediction error is ~{rmse_pct:.1f}% of stock price.</p>
            <p><strong>MAE% ≈ {mae_pct:.1f}%</strong> → More intuitive: predictions are ~{mae_pct:.1f}% off on average.</p>
        </div>
        """, unsafe_allow_html=True)

    # Display classification metrics
    st.subheader("Classification Metrics")
    with st.container(border=True):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        st.write("Model Evaluation Metrics:")
        col21, col22, col23, col24 = st.columns(4)
        with col21:
            st.metric("Accuracy", f"{accuracy:.2f}")
        with col22:
            st.metric("Precision", f"{precision:.2f}")
        with col23:
            st.metric("Recall", f"{recall:.2f}")
        with col24:
            st.metric("F1 Score", f"{f1:.2f}")
        st.write("Classification Report:")
        st.code(classification_report(y_test, y_pred))

    # Display confusion matrix
    st.subheader("Confusion Matrix")
    with st.container(border=True):
        cm = confusion_matrix(y_test, y_pred)
        group_names = ["True Neg (TN)", "False Pos (FP)", 
                       "False Neg (FN)", "True Pos (TP)"]
        group_counts = [f"{value}" for value in cm.flatten()]
        labels = [f"{name}\n{count}" for name, count in zip(group_names, group_counts)]
        labels = np.array(labels).reshape(2, 2)
        fig = plt.figure(figsize=(15, 6))
        sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix with TP / FP / FN / TN")
        st.pyplot(fig)


