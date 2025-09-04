# ...

# Streamlit app
st.title("Stock Price Analysis and Prediction")

# Upload data
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    # ...

    # Display data preview and charts in two columns
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Preview")
        st.write(df.head())
    with col2:
        st.subheader("Close Price Chart")
        fig = plt.figure(figsize=(8, 6))
        plt.style.use('dark_background')
        plt.plot(df["date"], df["close"], color='yellow')
        plt.xlabel("Date", color='white')
        plt.ylabel("Close Price", color='white')
        plt.title("Close Price Chart", color='white')
        plt.tick_params(axis='x', colors='white')
        plt.tick_params(axis='y', colors='white')
        st.pyplot(fig)

    # Display technical indicator charts
    st.subheader("Technical Indicators")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("SMA Chart")
        fig = plt.figure(figsize=(8, 6))
        plt.style.use('dark_background')
        plt.plot(df["date"], df["sma_20"], label="SMA 20", color='blue')
        plt.plot(df["date"], df["sma_50"], label="SMA 50", color='red')
        plt.plot(df["date"], df["close"], label="Close Price", color='yellow')
        plt.xlabel("Date", color='white')
        plt.ylabel("SMA", color='white')
        plt.title("SMA Chart", color='white')
        plt.tick_params(axis='x', colors='white')
        plt.tick_params(axis='y', colors='white')
        plt.legend()
        st.pyplot(fig)
    with col2:
        st.subheader("RSI Chart")
        fig = plt.figure(figsize=(8, 6))
        plt.style.use('dark_background')
        plt.plot(df["date"], df["rsi_14"], label="RSI 14", color='green')
        plt.axhline(y=30, color='r', linestyle='--')
        plt.axhline(y=70, color='g', linestyle='--')
        plt.xlabel("Date", color='white')
        plt.ylabel("RSI", color='white')
        plt.title("RSI Chart", color='white')
        plt.tick_params(axis='x', colors='white')
        plt.tick_params(axis='y', colors='white')
        plt.legend()
        st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("MACD Chart")
        fig = plt.figure(figsize=(8, 6))
        plt.style.use('dark_background')
        plt.plot(df["date"], df["macd"], label="MACD", color='blue')
        plt.plot(df["date"], df["macd_signal"], label="Signal", color='red')
        plt.bar(df["date"], df["macd_hist"], label="Histogram", color='green', alpha=0.5)
        plt.xlabel("Date", color='white')
        plt.ylabel("MACD", color='white')
        plt.title("MACD Chart", color='white')
        plt.tick_params(axis='x', colors='white')
        plt.tick_params(axis='y', colors='white')
        plt.legend()
        st.pyplot(fig)
    with col2:
        st.subheader("Volatility Chart")
        fig = plt.figure(figsize=(8, 6))
        plt.style.use('dark_background')
        plt.plot(df["date"], df["volatility_5"], label="Volatility", color='red')
        plt.xlabel("Date", color='white')
        plt.ylabel("Volatility", color='white')
        plt.title("Volatility Chart", color='white')
        plt.tick_params(axis='x', colors='white')
        plt.tick_params(axis='y', colors='white')
        plt.legend()
        st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Pct Change Chart")
        fig = plt.figure(figsize=(8, 6))
        plt.style.use('dark_background')
        plt.plot(df["date"], df["roc_5"], label="Pct Change", color='blue')
        plt.xlabel("Date", color='white')
        plt.ylabel("Pct Change", color='white')
        plt.title("Pct Change Chart", color='white')
        plt.tick_params(axis='x', colors='white')
        plt.tick_params(axis='y', colors='white')
        plt.legend()
        st.pyplot(fig)
    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        group_names = ["True Neg (TN)", "False Pos (FP)", 
                       "False Neg (FN)", "True Pos (TP)"]
        group_counts = [f"{value}" for value in cm.flatten()]
        labels = [f"{name}\n{count}" for name, count in zip(group_names, group_counts)]
        labels = np.array(labels).reshape(2, 2)
        fig = plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix with TP / FP / FN / TN")
        st.pyplot(fig)
