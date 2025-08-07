import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np


# --- 1. Data Fetching ---
def get_stock_data(ticker, start, end):
    try:
        print(f"📥 Downloading data for {ticker} from {start} to {end}...")
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            raise ValueError("Downloaded data is empty. Check ticker or date range.")
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return pd.DataFrame()

# --- 2. Preprocessing ---
def preprocess_data(data):
    try:
        if data.empty:
            raise ValueError("Empty data received for preprocessing.")

        print("⚙️ Preprocessing data...")

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns]

        rename_map = {
            'Close_KO': 'Close',
            'Open_KO': 'Open',
            'High_KO': 'High',
            'Low_KO': 'Low',
            'Volume_KO': 'Volume'
        }
        data.rename(columns=rename_map, inplace=True)
        data.ffill(inplace=True)
        data.fillna(0, inplace=True)

        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        data['Daily_Return'] = data['Close'].pct_change()
        data['Volatility'] = data['Daily_Return'].rolling(window=20).std()

        data.dropna(inplace=True)
        return data

    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        return pd.DataFrame()

# --- 3. Model Training ---
def train_model(data, model_type='Random Forest', n_estimators=100):
    features = ['Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50', 'Daily_Return', 'Volatility']
    target = 'Close'

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == 'Random Forest':
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif model_type == 'Linear Regression':
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    elif model_type == 'SVM':
        model = SVR()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    elif model_type == 'Decision Tree':
        model = DecisionTreeRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif model_type == 'SGD':
        model = SGDRegressor(max_iter=1000, tol=1e-3)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        raise ValueError("Unsupported model type")
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    comparison_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred
    }, index=y_test.index)

    return model, features, mae, mse, comparison_df


# --- 4. Prediction ---
def predict_live_price(model, features, ticker='KO'):
    try:
        print("🔍 Fetching live data for prediction...")
        live_data = yf.download(ticker, period='2d', interval='1m')
        if live_data.empty:
            raise ValueError("No live data fetched.")

        live_data['MA_20'] = live_data['Close'].rolling(window=20).mean()
        live_data['MA_50'] = live_data['Close'].rolling(window=50).mean()
        live_data['Daily_Return'] = live_data['Close'].pct_change()
        live_data['Volatility'] = live_data['Daily_Return'].rolling(window=20).std()
        live_data.fillna(0, inplace=True)

        latest_features = live_data[features].iloc[-1:].dropna()
        if latest_features.empty:
            raise ValueError("No valid data row available for prediction.")

        prediction = model.predict(latest_features)
        return prediction[0]

    except Exception as e:
        print(f"❌ Error in live prediction: {e}")
        return None

# --- 5. Streamlit App ---
def run_streamlit_app():
    st.set_page_config(page_title="Coca-Cola Stock Price Prediction", layout="wide")
    st.title("📈 Coca-Cola Stock Price Prediction App")

    # Sidebar controls
    st.sidebar.header("🔧 Configuration")
    ticker = st.sidebar.text_input("Enter Stock Ticker", value='KO')
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2015-01-01'))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime('2023-12-31'))
    model_type = st.sidebar.selectbox(
    "Select Model",
    ("Random Forest", "Linear Regression", "SVM", "Decision Tree", "SGD")
)


    if model_type == 'Random Forest':
        n_estimators = st.sidebar.slider("Number of Estimators", min_value=10, max_value=300, value=100, step=10)
    else:
        n_estimators = 100

    if start_date >= end_date:
        st.error("End date must be after start date.")
        return

    # --- Step 1: Fetch and Preprocess ---
    data = get_stock_data(ticker, start_date, end_date)
    if data.empty:
        st.error("Failed to download stock data.")
        return

    data = preprocess_data(data)
    if data.empty:
        st.error("Preprocessing failed.")
        return

    # --- Step 2: Train Model ---
    model, features, mae, mse, comparison_df = train_model(data, model_type, n_estimators)

    # --- Step 3: Predict Live Price ---
    prediction = predict_live_price(model, features, ticker)
    if prediction is None:
        st.warning("Could not generate live prediction.")
        return

    # --- Step 4: Layout and Visualizations ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Historical Price Chart")
        st.line_chart(data[['Close', 'MA_20', 'MA_50']])

        st.subheader("📌 Prediction")
        st.metric(label="Predicted Closing Price", value=f"${prediction:.2f}")

    with col2:
        st.subheader("📈 Actual vs Predicted")
        st.line_chart(comparison_df[['Actual', 'Predicted']])

        st.subheader("📉 Model Evaluation")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
        st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")

    # --- Correlation Heatmap ---
    st.subheader("🔍 Feature Correlation Heatmap")
    corr = data[features + ['Close']].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# --- Main Execution ---
def main():
    run_streamlit_app()

if __name__ == "__main__":
    main()
