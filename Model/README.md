# 📈 Coca-Cola Stock Price Prediction App

This project is a machine learning web app built using **Streamlit** to predict the **Coca-Cola (KO)** stock closing price. The app uses historical stock data and multiple regression models to forecast future values and show model accuracy.

---

## 🚀 Features

- Fetches historical data using **yfinance**
- Supports multiple ML models:
  - Random Forest
  - Linear Regression
  - SVM
  - Decision Tree
  - SGD
- Visualizes:
  - Actual vs Predicted Prices
  - Moving Averages
  - Correlation Heatmap
- Predicts **live closing price**
- Interactive UI with Streamlit

---

## 🛠️ Technologies Used

- Python
- Streamlit
- yfinance
- scikit-learn
- pandas
- matplotlib, seaborn

---

## 📦 Installation

```bash
git clone https://github.com/Laxmi249/coca-cola-stock-predictor.git
cd coca-cola-stock-predictor
pip install -r requirements.txt
streamlit run main.py

