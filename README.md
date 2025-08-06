# 📈 Coca-Cola Stock Price Predictor

This project is a machine learning web application that predicts the future stock price of Coca-Cola (Ticker: KO) using various models such as Linear Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), and LSTM. The application is built with Streamlit and is deployed for public use.

🔗 **Live Demo**: [Click here to view the deployed app](https://laxmi249-coca-cola-stock-predictor-app-3zs8ml.streamlit.app/)

---

## 🗂️ Project Structure

Coca-Cola/
│
├── Dataset/
│ ├── Coca-Cola_stock_info.csv
│ └── Coca-Cola_stock_history.csv
│
├── Jupyter Analysis/
│ ├── Coca Cola Stock Analysis.ipynb
│ └── Coca Cola Stock Analysis.pdf
│
├── Model/
│ ├── app.py
│ ├── requirements.txt
│ └── ... (some other folders)
│
└── README.md


---

## 🚀 Features

- 📊 **Stock Data Analysis**: Interactive historical data exploration  
- 🤖 **Machine Learning Models**: Predicts future stock prices using:
  - Linear Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
  - Long Short-Term Memory (LSTM)
- 📈 **Visualization**: Matplotlib & Plotly-based charts  
- 🌐 **Live Data**: Fetches latest Coca-Cola stock prices from Yahoo Finance

---

## 🛠️ How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/Laxmi249/coca-cola-stock-predictor.git
   cd coca-cola-stock-predictor/Model

2. **Install dependencies**
pip install -r requirements.txt

---

3. **Run the app**
streamlit run app.py

---

## 📚 Dataset

The stock data is downloaded from Yahoo Finance using the yfinance library.

---

## 📓 Notebooks
The folder Jupyter Analysis/ contains:

📘 Coca Cola Stock Analysis.ipynb: Full exploratory data analysis (EDA) of Coca-Cola's stock.

📄 PDF version also included for easy viewing.

---

## 🧠 Models Used
All models were trained and tested on the Coca-Cola stock dataset:

Model	                            Description
Linear Regression               	Baseline model for trend analysis
Decision Tree	                    Non-linear decision-based prediction
Random Forest	                    Ensemble of decision trees
Support Vector Machine (SVM)	     Works well for linear and non-linear data

---

## 📦 Requirements
Key dependencies listed in requirements.txt:

streamlit

yfinance

scikit-learn

pandas

matplotlib

plotly

tensorflow (for LSTM)

---

## 🧑‍💻 Author
Laxmi Sheoran
📍 Bhiwani, Haryana
🔗 LinkedIn  [https://www.linkedin.com/in/laxmi-sheoran-9b0813276/]
📬 Aspirant Data Analyst with strong passion in predictive analytics and visualization

---

## ⭐ If you like this project, consider giving it a star on GitHub! ⭐
