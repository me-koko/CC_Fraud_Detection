# CyberGuard AI - Credit Card Fraud Detection Center

A modern, full-featured AI-powered fraud detection dashboard for evaluating credit card transactions using 4 machine learning models (Logistic Regression, Decision Tree, Support Vector Machine, and K-Nearest Neighbors).

---

## 🚀 Quick Start

### 1. Launch the Dashboard
Double-click [`start_dashboard.bat`](file:///D:/CC_Fraud_Detection/start_dashboard.bat) or run in terminal:
```powershell
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in any browser.

---

## 🌟 Key UI/UX Features

### 1. 🛡️ Transaction Screener
- **1-Click Realistic Presets**: Test typical fraud attacks (Identity theft, Card skimming, Micro-probe) and normal everyday transactions (Coffee, Groceries, Electronics).
- **Kaggle Dataset Sampler**: Click **"Sample Real Dataset"** to randomly draw real transactions from the 284,807 European cardholder dataset and test the AI predictions against true ground truth.
- **Consensus Risk Meter (0 - 100)**: Aggregates votes across all 4 models to determine the threat level (`LOW RISK`, `ELEVATED`, `HIGH RISK`, `CRITICAL THREAT`).
- **Explainable AI (XAI)**: Displays the top anomalous PCA features and their standard deviation ($\sigma$) distances from normal transaction baselines.

### 2. ⚡ Live Gateway Feed
- Simulates real-time payment gateway card authorization traffic.
- Shows live transactions being screened in real time, with automatic fraud blocking and protected dollar counters.

### 3. 📊 Model Benchmarks
- Interactive Chart.js charts comparing Accuracy, Precision, Recall, and F1-Score.
- Full metrics matrix and model selection guidance.

### 4. 📖 Dataset Insights
- Visualizes the 99.83% vs 0.17% extreme class imbalance and explains PCA privacy transformations.
