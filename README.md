# Credit Card Fraud Detection Using Machine Learning

An end-to-end Machine Learning suite and interactive web dashboard for real-time credit card fraud detection. The system leverages multiple machine learning models to detect fraudulent transactions with high sensitivity while minimizing false declines.

---

## 📌 Abstract

Credit card fraud is a significant global challenge, causing billions of dollars in losses annually. This project develops and evaluates four supervised machine learning models trained on historical European credit card transaction data:
- **Logistic Regression (LR)**
- **Decision Tree (DT)**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**

The models are evaluated on balanced holdout test datasets and deployed with a modern, human-centered interactive web dashboard and real-time transaction screener.

---

## 🚀 Key Features

- **Multi-Model Consensus Engine**: Evaluates transactions across all 4 machine learning models simultaneously to compute a composite risk score (0–100%).
- **Interactive Web Dashboard**: Modern, calm user interface built with Flask and Bootstrap 5 for real-time transaction screening and model comparison.
- **Explainable AI (XAI)**: Identifies key anomalous features using standard deviation ($\sigma$) Z-scores.
- **Live Gateway Stream**: Simulates continuous card authorization streams with live interception and dollar-volume protection tracking.
- **Pre-trained Model Artifacts**: Trained `.joblib` models ready for instant deployment and inference.

---

## 🧠 Algorithms & Performance

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **94.92%** | **97.83%** | **91.84%** | **94.74%** | 97.98% |
| **Support Vector Machine (SVM)** | 92.89% | 96.67% | 88.78% | 92.55% | **98.28%** |
| **Decision Tree** | 92.39% | 92.78% | 91.84% | 92.31% | 92.38% |
| **K-Nearest Neighbors (KNN)** | 91.88% | 95.56% | 87.76% | 91.49% | 97.50% |

---

## 📊 Dataset

The dataset used is the European Cardholder Credit Card Fraud Detection benchmark (ULB Machine Learning Group):
- **Total Transactions**: 284,807
- **Fraudulent Transactions**: 492 (0.172% extreme class imbalance)
- **Features**: 30 numerical input features (`Time`, `Amount`, and 28 PCA-transformed components `V1`–`V28` for privacy preservation).
- **Download Link**: [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

*Note: Due to GitHub's file size limits (>100MB), the raw `creditcard.csv` dataset is excluded from version control and can be downloaded directly from Kaggle or HuggingFace.*

---

## 🛠️ Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/me-koko/CC_Fraud_Detection.git
cd CC_Fraud_Detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(or `pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn flask joblib`)*

### 3. Run the Web Dashboard
```bash
python app.py
```
Or double-click `start_dashboard.bat`.  
Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

### 4. Re-train Models (Optional)
```bash
python run_models.py
```

---

## 👤 Author & Contributor

- **Komal**
  - GitHub: [@me-koko](https://github.com/me-koko)
  - Repository: [CC_Fraud_Detection](https://github.com/me-koko/CC_Fraud_Detection)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) &copy; 2026 Komal.
