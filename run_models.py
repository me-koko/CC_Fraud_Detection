import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'creditcard.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_and_preprocess_data():
    if not os.path.exists(DATA_PATH):
        alt_path = os.path.join(BASE_DIR, 'Credit Card Fraud Detection Using Machine Learning', 'Code', 'creditcard.csv')
        if os.path.exists(alt_path):
            csv_path = alt_path
        else:
            raise FileNotFoundError(f"Dataset not found at {DATA_PATH} or {alt_path}")
    else:
        csv_path = DATA_PATH

    print(f"[1/4] Loading dataset from: {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"      Total records: {len(df):,}, Features: {df.shape[1]}")
    legit_count = (df['Class'] == 0).sum()
    fraud_count = (df['Class'] == 1).sum()
    print(f"      Legitimate: {legit_count:,} ({legit_count/len(df)*100:.2f}%)")
    print(f"      Fraudulent: {fraud_count:,} ({fraud_count/len(df)*100:.2f}%)")

    # Balanced dataset sampling (as per project methodology)
    legit = df[df.Class == 0]
    fraud = df[df.Class == 1]
    legit_sample = legit.sample(n=len(fraud), random_state=42)
    balanced_df = pd.concat([legit_sample, fraud], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

    X = balanced_df.drop(columns=['Class'])
    y = balanced_df['Class']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Save scaler
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    # Save a sample legit and fraud transaction for later testing
    sample_legit = legit.sample(n=1, random_state=42).drop(columns=['Class'])
    sample_fraud = fraud.sample(n=1, random_state=42).drop(columns=['Class'])
    sample_legit.to_json(os.path.join(RESULTS_DIR, 'sample_legit.json'), orient='records')
    sample_fraud.to_json(os.path.join(RESULTS_DIR, 'sample_fraud.json'), orient='records')

    return X_train, X_test, y_train, y_test, scaler, df


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    print("\n[2/4] Training and evaluating 4 Machine Learning models...")

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(criterion='entropy', random_state=42),
        'Support Vector Machine (SVM)': SVC(probability=True, kernel='rbf', random_state=42),
        'K-Nearest Neighbors (KNN)': KNeighborsClassifier(n_neighbors=3)
    }

    results = []

    for name, model in models.items():
        t0 = time.time()
        print(f"\n   -> Training {name}...")
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
        cm = confusion_matrix(y_test, y_pred)

        # Save model
        slug = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        model_path = os.path.join(MODEL_DIR, f"{slug}.joblib")
        joblib.dump(model, model_path)

        res = {
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1,
            'ROC-AUC': roc,
            'Train Time (s)': round(train_time, 3),
            'Confusion Matrix': cm.tolist()
        }
        results.append(res)

        print(f"      Accuracy:  {acc * 100:.2f}%")
        print(f"      Precision: {prec * 100:.2f}%")
        print(f"      Recall:    {rec * 100:.2f}%")
        print(f"      F1 Score:  {f1 * 100:.2f}%")
        print(f"      ROC-AUC:   {roc * 100:.2f}%")
        print(f"      Confusion Matrix:\n{cm}")

    return results


def plot_and_save_summary(results):
    print("\n[3/4] Generating performance comparison charts...")
    df_res = pd.DataFrame(results)
    
    # Save results to CSV
    df_res[['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'Train Time (s)']].to_csv(
        os.path.join(RESULTS_DIR, 'model_comparison.csv'), index=False
    )

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        sns.barplot(data=df_res, x='Model', y=metric, palette='viridis', ax=ax)
        ax.set_title(f'Model {metric}', fontsize=14, fontweight='bold')
        ax.set_ylim(0.8, 1.02)
        ax.set_xticklabels(df_res['Model'], rotation=25, ha='right')
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f"{height*100:.1f}%",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10, xytext=(0, 3),
                        textcoords='offset points')

    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, 'model_comparison_chart.png')
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"      Saved comparison chart to: {chart_path}")


def main():
    print("=" * 60)
    print(" Credit Card Fraud Detection - Model Execution & Evaluation")
    print("=" * 60)

    X_train, X_test, y_train, y_test, scaler, full_df = load_and_preprocess_data()
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    plot_and_save_summary(results)

    print("\n[4/4] Execution Complete!")
    print(f"      Models saved to: {MODEL_DIR}")
    print(f"      Results saved to: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
