import os
import json
import time
import random
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DATA_PATH = os.path.join(BASE_DIR, 'creditcard.csv')

scaler = None
models = {}
feature_stats = {'means': {}, 'stds': {}}
fraud_samples_pool = []
legit_samples_pool = []

PRESETS = [
    {
        'id': 'fraud_spike',
        'title': '🚨 Identity Theft Sudden Spike',
        'badge': 'Fraud Example',
        'category': 'fraud',
        'description': 'High-velocity unusual transaction pattern with extreme PCA deviations.',
        'amount': 529.00,
        'time': 406
    },
    {
        'id': 'fraud_skimming',
        'title': '🚨 Online Card Skimming Attack',
        'badge': 'Fraud Example',
        'category': 'fraud',
        'description': 'Typical unauthorized e-commerce purchase flagged across V14 and V12.',
        'amount': 239.93,
        'time': 472
    },
    {
        'id': 'fraud_probe',
        'title': '🚨 Micro-Authorization Probe ($1.00)',
        'badge': 'Fraud Example',
        'category': 'fraud',
        'description': 'Low-dollar transaction fraudsters use to verify if stolen card details work.',
        'amount': 1.00,
        'time': 85285
    },
    {
        'id': 'legit_coffee',
        'title': '☕ Morning Cafe Coffee',
        'badge': 'Legitimate Example',
        'category': 'legit',
        'description': 'Standard everyday micro-transaction with baseline PCA distribution.',
        'amount': 2.69,
        'time': 1
    },
    {
        'id': 'legit_grocery',
        'title': '🛒 Supermarket Groceries',
        'badge': 'Legitimate Example',
        'category': 'legit',
        'description': 'Typical household recurring weekly retail purchase.',
        'amount': 149.62,
        'time': 0
    },
    {
        'id': 'legit_electronics',
        'title': '💻 Electronics Store Purchase',
        'badge': 'Legitimate Example',
        'category': 'legit',
        'description': 'Higher ticket purchase with regular domestic cardholder characteristics.',
        'amount': 378.66,
        'time': 2
    }
]


def initialize_app():
    global scaler, models, feature_stats, fraud_samples_pool, legit_samples_pool

    # Load Scaler
    scaler_path = os.path.join(MODEL_DIR, 'scaler.joblib')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)

    # Load Models
    model_mapping = {
        'Logistic Regression': 'logistic_regression.joblib',
        'Decision Tree': 'decision_tree.joblib',
        'Support Vector Machine (SVM)': 'support_vector_machine_svm.joblib',
        'K-Nearest Neighbors (KNN)': 'k-nearest_neighbors_knn.joblib'
    }

    for name, fname in model_mapping.items():
        m_path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(m_path):
            models[name] = joblib.load(m_path)
            print(f"[Loaded Model] {name}")

    # Load dataset sample pools & compute anomaly baselines
    if os.path.exists(DATA_PATH):
        print("[Loading Dataset] Reading creditcard.csv for sampling & baselines...")
        df = pd.read_csv(DATA_PATH)
        
        # Baselines
        feature_stats['means'] = df.mean().to_dict()
        feature_stats['stds'] = df.std().to_dict()

        # Pools for instant random sampling
        frauds_df = df[df.Class == 1]
        legits_df = df[df.Class == 0]
        
        fraud_samples_pool = frauds_df.to_dict(orient='records')
        legit_samples_pool = legits_df.sample(n=min(1000, len(legits_df)), random_state=42).to_dict(orient='records')

        # Attach real PCA features to presets
        for p in PRESETS:
            if p['category'] == 'fraud':
                match = frauds_df[frauds_df['Amount'] == p['amount']]
                row = match.iloc[0].to_dict() if not match.empty else frauds_df.iloc[0].to_dict()
            else:
                match = legits_df[legits_df['Amount'] == p['amount']]
                row = match.iloc[0].to_dict() if not match.empty else legits_df.iloc[0].to_dict()
            
            p['features'] = {k: float(v) for k, v in row.items() if k != 'Class'}
            p['ground_truth'] = int(row.get('Class', 0))

        print(f"[Initialized] Loaded {len(fraud_samples_pool)} fraud samples and {len(legit_samples_pool)} legit samples.")


initialize_app()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/presets')
def get_presets():
    return jsonify(PRESETS)


@app.route('/api/metrics')
def get_metrics():
    comp_file = os.path.join(RESULTS_DIR, 'model_comparison.csv')
    if os.path.exists(comp_file):
        df_comp = pd.read_csv(comp_file)
        return jsonify({
            'status': 'ok',
            'models': df_comp.to_dict(orient='records'),
            'dataset': {
                'total': 284807,
                'legit': 284315,
                'fraud': 492,
                'fraud_pct': 0.17,
                'features_count': 31
            }
        })
    return jsonify({'status': 'error', 'message': 'Metrics not generated yet'}), 404


@app.route('/api/random_transaction')
def get_random_transaction():
    filter_type = request.args.get('type', 'any')  # any, fraud, legit

    if filter_type == 'fraud' and fraud_samples_pool:
        chosen = random.choice(fraud_samples_pool)
    elif filter_type == 'legit' and legit_samples_pool:
        chosen = random.choice(legit_samples_pool)
    else:
        # 30% chance fraud for interactive testing variety
        if random.random() < 0.3 and fraud_samples_pool:
            chosen = random.choice(fraud_samples_pool)
        else:
            chosen = random.choice(legit_samples_pool) if legit_samples_pool else random.choice(fraud_samples_pool)

    ground_truth = int(chosen['Class'])
    features = {k: float(v) for k, v in chosen.items() if k != 'Class'}

    return jsonify({
        'status': 'ok',
        'ground_truth': ground_truth,
        'ground_truth_label': 'FRAUD' if ground_truth == 1 else 'LEGITIMATE',
        'features': features
    })


@app.route('/api/predict_all', methods=['POST'])
def predict_all():
    data = request.get_json() or {}
    features = data.get('features', {})

    cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    row = [float(features.get(c, 0.0)) for c in cols]
    df_in = pd.DataFrame([row], columns=cols)

    if scaler is not None:
        scaled_in = scaler.transform(df_in)
    else:
        scaled_in = df_in.values

    model_results = {}
    fraud_votes = 0
    total_prob = 0.0
    prob_count = 0

    for name, model in models.items():
        t0 = time.time()
        pred = int(model.predict(scaled_in)[0])
        prob = None
        if hasattr(model, 'predict_proba'):
            prob = float(model.predict_proba(scaled_in)[0, 1])
            total_prob += prob
            prob_count += 1
        elif hasattr(model, 'decision_function'):
            df_val = float(model.decision_function(scaled_in)[0])
            prob = 1.0 / (1.0 + np.exp(-df_val))
            total_prob += prob
            prob_count += 1
        else:
            prob = 1.0 if pred == 1 else 0.0
            total_prob += prob
            prob_count += 1

        inf_time = round((time.time() - t0) * 1000, 2)

        if pred == 1:
            fraud_votes += 1

        model_results[name] = {
            'prediction': pred,
            'prediction_label': 'Fraud' if pred == 1 else 'Legitimate',
            'probability': round(prob, 4) if prob is not None else None,
            'probability_pct': round((prob * 100), 1) if prob is not None else None,
            'latency_ms': inf_time
        }

    # Ensemble / Consensus calculation
    consensus_prob = (total_prob / prob_count) if prob_count > 0 else (fraud_votes / len(models))
    risk_score = int(round(consensus_prob * 100))

    if risk_score >= 80:
        risk_level = 'CRITICAL'
        risk_action = 'BLOCKED'
        risk_badge = 'danger'
        status_message = 'Severe Threat Detected: Immediate Card Freeze & Transaction Blocked'
    elif risk_score >= 50:
        risk_level = 'HIGH RISK'
        risk_action = 'FLAGGED'
        risk_badge = 'warning'
        status_message = 'High Anomaly: Step-Up Authentication Required (SMS / App 2FA)'
    elif risk_score >= 25:
        risk_level = 'ELEVATED'
        risk_action = 'REVIEW'
        risk_badge = 'info'
        status_message = 'Moderate Anomaly: Sent to Risk Operations Queue for Post-Audit'
    else:
        risk_level = 'LOW RISK'
        risk_action = 'APPROVED'
        risk_badge = 'success'
        status_message = 'Normal Pattern: Instant Payment Authorization Approved'

    # Compute Feature Anomalies (Z-scores for Explainability)
    anomalies = []
    means = feature_stats.get('means', {})
    stds = feature_stats.get('stds', {})
    for col in [f'V{i}' for i in range(1, 29)]:
        val = float(features.get(col, 0.0))
        m = means.get(col, 0.0)
        s = stds.get(col, 1.0) or 1.0
        z_score = (val - m) / s
        anomalies.append({
            'feature': col,
            'value': round(val, 3),
            'z_score': round(z_score, 2),
            'deviation_magnitude': round(abs(z_score), 2)
        })

    anomalies.sort(key=lambda x: x['deviation_magnitude'], reverse=True)
    top_anomalies = anomalies[:5]

    return jsonify({
        'status': 'ok',
        'risk_score': risk_score,
        'risk_level': risk_level,
        'risk_action': risk_action,
        'risk_badge': risk_badge,
        'status_message': status_message,
        'fraud_votes': fraud_votes,
        'total_models': len(models),
        'consensus_prob': round(consensus_prob, 4),
        'models': model_results,
        'top_anomalies': top_anomalies
    })


@app.route('/api/stream_tick')
def stream_tick():
    # Simulated live streaming transaction
    is_fraud = random.random() < 0.15
    pool = fraud_samples_pool if (is_fraud and fraud_samples_pool) else legit_samples_pool
    chosen = random.choice(pool)

    ground_truth = int(chosen['Class'])
    amount = float(chosen['Amount'])
    tx_time = int(chosen['Time'])

    # Fast evaluation using Logistic Regression or ensemble
    cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    row = [float(chosen[c]) for c in cols]
    df_in = pd.DataFrame([row], columns=cols)
    scaled_in = scaler.transform(df_in) if scaler else df_in.values

    # Check primary model
    model = models.get('Logistic Regression') or list(models.values())[0]
    pred = int(model.predict(scaled_in)[0])
    prob = float(model.predict_proba(scaled_in)[0, 1]) if hasattr(model, 'predict_proba') else float(pred)

    tx_id = f"TX-{random.randint(100000, 999999)}"
    card_suffix = f"•••• {random.randint(1000, 9999)}"

    return jsonify({
        'tx_id': tx_id,
        'card': card_suffix,
        'time_sec': tx_time,
        'amount': round(amount, 2),
        'prediction': pred,
        'risk_pct': round(prob * 100, 1),
        'is_fraud': (pred == 1),
        'ground_truth': ground_truth,
        'status': 'BLOCKED' if pred == 1 else 'APPROVED'
    })


if __name__ == '__main__':
    print("=========================================================")
    print("  Starting Enhanced Fraud Detection Web Dashboard")
    print("  URL: http://127.0.0.1:5000")
    print("=========================================================")
    app.run(host='127.0.0.1', port=5000, debug=False)
