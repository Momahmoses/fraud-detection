"""
Fraud Detection — Model Training & Evaluation
Trains multiple ML models, evaluates, and saves the best one.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score,
    average_precision_score, precision_recall_curve
)
from imblearn.over_sampling import SMOTE

os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# ── Load & preprocess ──────────────────────────────────────────────────────────
df = pd.read_csv('data/fraud_dataset.csv')

le = LabelEncoder()
df['merchant_category_enc'] = le.fit_transform(df['merchant_category'])

features = [
    'transaction_amount', 'transaction_hour', 'distance_from_home_km',
    'distance_from_last_txn_km', 'ratio_to_median_purchase', 'repeat_retailer',
    'used_chip', 'used_pin', 'online_order', 'num_transactions_today',
    'merchant_category_enc'
]

X = df[features]
y = df['fraud']

# ── Split ──────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Handle class imbalance with SMOTE ─────────────────────────────────────────
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"After SMOTE — Legit: {(y_train_bal==0).sum():,} | Fraud: {(y_train_bal==1).sum():,}")

# ── Scale ──────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_bal)
X_test_sc  = scaler.transform(X_test)

# ── Models ─────────────────────────────────────────────────────────────────────
models = {
    'Logistic Regression':  LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':        DecisionTreeClassifier(max_depth=8, random_state=42),
    'Random Forest':        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting':    GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
print("\n── Model Comparison ─────────────────────────────────────────────────────")
print(f"{'Model':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>7} {'AUC':>7}")
print("-"*70)

for name, model in models.items():
    use_scaled = name == 'Logistic Regression'
    Xtr = X_train_sc if use_scaled else X_train_bal
    Xte = X_test_sc  if use_scaled else X_test

    model.fit(Xtr, y_train_bal)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)

    results[name] = {
        'model': model, 'scaled': use_scaled,
        'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc,
        'y_pred': y_pred, 'y_prob': y_prob
    }
    print(f"{name:<25} {acc:>9.3f} {prec:>10.3f} {rec:>8.3f} {f1:>7.3f} {auc:>7.3f}")

# ── Best model ─────────────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['f1'])
best = results[best_name]
print(f"\nBest model: {best_name} (F1: {best['f1']:.3f} | AUC: {best['auc']:.3f})")

print("\nDetailed Classification Report:")
print(classification_report(y_test, best['y_pred'], target_names=['Legitimate', 'Fraud']))

# ── Plot 7: Confusion Matrix ───────────────────────────────────────────────────
cm = confusion_matrix(y_test, best['y_pred'])
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud'], ax=ax)
ax.set_title(f'Confusion Matrix — {best_name}')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('plots/07_confusion_matrix.png', dpi=150)
plt.close()

# ── Plot 8: ROC Curves ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
ax.plot([0,1],[0,1],'k--', alpha=0.4)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Fraud Detection Models')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('plots/08_roc_curves.png', dpi=150)
plt.close()

# ── Plot 9: Precision-Recall Curve ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for name, res in results.items():
    prec_c, rec_c, _ = precision_recall_curve(y_test, res['y_prob'])
    ap = average_precision_score(y_test, res['y_prob'])
    ax.plot(rec_c, prec_c, label=f"{name} (AP={ap:.3f})")
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('plots/09_precision_recall.png', dpi=150)
plt.close()

# ── Plot 10: Feature Importance ────────────────────────────────────────────────
if hasattr(best['model'], 'feature_importances_'):
    imp = pd.Series(best['model'].feature_importances_, index=features).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.plot(kind='barh', color='steelblue', ax=ax)
    ax.set_title(f'Feature Importance — {best_name}')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('plots/10_feature_importance.png', dpi=150)
    plt.close()

# ── Save ───────────────────────────────────────────────────────────────────────
joblib.dump(best['model'], 'models/best_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le, 'models/label_encoder.pkl')
joblib.dump({'best_model': best_name, 'features': features, 'scaled': best['scaled']},
            'models/metadata.pkl')

print(f"\nModel saved  → models/best_model.pkl")
print(f"Plots saved  → plots/")
