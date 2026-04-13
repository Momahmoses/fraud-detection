"""
Exploratory Data Analysis — Fraud Detection Dataset
Generates visual insights saved to plots/
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('plots', exist_ok=True)
sns.set_theme(style='whitegrid')

df = pd.read_csv('data/fraud_dataset.csv')

print("="*55)
print("  FRAUD DETECTION — EXPLORATORY DATA ANALYSIS")
print("="*55)
print(f"\nShape       : {df.shape}")
print(f"\nClass split :\n{df['fraud'].value_counts().rename({0:'Legitimate',1:'Fraud'})}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nBasic stats:\n{df.describe().round(2)}")

# ── Plot 1: Class Distribution ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
counts = df['fraud'].value_counts()
bars = ax.bar(['Legitimate', 'Fraud'], counts.values, color=['steelblue', 'crimson'])
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f'{val:,}', ha='center', fontweight='bold')
ax.set_title('Class Distribution: Fraud vs Legitimate')
ax.set_ylabel('Number of Transactions')
plt.tight_layout()
plt.savefig('plots/01_class_distribution.png', dpi=150)
plt.close()

# ── Plot 2: Transaction Amount Distribution ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, label, color, title in zip(
    axes,
    [0, 1],
    ['steelblue', 'crimson'],
    ['Legitimate Transactions', 'Fraudulent Transactions']
):
    data = df[df['fraud'] == label]['transaction_amount']
    ax.hist(data, bins=50, color=color, edgecolor='white', alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel('Transaction Amount ($)')
    ax.set_ylabel('Count')
    ax.axvline(data.median(), color='black', linestyle='--', label=f'Median: ${data.median():.0f}')
    ax.legend()
plt.suptitle('Transaction Amount Distribution', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/02_amount_distribution.png', dpi=150)
plt.close()

# ── Plot 3: Fraud by Hour ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
fraud_by_hour = df.groupby('transaction_hour')['fraud'].mean() * 100
ax.bar(fraud_by_hour.index, fraud_by_hour.values, color='crimson', alpha=0.8)
ax.set_title('Fraud Rate by Hour of Day')
ax.set_xlabel('Hour (24h)')
ax.set_ylabel('Fraud Rate (%)')
ax.set_xticks(range(24))
plt.tight_layout()
plt.savefig('plots/03_fraud_by_hour.png', dpi=150)
plt.close()

# ── Plot 4: Fraud by Merchant Category ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
fraud_by_cat = df.groupby('merchant_category')['fraud'].mean().sort_values(ascending=False) * 100
bars = ax.bar(fraud_by_cat.index, fraud_by_cat.values, color='darkorange', alpha=0.85)
for bar, val in zip(bars, fraud_by_cat.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}%', ha='center', fontsize=9)
ax.set_title('Fraud Rate by Merchant Category')
ax.set_ylabel('Fraud Rate (%)')
plt.tight_layout()
plt.savefig('plots/04_fraud_by_category.png', dpi=150)
plt.close()

# ── Plot 5: Correlation Heatmap ────────────────────────────────────────────────
num_cols = ['transaction_amount','distance_from_home_km','distance_from_last_txn_km',
            'ratio_to_median_purchase','num_transactions_today','used_chip',
            'used_pin','online_order','repeat_retailer','fraud']
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, linewidths=0.5)
ax.set_title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('plots/05_correlation_heatmap.png', dpi=150)
plt.close()

# ── Plot 6: Key Features Boxplot ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
features = ['transaction_amount', 'distance_from_home_km', 'ratio_to_median_purchase']
titles   = ['Transaction Amount ($)', 'Distance from Home (km)', 'Ratio to Median Purchase']
for ax, feat, title in zip(axes, features, titles):
    data = [df[df['fraud']==0][feat], df[df['fraud']==1][feat]]
    bp = ax.boxplot(data, labels=['Legit', 'Fraud'], patch_artist=True,
                    notch=True, showfliers=False)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('crimson')
    ax.set_title(title)
plt.suptitle('Key Feature Distributions: Fraud vs Legitimate', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/06_feature_boxplots.png', dpi=150)
plt.close()

print("\nAll EDA plots saved to plots/")
