"""
Generates a synthetic credit card fraud dataset.
Features are based on real-world fraud detection patterns.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 10000

# ── Legitimate transactions (90%) ──────────────────────────────────────────────
n_legit = int(N * 0.90)
n_fraud = N - n_legit

def make_legit(n):
    return {
        'transaction_amount':        np.random.exponential(80, n).clip(1, 5000),
        'transaction_hour':          np.random.choice(range(6, 23), n),
        'distance_from_home_km':     np.random.exponential(10, n).clip(0.1, 200),
        'distance_from_last_txn_km': np.random.exponential(5, n).clip(0.1, 100),
        'ratio_to_median_purchase':  np.random.normal(1.0, 0.3, n).clip(0.1, 5),
        'repeat_retailer':           np.random.binomial(1, 0.75, n),
        'used_chip':                 np.random.binomial(1, 0.80, n),
        'used_pin':                  np.random.binomial(1, 0.70, n),
        'online_order':              np.random.binomial(1, 0.25, n),
        'num_transactions_today':    np.random.randint(1, 6, n),
        'merchant_category':         np.random.choice(
                                        ['grocery','retail','restaurant',
                                         'travel','entertainment','online'],
                                        n, p=[0.30,0.25,0.20,0.10,0.10,0.05]),
        'fraud': np.zeros(n, dtype=int)
    }

def make_fraud(n):
    return {
        'transaction_amount':        np.random.exponential(300, n).clip(50, 10000),
        'transaction_hour':          np.random.choice(list(range(0, 6)) + list(range(22, 24)), n),
        'distance_from_home_km':     np.random.exponential(80, n).clip(5, 5000),
        'distance_from_last_txn_km': np.random.exponential(50, n).clip(1, 5000),
        'ratio_to_median_purchase':  np.random.normal(3.5, 1.5, n).clip(0.5, 20),
        'repeat_retailer':           np.random.binomial(1, 0.15, n),
        'used_chip':                 np.random.binomial(1, 0.20, n),
        'used_pin':                  np.random.binomial(1, 0.10, n),
        'online_order':              np.random.binomial(1, 0.70, n),
        'num_transactions_today':    np.random.randint(3, 15, n),
        'merchant_category':         np.random.choice(
                                        ['grocery','retail','restaurant',
                                         'travel','entertainment','online'],
                                        n, p=[0.05,0.10,0.05,0.20,0.15,0.45]),
        'fraud': np.ones(n, dtype=int)
    }

df = pd.concat([
    pd.DataFrame(make_legit(n_legit)),
    pd.DataFrame(make_fraud(n_fraud))
], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv('data/fraud_dataset.csv', index=False)
print(f"Dataset created: {len(df):,} records")
print(f"Fraud cases   : {df['fraud'].sum():,} ({df['fraud'].mean()*100:.1f}%)")
print(f"Legit cases   : {(df['fraud']==0).sum():,} ({(df['fraud']==0).mean()*100:.1f}%)")
print(df.head())
