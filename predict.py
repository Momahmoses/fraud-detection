"""
Fraud Detection — Interactive Prediction Tool
Run: python3 predict.py
"""

import joblib
import numpy as np

model    = joblib.load('models/best_model.pkl')
scaler   = joblib.load('models/scaler.pkl')
le       = joblib.load('models/label_encoder.pkl')
metadata = joblib.load('models/metadata.pkl')

CATEGORIES = ['grocery', 'retail', 'restaurant', 'travel', 'entertainment', 'online']

THRESHOLD = 0.40  # lower threshold = catch more fraud (higher recall)

def get_input(prompt, type_fn, valid=None, min_val=None, max_val=None):
    while True:
        try:
            raw = input(prompt).strip()
            val = type_fn(raw)
            if valid is not None and val not in valid:
                print(f"  Please enter one of: {valid}")
                continue
            if min_val is not None and val < min_val:
                print(f"  Value must be >= {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"  Value must be <= {max_val}")
                continue
            return val
        except (ValueError, KeyError):
            print("  Invalid input, please try again.")

def predict():
    print("\n" + "="*60)
    print("       FRAUD DETECTION & PREDICTION SYSTEM")
    print("="*60)
    print("Enter transaction details below:\n")

    amount    = get_input("Transaction amount ($): ", float, min_val=0.01)
    hour      = get_input("Transaction hour (0-23): ", int, min_val=0, max_val=23)
    dist_home = get_input("Distance from home (km): ", float, min_val=0)
    dist_last = get_input("Distance from last transaction (km): ", float, min_val=0)
    ratio     = get_input("Ratio to your median purchase (e.g. 1.0 = normal, 3.0 = 3x usual): ", float, min_val=0)
    repeat    = get_input("Is this a repeat retailer? (1=Yes, 0=No): ", int, valid=[0,1])
    chip      = get_input("Was chip used? (1=Yes, 0=No): ", int, valid=[0,1])
    pin       = get_input("Was PIN used? (1=Yes, 0=No): ", int, valid=[0,1])
    online    = get_input("Is this an online order? (1=Yes, 0=No): ", int, valid=[0,1])
    num_txns  = get_input("Number of transactions made today so far: ", int, min_val=1)

    print(f"\nMerchant categories: {', '.join(f'{i+1}.{c}' for i,c in enumerate(CATEGORIES))}")
    cat_idx  = get_input("Select category number: ", int, min_val=1, max_val=len(CATEGORIES)) - 1
    category = CATEGORIES[cat_idx]
    cat_enc  = le.transform([category])[0]

    features = np.array([[
        amount, hour, dist_home, dist_last, ratio,
        repeat, chip, pin, online, num_txns, cat_enc
    ]])

    if metadata['scaled']:
        features = scaler.transform(features)

    prob      = model.predict_proba(features)[0][1]
    predicted = int(prob >= THRESHOLD)

    # Risk level
    if prob < 0.25:
        risk_label = "LOW"
        risk_color = "SAFE"
    elif prob < 0.50:
        risk_label = "MEDIUM"
        risk_color = "CAUTION"
    elif prob < 0.75:
        risk_label = "HIGH"
        risk_color = "WARNING"
    else:
        risk_label = "CRITICAL"
        risk_color = "ALERT"

    print("\n" + "="*60)
    print(f"  TRANSACTION ANALYSIS RESULT")
    print("="*60)
    print(f"  Amount          : ${amount:,.2f}")
    print(f"  Category        : {category.capitalize()}")
    print(f"  Fraud Probability: {prob*100:.1f}%")
    print(f"  Risk Level      : {risk_label} [{risk_color}]")
    print(f"  Decision        : {'>>> FRAUDULENT — BLOCK TRANSACTION <<<' if predicted else 'LEGITIMATE — APPROVE TRANSACTION'}")
    print("="*60)

    if predicted:
        print("\n  RED FLAGS DETECTED:")
        if amount > 500:
            print(f"  - Unusually high amount (${amount:,.2f})")
        if dist_home > 50:
            print(f"  - Transaction far from home ({dist_home:.1f} km)")
        if ratio > 2.5:
            print(f"  - Amount is {ratio:.1f}x your usual spending")
        if online and not chip and not pin:
            print("  - Online purchase with no chip/PIN verification")
        if num_txns > 5:
            print(f"  - High number of transactions today ({num_txns})")
        if not repeat:
            print("  - First-time retailer")
        print("\n  RECOMMENDED ACTIONS:")
        print("  - Block this transaction immediately")
        print("  - Send OTP verification to the card holder")
        print("  - Flag account for manual review")
        print("  - Contact card holder to verify activity")
    else:
        print("\n  Transaction appears legitimate.")
        if prob > 0.25:
            print("  Note: Some mild risk indicators present — monitor account.")
    print("="*60)

    again = input("\nCheck another transaction? (y/n): ").strip().lower()
    if again == 'y':
        predict()

if __name__ == '__main__':
    predict()
