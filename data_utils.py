#!/usr/bin/env python3
import pandas as pd
import numpy as np

def load_or_create_dataset(path="telco_churn.csv", n=500):
    """
    Tries to load a real dataset from `path`. If not found, creates a demo synthetic dataset
    with the columns the pipeline expects and saves it to `telco_churn.csv`.
    """
    try:
        df = pd.read_csv(path)
        print(f"📁 Loaded {path}")
        return df
    except FileNotFoundError:
        print(f"⚠️ {path} not found — creating demo dataset")

    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "Tenure": rng.integers(1, 72, size=n),
        "MonthlyCharges": rng.integers(20, 120, size=n),
        "TotalCharges": rng.integers(20, 5000, size=n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "None"], size=n),
        "Churn": rng.choice([0, 1], size=n, p=[0.73, 0.27])  # realistic-ish churn ratio
    })
    df.to_csv(path, index=False)
    print(f"✅ Demo dataset saved -> {path}")
    return df