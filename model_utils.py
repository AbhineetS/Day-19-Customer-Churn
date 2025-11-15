#!/usr/bin/env python3
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def _get_feature_names(column_transformer, numeric_cols, categorical_cols):
    # column_transformer: fitted ColumnTransformer
    feature_names = []
    for name, trans, cols in column_transformer.transformers_:
        if name == "num":
            feature_names.extend(numeric_cols)
        elif name == "cat":
            # OneHotEncoder used
            ohe = trans
            # get_feature_names_out needs fitted encoder
            names = list(ohe.get_feature_names_out(cols))
            feature_names.extend(names)
    return feature_names

def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Returns: X_train_transformed, X_test_transformed, y_train, y_test, fitted_preprocessor, feature_names
    """
    # target
    if "Churn" not in df.columns:
        raise ValueError("Input dataframe must have 'Churn' column")

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # define columns (adjust if your dataset columns differ)
    categorical = ["Contract", "InternetService"]
    numeric = ["Tenure", "MonthlyCharges", "TotalCharges"]

    # ensure columns present (if demo dataset missing columns, fallback)
    present_numeric = [c for c in numeric if c in X.columns]
    present_categorical = [c for c in categorical if c in X.columns]

    # ColumnTransformer with OneHotEncoder (return dense arrays)
    cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preprocessor = ColumnTransformer([
        ("cat", cat_encoder, present_categorical),
        ("num", StandardScaler(), present_numeric)
    ], remainder="drop")

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique()>1 else None
    )

    # fit preprocessor on training data and transform both
    preprocessor.fit(X_train)
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # derive readable feature names
    feature_names = _get_feature_names(preprocessor, present_numeric, present_categorical)

    # If transformer returns numpy arrays, keep them as-is
    # Convert y to numpy
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    return X_train_t, X_test_t, y_train, y_test, preprocessor, feature_names

def train_model(model_name, X_train, y_train):
    """Train and return the model (currently RandomForest)."""
    if model_name != "random_forest":
        print("⚠️ Only 'random_forest' supported; using RandomForestClassifier")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    # Save model
    joblib.dump(model, "churn_model.pkl")
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🔍 Accuracy: {acc:.4f}")
    print("📋 Classification report:")
    print(classification_report(y_test, y_pred, digits=4))
    return y_pred

def save_artifacts(model, preprocessor):
    joblib.dump(model, "churn_model.pkl")
    joblib.dump(preprocessor, "preprocessor.pkl")