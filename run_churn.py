#!/usr/bin/env python3
import argparse
from data_utils import load_or_create_dataset
from model_utils import preprocess_data, train_model, evaluate_model, save_artifacts
from viz_utils import plot_confusion_matrix, plot_feature_importance

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="random_forest", help="Model type (only random_forest supported)")
    args = parser.parse_args()

    print("📥 Loading dataset...")
    df = load_or_create_dataset()

    print("⚙️ Preprocessing (fit on train)...")
    X_train_t, X_test_t, y_train, y_test, preprocessor, feature_names = preprocess_data(df)

    print("🎯 Training model...")
    model = train_model(args.model, X_train_t, y_train)

    print("📊 Evaluating model...")
    y_pred = evaluate_model(model, X_test_t, y_test)

    print("💾 Saving artifacts...")
    save_artifacts(model, preprocessor)

    print("📌 Saving visualizations...")
    plot_confusion_matrix(y_test, y_pred, out="confusion_matrix.png")
    plot_feature_importance(model, feature_names, out="feature_importance.png")

    print("✅ DONE — Files generated:")
    print("   - confusion_matrix.png")
    print("   - feature_importance.png")
    print("   - churn_model.pkl")
    print("   - preprocessor.pkl")
    print("   - telco_churn.csv (if created)")

if __name__ == "__main__":
    main()