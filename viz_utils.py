#!/usr/bin/env python3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_confusion_matrix(y_true, y_pred, out="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=0.5)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"📁 Saved -> {out}")

def plot_feature_importance(model, feature_names, out="feature_importance.png", top_n=30):
    try:
        importances = model.feature_importances_
    except AttributeError:
        print("⚠️ Model has no feature_importances_. Skipping feature importance plot.")
        return

    # ensure feature_names length matches importances
    if len(feature_names) != len(importances):
        # fallback to indices
        labels = [f"f{i}" for i in range(len(importances))]
    else:
        labels = feature_names

    # sort by importance
    idx = np.argsort(importances)[::-1][:top_n]
    sorted_imp = importances[idx]
    sorted_labels = [labels[i] for i in idx]

    plt.figure(figsize=(8, max(4, 0.2 * len(sorted_labels))))
    sns.barplot(x=sorted_imp, y=sorted_labels)
    plt.xlabel("Importance")
    plt.title("Feature Importance (top {})".format(min(top_n, len(sorted_labels))))
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"📁 Saved -> {out}")