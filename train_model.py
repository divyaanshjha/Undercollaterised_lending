"""
train_model.py
Trains a logistic regression on wallet features to predict default probability.
The predicted probability IS the reputation signal — no black box needed.
Saves the model and outputs AUC-ROC curve for the paper.
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline

FEATURE_COLS = [
    "repayment_rate",
    "log_repay_score",
    "recency_weight",
    "wallet_age_days",
    "n_assets",
    "n_borrows",
    "survived_luna",
    "survived_covid",
]


def load_features(path: str = "data/wallet_features.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # Drop wallets with no meaningful history
    df = df[df["total_borrowed"] > 10].copy()
    print(f"Wallets after filtering: {len(df):,}  |  Default rate: {df['defaulted'].mean():.1%}")
    return df


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(
            C=1.0,
            class_weight="balanced",   # handles class imbalance (few defaults)
            max_iter=1000,
            solver="lbfgs",
            random_state=42
        ))
    ])


def plot_roc(y_test, y_prob, auc: float, out_dir: str):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#4F8EF7", lw=2.5,
            label=f"Reputation model  (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Default Prediction Model", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{out_dir}/roc_curve.png", dpi=150)
    plt.close()
    print(f"  Saved {out_dir}/roc_curve.png")


def plot_score_distribution(df: pd.DataFrame, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Reputation score by default status
    for label, group in df.groupby("defaulted"):
        name = "Defaulted" if label == 1 else "Repaid"
        color = "#E05C5C" if label == 1 else "#4CAF7D"
        axes[0].hist(group["reputation_score"], bins=30, alpha=0.65,
                     label=name, color=color, density=True)
    axes[0].set_xlabel("Reputation Score")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Score Distribution by Default Status")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Collateral ratio distribution
    axes[1].hist(df["ml_collateral_ratio"], bins=30, color="#4F8EF7", alpha=0.8, edgecolor="white")
    axes[1].axvline(1.50, color="red", linestyle="--", lw=2, label="Aave baseline (150%)")
    axes[1].axvline(df["ml_collateral_ratio"].mean(), color="green",
                    linestyle="--", lw=2, label=f"Protocol mean ({df['ml_collateral_ratio'].mean():.0%})")
    axes[1].set_xlabel("Collateral Ratio")
    axes[1].set_ylabel("Wallet Count")
    axes[1].set_title("Personalised Collateral Ratio Distribution")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{out_dir}/score_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved {out_dir}/score_distribution.png")


def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models",  exist_ok=True)

    df = load_features()

    # ── Train / test split (stratified to preserve default rate) ─────────────
    X = df[FEATURE_COLS].fillna(0)
    y = df["defaulted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── Fit ──────────────────────────────────────────────────────────────────
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    y_prob = pipe.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)

    cv_aucs = cross_val_score(pipe, X, y, cv=StratifiedKFold(5),
                               scoring="roc_auc", n_jobs=-1)

    print(f"\n{'─'*40}")
    print(f"  Hold-out AUC-ROC : {auc:.4f}")
    print(f"  5-fold CV AUC    : {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")
    print(f"{'─'*40}\n")
    print(classification_report(y_test, pipe.predict(X_test), target_names=["Repaid","Defaulted"]))

    # ── Attach model-predicted score back to full dataset ────────────────────
    df["default_prob"]      = pipe.predict_proba(X)[:, 1]
    # Higher default_prob → lower reputation → higher collateral ratio
    df["ml_reputation"]     = 1.0 - df["default_prob"]
    df["ml_collateral_ratio"] = 1.20 + df["default_prob"] * 0.30   # maps [0,1] → [1.20, 1.50]
    df.to_csv("data/wallet_features.csv", index=False)

    # ── Feature coefficients ─────────────────────────────────────────────────
    coef = pd.Series(
        pipe.named_steps["model"].coef_[0],
        index=FEATURE_COLS
    ).sort_values()
    print("\nFeature coefficients (positive = raises default risk):")
    print(coef.to_string())

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_roc(y_test, y_prob, auc, "outputs")
    plot_score_distribution(df, "outputs")

    # ── Save model ───────────────────────────────────────────────────────────
    with open("models/reputation_model.pkl", "wb") as f:
        pickle.dump({"pipeline": pipe, "features": FEATURE_COLS, "auc": auc}, f)
    print("\n  Saved models/reputation_model.pkl")
    print(f"\nCapital efficiency snapshot:")
    print(f"  Baseline (all wallets at 150%)       : {1.50:.2f}x avg collateral")
    print(f"  Protocol (personalised):               {df['ml_collateral_ratio'].mean():.2f}x avg collateral")
    print(f"  Efficiency gain (lower = better):      {(1.50 - df['ml_collateral_ratio'].mean()) / 1.50:.1%} reduction")


if __name__ == "__main__":
    main()
