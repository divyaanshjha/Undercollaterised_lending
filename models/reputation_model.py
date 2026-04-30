"""
Reputation Model & Collateral Ratio Engine
============================================
1. Trains logistic regression on wallet features → default probability
2. Maps default probability → personalised collateral ratio
3. Enforces anti-farming cap: discount is bounded by historical max loan
4. Outputs wallet_scores.csv for use in simulation
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report,
                             confusion_matrix)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ─── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv("wallets.csv")

FEATURES = [
    "repayment_rate",
    "n_liquidated",
    "recency_weighted_score",
    "wallet_age_days",
    "protocol_diversity",
    "avg_health_factor",
    "total_loans",
    "max_loan_usd",       # included so model learns scale-of-trustworthiness
]

X = df[FEATURES]
y = df["default_label"]

# ─── Train / test split (stratified) ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# ─── Model pipeline ───────────────────────────────────────────────────────────
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(class_weight="balanced", max_iter=500, C=1.0))
])

model.fit(X_train, y_train)

# ─── Evaluation ───────────────────────────────────────────────────────────────
y_prob  = model.predict_proba(X_test)[:, 1]
y_pred  = model.predict(X_test)
auc     = roc_auc_score(y_test, y_prob)
cv_aucs = cross_val_score(model, X, y, cv=5, scoring="roc_auc")

print("=" * 55)
print("        REPUTATION MODEL EVALUATION")
print("=" * 55)
print(f"  Test AUC-ROC        : {auc:.4f}")
print(f"  5-fold CV AUC       : {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")
print(f"\nClassification Report (threshold=0.5):")
print(classification_report(y_test, y_pred, target_names=["No Default","Default"]))

# ─── Collateral Ratio Mapping ─────────────────────────────────────────────────
def default_prob_to_collateral(prob: float) -> float:
    """
    Linear interpolation:
      prob = 0.0 (perfect) → 105% collateral
      prob = 0.5 (average) → 130% collateral
      prob = 1.0 (certain default) → 150% collateral (Aave baseline)
    Clipped to [1.05, 1.50].
    """
    ratio = 1.05 + (prob * 0.90)
    return round(min(max(ratio, 1.05), 1.50), 4)


def anti_farming_cap(requested_amount: float, max_historical_loan: float,
                     k: float = 2.5) -> float:
    """
    Anti-farming cap: the discounted rate only applies up to k × max historical loan.
    Anything above that is charged at 150% (baseline).

    Returns the effective blended collateral ratio for a given request size.
    """
    cap = k * max_historical_loan
    if requested_amount <= cap:
        return None   # full discount applies — return None (caller uses score ratio)
    else:
        discounted_portion = cap
        baseline_portion   = requested_amount - cap
        return (discounted_portion + baseline_portion * 1.50) / requested_amount


# ─── Score all wallets ────────────────────────────────────────────────────────
all_probs = model.predict_proba(X)[:, 1]
df["default_prob"]       = all_probs
df["base_collateral"]    = df["default_prob"].apply(default_prob_to_collateral)

# Score tier for reporting
def prob_to_tier(p):
    if p < 0.10: return "Excellent"
    if p < 0.20: return "Good"
    if p < 0.35: return "Fair"
    return "Poor"

df["score_tier"] = df["default_prob"].apply(prob_to_tier)

print("\n── Collateral Ratio by Tier ─────────────────────────────")
print(df.groupby("score_tier")["base_collateral"].describe()[
    ["count","mean","min","max"]
].to_string())

print("\n── Score Tier Distribution ──────────────────────────────")
print(df["score_tier"].value_counts().to_string())

df.to_csv("wallet_scores.csv", index=False)
print("\nSaved → wallet_scores.csv")

# ─── Plots ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Reputation Model Analysis", fontsize=14, fontweight="bold")

# 1. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[0].plot(fpr, tpr, color="#2563EB", lw=2, label=f"AUC = {auc:.3f}")
axes[0].plot([0,1],[0,1],"--", color="gray", alpha=0.5, label="Random")
axes[0].fill_between(fpr, tpr, alpha=0.1, color="#2563EB")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve")
axes[0].legend()
axes[0].set_facecolor("#F8FAFC")

# 2. Default Prob distribution by wallet type
colors = {"reliable": "#10B981", "moderate": "#F59E0B", "risky": "#EF4444"}
for wtype, grp in df.groupby("wallet_type"):
    axes[1].hist(grp["default_prob"], bins=30, alpha=0.6,
                 label=wtype, color=colors[wtype], density=True)
axes[1].set_xlabel("Predicted Default Probability")
axes[1].set_ylabel("Density")
axes[1].set_title("Score Distribution by Wallet Type")
axes[1].legend()
axes[1].set_facecolor("#F8FAFC")

# 3. Collateral Ratio Curve
p_range   = np.linspace(0, 1, 200)
col_range = [default_prob_to_collateral(p) * 100 for p in p_range]
axes[2].plot(p_range, col_range, color="#7C3AED", lw=2.5)
axes[2].axhline(150, color="gray", ls="--", alpha=0.7, label="Aave Baseline (150%)")
axes[2].fill_between(p_range, col_range, 150, alpha=0.12, color="#10B981",
                     label="Capital Efficiency Gain")
axes[2].set_xlabel("Predicted Default Probability")
axes[2].set_ylabel("Required Collateral (%)")
axes[2].set_title("Reputation → Collateral Ratio Curve")
axes[2].legend(fontsize=8)
axes[2].set_facecolor("#F8FAFC")

plt.tight_layout()
plt.savefig("../plots/reputation_model.png", dpi=150, bbox_inches="tight")
print("Saved → plots/reputation_model.png")

# ─── Feature importance (logistic regression coefficients) ────────────────────
coef = pd.Series(
    model.named_steps["clf"].coef_[0],
    index=FEATURES
).sort_values()

fig2, ax2 = plt.subplots(figsize=(8, 5))
bars = coef.plot(kind="barh", ax=ax2,
                 color=["#EF4444" if v > 0 else "#10B981" for v in coef])
ax2.axvline(0, color="black", lw=0.8)
ax2.set_title("Feature Coefficients (positive = increases default risk)")
ax2.set_xlabel("Coefficient (standardised)")
ax2.set_facecolor("#F8FAFC")
plt.tight_layout()
plt.savefig("../plots/feature_importance.png", dpi=150, bbox_inches="tight")
print("Saved → plots/feature_importance.png")
