"""
build_features.py
Computes per-wallet reputation features from the raw borrow/repay/liquidation CSVs.

Anti-farming design:
  - Reputation contribution is log-weighted by loan size (micro-loans ≈ worthless)
  - max_loan_usd is stored per wallet so the contract can cap discounts
"""

import pandas as pd
import numpy as np
import os

# Unix timestamps for key market stress events
LUNA_CRASH    = 1652140800   # ~May 10 2022
COVID_CRASH   = 1583798400   # ~Mar 10 2020
NOW           = 1700000000   # reference "now" (Nov 2023) for recency calc


def load_data(data_dir: str = "data") -> tuple:
    borrows      = pd.read_csv(f"{data_dir}/borrows.csv")
    repays       = pd.read_csv(f"{data_dir}/repays.csv")
    liquidations = pd.read_csv(f"{data_dir}/liquidations.csv")
    return borrows, repays, liquidations


def compute_features(borrows: pd.DataFrame,
                     repays: pd.DataFrame,
                     liquidations: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per wallet with all reputation features.
    """
    wallets = borrows["wallet"].unique()
    print(f"Computing features for {len(wallets):,} wallets...")

    records = []
    for w in wallets:
        b = borrows[borrows["wallet"] == w].copy()
        r = repays[repays["wallet"] == w].copy()
        liq = liquidations[liquidations["wallet"] == w]

        # ── Basic counts ─────────────────────────────────────────────────────
        n_borrows      = len(b)
        n_repays       = len(r)
        n_liquidations = len(liq)

        # ── Repayment rate (capped at 1.0) ───────────────────────────────────
        total_borrowed = b["amount_usd"].sum()
        total_repaid   = r["amount_usd"].sum()
        repayment_rate = min(total_repaid / total_borrowed, 1.0) if total_borrowed > 0 else 0.0

        # ── Log-weighted repayment score (anti-farming core) ─────────────────
        # Each repay contributes log(amount_usd + 1), normalised
        # This means repaying $10 adds ≈2.4 pts; $10,000 adds ≈9.2 pts
        # A farmer repaying 1000×$10 gets 2400 pts; one $10k repayment = 9200 pts
        if len(r) > 0:
            log_repay_score = r["amount_usd"].apply(lambda x: np.log1p(max(x, 0))).sum()
        else:
            log_repay_score = 0.0

        # ── Recency weight: recent activity matters more ──────────────────────
        if len(r) > 0:
            latest_repay   = r["timestamp"].max()
            days_since     = (NOW - latest_repay) / 86400
            recency_weight = np.exp(-days_since / 365)   # exponential decay, 1yr half-life
        else:
            recency_weight = 0.0

        weighted_score = log_repay_score * recency_weight

        # ── Wallet age (days since first borrow) ─────────────────────────────
        wallet_age_days = (NOW - b["timestamp"].min()) / 86400

        # ── Max single loan — used by contract to cap discount ───────────────
        max_loan_usd = b["amount_usd"].max()

        # ── Protocol diversity (number of distinct assets borrowed) ──────────
        n_assets = b["asset"].nunique()

        # ── Stress-period behaviour ───────────────────────────────────────────
        # Did the wallet have active loans during crash windows?
        # If they repaid during a crash without being liquidated → strong signal
        luna_borrows = b[b["timestamp"] < LUNA_CRASH]
        luna_repays  = r[(r["timestamp"] >= LUNA_CRASH) &
                         (r["timestamp"] <= LUNA_CRASH + 30*86400)]
        
        LUNA_WINDOW_END = LUNA_CRASH + 30 * 86400
        
        luna_liquidations = liq[(liq['timestamp'] >= LUNA_CRASH) &(liq['timestamp'] <= LUNA_WINDOW_END)] 
        
        survived_luna = int(len(luna_borrows) > 0 and len(luna_repays) > 0 and len(luna_liquidations) == 0)
        
        COVID_WINDOW_END = COVID_CRASH + 30 * 86400
        
        covid_liquidations = liq[(liq['timestamp'] >= COVID_CRASH) & (liq['timestamp'] <= COVID_WINDOW_END)] 
        
        survived_covid = int(len(covid_borrows) > 0 and len(covid_repays) > 0 and len(covid_liquidations) == 0)

        # ── Label: was this wallet ever liquidated? ───────────────────────────
        # This is the binary target for the ML model
        defaulted = int(n_liquidations > 0)

        records.append({
            "wallet":           w,
            "n_borrows":        n_borrows,
            "n_repays":         n_repays,
            "n_liquidations":   n_liquidations,
            "repayment_rate":   repayment_rate,
            "log_repay_score":  log_repay_score,
            "recency_weight":   recency_weight,
            "weighted_score":   weighted_score,
            "wallet_age_days":  wallet_age_days,
            "max_loan_usd":     max_loan_usd,
            "n_assets":         n_assets,
            "total_borrowed":   total_borrowed,
            "total_repaid":     total_repaid,
            "survived_luna":    survived_luna,
            "survived_covid":   survived_covid,
            "defaulted":        defaulted,   # ← ML target
        })

    df = pd.DataFrame(records)

    # ── Normalise weighted_score to [0, 1] for interpretability ──────────────
    max_ws = df["weighted_score"].max()
    df["reputation_score"] = df["weighted_score"] / max_ws if max_ws > 0 else 0.0

    print(f"  Default rate in dataset: {df['defaulted'].mean():.1%}")
    print(f"  Median wallet age:       {df['wallet_age_days'].median():.0f} days")
    print(f"  Median max loan USD:     ${df['max_loan_usd'].median():,.0f}")

    return df


def score_to_collateral_ratio(score: float) -> float:
    """
    Maps reputation_score ∈ [0,1] → collateral ratio ∈ [1.20, 1.50].
    Curve: 150% for unknown wallets, drops to 105% for perfect score.
    Linear interpolation — easy to argue and explain.
    """
    min_ratio = 1.20   # best case: 105% (undercollateralised)
    max_ratio = 1.50   # baseline: 150% (Aave standard)
    return max_ratio - score * (max_ratio - min_ratio)


def main():
    os.makedirs("data", exist_ok=True)
    borrows, repays, liquidations = load_data()
    df = compute_features(borrows, repays, liquidations)
    df["collateral_ratio"] = df["reputation_score"].apply(score_to_collateral_ratio)
    df.to_csv("data/wallet_features.csv", index=False)
    print(f"\nSaved data/wallet_features.csv  ({len(df):,} wallets)")
    print(df[["wallet","reputation_score","collateral_ratio","max_loan_usd","defaulted"]].head(10).to_string())


if __name__ == "__main__":
    main()
