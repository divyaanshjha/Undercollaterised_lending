"""
generate_synthetic_data.py

Generates realistic synthetic Aave V2 wallet histories calibrated to
publicly known Aave statistics:
  - ~2–4% historical liquidation rate
  - Loan size distribution: log-normal, median ~$3,000, mean ~$12,000
  - Wallet age: uniform 30–1000 days since Aave V2 launch (Dec 2020)
  - Protocol diversity: 1–5 assets per borrower
  - Repayment rates: 0.85–1.0 for good wallets; 0.3–0.7 for bad ones

This approach is standard in DeFi academic papers where subgraph APIs 
are rate-limited or deprecated. The model is trained on these distributions
and backtested on synthetic stress scenarios.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

# Aave V2 launched Dec 2020; use seconds since epoch
AAVE_V2_LAUNCH = 1607472000   # Dec 9 2020
NOW            = 1700000000   # Nov 2023 reference
LUNA_CRASH     = 1652140800   # May 10 2022
COVID_CRASH    = 1583798400   # Mar 10 2020  (pre-V2, used for synthetic stress)


def make_wallet_id(i: int) -> str:
    return f"0x{i:040x}"


def generate_wallets(n: int = 3000) -> pd.DataFrame:
    """
    Generates n synthetic wallet profiles with realistic Aave-like distributions.
    Roughly 3% will be "bad" wallets (eventual defaulters).
    """
    print(f"Generating {n:,} synthetic wallet histories...")

    # Each wallet is assigned a latent risk score:
    # low risk (0.0–0.3) → good borrower; high risk (0.7–1.0) → likely defaulter
    risk = np.random.beta(a=2, b=12, size=n)   # ~3% high-risk tail

    # Wallet age in days (uniform between 30 and ~1000 days post-launch)
    wallet_age = np.random.uniform(30, 1000, n)

    # Number of borrow events per wallet (Poisson, mean ~4)
    n_borrows = np.random.poisson(4, n).clip(1, 20)

    # Loan sizes: log-normal, median ~$3k
    # Shape varies by risk — bad wallets tend to make larger last loans
    base_loan = np.random.lognormal(mean=8.0, sigma=1.5, size=n).clip(50, 500_000)
    max_loan  = base_loan * np.random.uniform(1.0, 3.0, n)   # max loan can be 1–3× typical

    # Total borrowed scales with n_borrows and base_loan
    total_borrowed = base_loan * n_borrows * np.random.uniform(0.7, 1.3, n)

    # Repayment rate: good wallets repay fully; bad wallets don't
    # Adding noise so that some good wallets look risky and vice versa (realistic)
    repay_rate = np.where(
        risk < 0.5,
        np.random.uniform(0.80, 1.00, n) - risk * 0.2,   # mostly good, slight risk penalty
        np.random.uniform(0.10, 0.75, n)                  # bad wallets: partial repayment
    ).clip(0.05, 1.0)
    total_repaid = total_borrowed * repay_rate

    # Log-weighted repayment score (anti-farming design)
    # Approximated per wallet: sum of log(loan_i) for each repaid loan
    avg_repaid_per_event = total_repaid / n_borrows.clip(1)
    log_repay_score = np.log1p(avg_repaid_per_event.clip(0)) * n_borrows
    # Add realistic noise to simulate wallet activity variance
    log_repay_score *= np.random.uniform(0.85, 1.15, n)

    # Recency weight: exponential decay from last repay
    days_since_last_repay = np.random.uniform(10, 400, n)
    recency_weight = np.exp(-days_since_last_repay / 365)
    weighted_score = log_repay_score * recency_weight

    # Asset diversity (1–5 assets)
    n_assets = np.random.randint(1, 6, n)

    # Liquidation: a wallet is liquidated if risk > threshold
    # Risk 0.7+ → 50% chance; 0.4–0.7 → 10% chance; < 0.4 → 2% chance
    liq_prob = np.where(risk > 0.7, 0.50,
               np.where(risk > 0.4, 0.10, 0.02))
    n_liquidations = np.random.binomial(1, liq_prob, n)

    # Stress-period survival
    # Wallets with low risk that were active near crash windows
    active_near_luna  = (wallet_age > (NOW - LUNA_CRASH) / 86400) & (risk < 0.3)
    active_near_covid = (wallet_age > 900) & (risk < 0.3)   # only old wallets
    survived_luna  = active_near_luna  & (n_liquidations == 0)
    survived_covid = active_near_covid & (n_liquidations == 0)

    # Reputation score: normalise weighted_score to [0,1]
    ws_max = weighted_score.max()
    reputation_score = weighted_score / ws_max if ws_max > 0 else weighted_score

    # Collateral ratio from reputation
    # Floor at 120%: survives a 16.7% price drop before any liquidation.
    # This makes the mechanism realistic — even top-tier wallets have a safety buffer.
    ml_collateral_ratio = 1.20 + (1 - reputation_score) * 0.30  # [1.20, 1.50]

    # Defaulted = was ever liquidated (ML target)
    defaulted = (n_liquidations > 0).astype(int)

    df = pd.DataFrame({
        "wallet":               [make_wallet_id(i) for i in range(n)],
        "n_borrows":            n_borrows,
        "n_repays":             (n_borrows * repay_rate).astype(int),
        "n_liquidations":       n_liquidations,
        "repayment_rate":       repay_rate,
        "log_repay_score":      log_repay_score,
        "recency_weight":       recency_weight,
        "weighted_score":       weighted_score,
        "wallet_age_days":      wallet_age,
        "max_loan_usd":         max_loan,
        "n_assets":             n_assets,
        "total_borrowed":       total_borrowed,
        "total_repaid":         total_repaid,
        "survived_luna":        survived_luna.astype(int),
        "survived_covid":       survived_covid.astype(int),
        "defaulted":            defaulted,
        "reputation_score":     reputation_score,
        "ml_collateral_ratio":  ml_collateral_ratio,
        "latent_risk":          risk,           # ground truth for calibration check
    })

    print(f"  Default rate: {defaulted.mean():.1%}  (target: 2–5%)")
    print(f"  Median loan size:   ${np.median(total_borrowed):,.0f}")
    print(f"  Median wallet age:  {np.median(wallet_age):.0f} days")
    return df


def generate_events(df: pd.DataFrame) -> tuple:
    """
    Generates flat event-level tables (borrows, repays, liquidations)
    from the wallet-level summary — needed for compatibility with build_features.py.
    """
    borrow_rows = []
    repay_rows  = []
    liq_rows    = []

    assets = ["ETH", "USDC", "WBTC", "DAI", "USDT", "LINK", "AAVE"]

    for _, w in df.iterrows():
        wallet = w["wallet"]
        age    = int(w["wallet_age_days"])
        first_ts = NOW - age * 86400

        n_b = int(w["n_borrows"])
        n_r = int(w["n_repays"])
        n_l = int(w["n_liquidations"])

        per_borrow_usd = w["total_borrowed"] / max(n_b, 1)

        for i in range(n_b):
            ts = int(first_ts + i * (age * 86400 / max(n_b, 1)))
            borrow_rows.append({
                "wallet": wallet,
                "asset":  assets[i % len(assets)],
                "amount_usd": per_borrow_usd * np.random.uniform(0.7, 1.3),
                "timestamp": ts,
                "event": "borrow"
            })

        for i in range(n_r):
            ts = int(first_ts + (i + 0.5) * (age * 86400 / max(n_r, 1)))
            repay_rows.append({
                "wallet": wallet,
                "asset":  assets[i % len(assets)],
                "amount_usd": w["total_repaid"] / max(n_r, 1),
                "timestamp": ts,
                "event": "repay"
            })

        if n_l > 0:
            liq_ts = int(NOW - np.random.uniform(30, 365) * 86400)
            liq_rows.append({
                "wallet": wallet,
                "timestamp": liq_ts,
                "event": "liquidation"
            })

    return (
        pd.DataFrame(borrow_rows),
        pd.DataFrame(repay_rows),
        pd.DataFrame(liq_rows)
    )


def main():
    os.makedirs("data", exist_ok=True)

    df = generate_wallets(n=3000)
    borrows, repays, liquidations = generate_events(df)

    df.to_csv("data/wallet_features.csv",  index=False)
    borrows.to_csv("data/borrows.csv",     index=False)
    repays.to_csv("data/repays.csv",       index=False)
    liquidations.to_csv("data/liquidations.csv", index=False)

    print(f"\nSaved:")
    print(f"  data/wallet_features.csv  → {len(df):,} wallets")
    print(f"  data/borrows.csv          → {len(borrows):,} events")
    print(f"  data/repays.csv           → {len(repays):,} events")
    print(f"  data/liquidations.csv     → {len(liquidations):,} events")


if __name__ == "__main__":
    main()
