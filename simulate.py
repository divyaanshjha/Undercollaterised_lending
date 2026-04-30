"""
simulate.py — Before/after simulation with correct per-wallet liquidation buffers.

Liquidation buffer = 1 - (1 / collateral_ratio)
  → 150% collateral: liquidated only if price drops > 33.3%
  → 120% collateral: liquidated only if price drops > 16.7%
  → 105% collateral: liquidated only if price drops >  4.8%

Bad debt = max(0, loan_value - post_crash_collateral_value)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os

ANTI_FARM_K = 2.5

NAMED_SCENARIOS = {
    "Mild Correction (-15%)":   0.15,
    "Moderate Selloff (-25%)":  0.25,
    "COVID Mar 2020 (-50%)":    0.50,
    "LUNA May 2022 (-60%)":     0.60,
}

SWEEP_DROPS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]


def load_wallets(path="data/wallet_features.csv"):
    df = pd.read_csv(path)
    df = df[df["total_borrowed"] > 10].copy()
    if "ml_collateral_ratio" not in df.columns:
        df["ml_collateral_ratio"] = 1.50 - df["reputation_score"] * 0.45
    return df


def apply_anti_farming_cap(df, avg_loan):
    cap         = df["max_loan_usd"] * ANTI_FARM_K
    capped_frac = np.minimum(avg_loan, cap) / avg_loan
    blended     = df["ml_collateral_ratio"] * capped_frac + 1.50 * (1 - capped_frac)
    return blended.clip(1.05, 1.50)


def simulate_crash(loan_values, collateral_ratios, price_drop):
    coll_values  = loan_values * collateral_ratios
    liq_buffers  = 1.0 - (1.0 / collateral_ratios)        # price drop needed to trigger liquidation
    is_liq       = price_drop > liq_buffers
    post_crash   = coll_values * (1 - price_drop)
    bad_debt     = np.maximum(loan_values - post_crash, 0) * is_liq
    total_loans  = loan_values.sum()
    return {
        "price_drop":         price_drop,
        "n_liquidated":       int(is_liq.sum()),
        "liquidation_rate":   float(is_liq.mean()),
        "bad_debt_usd":       float(bad_debt.sum()),
        "bad_debt_ratio":     float(bad_debt.sum() / total_loans) if total_loans else 0,
        "capital_efficiency": float(total_loans / coll_values.sum()) if coll_values.sum() else 0,
        "avg_collateral":     float(collateral_ratios.mean()),
    }


def demand_model(df):
    avg_protocol   = df["ml_collateral_ratio"].mean()
    pct_reduction  = (1.50 - avg_protocol) / 1.50
    gain_pct       = pct_reduction * 0.8                   # elasticity = 0.8
    new_borrowers  = int(len(df) * gain_pct)
    return {
        "baseline":        len(df),
        "new_borrowers":   new_borrowers,
        "gain_pct":        gain_pct,
        "new_volume":      new_borrowers * df["total_borrowed"].median(),
        "avg_protocol":    avg_protocol,
        "ratio_reduction": pct_reduction,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_named_scenarios(results, out_dir):
    df        = pd.DataFrame(results)
    scenarios = list(NAMED_SCENARIOS.keys())
    x         = np.arange(len(scenarios))
    w         = 0.35
    colors    = {"Baseline (150%)": "#E05C5C", "Protocol (Dynamic)": "#4CAF7D"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Protocol vs Baseline: Stress Scenario Comparison",
                 fontsize=13, fontweight="bold")

    for ax, metric, ylabel, title in [
        (axes[0], "bad_debt_ratio",   "Bad Debt / Total Loans (%)", "Bad Debt Ratio"),
        (axes[1], "liquidation_rate", "% Wallets Liquidated",       "Liquidation Rate"),
    ]:
        for i, system in enumerate(["Baseline (150%)", "Protocol (Dynamic)"]):
            vals = []
            for s in scenarios:
                row = df[(df["scenario"] == s) & (df["system"] == system)]
                vals.append(row[metric].values[0] if len(row) else 0)
            offset = (i - 0.5) * w
            bars = ax.bar(x + offset, [v * 100 for v in vals], w,
                          color=colors[system], label=system, edgecolor="white")
            for bar, v in zip(bars, vals):
                if v > 0.005:
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.3,
                            f"{v:.1%}", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, fontsize=8, rotation=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{out_dir}/named_scenarios.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_dir}/named_scenarios.png")


def plot_pareto_frontier(sweep, out_dir):
    df     = pd.DataFrame(sweep)
    colors = {"Baseline (150%)": "#E05C5C", "Protocol (Dynamic)": "#4CAF7D"}

    fig, ax = plt.subplots(figsize=(9, 6))
    for system, group in df.groupby("system"):
        ax.plot(group["capital_efficiency"] * 100,
                group["bad_debt_ratio"] * 100,
                "o-", color=colors[system], label=system, lw=2.2, markersize=7)
        for _, row in group[group["price_drop"].isin([0.15, 0.25, 0.35, 0.50])].iterrows():
            ax.annotate(f"{row['price_drop']:.0%}",
                        (row["capital_efficiency"] * 100, row["bad_debt_ratio"] * 100),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=8.5, color=colors[system])

    ax.set_xlabel("Capital Efficiency — Borrowed / Locked (%)", fontsize=11)
    ax.set_ylabel("Bad Debt Ratio (%)", fontsize=11)
    ax.set_title("Pareto Frontier: Capital Efficiency vs Solvency Risk\n"
                 "Labels show crash severity (up-left corner = ideal)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{out_dir}/pareto_frontier.png", dpi=150)
    plt.close()
    print(f"  Saved {out_dir}/pareto_frontier.png")


def plot_collateral_distribution(df, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["ml_collateral_ratio"] * 100, bins=40, color="#4CAF7D",
            alpha=0.8, edgecolor="white", label="Protocol (personalised)")
    ax.axvline(150, color="#E05C5C", lw=2.5, linestyle="--", label="Aave baseline (150%)")
    ax.axvline(df["ml_collateral_ratio"].mean() * 100, color="#2196F3",
               lw=2, linestyle="--",
               label=f"Protocol mean ({df['ml_collateral_ratio'].mean():.1%})")
    ax.set_xlabel("Collateral Ratio (%)", fontsize=11)
    ax.set_ylabel("Number of Wallets", fontsize=11)
    ax.set_title("Distribution of Personalised Collateral Ratios\n"
                 "Protocol (dynamic) vs Aave Fixed Rate", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{out_dir}/collateral_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved {out_dir}/collateral_distribution.png")


def plot_demand(demand, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].bar(["Baseline\nBorrowers", "New Borrowers\n(Protocol)"],
                [demand["baseline"], demand["new_borrowers"]],
                color=["#4F8EF7", "#4CAF7D"], width=0.45, edgecolor="white")
    for bar, val in zip(axes[0].patches,
                        [demand["baseline"], demand["new_borrowers"]]):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 5, f"{val:,}",
                     ha="center", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Wallet Count")
    axes[0].set_title(f"Participation Increase\n"
                      f"(+{demand['gain_pct']:.1%} borrowers from "
                      f"{demand['ratio_reduction']:.1%} ratio reduction)")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].barh(["Aave (Baseline)", "Protocol (Avg)"],
                 [150, demand["avg_protocol"] * 100],
                 color=["#E05C5C", "#4CAF7D"], edgecolor="white")
    axes[1].set_xlabel("Collateral Ratio (%)")
    axes[1].set_title("Average Collateral Requirement")
    axes[1].axvline(100, color="black", lw=1.2, linestyle="--", label="100% (loan value)")
    axes[1].legend()
    axes[1].grid(axis="x", alpha=0.3)
    for i, v in enumerate([150, demand["avg_protocol"] * 100]):
        axes[1].text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(f"{out_dir}/demand_and_efficiency.png", dpi=150)
    plt.close()
    print(f"  Saved {out_dir}/demand_and_efficiency.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("outputs", exist_ok=True)
    df = load_wallets()

    avg_loan         = df["total_borrowed"].mean()
    loan_values      = df["total_borrowed"].values
    baseline_ratios  = np.full(len(df), 1.50)
    protocol_ratios  = apply_anti_farming_cap(df, avg_loan).values

    print(f"{'═'*60}")
    print(f"  DEFI REPUTATION PROTOCOL — SIMULATION RESULTS")
    print(f"{'═'*60}\n")
    print(f"  Wallets simulated:             {len(df):,}")
    print(f"  Avg collateral (Baseline):     150.0%")
    print(f"  Avg collateral (Protocol):     {protocol_ratios.mean():.1%}")
    print(f"  Capital efficiency gain:        {(1.50 - protocol_ratios.mean()) / 1.50:.1%} better\n")

    # Named scenarios
    named_results = []
    for scenario_name, drop in NAMED_SCENARIOS.items():
        print(f"  ── {scenario_name} ──")
        for system, ratios in [("Baseline (150%)", baseline_ratios),
                                ("Protocol (Dynamic)", protocol_ratios)]:
            r = simulate_crash(loan_values, ratios, drop)
            r["scenario"] = scenario_name
            r["system"]   = system
            named_results.append(r)
            print(f"    {system:25s} | "
                  f"Liquidated: {r['liquidation_rate']:.1%} | "
                  f"Bad debt: {r['bad_debt_ratio']:.2%} | "
                  f"Efficiency: {r['capital_efficiency']:.3f}")
        print()

    # Pareto sweep
    sweep_results = []
    for drop in SWEEP_DROPS:
        for system, ratios in [("Baseline (150%)", baseline_ratios),
                                ("Protocol (Dynamic)", protocol_ratios)]:
            r = simulate_crash(loan_values, ratios, drop)
            r["scenario"] = f"{drop:.0%}"
            r["system"]   = system
            sweep_results.append(r)

    # Demand model
    demand = demand_model(df)
    print(f"  {'─'*58}")
    print(f"  DEMAND MODEL")
    print(f"  {'─'*58}")
    print(f"  Collateral reduction:       {demand['ratio_reduction']:.1%}")
    print(f"  New borrowers (projected):  +{demand['new_borrowers']:,} (+{demand['gain_pct']:.1%})")
    print(f"  Additional loan volume:     ${demand['new_volume']:,.0f}")

    # All plots
    plot_named_scenarios(named_results, "outputs")
    plot_pareto_frontier(sweep_results, "outputs")
    plot_collateral_distribution(df,    "outputs")
    plot_demand(demand,                 "outputs")
    print("\n  All plots saved to outputs/")


if __name__ == "__main__":
    main()
