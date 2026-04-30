"""
Protocol Simulation: Before vs After
======================================
Simulates two scenarios across three market conditions:

  BASELINE   — Aave-style fixed 150% collateral for all borrowers
  PROTOCOL   — Reputation-gated collateral (105%–150%) + anti-farming cap

Market conditions:
  - Normal      : ETH price stable
  - March 2020  : ETH drops 50% in 48 hours (COVID crash)
  - LUNA 2022   : ETH drops 60% over 72 hours (LUNA/UST collapse)

Key outputs:
  1. Bad debt incurred as % of total borrowing (solvency)
  2. Capital efficiency (total borrowed / total collateral locked)
  3. Participation rate (how many wallets borrow at all)
  4. Pareto frontier: solvency vs efficiency
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import json, warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─── Load scored wallets ──────────────────────────────────────────────────────
df = pd.read_csv("wallet_scores.csv")

# Re-load raw loan data for simulation
raw_df = pd.read_csv("loans.csv")


# ─── Market crash parameters ──────────────────────────────────────────────────
SCENARIOS = {
    "Normal Market": {
        "price_drop": 0.00,          # no crash
        "description": "No crash"
    },
    "March 2020 (COVID)": {
        "price_drop": 0.50,          # ETH -50% in 48 hrs
        "description": "ETH -50%"
    },
    "LUNA Crash (May 2022)": {
        "price_drop": 0.60,          # ETH -60% over 72 hrs
        "description": "ETH -60%"
    },
}

# Anti-farming cap multiplier
K_CAP = 2.5
BASELINE_COLLATERAL = 1.50

# Demand curve: participation elasticity
# Empirically, DeFi lending studies show ~15-25% more borrowers per 10pp collateral reduction
DEMAND_ELASTICITY = 0.20  # 20% more participation per 10pp collateral reduction


def anti_farming_effective_ratio(base_ratio: float, requested: float,
                                 max_historical: float, k: float = K_CAP) -> float:
    """
    Blended effective collateral ratio after applying the anti-farming cap.
    - Discounted rate applies up to k × max_historical
    - Anything above that is charged at baseline 150%
    """
    cap = k * max_historical
    if requested <= cap:
        return base_ratio
    discounted = cap * base_ratio
    baseline   = (requested - cap) * BASELINE_COLLATERAL
    return (discounted + baseline) / requested


def simulate_scenario(scenario_name: str, price_drop: float,
                      loan_request_multiple: float = 1.0) -> dict:
    """
    Simulate protocol under a given price crash.
    Each wallet requests a loan equal to loan_request_multiple × their historical max loan.

    Returns metrics dict for baseline and protocol.
    """
    results = {"baseline": {}, "protocol": {}}

    baseline_borrows    = []
    baseline_collateral_locked = []
    baseline_bad_debt   = []
    baseline_participated = 0

    protocol_borrows    = []
    protocol_collateral_locked = []
    protocol_bad_debt   = []
    protocol_participated = 0

    for _, wallet in df.iterrows():
        requested = wallet["max_loan_usd"] * loan_request_multiple
        max_hist  = wallet["max_loan_usd"]
        base_col  = wallet["base_collateral"]   # from reputation model (1.05–1.50)
        default_p = wallet["default_prob"]

        # ── Baseline (Aave) ──────────────────────────────────────────────────
        # All wallets borrow at 150%. Participation: everyone qualifies.
        b_collateral = requested * BASELINE_COLLATERAL
        baseline_collateral_locked.append(b_collateral)
        baseline_borrows.append(requested)
        baseline_participated += 1

        # Liquidation: if price drops enough to breach collateral buffer
        # Health factor = (collateral × current_price) / loan
        # Initial health = BASELINE_COLLATERAL (at time of borrow)
        # After crash: health = BASELINE_COLLATERAL × (1 - price_drop)
        b_health_after = BASELINE_COLLATERAL * (1 - price_drop)
        if b_health_after < 1.0:
            # Liquidated: bad debt = loan - recovered (collateral post-crash)
            recovered = b_collateral * (1 - price_drop)
            bad = max(0, requested - recovered)
            baseline_bad_debt.append(bad)
        elif np.random.random() < default_p * 0.3:
            # Random default (non-crash related, based on reputation)
            baseline_bad_debt.append(requested * 0.15)
        else:
            baseline_bad_debt.append(0)

        # ── Protocol (reputation-gated) ──────────────────────────────────────
        # Effective collateral ratio after anti-farming cap
        eff_ratio = anti_farming_effective_ratio(base_col, requested, max_hist)

        # Demand effect: lower collateral attracts more borrowers
        # Model: participation prob = 1 for all (everyone already in dataset)
        # But additionally, wallets who wouldn't participate at 150% now join
        # (modeled as extra synthetic wallets — see demand analysis below)
        p_collateral = requested * eff_ratio
        protocol_collateral_locked.append(p_collateral)
        protocol_borrows.append(requested)
        protocol_participated += 1

        # Liquidation: same logic but with lower collateral buffer
        p_health_after = eff_ratio * (1 - price_drop)
        if p_health_after < 1.0:
            recovered = p_collateral * (1 - price_drop)
            bad = max(0, requested - recovered)
            protocol_bad_debt.append(bad)
        elif np.random.random() < default_p * 0.3:
            protocol_bad_debt.append(requested * 0.15)
        else:
            protocol_bad_debt.append(0)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def agg(borrows, collateral, bad_debt):
        total_borrowed  = sum(borrows)
        total_locked    = sum(collateral)
        total_bad       = sum(bad_debt)
        return {
            "total_borrowed_M":    round(total_borrowed / 1e6, 2),
            "total_collateral_M":  round(total_locked / 1e6, 2),
            "total_bad_debt_M":    round(total_bad / 1e6, 2),
            "bad_debt_pct":        round(total_bad / total_borrowed * 100, 3) if total_borrowed > 0 else 0,
            "capital_efficiency":  round(total_borrowed / total_locked, 4) if total_locked > 0 else 0,
            "avg_collateral_ratio":round(total_locked / total_borrowed * 100, 2) if total_borrowed > 0 else 0,
        }

    results["baseline"] = agg(baseline_borrows, baseline_collateral_locked, baseline_bad_debt)
    results["protocol"] = agg(protocol_borrows, protocol_collateral_locked, protocol_bad_debt)
    results["baseline"]["participants"] = baseline_participated
    results["protocol"]["participants"] = protocol_participated

    return results


def demand_curve_analysis():
    """
    Model the participation uplift from reduced collateral requirements.

    Approach: wallets whose max_loan demanded 140-150% utilization in Aave
    represent marginal borrowers who would be deterred at 150% but participate at 120%.
    We estimate additional participation using elasticity model.
    """
    avg_protocol_ratio = df["base_collateral"].mean()
    avg_baseline_ratio = BASELINE_COLLATERAL

    # Percentage points reduction
    reduction_pp = (avg_baseline_ratio - avg_protocol_ratio) * 100

    # New participants = existing × elasticity × (reduction / 10)
    uplift_pct = DEMAND_ELASTICITY * (reduction_pp / 10)
    new_participants = int(len(df) * uplift_pct)

    return {
        "avg_protocol_collateral_pct": round(avg_protocol_ratio * 100, 1),
        "avg_baseline_collateral_pct": round(avg_baseline_ratio * 100, 1),
        "collateral_reduction_pp":     round(reduction_pp, 1),
        "participation_uplift_pct":    round(uplift_pct * 100, 1),
        "new_participants_estimated":  new_participants,
    }


# ─── Run all scenarios ────────────────────────────────────────────────────────
print("=" * 65)
print("      BEFORE vs AFTER SIMULATION RESULTS")
print("=" * 65)

all_results = {}
for scenario_name, params in SCENARIOS.items():
    r = simulate_scenario(scenario_name, params["price_drop"])
    all_results[scenario_name] = r

    b = r["baseline"]
    p = r["protocol"]
    print(f"\n{'─'*65}")
    print(f"  Scenario: {scenario_name} ({params['description']})")
    print(f"{'─'*65}")
    print(f"  {'Metric':<30} {'Baseline (Aave)':>15} {'Protocol':>12}")
    print(f"  {'─'*55}")
    print(f"  {'Total Borrowed ($M)':<30} {b['total_borrowed_M']:>15.2f} {p['total_borrowed_M']:>12.2f}")
    print(f"  {'Collateral Locked ($M)':<30} {b['total_collateral_M']:>15.2f} {p['total_collateral_M']:>12.2f}")
    print(f"  {'Avg Collateral Ratio':<30} {b['avg_collateral_ratio']:>14.1f}% {p['avg_collateral_ratio']:>11.1f}%")
    print(f"  {'Bad Debt ($M)':<30} {b['total_bad_debt_M']:>15.3f} {p['total_bad_debt_M']:>12.3f}")
    print(f"  {'Bad Debt %':<30} {b['bad_debt_pct']:>14.3f}% {p['bad_debt_pct']:>11.3f}%")
    print(f"  {'Capital Efficiency':<30} {b['capital_efficiency']:>15.4f} {p['capital_efficiency']:>12.4f}")

demand = demand_curve_analysis()
print(f"\n{'─'*65}")
print(f"  DEMAND CURVE ANALYSIS")
print(f"{'─'*65}")
print(f"  Avg collateral (baseline):       {demand['avg_baseline_collateral_pct']}%")
print(f"  Avg collateral (protocol):       {demand['avg_protocol_collateral_pct']}%")
print(f"  Reduction:                       {demand['collateral_reduction_pp']} pp")
print(f"  Estimated new participants:      +{demand['new_participants_estimated']} wallets")
print(f"  Participation uplift:            +{demand['participation_uplift_pct']}%")

# ─── Anti-farming demonstration ───────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"  ANTI-FARMING CAP DEMONSTRATION")
print(f"{'─'*65}")
print(f"  Scenario: Wallet with max historical loan = $1,000 (farmer)")
max_hist = 1000
base_col = 1.08  # excellent score
for req in [1_000, 2_500, 5_000, 20_000, 100_000]:
    eff = anti_farming_effective_ratio(base_col, req, max_hist)
    discount = (BASELINE_COLLATERAL - eff) / BASELINE_COLLATERAL * 100
    print(f"  Requested ${req:>8,.0f}  →  Effective ratio: {eff*100:.1f}%  "
          f"(discount from baseline: {discount:.1f}%)")

# ─── Plots ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
gs  = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
fig.suptitle("DeFi Reputation Protocol: Before vs After Simulation",
             fontsize=15, fontweight="bold", y=0.98)

COLOR_BASE = "#6B7280"   # gray
COLOR_PROT = "#2563EB"   # blue

scenarios = list(SCENARIOS.keys())
labels    = ["Normal", "March 2020\n(−50%)", "LUNA 2022\n(−60%)"]
x         = np.arange(len(scenarios))
width     = 0.35

# ── 1. Bad Debt % ─────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
b_bad = [all_results[s]["baseline"]["bad_debt_pct"] for s in scenarios]
p_bad = [all_results[s]["protocol"]["bad_debt_pct"] for s in scenarios]
ax1.bar(x - width/2, b_bad, width, label="Baseline (Aave)", color=COLOR_BASE, alpha=0.85)
ax1.bar(x + width/2, p_bad, width, label="Protocol", color=COLOR_PROT, alpha=0.85)
ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
ax1.set_ylabel("Bad Debt (% of total borrowed)")
ax1.set_title("Bad Debt Rate\n(lower = more stable)", fontweight="bold")
ax1.legend(fontsize=8); ax1.set_facecolor("#F8FAFC")

# ── 2. Capital Efficiency ─────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
b_eff = [all_results[s]["baseline"]["capital_efficiency"] for s in scenarios]
p_eff = [all_results[s]["protocol"]["capital_efficiency"] for s in scenarios]
ax2.bar(x - width/2, b_eff, width, label="Baseline", color=COLOR_BASE, alpha=0.85)
ax2.bar(x + width/2, p_eff, width, label="Protocol", color=COLOR_PROT, alpha=0.85)
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
ax2.set_ylabel("Borrowed / Collateral Locked")
ax2.set_title("Capital Efficiency\n(higher = more efficient)", fontweight="bold")
ax2.legend(fontsize=8); ax2.set_facecolor("#F8FAFC")

# ── 3. Avg Collateral Ratio ───────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
b_col = [all_results[s]["baseline"]["avg_collateral_ratio"] for s in scenarios]
p_col = [all_results[s]["protocol"]["avg_collateral_ratio"] for s in scenarios]
ax3.bar(x - width/2, b_col, width, label="Baseline", color=COLOR_BASE, alpha=0.85)
ax3.bar(x + width/2, p_col, width, label="Protocol", color=COLOR_PROT, alpha=0.85)
ax3.axhline(100, color="red", ls="--", lw=1, label="Undercollateralised threshold")
ax3.set_xticks(x); ax3.set_xticklabels(labels, fontsize=8)
ax3.set_ylabel("Average Collateral Ratio (%)")
ax3.set_title("Average Collateral Ratio\n(reduction = capital freed)", fontweight="bold")
ax3.legend(fontsize=8); ax3.set_facecolor("#F8FAFC")

# ── 4. Anti-farming cap illustration ─────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
loan_amounts = np.logspace(2, 6, 200)   # $100 to $1M
for max_h, label, color in [(500, "Max hist $500 (micro-farmer)", "#EF4444"),
                              (5000,  "Max hist $5k  (small user)",  "#F59E0B"),
                              (50000, "Max hist $50k (real user)",   "#10B981")]:
    ratios = [anti_farming_effective_ratio(1.08, req, max_h) * 100
              for req in loan_amounts]
    ax4.plot(loan_amounts / 1000, ratios, label=label, lw=2, color=color)
ax4.axhline(150, color="gray", ls="--", lw=1.5, label="Baseline (150%)")
ax4.set_xscale("log")
ax4.set_xlabel("Requested Loan ($k)")
ax4.set_ylabel("Effective Collateral Ratio (%)")
ax4.set_title("Anti-Farming Cap in Action\n(score = Excellent, cap = 2.5×)", fontweight="bold")
ax4.legend(fontsize=7); ax4.set_facecolor("#F8FAFC")

# ── 5. Demand curve: participation vs collateral ──────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
col_levels = np.linspace(100, 155, 100)
# Demand: exponential growth as collateral falls (based on elasticity model)
participation_index = 100 * np.exp(DEMAND_ELASTICITY * (150 - col_levels) / 10)
ax5.plot(col_levels, participation_index, color="#7C3AED", lw=2.5)
ax5.axvline(150, color=COLOR_BASE, ls="--", lw=1.5, label=f"Baseline (150%)")
avg_prot = demand["avg_protocol_collateral_pct"]
ax5.axvline(avg_prot, color=COLOR_PROT, ls="--", lw=1.5, label=f"Protocol avg ({avg_prot}%)")
ax5.fill_betweenx(participation_index,
                   [avg_prot]*len(col_levels), [150]*len(col_levels),
                   where=[c >= avg_prot and c <= 150 for c in col_levels],
                   alpha=0.15, color=COLOR_PROT, label="Participation gain")
ax5.set_xlabel("Required Collateral Ratio (%)")
ax5.set_ylabel("Relative Participation (index = 100 at 150%)")
ax5.set_title("Demand Curve: Lower Collateral → More Borrowers", fontweight="bold")
ax5.legend(fontsize=8); ax5.set_facecolor("#F8FAFC")

# ── 6. Pareto frontier: solvency vs efficiency ────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
# Sweep collateral multiplier and compute solvency vs efficiency
multipliers = np.linspace(1.0, 1.5, 30)
march_drop  = 0.50
efficiencies, bad_debts_pct = [], []
for m in multipliers:
    tot_b, tot_c, tot_bd = 0, 0, 0
    for _, w in df.iterrows():
        req = w["max_loan_usd"]
        col = req * m
        health_after = m * (1 - march_drop)
        tot_b += req
        tot_c += col
        if health_after < 1.0:
            recovered = col * (1 - march_drop)
            tot_bd += max(0, req - recovered)
    efficiencies.append(tot_b / tot_c)
    bad_debts_pct.append(tot_bd / tot_b * 100)

sc = ax6.scatter(efficiencies, bad_debts_pct, c=multipliers * 100,
                 cmap="RdYlGn_r", s=60, zorder=3)
# Mark baseline and protocol
baseline_eff = all_results["March 2020 (COVID)"]["baseline"]["capital_efficiency"]
baseline_bad = all_results["March 2020 (COVID)"]["baseline"]["bad_debt_pct"]
protocol_eff = all_results["March 2020 (COVID)"]["protocol"]["capital_efficiency"]
protocol_bad = all_results["March 2020 (COVID)"]["protocol"]["bad_debt_pct"]

ax6.scatter([baseline_eff], [baseline_bad], marker="*", s=300, color=COLOR_BASE,
            zorder=5, label="Baseline")
ax6.scatter([protocol_eff], [protocol_bad], marker="*", s=300, color=COLOR_PROT,
            zorder=5, label="Protocol")
plt.colorbar(sc, ax=ax6, label="Collateral Ratio (%)")
ax6.set_xlabel("Capital Efficiency (Borrowed / Locked)")
ax6.set_ylabel("Bad Debt % (March 2020 crash)")
ax6.set_title("Pareto Frontier\nSolvency vs Capital Efficiency", fontweight="bold")
ax6.legend(fontsize=8); ax6.set_facecolor("#F8FAFC")

plt.savefig("../plots/simulation_results.png", dpi=150, bbox_inches="tight")
print("\nSaved → plots/simulation_results.png")
