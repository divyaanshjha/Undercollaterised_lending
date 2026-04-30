"""
fetch_data.py
Pulls Aave V2 borrow, repay, and liquidation events from The Graph (hosted service).
Saves raw data to data/ as CSVs so you only need to run this once.
"""

import requests
import pandas as pd
import time
import json
import os

SUBGRAPH_URL = "https://api.thegraph.com/subgraphs/name/aave/protocol-v2"

HEADERS = {"Content-Type": "application/json"}

# ── GraphQL queries ──────────────────────────────────────────────────────────

BORROW_QUERY = """
query getBorrows($skip: Int!, $first: Int!) {
  borrows(
    first: $first
    skip: $skip
    orderBy: timestamp
    orderDirection: asc
  ) {
    id
    user { id }
    amount
    reserve { symbol decimals }
    timestamp
    assetPriceUSD
  }
}
"""

REPAY_QUERY = """
query getRepays($skip: Int!, $first: Int!) {
  repays(
    first: $first
    skip: $skip
    orderBy: timestamp
    orderDirection: asc
  ) {
    id
    user { id }
    amount
    reserve { symbol decimals }
    timestamp
    assetPriceUSD
  }
}
"""

LIQUIDATION_QUERY = """
query getLiquidations($skip: Int!, $first: Int!) {
  liquidationCalls(
    first: $first
    skip: $skip
    orderBy: timestamp
    orderDirection: asc
  ) {
    id
    user { id }
    collateralAmount
    collateralReserve { symbol decimals }
    principalAmount
    principalReserve { symbol decimals }
    timestamp
  }
}
"""

# ── Paginated fetcher ────────────────────────────────────────────────────────

def fetch_paginated(query: str, key: str, max_records: int = 5000) -> list:
    """
    Fetches up to max_records entries from The Graph using skip-based pagination.
    The Graph caps skip at 5000, so we use first=1000 and paginate.
    """
    results = []
    skip = 0
    batch = 1000

    print(f"  Fetching '{key}'...")
    while len(results) < max_records:
        payload = {
            "query": query,
            "variables": {"skip": skip, "first": batch}
        }
        try:
            resp = requests.post(SUBGRAPH_URL, json=payload, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if "errors" in data:
                print(f"  GraphQL error: {data['errors']}")
                break

            batch_data = data.get("data", {}).get(key, [])
            if not batch_data:
                break

            results.extend(batch_data)
            skip += len(batch_data)
            print(f"    {len(results)} records fetched...", end="\r")

            if len(batch_data) < batch:
                break  # last page

            time.sleep(0.3)  # be polite to the free endpoint

        except Exception as e:
            print(f"\n  Error at skip={skip}: {e}")
            break

    print(f"    Done: {len(results)} '{key}' records")
    return results


# ── Normalise raw records into flat DataFrames ───────────────────────────────

def normalise_borrows(raw: list) -> pd.DataFrame:
    rows = []
    for r in raw:
        try:
            decimals = int(r["reserve"]["decimals"])
            amount_usd = (int(r["amount"]) / 10**decimals) * float(r.get("assetPriceUSD") or 0)
            rows.append({
                "wallet":    r["user"]["id"].lower(),
                "asset":     r["reserve"]["symbol"],
                "amount_usd": amount_usd,
                "timestamp": int(r["timestamp"]),
                "event":     "borrow"
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def normalise_repays(raw: list) -> pd.DataFrame:
    rows = []
    for r in raw:
        try:
            decimals = int(r["reserve"]["decimals"])
            amount_usd = (int(r["amount"]) / 10**decimals) * float(r.get("assetPriceUSD") or 0)
            rows.append({
                "wallet":    r["user"]["id"].lower(),
                "asset":     r["reserve"]["symbol"],
                "amount_usd": amount_usd,
                "timestamp": int(r["timestamp"]),
                "event":     "repay"
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def normalise_liquidations(raw: list) -> pd.DataFrame:
    rows = []
    for r in raw:
        try:
            rows.append({
                "wallet":    r["user"]["id"].lower(),
                "timestamp": int(r["timestamp"]),
                "event":     "liquidation"
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("data", exist_ok=True)

    print("=== Fetching Aave V2 data from The Graph ===\n")

    raw_borrows      = fetch_paginated(BORROW_QUERY,      "borrows",          max_records=5000)
    raw_repays       = fetch_paginated(REPAY_QUERY,       "repays",           max_records=5000)
    raw_liquidations = fetch_paginated(LIQUIDATION_QUERY, "liquidationCalls", max_records=2000)

    df_borrows      = normalise_borrows(raw_borrows)
    df_repays       = normalise_repays(raw_repays)
    df_liquidations = normalise_liquidations(raw_liquidations)

    df_borrows.to_csv("data/borrows.csv",           index=False)
    df_repays.to_csv("data/repays.csv",             index=False)
    df_liquidations.to_csv("data/liquidations.csv", index=False)

    print(f"\nSaved:")
    print(f"  data/borrows.csv      → {len(df_borrows):,} rows")
    print(f"  data/repays.csv       → {len(df_repays):,} rows")
    print(f"  data/liquidations.csv → {len(df_liquidations):,} rows")
    print(f"\nUnique wallets in borrows: {df_borrows['wallet'].nunique():,}")


if __name__ == "__main__":
    main()
