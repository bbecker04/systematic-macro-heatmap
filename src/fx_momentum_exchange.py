import re
import requests
import pandas as pd
from datetime import date, timedelta

def fetch_exchange_api_json(base="EUR", day: date | None = None):
    base_l = base.lower()

    if day is None:
        urls = [
            f"https://latest.currency-api.pages.dev/v1/currencies/{base_l}.json",
            f"https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/{base_l}.json",
        ]
    else:
        d = day.isoformat()
        urls = [
            f"https://latest.currency-api.pages.dev/v1/{d}/currencies/{base_l}.json",
            f"https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@{d}/v1/currencies/{base_l}.json",
        ]

    last_err = None
    for url in urls:
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return r.json(), url
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Both URLs failed. Last error: {last_err}")

def compute_momentum_scores(base="EUR", lookback_days=30):
    # 1) Only keep currencies that actually show up in our country mapping
    cc = pd.read_csv("data/country_currency.csv")
    needed = sorted(set(cc["primary_currency"].dropna().astype(str).str.upper()))

    # 2) Pull latest + one past date (simple and fast)
    today = date.today()
    past = today - timedelta(days=lookback_days)

    latest_json, latest_url = fetch_exchange_api_json(base=base, day=None)
    past_json, past_url = fetch_exchange_api_json(base=base, day=past)

    base_l = base.lower()
    latest_rates_raw = latest_json.get(base_l, {})
    past_rates_raw = past_json.get(base_l, {})

    # 3) Normalize keys and keep only needed currency codes (this also removes crypto tickers)
    latest = {k.upper(): v for k, v in latest_rates_raw.items() if re.fullmatch(r"[a-z]{3}", k)}
    pastd  = {k.upper(): v for k, v in past_rates_raw.items() if re.fullmatch(r"[a-z]{3}", k)}

    # Add base itself
    latest[base] = 1.0
    pastd[base] = 1.0

    rows = []
    for cur in needed:
        if cur in latest and cur in pastd:
            try:
                pct = (float(latest[cur]) / float(pastd[cur]) - 1.0) * 100.0
                rows.append({"currency": cur, "fx_pct_change": pct})
            except Exception:
                pass

    df = pd.DataFrame(rows)
    df["fx_score_mom"] = df["fx_pct_change"].rank(pct=True) * 100.0
    df = df.sort_values("fx_score_mom", ascending=False).reset_index(drop=True)

    meta = {
        "latest_date": latest_json.get("date"),
        "latest_url": latest_url,
        "past_date": past_json.get("date"),
        "past_url": past_url,
        "lookback_days": lookback_days,
        "base": base,
        "currencies_scored": len(df),
    }
    return df, meta

if __name__ == "__main__":
    df, meta = compute_momentum_scores(base="EUR", lookback_days=30)

    print("Base:", meta["base"])
    print("Lookback days:", meta["lookback_days"])
    print("Latest date:", meta["latest_date"])
    print("Past date:", meta["past_date"])
    print("Currencies scored:", meta["currencies_scored"])
    print(df.head(15).to_string(index=False))

    df.to_csv("data/fx_momentum_exchange.csv", index=False)
    print("Saved: data/fx_momentum_exchange.csv")
