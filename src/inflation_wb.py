import re
import requests
import pandas as pd
from pathlib import Path

INDICATOR = "FP.CPI.TOTL.ZG"  # Inflation, consumer prices (annual %)

def fetch_worldbank_indicator_mrv2(indicator: str) -> pd.DataFrame:
    """
    Pull most recent 2 observations per country from World Bank API (V2).
    Docs: MRV fetches most recent values; per_page controls pagination.
    """
    url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator}"
    params = {
        "format": "json",
        "per_page": 20000,
        "mrv": 2,      # most recent values
        "page": 1,
    }

    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    payload = r.json()

    meta = payload[0]
    pages = int(meta.get("pages", 1))

    rows = payload[1] if len(payload) > 1 else []

    for p in range(2, pages + 1):
        params["page"] = p
        rp = requests.get(url, params=params, timeout=60)
        rp.raise_for_status()
        part = rp.json()
        if len(part) > 1 and part[1]:
            rows.extend(part[1])

    df = pd.json_normalize(rows)

    # Keep ISO3 country codes only (exclude aggregates like "WLD")
    df["countryiso3code"] = df.get("countryiso3code")
    df = df[df["countryiso3code"].astype(str).str.match(r"^[A-Z]{3}$", na=False)]

    # Year
    df["year"] = pd.to_numeric(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df[["countryiso3code", "year", "value"]].dropna(subset=["countryiso3code", "year"])

def winsorize(s: pd.Series, p_low=0.05, p_high=0.95) -> pd.Series:
    lo = s.quantile(p_low)
    hi = s.quantile(p_high)
    return s.clip(lower=lo, upper=hi)

if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)

    df = fetch_worldbank_indicator_mrv2(INDICATOR)

    # Get latest + previous non-null per country
    df = df.dropna(subset=["value"]).sort_values(["countryiso3code", "year"], ascending=[True, False])

    latest = df.groupby("countryiso3code").nth(0).reset_index()
    prev   = df.groupby("countryiso3code").nth(1).reset_index()

    out = latest.merge(prev, on="countryiso3code", how="left", suffixes=("_latest", "_prev"))

    out.rename(columns={
        "value_latest": "infl_latest",
        "year_latest": "infl_year_latest",
        "value_prev": "infl_prev",
        "year_prev": "infl_year_prev",
    }, inplace=True)

    # Inflation momentum (positive = accelerating inflation, negative = disinflation)
    out["infl_momentum"] = out["infl_latest"] - out["infl_prev"]

    # Build 0–100 scores:
    # - Lower inflation level is better (invert rank)
    # - More negative momentum (disinflation) is better (invert rank)
    lvl = winsorize(out["infl_latest"])
    mom = winsorize(out["infl_momentum"].fillna(0.0))

    lvl_score = (1.0 - lvl.rank(pct=True)) * 100.0
    mom_score = (1.0 - mom.rank(pct=True)) * 100.0

    # Blend (you can change weights later)
    out["infl_score"] = 0.7 * lvl_score + 0.3 * mom_score

    out = out.sort_values("infl_score", ascending=False).reset_index(drop=True)

    out_path = Path("data/inflation_wb.csv")
    out.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print("Countries with latest inflation:", out["infl_latest"].notna().sum())
    print("Countries with prev inflation:", out["infl_prev"].notna().sum())
    print(out.head(10).to_string(index=False))
