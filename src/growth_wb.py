import requests
import pandas as pd
from pathlib import Path

INDICATOR = "NY.GDP.MKTP.KD.ZG"  # GDP growth (annual %)

def fetch_worldbank_indicator_mrv2(indicator: str) -> pd.DataFrame:
    url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator}"
    params = {"format": "json", "per_page": 20000, "mrv": 2, "page": 1}

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

    # ISO3 only (exclude aggregates like WLD)
    df = df[df["countryiso3code"].astype(str).str.match(r"^[A-Z]{3}$", na=False)]

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
    df = df.dropna(subset=["value"]).sort_values(["countryiso3code", "year"], ascending=[True, False])

    latest = df.groupby("countryiso3code").nth(0).reset_index()
    prev   = df.groupby("countryiso3code").nth(1).reset_index()

    out = latest.merge(prev, on="countryiso3code", how="left", suffixes=("_latest", "_prev"))
    out.rename(columns={
        "value_latest": "growth_latest",
        "year_latest": "growth_year_latest",
        "value_prev": "growth_prev",
        "year_prev": "growth_year_prev",
    }, inplace=True)

    # Growth momentum (positive = accelerating growth)
    out["growth_momentum"] = out["growth_latest"] - out["growth_prev"]

    # Scores (higher growth is better, higher momentum is better)
    lvl = winsorize(out["growth_latest"])
    mom = winsorize(out["growth_momentum"].fillna(0.0))

    lvl_score = lvl.rank(pct=True) * 100.0
    mom_score = mom.rank(pct=True) * 100.0

    out["growth_score"] = 0.7 * lvl_score + 0.3 * mom_score
    out = out.sort_values("growth_score", ascending=False).reset_index(drop=True)

    out_path = Path("data/growth_wb.csv")
    out.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print("Countries with latest growth:", out["growth_latest"].notna().sum())
    print("Countries with prev growth:", out["growth_prev"].notna().sum())
    print(out.head(10).to_string(index=False))
