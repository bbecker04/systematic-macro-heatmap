import requests
import pandas as pd
from pathlib import Path
import time

def get_with_retries(url, params, timeout=180, retries=3, sleep_seconds=3):
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            time.sleep(sleep_seconds)
    raise last_err


INDICATOR = "FR.INR.RINR"  # Real interest rate (%), annual

def fetch_worldbank_indicator_mrv2(indicator: str) -> pd.DataFrame:
    url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator}"
    params = {"format": "json", "per_page": 20000, "mrv": 2, "page": 1}

    r = requests.get(url, params=params, timeout=180)
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
        "value_latest": "real_rate_latest",
        "year_latest": "real_rate_year_latest",
        "value_prev": "real_rate_prev",
        "year_prev": "real_rate_year_prev",
    }, inplace=True)

    # “Policy momentum” = change in real rate (higher = tightening / more supportive carry)
    out["real_rate_momentum"] = out["real_rate_latest"] - out["real_rate_prev"]

    lvl = winsorize(out["real_rate_latest"])
    mom = winsorize(out["real_rate_momentum"].fillna(0.0))

    # Higher real rates better; higher momentum better
    lvl_score = lvl.rank(pct=True) * 100.0
    mom_score = mom.rank(pct=True) * 100.0

    out["policy_score"] = 0.7 * lvl_score + 0.3 * mom_score
    out = out.sort_values("policy_score", ascending=False).reset_index(drop=True)

    out_path = Path("data/policy_wb.csv")
    out.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print("Countries with latest real rate:", out["real_rate_latest"].notna().sum())
    print("Countries with prev real rate:", out["real_rate_prev"].notna().sum())
    print(out.head(10).to_string(index=False))
