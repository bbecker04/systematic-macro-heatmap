import requests
import pandas as pd

def get_frankfurter_timeseries(base="EUR", start="2025-11-01", end=None):
    # If end is None, Frankfurter uses latest available
    url = f"https://api.frankfurter.app/{start}..{'' if end is None else end}"
    params = {"from": base}
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    # data["rates"] is a dict: { "YYYY-MM-DD": {"USD": 1.08, ...}, ... }
    df = pd.DataFrame.from_dict(data["rates"], orient="index").sort_index()
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    # add base currency itself
    df[base] = 1.0
    return data.get("start_date", start), data.get("end_date", None), df

def momentum_score(df_rates: pd.DataFrame, lookback_days=30):
    """
    df_rates columns are currencies; values are 1 unit of base buys X of currency.
    We compute pct change from first to last available row in the window.
    """
    if len(df_rates) < 2:
        raise ValueError("Not enough history to compute momentum.")

    first = df_rates.iloc[0]
    last = df_rates.iloc[-1]
    pct = (last / first - 1.0) * 100.0
    out = pct.to_frame("pct_change")
    out.index.name = "currency"
    out = out.reset_index()

    # simple 0-100 score by ranking (stronger = higher score)
    out["score_fx_mom"] = out["pct_change"].rank(pct=True) * 100.0
    return out.sort_values("score_fx_mom", ascending=False)

if __name__ == "__main__":
    # Choose a recent window; Frankfurter is daily so this is stable.
    # We'll pick ~45 days back to ensure we get at least ~30 trading days of data.
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=45)
    start_str = start.strftime("%Y-%m-%d")

    _, _, rates = get_frankfurter_timeseries(base="EUR", start=start_str)

    # Keep only currencies that exist in the last row (available latest)
    latest = rates.iloc[-1].dropna().index.tolist()
    rates = rates[latest]

    mom = momentum_score(rates, lookback_days=30)

    print("Currencies with data:", len(latest))
    print(mom.head(15).to_string(index=False))
