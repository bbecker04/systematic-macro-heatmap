from pathlib import Path
import pandas as pd
import geopandas as gpd
import requests

def get_frankfurter_timeseries(base="EUR", start="2025-11-01", end=None):
    url = f"https://api.frankfurter.app/{start}..{'' if end is None else end}"
    params = {"from": base}
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame.from_dict(data["rates"], orient="index").sort_index()
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    # add base currency itself
    df[base] = 1.0
    return data.get("start_date", start), data.get("end_date", None), df

def momentum_score(df_rates: pd.DataFrame):
    first = df_rates.iloc[0]
    last = df_rates.iloc[-1]
    pct = (last / first - 1.0) * 100.0

    out = pct.to_frame("fx_pct_change")
    out.index.name = "currency"
    out = out.reset_index()

    # simple 0-100 score by rank (higher = stronger)
    out["fx_score_mom"] = out["fx_pct_change"].rank(pct=True) * 100.0
    return out

if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)

    # 1) Compute FX momentum table
    start = (pd.Timestamp.today().normalize() - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    _, _, rates = get_frankfurter_timeseries(base="EUR", start=start)

    # keep currencies that exist in latest row
    latest_currencies = rates.iloc[-1].dropna().index.tolist()
    rates = rates[latest_currencies]

    mom = momentum_score(rates)
    mom.to_csv("data/fx_momentum.csv", index=False)

    # 2) Load your merged world layer (has primary_currency already)
    world = gpd.read_file("data/world_scored.gpkg", layer="countries_scored")

    # 3) Merge FX momentum into countries by currency
    out = world.merge(mom, left_on="primary_currency", right_on="currency", how="left")

    # 4) Quick coverage stats
    total = len(out)
    has_fx = out["fx_score_mom"].notna().sum()
    print("Total country rows:", total)
    print("Rows with FX momentum score:", has_fx)

    # 5) Save for the next stage (mapping)
    out.to_file("data/world_fx.gpkg", layer="countries_fx", driver="GPKG")
    print("Saved: data/world_fx.gpkg (layer=countries_fx)")
    print("Also saved: data/fx_momentum.csv")
