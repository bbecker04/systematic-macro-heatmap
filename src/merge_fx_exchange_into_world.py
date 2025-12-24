from pathlib import Path
import pandas as pd
import geopandas as gpd

if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)

    # This layer already has country shapes + primary_currency (from your earlier join pipeline)
    world = gpd.read_file("data/world_scored.gpkg", layer="countries_scored")

    # New broader-coverage FX momentum scores (created by fx_momentum_exchange.py)
    fx = pd.read_csv("data/fx_momentum_exchange.csv")

    # Merge on currency code
    out = world.merge(fx, left_on="primary_currency", right_on="currency", how="left")

    total = len(out)
    has_currency = out["primary_currency"].notna().sum()
    has_fx = out["fx_score_mom"].notna().sum()

    print("Total country rows:", total)
    print("Rows with currency:", has_currency)
    print("Rows with FX momentum score:", has_fx)

    out.to_file("data/world_fx_exchange.gpkg", layer="countries_fx", driver="GPKG")
    print("Saved: data/world_fx_exchange.gpkg (layer=countries_fx)")
