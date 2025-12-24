import pandas as pd
import geopandas as gpd

def pick_iso3_column(gdf: gpd.GeoDataFrame) -> str:
    for c in ["ADM0_A3", "ISO_A3", "SOV_A3"]:
        if c in gdf.columns:
            return c
    raise ValueError("No ISO3 column found (ISO_A3 / ADM0_A3 / SOV_A3).")

if __name__ == "__main__":
    world = gpd.read_file("data/world_fx_exchange_infl_growth.gpkg", layer="countries_fx")
    iso_col = pick_iso3_column(world)

    pol = pd.read_csv("data/policy_wb.csv")
    pol = pol[["countryiso3code", "real_rate_latest", "real_rate_prev", "real_rate_momentum", "policy_score"]]
    pol = pol.rename(columns={"countryiso3code": "iso3"})

    out = world.merge(pol, left_on=iso_col, right_on="iso3", how="left")

    print("ISO join column:", iso_col)
    print("Rows:", len(out))
    print("Rows with policy_score:", out["policy_score"].notna().sum())

    out.to_file("data/world_fx_exchange_infl_growth_policy.gpkg", layer="countries_fx", driver="GPKG")
    print("Saved: data/world_fx_exchange_infl_growth_policy.gpkg (layer=countries_fx)")
