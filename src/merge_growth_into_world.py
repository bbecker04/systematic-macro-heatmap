import pandas as pd
import geopandas as gpd

def pick_iso3_column(gdf):
    # Prefer ADM0_A3 because ISO_A3 is sometimes -99 (France/Norway issue)
    for c in ["ADM0_A3", "ISO_A3", "SOV_A3"]:
        if c in gdf.columns:
            return c
    raise ValueError("No ISO3 column found (ADM0_A3 / ISO_A3 / SOV_A3).")

if __name__ == "__main__":
    world = gpd.read_file("data/world_fx_exchange_infl.gpkg", layer="countries_fx")
    iso_col = pick_iso3_column(world)

    growth = pd.read_csv("data/growth_wb.csv")
    growth = growth[["countryiso3code", "growth_latest", "growth_prev", "growth_momentum", "growth_score"]]
    growth = growth.rename(columns={"countryiso3code": "iso3"})

    out = world.merge(growth, left_on=iso_col, right_on="iso3", how="left")

    print("ISO join column:", iso_col)
    print("Rows:", len(out))
    print("Rows with growth_score:", out["growth_score"].notna().sum())

    out.to_file("data/world_fx_exchange_infl_growth.gpkg", layer="countries_fx", driver="GPKG")
    print("Saved: data/world_fx_exchange_infl_growth.gpkg (layer=countries_fx)")
