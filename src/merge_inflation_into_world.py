import pandas as pd
import geopandas as gpd

def pick_iso3_column(gdf: gpd.GeoDataFrame) -> str:
    for c in ["ADM0_A3", "ISO_A3", "SOV_A3"]:
        if c in gdf.columns:
            return c
    raise ValueError(f"No ISO3 column found. Columns: {list(gdf.columns)}")

if __name__ == "__main__":
    world = gpd.read_file("data/world_fx_exchange.gpkg", layer="countries_fx")
    iso_col = pick_iso3_column(world)

    infl = pd.read_csv("data/inflation_wb.csv")
    infl = infl[["countryiso3code", "infl_latest", "infl_prev", "infl_momentum", "infl_score"]]
    infl = infl.rename(columns={"countryiso3code": "iso3"})

    out = world.merge(infl, left_on=iso_col, right_on="iso3", how="left")

    print("ISO join column:", iso_col)
    print("Rows:", len(out))
    print("Rows with infl_score:", out["infl_score"].notna().sum())

    out.to_file("data/world_fx_exchange_infl.gpkg", layer="countries_fx", driver="GPKG")
    print("Saved: data/world_fx_exchange_infl.gpkg (layer=countries_fx)")
