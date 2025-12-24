import pandas as pd
import geopandas as gpd

def pick_iso3_column(gdf: gpd.GeoDataFrame) -> str:
    # Prefer ADM0_A3 because ISO_A3 is sometimes -99 (France/Norway issue)
    for c in ["ADM0_A3", "ISO_A3", "SOV_A3"]:
        if c in gdf.columns:
            return c
    raise ValueError("No ISO3 column found (ADM0_A3 / ISO_A3 / SOV_A3).")

def pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

if __name__ == "__main__":
    # Start from the policy-enriched world file you just produced
    world = gpd.read_file("data/world_fx_exchange_infl_growth_policy.gpkg", layer="countries_fx")
    iso_col = pick_iso3_column(world)

    # Risk data (WGI) pulled by risk_wgi.py
    risk = pd.read_csv("data/risk_wgi.csv")

    risk_iso = pick_col(risk, ["ISO_A3", "iso3", "ADM0_A3", "countryiso3code", "Country Code", "CODE"])
    risk_score = pick_col(risk, ["risk_score", "risk", "wgi_score"])

    if risk_iso is None or risk_score is None:
        raise ValueError(f"Could not find needed columns in risk_wgi.csv. Columns are: {list(risk.columns)}")

    risk = risk[[risk_iso, risk_score]].rename(columns={risk_iso: "iso3", risk_score: "risk_score"})


# Clean up leftover ISO columns from prior merges (prevents iso3_x / iso3_y collisions)
for col in ["iso3", "iso3_x", "iso3_y"]:
    if col in world.columns:
        world = world.drop(columns=[col])


    out = world.merge(risk, left_on=iso_col, right_on="iso3", how="left")


# Drop the right-side join key to keep the output clean
if "iso3" in out.columns:
    out = out.drop(columns=["iso3"])


    print("ISO join column:", iso_col)
    print("Rows:", len(out))
    print("Rows with risk_score:", out["risk_score"].notna().sum())

    out.to_file(
        "data/world_fx_exchange_infl_growth_policy_risk.gpkg",
        layer="countries_fx",
        driver="GPKG"
    )
    print("Saved: data/world_fx_exchange_infl_growth_policy_risk.gpkg (layer=countries_fx)")

