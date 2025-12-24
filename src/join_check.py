import pandas as pd
import geopandas as gpd
import requests
def get_frankfurter_latest(base="EUR"):
    url = f"https://api.frankfurter.app/latest?from={base}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    rates = pd.Series(data["rates"], name="rate")

    # Add the base currency itself so it has a rate too (e.g., EUR=1.0)
    rates.loc[base] = 1.0

    return data["date"], rates

def pick_iso3_column(world: gpd.GeoDataFrame) -> str:
    # Preference order: best join keys first
    for c in ["ISO_A3", "ADM0_A3", "SOV_A3"]:
        if c in world.columns:
            return c
    raise ValueError(f"No ISO-3 column found. Available columns: {list(world.columns)}")

if __name__ == "__main__":
    world = gpd.read_file("data/world.gpkg", layer="countries")
    iso_col = pick_iso3_column(world)

    cc = pd.read_csv("data/country_currency.csv")  # from REST Countries
    cc = cc[["cca3", "country_name", "primary_currency"]].dropna(subset=["cca3"])

    fx_date, rates = get_frankfurter_latest("EUR")
    rates.index.name = "currency"
    rates = rates.reset_index()

    # Merge world -> currency
    merged = world.merge(cc, left_on=iso_col, right_on="cca3", how="left")

    # Merge currency -> FX rate
    merged = merged.merge(rates, left_on="primary_currency", right_on="currency", how="left")

    total = len(merged)
    has_cca3 = merged["cca3"].notna().sum()
    has_curr = merged["primary_currency"].notna().sum()
    has_rate = merged["rate"].notna().sum()

    print("FX date (Frankfurter):", fx_date)
    print("World rows:", total)
    print("Join key used:", iso_col)
    print("Rows with cca3 match:", has_cca3)
    print("Rows with currency:", has_curr)
    print("Rows with FX rate available:", has_rate)

    # Show some missing examples
    missing_rate = merged.loc[merged["primary_currency"].notna() & merged["rate"].isna(),
                              [iso_col, "NAME", "country_name", "primary_currency"]].head(20)
    if len(missing_rate):
        print("\nExamples where currency exists but Frankfurter has no rate (first 20):")
        print(missing_rate.to_string(index=False))

    # Save merged layer for later mapping
    merged.to_file("data/world_scored.gpkg", layer="countries_scored", driver="GPKG")
    print("\nSaved: data/world_scored.gpkg (layer=countries_scored)")
