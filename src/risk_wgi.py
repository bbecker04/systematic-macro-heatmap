import os
import time
from datetime import datetime

import requests
import pandas as pd
import geopandas as gpd


# ---------------------------
# CONFIG
# ---------------------------
WORLD_GPKG = r"data/world_fx_exchange_infl_growth_policy.gpkg"
LAYER = "countries_fx"

OUT_CSV = r"data/risk_wgi.csv"
OUT_GPKG = r"data/world_fx_exchange_infl_growth_policy_risk.gpkg"

# World Bank WGI:
# Political Stability and Absence of Violence/Terrorism - Percentile Rank (0–100, higher is better)
INDICATOR = "PV.PER.RNK"

BASE_URL = "https://api.worldbank.org/v2/country/all/indicator/{}"
PER_PAGE = 1000

MIN_COUNTRIES_WITH_SCORE = 140
MAX_YEARS_BACK = 15

TIMEOUT = (10, 120)   # (connect, read)
RETRIES = 4


# ---------------------------
# HTTP helper
# ---------------------------
def get_json(url, params, retries=RETRIES, timeout=TIMEOUT):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers={"User-Agent": "fx-heatmap/0.1 (requests)"},
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"World Bank request failed after {retries} retries. Last error: {last_err}")


def fetch_indicator_for_year(indicator: str, year: int, per_page: int = PER_PAGE) -> pd.DataFrame:
    """
    Fetch one year of indicator data (smaller + more reliable).
    Returns columns: ISO_A3, risk_year, risk_score
    """
    url = BASE_URL.format(indicator)
    rows = []

    params = {"format": "json", "per_page": per_page, "page": 1, "date": f"{year}:{year}"}
    j = get_json(url, params)

    if not isinstance(j, list) or len(j) < 2 or not isinstance(j[0], dict):
        return pd.DataFrame(columns=["ISO_A3", "risk_year", "risk_score"])

    pages = int(j[0].get("pages", 1))

    def parse_items(items):
        for item in items:
            if not item:
                continue
            iso3 = item.get("countryiso3code")
            yr = item.get("date")
            val = item.get("value")
            if not iso3 or not yr:
                continue
            try:
                y = int(yr)
            except Exception:
                continue
            try:
                v = float(val) if val is not None else None
            except Exception:
                v = None
            rows.append({"ISO_A3": str(iso3).strip().upper(), "risk_year": y, "risk_score": v})

    parse_items(j[1])

    for p in range(2, pages + 1):
        jp = get_json(url, {"format": "json", "per_page": per_page, "page": p, "date": f"{year}:{year}"})
        if isinstance(jp, list) and len(jp) >= 2:
            parse_items(jp[1])

    return pd.DataFrame(rows)


def main():
    os.makedirs("data", exist_ok=True)

    print("Loading world file:", WORLD_GPKG)
    gdf = gpd.read_file(WORLD_GPKG, layer=LAYER)

    if "ISO_A3" not in gdf.columns:
        raise KeyError("Your world layer must contain ISO_A3 to merge WGI risk.")

    # Make ISO_A3 clean
    gdf["ISO_A3"] = gdf["ISO_A3"].astype(str).str.strip().str.upper()

    # IMPORTANT: remove old leftover columns that can cause merge suffix collisions
    cols_to_drop = []
    for c in gdf.columns:
        cl = str(c).lower()
        if cl.startswith("iso3"):
            cols_to_drop.append(c)
    # Also drop any previous risk columns so reruns are clean
    for c in ["risk_score", "risk_year"]:
        if c in gdf.columns:
            cols_to_drop.append(c)

    if cols_to_drop:
        gdf = gdf.drop(columns=sorted(set(cols_to_drop)))
        print("Dropped existing columns to avoid merge conflicts:", sorted(set(cols_to_drop)))

    # Build target set
    target_iso3 = set(gdf["ISO_A3"].dropna().unique().tolist())
    print("World ISO_A3 count:", len(target_iso3))

    this_year = datetime.utcnow().year
    best = {}  # ISO_A3 -> (risk_year, risk_score)

    years_tried = 0
    for y in range(this_year, this_year - MAX_YEARS_BACK - 1, -1):
        years_tried += 1
        print(f"\nDownloading WGI {INDICATOR} for year {y} ...")

        try:
            dfy = fetch_indicator_for_year(INDICATOR, y)
        except Exception as e:
            print(f"  Year {y} failed: {e}")
            continue

        if dfy.empty:
            print(f"  No rows returned for {y}.")
            continue

        dfy = dfy.dropna(subset=["risk_score"])
        print(f"  Non-null scores returned: {len(dfy)}")

        for r in dfy.itertuples(index=False):
            iso3 = r.ISO_A3
            if iso3 in target_iso3 and iso3 not in best:
                best[iso3] = (int(r.risk_year), float(r.risk_score))

        covered = len(best)
        print(f"  Coverage so far: {covered} countries")

        if covered >= MIN_COUNTRIES_WITH_SCORE:
            print("  Coverage target reached.")
            break

    out = pd.DataFrame(
        [{"ISO_A3": k, "risk_year": v[0], "risk_score": v[1]} for k, v in best.items()]
    ).sort_values("ISO_A3")

    out.to_csv(OUT_CSV, index=False)
    print("\nSaved:", OUT_CSV)
    print("Countries with risk_score:", int(out["risk_score"].notna().sum()))
    print("Years tried:", years_tried)

    # Merge on ISO_A3 ONLY (no iso3_x/iso3_y possible)
    gdf2 = gdf.merge(out, on="ISO_A3", how="left", validate="m:1")

    gdf2.to_file(OUT_GPKG, layer=LAYER, driver="GPKG")
    print("Saved:", OUT_GPKG, f"(layer={LAYER})")
    print("Rows with risk_score in merged world:", int(gdf2["risk_score"].notna().sum()))


if __name__ == "__main__":
    main()
