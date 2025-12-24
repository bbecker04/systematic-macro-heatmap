import time
import requests
import pandas as pd
import numpy as np
import geopandas as gpd

# ---------------------------
# Settings you can tweak
# ---------------------------
WORLD_GPKG = r"data/world_fx_exchange_infl_growth_policy.gpkg"
LAYER = "countries_fx"
OUT_CSV = r"data/risk_gdelt.csv"
OUT_GPKG = r"data/world_fx_exchange_infl_growth_policy_risk.gpkg"

TIMESPAN = "30d"  # try: "7d", "14d", "30d"

# Keep query reasonably sized
RISK_QUERY_OR = "(protest OR riot OR coup OR unrest OR terror OR bombing OR attack OR war OR invasion OR sanctions)"

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

HEADERS = {
    # Helps avoid some non-JSON / blocked responses
    "User-Agent": "fx-heatmap/0.1 (Windows; requests)"
}


# ---------------------------
# Helpers
# ---------------------------
def safe_get_json(params, timeout=60, retries=3, sleep_s=1.5):
    """HTTP GET with retries; returns (json_dict_or_None, status_code_or_None, text_preview)."""
    last_status = None
    last_preview = ""
    last_err = None

    for attempt in range(1, retries + 1):
        try:
            r = requests.get(GDELT_DOC_URL, params=params, headers=HEADERS, timeout=timeout)
            last_status = r.status_code
            txt = (r.text or "").strip()
            last_preview = txt[:200].replace("\n", " ")

            # If not 200, retry with backoff
            if r.status_code != 200:
                time.sleep(sleep_s * attempt)
                continue

            # Sometimes GDELT returns HTML/text even with 200
            if not txt.startswith("{"):
                # Try once more after a short wait
                time.sleep(sleep_s * attempt)
                continue

            return (r.json(), r.status_code, last_preview)

        except Exception as e:
            last_err = e
            time.sleep(sleep_s * attempt)

    # Final failure
    msg = f"Request failed after {retries} retries."
    if last_status is not None:
        msg += f" Last HTTP status: {last_status}."
    if last_preview:
        msg += f" Preview: {last_preview}"
    if last_err is not None:
        msg += f" Last error: {last_err}"
    print(msg)

    return (None, last_status, last_preview)


def timeline_counts(query, timespan=TIMESPAN):
    """
    Calls GDELT DOC API in timelinevolraw mode.
    Returns (match_total, all_total) based on summing returned timeline buckets.
    """
    params = {
        "query": query,
        "mode": "timelinevolraw",
        "format": "json",
        "timespan": timespan,
    }

    j, status, preview = safe_get_json(params)
    if not j or "timeline" not in j:
        return (pd.NA, pd.NA)

    match_sum = 0.0
    all_sum = 0.0

    for row in j.get("timeline", []):
        v = row.get("value", 0)
        a = row.get("all", row.get("total", 0))

        try:
            match_sum += float(v) if v is not None else 0.0
        except Exception:
            pass

        try:
            all_sum += float(a) if a is not None else 0.0
        except Exception:
            pass

    if all_sum <= 0:
        return (pd.NA, pd.NA)

    return (match_sum, all_sum)


def score_from_ratio(series_ratio):
    """
    Convert risk_ratio (higher = worse) -> risk_score (higher = better), 0–100.
    Uses 5th/95th percentiles to reduce outlier impact.
    """
    s = pd.to_numeric(series_ratio, errors="coerce")
    if s.notna().sum() < 5:
        return pd.Series([pd.NA] * len(series_ratio), index=series_ratio.index)

    lo = np.nanpercentile(s, 5)
    hi = np.nanpercentile(s, 95)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(s))
        hi = float(np.nanmax(s))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return pd.Series([pd.NA] * len(series_ratio), index=series_ratio.index)

    clipped = s.clip(lower=lo, upper=hi)
    norm = (clipped - lo) / (hi - lo)   # 0=low risk, 1=high risk
    score = 100.0 * (1.0 - norm)        # invert so higher score = safer
    return score


# ---------------------------
# Main
# ---------------------------
def main():
    gdf = gpd.read_file(WORLD_GPKG, layer=LAYER)
    print(f"Using world file: {WORLD_GPKG}")

    # Prefer ISO_A2 if present
    if "ISO_A2" in gdf.columns:
        code_col = "ISO_A2"
    elif "iso_a2" in gdf.columns:
        code_col = "iso_a2"
    else:
        raise KeyError("Could not find ISO_A2 / iso_a2 in world data. We need 2-letter country codes.")

    print(f"Using 2-letter code column: {code_col}")

    # Pre-flight test so we don't loop 172 countries if GDELT is down
    print("Pre-flight test (GDELT): sourcecountry:US ...")
    test_match, test_all = timeline_counts("sourcecountry:US", timespan="7d")
    if pd.isna(test_match) or pd.isna(test_all):
        print("Pre-flight failed. GDELT is likely throttling/returning non-JSON right now.")
        print("Try again in a few minutes, or switch TIMESPAN to '7d' to reduce load.")
        return

    countries = (
        gdf[[code_col, "ISO_A3"] + (["display_name"] if "display_name" in gdf.columns else [])]
        .drop_duplicates()
        .rename(columns={code_col: "cc2"})
    )

    # Clean codes like "-99" or missing
    countries["cc2"] = countries["cc2"].astype(str).str.strip().str.upper()
    countries["cc2"] = countries["cc2"].replace({"-99": pd.NA, "": pd.NA, "NONE": pd.NA, "NAN": pd.NA})
    countries = countries.dropna(subset=["cc2"])

    rows = []
    n = len(countries)

    for i, row in enumerate(countries.itertuples(index=False), start=1):
        cc2 = getattr(row, "cc2")
        iso_a3 = getattr(row, "ISO_A3", pd.NA)
        name = getattr(row, "display_name", cc2)

        print(f"[{i}/{n}] {name} ({iso_a3}) cc2={cc2}")

        # Numerator: "risk" articles where the SOURCE country is cc2
        q_risk = f"{RISK_QUERY_OR} sourcecountry:{cc2}"
        risk_count, _ = timeline_counts(q_risk, timespan=TIMESPAN)

        # Denominator: ALL articles where the SOURCE country is cc2
        q_all = f"sourcecountry:{cc2}"
        all_count, _ = timeline_counts(q_all, timespan=TIMESPAN)

        if pd.isna(risk_count) or pd.isna(all_count) or float(all_count) <= 0:
            risk_ratio = pd.NA
        else:
            risk_ratio = float(risk_count) / float(all_count)

        rows.append(
            {
                "cc2": cc2,
                "ISO_A3": iso_a3,
                "display_name": name,
                "risk_count": risk_count,
                "all_count": all_count,
                "risk_ratio": risk_ratio,
            }
        )

        time.sleep(0.35)  # gentler rate limit

    df = pd.DataFrame(rows)
    df["risk_score"] = score_from_ratio(df["risk_ratio"])

    df.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")

    # Merge into world
    gdf2 = gdf.merge(df[["cc2", "risk_score", "risk_ratio"]], left_on=code_col, right_on="cc2", how="left")
    gdf2.drop(columns=["cc2"], inplace=True, errors="ignore")

    gdf2.to_file(OUT_GPKG, layer=LAYER, driver="GPKG")
    print(f"Saved: {OUT_GPKG} (layer={LAYER})")


if __name__ == "__main__":
    main()
