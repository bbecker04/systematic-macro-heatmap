import requests
import pandas as pd
from pathlib import Path

RESTCOUNTRIES_ALL = "https://restcountries.com/v3.1/all"

def fetch_country_currencies():
    # REST Countries requires fields on /all; keep it small (faster + polite)
    params = {"fields": "cca3,name,currencies"}
    r = requests.get(RESTCOUNTRIES_ALL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    rows = []
    for c in data:
        cca3 = c.get("cca3")
        name = (c.get("name") or {}).get("common")
        currencies = c.get("currencies") or {}
        currency_codes = sorted(list(currencies.keys()))  # e.g. ["USD"]
        primary = currency_codes[0] if currency_codes else None

        if cca3:
            rows.append(
                {
                    "cca3": cca3,
                    "country_name": name,
                    "primary_currency": primary,
                    "all_currencies": ",".join(currency_codes),
                }
            )

    df = pd.DataFrame(rows).drop_duplicates(subset=["cca3"]).sort_values("cca3")
    return df

if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)

    df = fetch_country_currencies()
    out = Path("data/country_currency.csv")
    df.to_csv(out, index=False)

    print("Saved:", out)
    print("Rows:", len(df))
    print(df.head(10).to_string(index=False))
