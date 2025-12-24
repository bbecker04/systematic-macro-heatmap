import re
import requests

def fetch_exchange_api_latest(base="EUR"):
    base_l = base.lower()

    # Primary + fallback (recommended by the project)
    urls = [
        f"https://latest.currency-api.pages.dev/v1/currencies/{base_l}.json",
        f"https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/{base_l}.json",
    ]

    last_err = None
    for url in urls:
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            data = r.json()
            date = data.get("date")
            rates_raw = data.get(base_l, {})
            return date, rates_raw, url
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Both URLs failed. Last error: {last_err}")

if __name__ == "__main__":
    date, rates_raw, used_url = fetch_exchange_api_latest("EUR")

    # Keep only normal 3-letter currency codes (ignore crypto/metals like btc/xau/etc if you want later)
    # For now: strict ISO-ish filter = exactly 3 letters
    rates_iso = {k.upper(): v for k, v in rates_raw.items() if re.fullmatch(r"[a-z]{3}", k)}

    print("Source URL:", used_url)
    print("Data date :", date)
    print("ISO-ish 3-letter codes:", len(rates_iso))
    print("Has USD?", "USD" in rates_iso)
    print("Sample (first 15):", list(sorted(rates_iso.keys()))[:15])
