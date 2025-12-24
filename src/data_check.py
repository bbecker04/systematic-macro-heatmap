import streamlit as st

st.set_page_config(page_title="FX Heatmap", layout="wide")

st.title("FX Heatmap – Setup Check ✅")
st.write("If you can see this page, Streamlit is working.")



import requests
import pandas as pd

def get_frankfurter_latest(base="USD"):
    url = f"https://api.frankfurter.app/latest?from={base}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    rates = pd.Series(data["rates"], name="rate").sort_index()
    return data["date"], rates

if __name__ == "__main__":
    date, rates = get_frankfurter_latest("USD")
    print("Frankfurter date:", date)
    print("Number of currencies:", len(rates))
    print(rates.head(10))
