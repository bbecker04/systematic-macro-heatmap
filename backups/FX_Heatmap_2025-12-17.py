# app.py
# FX Heatmap (Streamlit)
# ------------------------------------------------------------
# Organized into clear sections:
#   1) Imports + config
#   2) Helpers (FX backtest, scoring, map)
#   3) Sidebar = heatmap settings only
#   4) Main page = map + country details + FX momentum + backtest
# ------------------------------------------------------------

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st

import folium
from folium import Element
import branca.colormap as cm
from streamlit_folium import st_folium
from shapely.geometry import Point


# ============================================================
# Page config + CSS
# ============================================================

st.set_page_config(page_title="FX Heatmap", layout="wide")

st.markdown(
    """
<style>
/* Pull main content higher */
section.main > div { padding-top: 0.5rem; }

/* Optional: hide Streamlit top toolbar spacing a bit */
header[data-testid="stHeader"] { height: 0rem; }

/* Make the page use more width (Streamlit's main container) */
div.block-container {
    max-width: 100% !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

/* Make embedded iframes (folium map) stretch full width */
iframe { width: 100% !important; }

/* Force Streamlit iframe blocks to use full width */
div[data-testid="stIFrame"] { width: 100% !important; }
div[data-testid="stIFrame"] iframe {
    width: 100% !important;
    min-width: 100% !important;
}

/* Folium attribution sizing (if it shows up) */
.leaflet-control-attribution {
    font-size: 9px !important;
    opacity: 0.6 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.caption("Gray countries = N/A (missing data for the selected metric / settings).")


# ============================================================
# Constants + paths
# ============================================================

DATA_PATH = Path("data/world_fx_exchange_infl_growth_policy_risk.gpkg")
LAYER = "countries_fx"

MAP_HEIGHT = 520

EXPECTED_SCORE_COLS = [
    "fx_score_mom",
    "fx_pct_change",
    "infl_score",
    "growth_score",
    "policy_score",
    "risk_score",
]

DATA_MTIME = DATA_PATH.stat().st_mtime if DATA_PATH.exists() else None


def format_mtime(ts: float | None) -> str:
    if ts is None:
        return "N/A (data file not found)"
    dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone()
    return dt.strftime("%Y-%m-%d %H:%M %Z")


# ============================================================
# Session state defaults
# (important because the map build uses these values)
# ============================================================

if "use_live_fx" not in st.session_state:
    st.session_state["use_live_fx"] = True

if "fx_base" not in st.session_state:
    st.session_state["fx_base"] = "USD"

if "fx_lookback" not in st.session_state:
    st.session_state["fx_lookback"] = 30


# ============================================================
# Pair backtest helpers (Yahoo Finance via yfinance)
# ============================================================

@st.cache_data(ttl=60 * 60, show_spinner=False)
def _fetch_fx_series_yahoo(ccy: str, start: str, end: str) -> pd.Series:
    """
    Returns a price series representing USD value of 1 unit of the currency (CCYUSD).
    Uses Yahoo Finance via yfinance.

    - If Yahoo has CCYUSD=X, use it directly.
    - Else try USDCCY=X and invert it (1 / price).
    """
    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError("Missing dependency: yfinance. Install it with: pip install yfinance") from e

    ccy = ccy.upper().strip()

    # Try direct CCYUSD first (e.g., EURUSD=X)
    t1 = f"{ccy}USD=X"
    df1 = yf.download(t1, start=start, end=end, progress=False)
    if df1 is not None and not df1.empty and "Adj Close" in df1:
        s = df1["Adj Close"].dropna()
        s.name = ccy
        return s

    # Fallback: USDCCY (e.g., USDJPY=X) then invert
    t2 = f"USD{ccy}=X"
    df2 = yf.download(t2, start=start, end=end, progress=False)
    if df2 is not None and not df2.empty and "Adj Close" in df2:
        s = df2["Adj Close"].dropna()
        s = 1.0 / s
        s.name = ccy
        return s

    raise RuntimeError(f"Could not download FX series for {ccy} from Yahoo Finance.")


def run_pair_backtest(long_ccy: str, short_ccy: str, start: str, end: str) -> pd.DataFrame:
    """Equal-weight daily backtest: port_ret = ret(long_ccyUSD) - ret(short_ccyUSD)."""
    long_px = _fetch_fx_series_yahoo(long_ccy, start, end)
    short_px = _fetch_fx_series_yahoo(short_ccy, start, end)

    df = pd.concat([long_px, short_px], axis=1).dropna()
    df.columns = ["long_px", "short_px"]

    df["long_ret"] = df["long_px"].pct_change()
    df["short_ret"] = df["short_px"].pct_change()
    df["port_ret"] = df["long_ret"] - df["short_ret"]
    df = df.dropna()

    df["equity"] = (1.0 + df["port_ret"]).cumprod()
    return df


def summarize_backtest(bt: pd.DataFrame) -> dict:
    r = bt["port_ret"]
    if r.empty:
        return {}

    ann_factor = 252.0
    ann_ret = (1.0 + r.mean()) ** ann_factor - 1.0
    ann_vol = r.std() * np.sqrt(ann_factor)
    sharpe = (r.mean() / r.std()) * np.sqrt(ann_factor) if r.std() != 0 else np.nan

    equity = bt["equity"]
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    max_dd = dd.min()

    return {
        "Annualized return": ann_ret,
        "Annualized vol": ann_vol,
        "Sharpe (naive)": sharpe,
        "Max drawdown": max_dd,
        "Start equity": float(equity.iloc[0]),
        "End equity": float(equity.iloc[-1]),
        "Days": int(len(bt)),
    }


# ============================================================
# Data loading + scoring
# ============================================================

@st.cache_data(show_spinner=False)
def load_world_fx(_data_mtime: float | None) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(DATA_PATH, layer=LAYER)

    # Ensure CRS is WGS84 for mapping/clicking
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    else:
        gdf = gdf.to_crs(4326)

    # Name column -> display_name
    if "display_name" in gdf.columns:
        gdf["display_name"] = gdf["display_name"].astype(str)
    elif "NAME" in gdf.columns:
        gdf["display_name"] = gdf["NAME"].astype(str)
    elif "ADMIN" in gdf.columns:
        gdf["display_name"] = gdf["ADMIN"].astype(str)
    else:
        gdf["display_name"] = "Unknown"

    # Ensure expected numeric score columns exist (missing -> NaN)
    for col in EXPECTED_SCORE_COLS:
        if col not in gdf.columns:
            gdf[col] = pd.NA
        gdf[col] = pd.to_numeric(gdf[col], errors="coerce")

    _ = gdf.sindex  # build spatial index
    return gdf


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fx_momentum_live(base: str, lookback_days: int):
    """
    Uses src/fx_momentum_exchange.py (free daily FX snapshot API).
    Cached for 1 hour.
    """
    from src.fx_momentum_exchange import compute_momentum_scores

    df, meta = compute_momentum_scores(base=base, lookback_days=lookback_days)
    keep = [c for c in ["currency", "fx_pct_change", "fx_score_mom"] if c in df.columns]
    return df[keep].copy(), meta


def overlay_live_fx(gdf: gpd.GeoDataFrame, use_live: bool, base: str, lookback_days: int):
    if not use_live:
        return gdf, None
    try:
        fx, meta = fetch_fx_momentum_live(base=base, lookback_days=lookback_days)
        gdf2 = gdf.drop(columns=["currency", "fx_pct_change", "fx_score_mom"], errors="ignore")
        gdf2 = gdf2.merge(fx, left_on="primary_currency", right_on="currency", how="left")
        gdf2.drop(columns=["currency"], inplace=True, errors="ignore")
        _ = gdf2.sindex  # rebuild after merge
        return gdf2, meta
    except Exception as e:
        return gdf, {"error": str(e)}


def compute_aggregate_scores(gdf: gpd.GeoDataFrame, weights: dict, fill_missing_with_neutral: bool) -> gpd.GeoDataFrame:
    out = gdf.copy()

    factors = {
        "fx": ("fx_score_mom", float(weights["fx"])),
        "infl": ("infl_score", float(weights["infl"])),
        "growth": ("growth_score", float(weights["growth"])),
        "policy": ("policy_score", float(weights["policy"])),
        "risk": ("risk_score", float(weights["risk"])),
    }

    score_df = pd.DataFrame({k: pd.to_numeric(out[col], errors="coerce") for k, (col, _) in factors.items()})
    w = pd.Series({k: wgt for k, (_, wgt) in factors.items()})

    if w.sum() == 0:
        out["aggregate_score"] = pd.NA
        return out

    if fill_missing_with_neutral:
        score_df_filled = score_df.fillna(50.0)
        out["aggregate_score"] = score_df_filled.mul(w, axis=1).sum(axis=1) / w.sum()
        return out

    mask = ~score_df.isna()
    weighted_sum = score_df.mul(w, axis=1).where(mask).sum(axis=1)
    weight_sum = mask.mul(w, axis=1).sum(axis=1)

    out["aggregate_score"] = weighted_sum / weight_sum
    out.loc[weight_sum == 0, "aggregate_score"] = pd.NA
    return out


# ============================================================
# Formatting + AI-style note
# ============================================================

def fmt_score(x):
    try:
        return "N/A" if pd.isna(x) else f"{float(x):.1f}"
    except Exception:
        return "N/A"


def driver_summary(row: pd.Series, weights: dict):
    comps = {
        "FX": ("fx_score_mom", weights["fx"]),
        "Inflation": ("infl_score", weights["infl"]),
        "Growth": ("growth_score", weights["growth"]),
        "Policy": ("policy_score", weights["policy"]),
        "Risk": ("risk_score", weights["risk"]),
    }

    items = []
    for label, (col, wgt) in comps.items():
        s = row.get(col, pd.NA)
        if pd.isna(s) or wgt == 0:
            continue
        s = float(s)
        items.append((label, float(wgt) * (s - 50.0), s))

    if not items:
        return [], []

    items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
    positives = [(lbl, contrib, s) for (lbl, contrib, s) in items_sorted if contrib > 0][:2]
    negatives = [(lbl, contrib, s) for (lbl, contrib, s) in reversed(items_sorted) if contrib < 0][:2]
    return positives, negatives


def format_note(row: pd.Series, weights: dict, fill_missing: bool) -> str:
    name = row.get("display_name", "This country")
    curr = row.get("primary_currency", None)

    agg = row.get("aggregate_score", pd.NA)
    fx = row.get("fx_score_mom", pd.NA)

    if pd.isna(curr):
        return (
            f"**{name}**: Currency mapping is missing for this shape, so we can’t score it yet.\n\n"
            "Next step is improving coverage / joins."
        )

    if pd.isna(agg):
        return (
            f"**{name} ({curr})**: This country has no available factor scores under the current settings.\n\n"
            "Tip: turn on **“Fill missing with neutral 50”** if you want every country colored while we build out macro/risk."
        )

    agg = float(agg)
    stance = "constructive" if agg >= 70 else ("mixed" if agg >= 40 else "cautious")

    pos, neg = driver_summary(row, weights)

    lines = [f"**{name} ({curr})** looks **{stance}** on the current aggregate model."]

    if pos:
        lines.append("\n**Tailwinds:** " + ", ".join([f"{lbl} ({s:.0f})" for lbl, _, s in pos]))
    if neg:
        lines.append("\n**Headwinds:** " + ", ".join([f"{lbl} ({s:.0f})" for lbl, _, s in neg]))

    fx_txt = "N/A" if pd.isna(fx) else f"{float(fx):.1f}"
    lines.append(f"\n\n- Aggregate score (0–100): **{agg:.1f}**")
    lines.append(f"- FX score (0–100): **{fx_txt}**")
    lines.append("\nThis text will get more meaningful as we populate inflation, growth, policy, and risk scores with real data.")

    if fill_missing:
        lines.append("\n\n*(Missing factors are currently filled with a neutral 50 to color every country.)*")

    return "\n".join(lines)


def _first_existing_value(row, candidates):
    if row is None:
        return pd.NA

    for c in candidates:
        if c in row.index:
            v = row.get(c, pd.NA)
            if not pd.isna(v):
                return v
    return pd.NA


# ============================================================
# Map rendering (keeps your “one world” + legend styling)
# ============================================================

def build_map(gdf: gpd.GeoDataFrame, value_col: str, title: str, show_outlines: bool = True):
    bounds = [[-60.0, -180.0], [85.0, 180.0]]

    m = folium.Map(
        location=[20, 0],
        zoom_start=3,
        min_zoom=3,
        max_zoom=8,
        tiles=None,
        width="100%",
        height=MAP_HEIGHT,
    )

    # Hide attribution
    m.get_root().header.add_child(Element("<style>.leaflet-control-attribution { display: none !important; }</style>"))
    # Dark background where there are no tiles
    m.get_root().header.add_child(Element("<style>.leaflet-container { background: #111 !important; }</style>"))

    # Basemap tiles
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="© OpenStreetMap contributors © CARTO",
        control=False,
        no_wrap=True,
    ).add_to(m)

    m.fit_bounds(bounds)
    m.options["maxBounds"] = bounds
    m.options["maxBoundsViscosity"] = 1.0

    colormap = cm.linear.RdYlGn_11.scale(0, 100)
    colormap.caption = title
    colormap.add_to(m)

    # Legend placement styling
    legend_id = colormap.get_name()
    m.get_root().header.add_child(
        folium.Element(
            f"""
<style>
#{legend_id} {{
    position: fixed !important;
    top: auto !important;
    bottom: 30px !important;
    left: auto !important;
    right: 30px !important;
    z-index: 9999 !important;
    background: rgba(0,0,0,0.55) !important;
    padding: 10px 12px !important;
    border-radius: 10px !important;
}}
#{legend_id} .caption {{
    color: white !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}}
</style>
"""
        )
    )

    tooltip = folium.features.GeoJsonTooltip(
        fields=[
            "display_name",
            "primary_currency",
            "aggregate_score_disp",
            "fx_score_mom_disp",
            "infl_score_disp",
            "growth_score_disp",
            "policy_score_disp",
            "risk_score_disp",
        ],
        aliases=[
            "Country",
            "Currency",
            "Aggregate",
            "FX score",
            "Inflation score",
            "Growth score",
            "Policy score",
            "Risk score",
        ],
        localize=True,
        sticky=False,
        labels=True,
    )

    def style_function(feature):
        raw = feature["properties"].get(value_col, None)
        try:
            val = float(raw)
        except Exception:
            val = None

        is_missing = val is None
        fill = "#666666" if is_missing else colormap(val)

        if show_outlines:
            border_color = "#111111"
            border_weight = 0.6 if not is_missing else 0.2
        else:
            border_color = fill
            border_weight = 0

        return {"fillColor": fill, "color": border_color, "weight": border_weight, "fillOpacity": 0.80}

    folium.GeoJson(
        data=gdf.to_json(),
        name="Scores",
        style_function=style_function,
        tooltip=tooltip,
        highlight_function=lambda x: {"weight": 2, "fillOpacity": 0.90},
    ).add_to(m)

    return m


# ============================================================
# Sidebar: HEATMAP SETTINGS ONLY
# ============================================================

st.sidebar.markdown("### Data Status")
st.sidebar.caption(f"Last updated: **{format_mtime(DATA_MTIME)}**")

if st.sidebar.button("Reset settings"):
    for k in ["use_live_fx", "fx_base", "fx_lookback", "fill_missing", "preset_name",
              "w_fx", "w_infl", "w_growth", "w_policy", "w_risk", "country_search"]:
        st.session_state.pop(k, None)
    st.rerun()


if st.sidebar.button("Refresh Data Now"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("Heatmap Settings")

show_outlines = st.sidebar.checkbox("Show Borders", value=True)

st.sidebar.subheader("Weights (Normalized Automatically)")
presets = {
    "Balanced (default)": {"fx": 35, "infl": 15, "growth": 15, "policy": 20, "risk": 15},
    "Momentum tilt": {"fx": 60, "infl": 10, "growth": 10, "policy": 10, "risk": 10},
    "Equal weights": {"fx": 20, "infl": 20, "growth": 20, "policy": 20, "risk": 20},
    "Risk-off": {"fx": 25, "infl": 15, "growth": 15, "policy": 15, "risk": 30},
}

preset_name = st.sidebar.selectbox("Weight preset", list(presets.keys()), index=0)

# Initialize / update slider values when preset changes
if "preset_name" not in st.session_state:
    st.session_state["preset_name"] = preset_name
    for k, v in presets[preset_name].items():
        st.session_state[f"w_{k}"] = v
elif preset_name != st.session_state["preset_name"]:
    st.session_state["preset_name"] = preset_name
    for k, v in presets[preset_name].items():
        st.session_state[f"w_{k}"] = v

w_fx = st.sidebar.slider("FX weight", 0, 100, step=5, key="w_fx")
w_infl = st.sidebar.slider("Inflation weight", 0, 100, step=5, key="w_infl")
w_growth = st.sidebar.slider("Growth weight", 0, 100, step=5, key="w_growth")
w_policy = st.sidebar.slider("Policy weight", 0, 100, step=5, key="w_policy")
w_risk = st.sidebar.slider("Risk weight", 0, 100, step=5, key="w_risk")

weights = {"fx": w_fx, "infl": w_infl, "growth": w_growth, "policy": w_policy, "risk": w_risk}

fill_missing = st.sidebar.checkbox(
    "Fill missing factors with neutral 50 (colors every country)",
    value=False,
    key="fill_missing",
)

metric_label_to_col = {
    "Aggregate Score": "aggregate_score",
    "FX Momentum Score": "fx_score_mom",
    "Inflation Score": "infl_score",
    "Growth Score": "growth_score",
    "Policy Score": "policy_score",
    "Risk Score": "risk_score",
}

metric_label = st.sidebar.selectbox("Color map by", list(metric_label_to_col.keys()), index=0)
metric = metric_label_to_col[metric_label]

metric_title_map = {
    "Aggregate Score": "Aggregate Score (0–100)",
    "FX Momentum Score": "FX Momentum Score (0–100)",
    "Inflation Score": "Inflation Score (0–100)",
    "Growth Score": "Growth Score (0–100)",
    "Policy Score": "Policy Score (0–100)",
    "Risk Score": "Risk Score (0–100)",
}
metric_title = metric_title_map.get(metric_label, metric_label)


# ============================================================
# Build dataset used by map + details
# ============================================================

try:
    base_gdf = load_world_fx(DATA_MTIME)
except Exception as e:
    st.error("Could not load the map dataset. Make sure DATA_PATH exists and has the expected layer.")
    st.code(str(e))
    st.stop()

use_live_fx = bool(st.session_state["use_live_fx"])
fx_base = str(st.session_state["fx_base"])
fx_lookback = int(st.session_state["fx_lookback"])

# Live FX overlay BEFORE aggregate scoring
gdf, fx_meta = overlay_live_fx(base_gdf, use_live=use_live_fx, base=fx_base, lookback_days=fx_lookback)

# Aggregate scoring (now actually uses the checkbox)
gdf = compute_aggregate_scores(gdf, weights=weights, fill_missing_with_neutral=fill_missing)

# Tooltip display columns
for col in ["aggregate_score"] + EXPECTED_SCORE_COLS:
    if col not in gdf.columns:
        gdf[col] = pd.NA
    gdf[col + "_disp"] = gdf[col].apply(fmt_score)

st.sidebar.caption(f"Values available: {int(gdf[metric].notna().sum())} / {len(gdf)}")


# ============================================================
# Map
# ============================================================

m = build_map(gdf, value_col=metric, title=metric_title, show_outlines=show_outlines)
map_state = st_folium(m, height=MAP_HEIGHT, use_container_width=True)


# ============================================================
# Country details (MAIN PAGE, not sidebar)
# ============================================================

st.markdown("---")
st.subheader("Country details")

q = st.text_input("Search country (type to filter)", value="", key="country_search")
names_df = pd.DataFrame({"display_name": sorted(gdf["display_name"].dropna().unique().tolist())})
if q.strip():
    names_df = names_df[names_df["display_name"].str.contains(q.strip(), case=False, na=False)]
names = names_df["display_name"].tolist()

selected_name = st.selectbox("Jump to a country (optional)", ["(none)"] + names, index=0)

selected_row = None

# 1) Dropdown selection
if selected_name != "(none)":
    hit = gdf.loc[gdf["display_name"] == selected_name]
    if not hit.empty:
        selected_row = hit.iloc[0]

# 2) Map click selection
elif map_state and map_state.get("last_clicked"):
    lat = map_state["last_clicked"]["lat"]
    lng = map_state["last_clicked"]["lng"]
    pt = Point(lng, lat)

    candidates_idx = list(gdf.sindex.query(pt, predicate="intersects"))
    if candidates_idx:
        candidates = gdf.iloc[candidates_idx]
        hit = candidates[candidates.geometry.contains(pt)]
        if hit.empty:
            hit = candidates[candidates.geometry.intersects(pt)]
        if not hit.empty:
            selected_row = hit.iloc[0]

if selected_row is None:
    st.info("Select a country from the dropdown or click a country on the map.")
else:
    cA, cB = st.columns([1, 2])

    with cA:
        st.write(f"### {selected_row.get('display_name', 'Unknown')}")
        st.write(f"Currency: `{selected_row.get('primary_currency', 'N/A')}`")

        score_specs = [
            ("Aggregate score", ["aggregate_score"]),
            ("FX score", ["fx_score_mom"]),
            ("Inflation score", ["infl_score"]),
            ("Growth score", ["growth_score"]),
            ("Policy score", ["policy_score"]),
            ("Risk score", ["risk_score"]),
        ]

        for label, candidates in score_specs:
            val = _first_existing_value(selected_row, candidates)
            st.write(f"{label}: **{('N/A' if pd.isna(val) else f'{float(val):.1f}') }**")

    with cB:
        st.markdown("### AI-style note")
        st.markdown(format_note(selected_row, weights=weights, fill_missing=fill_missing))


# ============================================================
# FX Momentum section (controls live here)
# ============================================================

st.divider()
st.subheader("FX Momentum")

with st.expander("Live FX Momentum settings", expanded=False):
    st.checkbox("Use live FX momentum (cached 1h)", key="use_live_fx")
    st.selectbox("FX base currency", ["USD", "EUR"], key="fx_base", index=0)
    st.slider("FX lookback (days)", min_value=5, max_value=252, value=30, key="fx_lookback")

    if fx_meta and isinstance(fx_meta, dict) and fx_meta.get("error"):
        st.warning(f"Live FX error: {fx_meta['error']}")

    if st.button("Refresh FX data now"):
        st.cache_data.clear()
        st.rerun()


# ============================================================
# Pair Backtest (Long/Short)
# ============================================================

st.subheader("Pair Backtest (Long/Short)")

available_ccys = sorted(pd.Series(gdf["primary_currency"]).dropna().unique().tolist())
available_ccys = [c for c in available_ccys if isinstance(c, str) and len(c) == 3]

c1, c2 = st.columns(2)
with c1:
    long_ccy = st.selectbox("Long", ["(select)"] + available_ccys, index=0)
with c2:
    short_ccy = st.selectbox("Short", ["(select)"] + available_ccys, index=0)

c3, c4 = st.columns(2)
with c3:
    start_date = st.date_input("Start date", value=pd.to_datetime("2015-01-01").date())
with c4:
    end_date = st.date_input("End date", value=pd.Timestamp.today().date())

if long_ccy != "(select)" and short_ccy != "(select)":
    if long_ccy == short_ccy:
        st.warning("Long and Short can’t be the same currency.")
    else:
        with st.spinner(f"Running backtest: Long {long_ccy} / Short {short_ccy} ..."):
            try:
                bt = run_pair_backtest(long_ccy=long_ccy, short_ccy=short_ccy, start=str(start_date), end=str(end_date))
                stats = summarize_backtest(bt)

                st.line_chart(bt["equity"])

                if stats:
                    stats_df = pd.DataFrame({"Metric": list(stats.keys()), "Value": list(stats.values())})
                    st.dataframe(stats_df, use_container_width=True)

                with st.expander("Show daily returns table"):
                    st.dataframe(bt[["long_ret", "short_ret", "port_ret", "equity"]].tail(50), use_container_width=True)

            except Exception as e:
                st.error(str(e))
                st.info("If the error mentions yfinance, install it with: pip install yfinance")
