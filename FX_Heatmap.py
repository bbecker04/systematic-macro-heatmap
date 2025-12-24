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

/* Force Streamlit iframe blocks + their containers to use full width/height */
div[data-testid="stIFrame"],
div[data-testid="stIFrame"] > div {
    width: 100% !important;
}

div[data-testid="stIFrame"] iframe {
    width: 100% !important;
    min-width: 100% !important;
    height: 80vh !important;
    min-height: 700px !important;
    max-height: 90vh !important;
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


#================================================================
#Build AI Note
#================================================================

def build_ai_note(country: str, ccy: str, scores: dict, agg: float) -> str:
    """
    Systematic, template-based note (no external AI).
    scores expects keys:
      fx_score_mom, infl_score, growth_score, policy_score, risk_score
    All are 0-100 where higher is better.
    """
    fx = float(scores.get("fx_score_mom", float("nan")))
    infl = float(scores.get("infl_score", float("nan")))
    growth = float(scores.get("growth_score", float("nan")))
    policy = float(scores.get("policy_score", float("nan")))
    risk = float(scores.get("risk_score", float("nan")))

    factors = {
        "FX momentum": fx,
        "Growth": growth,
        "Policy / real rates & liquidity": policy,
        "Inflation momentum": infl,
        "Risk / governance & geopolitics": risk,
    }

    # Sort factors into tailwinds/headwinds (ignore NaNs)
    clean = [(k, v) for k, v in factors.items() if pd.notna(v)]
    ranked = sorted(clean, key=lambda x: x[1], reverse=True)
    top = ranked[:2]
    bottom = sorted(clean, key=lambda x: x[1])[:2]

    def stance_from_agg(a: float) -> tuple[str, str]:
        if a >= 75:
            return "attractive", "The mix of factors is broadly supportive and the model is signaling a high-conviction setup."
        if a >= 65:
            return "constructive", "The setup is broadly positive, but there are still meaningful risks that can overwhelm the signal."
        if a >= 55:
            return "slightly positive", "There are more positives than negatives, though the edge is not strong."
        if a >= 45:
            return "mixed", "Signals are conflicting; outcomes will likely depend on which macro factor dominates next."
        if a >= 35:
            return "cautious", "Headwinds dominate; the bar for a positive surprise is higher."
        return "unfavorable", "The model is signaling weak fundamentals/momentum and elevated downside risks."

    stance, stance_line = stance_from_agg(float(agg))

    top_txt = ", ".join([f"**{k}** ({v:.0f})" for k, v in top]) if top else "N/A"
    bot_txt = ", ".join([f"**{k}** ({v:.0f})" for k, v in bottom]) if bottom else "N/A"

    # Translate low scores into plain-English “why”
    def interpret(name: str, val: float) -> str:
        if pd.isna(val):
            return f"{name.lower()} data is missing."
        if val >= 70:
            return f"{name.lower()} is a clear tailwind."
        if val >= 55:
            return f"{name.lower()} is mildly supportive."
        if val >= 45:
            return f"{name.lower()} is neutral."
        if val >= 30:
            return f"{name.lower()} is a headwind."
        return f"{name.lower()} is a strong headwind."

    why_lines = [
        interpret("FX momentum", fx),
        interpret("Growth", growth),
        interpret("Policy / real rates & liquidity", policy),
        interpret("Inflation momentum", infl),
        interpret("Risk / governance & geopolitics", risk),
    ]

    note = f"""
**{country} ({ccy}) screens _{stance}_ on the current model (aggregate score **{agg:.1f}/100**).**  
{stance_line}

**What’s working:** {top_txt}.  
These are the main supportive drivers right now, implying better relative carry/real-rate dynamics, healthier growth impulse, or a positive price trend depending on which factor is leading.

**What’s not:** {bot_txt}.  
These are the main constraints — they often show up as inflation pressure, policy credibility concerns, or elevated political/geopolitical risk that can cap FX upside even when momentum looks good.

**How to use this signal (practically):** This is most actionable when the top drivers are stable for several months and the weakest drivers are improving.  
Watch-list: next inflation prints, central bank stance/real yields, and any risk headlines that could invalidate the setup.

*Note: This is systematic, informational analysis — not financial advice.*
"""
    return note

# ============================================================
# Constants + paths
# ============================================================

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = Path("data/world_fx_exchange_infl_growth_policy_risk.geojson")
LAYER = None  # GeoJSON has no layers

MAP_HEIGHT = 780

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
import yfinance as yf

@st.cache_data(ttl=60 * 60, show_spinner=False)
def _fetch_fx_series_yahoo(ccy: str, start: str, end: str) -> pd.Series:
    """
    Fetch CCY vs USD daily series.
    Tries Yahoo Finance via yfinance first, then falls back to Stooq CSV if Yahoo returns empty.
    Returns a Series indexed by date.
    """
    ccy = (ccy or "").upper().strip()

    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    # Guard: if end < start, swap (prevents silent empties)
    if end_dt < start_dt:
        start_dt, end_dt = end_dt, start_dt

    # USD has no meaningful USDUSD; treat as flat
    if ccy == "USD":
        idx = pd.date_range(start=start_dt, end=end_dt, freq="B")
        return pd.Series(1.0, index=idx, name="USD")

    # ---------- 1) Yahoo via yfinance ----------
    def _yf_get(ticker: str) -> pd.Series:
        try:
            df = yf.download(ticker, start=start_dt, end=end_dt, progress=False, auto_adjust=False, threads=False)
            if df is None or df.empty:
                # Sometimes download() returns empty; try history() as a second method
                hist = yf.Ticker(ticker).history(start=start_dt, end=end_dt, auto_adjust=False)
                df = hist

            if df is None or df.empty:
                return pd.Series(dtype=float)

            col = "Adj Close" if "Adj Close" in df.columns else "Close"

            s = df[col]
            # If df[col] comes back as a 1-column DataFrame, convert to Series
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]

            s = s.dropna()
            if s.empty:
                return pd.Series(dtype=float)


            # Ensure datetime index
            s.index = pd.to_datetime(s.index)
            s = s.sort_index()
            return s
        except Exception:
            return pd.Series(dtype=float)

    # Prefer CCYUSD=X, fall back to USDCCY=X inverted
    s1 = _yf_get(f"{ccy}USD=X")
    if not s1.empty:
        s1.name = ccy
        return s1

    s2 = _yf_get(f"USD{ccy}=X")
    if not s2.empty:
        s2 = 1.0 / s2
        s2.name = ccy
        return s2

    # ---------- 2) Fallback: Stooq CSV ----------
    # Stooq symbols are like: eurusd, gbpusd, usdjpy, etc.

    def _stooq_get(symbol: str) -> pd.Series:
        try:
            url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
            df = pd.read_csv(url)
            if df is None or df.empty or "Date" not in df.columns:
                return pd.Series(dtype=float)

            # Stooq columns: Date, Open, High, Low, Close
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")

            if "Close" not in df.columns:
                return pd.Series(dtype=float)

            s = df.set_index("Date")["Close"].dropna()
            return s
        except Exception:
            return pd.Series(dtype=float)

    # Try CCYUSD then USDCCY inverted
    s3 = _stooq_get(f"{ccy.lower()}usd")
    if not s3.empty:
        s3.name = ccy
        # Stooq often includes very old data; slice to requested window
        s3 = s3.loc[(s3.index >= start_dt) & (s3.index <= end_dt)]
        return s3

    s4 = _stooq_get(f"usd{ccy.lower()}")
    if not s4.empty:
        s4 = (1.0 / s4).dropna()
        s4.name = ccy
        s4 = s4.loc[(s4.index >= start_dt) & (s4.index <= end_dt)]
        return s4

    raise ValueError(f"Could not fetch FX series for {ccy}. Tried: {ccy}USD=X, USD{ccy}=X (Yahoo) and {ccy.lower()}usd / usd{ccy.lower()} (Stooq).")

    """
    Daily long/short backtest:
      port_ret = ret(long_ccy_vs_USD) - ret(short_ccy_vs_USD)
      equity = cumulative product of (1 + port_ret)
    """
    long_px = _fetch_fx_series_yahoo(long_ccy, start, end)
    short_px = _fetch_fx_series_yahoo(short_ccy, start, end)

    df = pd.concat([long_px, short_px], axis=1).dropna()
    if df.empty or df.shape[0] < 5:
        raise ValueError("Not enough overlapping FX data for this pair/date range.")

    long_ret = df[long_px.name].pct_change().fillna(0.0)
    short_ret = df[short_px.name].pct_change().fillna(0.0)

    port_ret = (long_ret - short_ret).astype(float)
    equity = (1.0 + port_ret).cumprod()

    out = pd.DataFrame(
        {
            "long_ret": long_ret,
            "short_ret": short_ret,
            "port_ret": port_ret,
            "equity": equity,
        },
        index=df.index,
    )
    return out

def _to_1d_series(x, name: str) -> pd.Series:
    """Ensure x is a 1D pandas Series (not a 1-col DataFrame)."""
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(f"{name}: expected 1 column, got {x.shape[1]}")
        x = x.iloc[:, 0]
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    x = x.dropna()
    x.name = name
    return x

def run_pair_backtest(long_ccy: str, short_ccy: str, start: str, end: str) -> pd.DataFrame:
    """
    Daily long/short backtest:
      port_ret = ret(Long_vs_USD) - ret(Short_vs_USD)
      equity = cumprod(1 + port_ret)

    Robust to data arriving as Series OR single-column DataFrame.
    """
    long_ccy = (long_ccy or "").upper().strip()
    short_ccy = (short_ccy or "").upper().strip()

    long_px_raw = _fetch_fx_series_yahoo(long_ccy, start, end)
    short_px_raw = _fetch_fx_series_yahoo(short_ccy, start, end)

    long_px = _to_1d_series(long_px_raw, "Long")
    short_px = _to_1d_series(short_px_raw, "Short")

    df = pd.concat([long_px, short_px], axis=1).dropna()
    if df.empty or df.shape[0] < 5:
        raise ValueError("Not enough overlapping FX data for this pair/date range.")

    long_ret = df["Long"].pct_change().fillna(0.0)
    short_ret = df["Short"].pct_change().fillna(0.0)

    port_ret = (long_ret - short_ret).astype(float)
    equity = (1.0 + port_ret).cumprod()

    out = pd.DataFrame(
        {
            "long_ret": long_ret,
            "short_ret": short_ret,
            "port_ret": port_ret,
            "equity": equity,
        },
        index=df.index,
    )
    return out

def summarize_backtest(bt: pd.DataFrame) -> dict:
    """Simple stats for the backtest."""
    r = bt["port_ret"].dropna().astype(float)
    if r.empty:
        return {}

    ann = 252.0
    vol = float(r.std() * np.sqrt(ann))
    cagr = float(bt["equity"].iloc[-1] ** (ann / max(len(r), 1)) - 1.0)
    sharpe = float((r.mean() * ann) / vol) if vol > 0 else np.nan

    eq = bt["equity"].astype(float)
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    max_dd = float(dd.min())

    return {
        "CAGR": cagr,
        "Vol (ann.)": vol,
        "Sharpe (rf=0)": sharpe,
        "Max Drawdown": max_dd,
        "Days": int(len(r)),
    }



# ============================================================
# Data loading + scoring
# ============================================================

@st.cache_data(show_spinner=False)
def load_world_fx(_data_mtime: float | None) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(DATA_PATH)

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
    """
    Systematic, template-based "investment note" using the factor scores + weights.
    No external AI calls.
    """
    name = row.get("display_name", "This country")
    curr = row.get("primary_currency", None)

    agg = row.get("aggregate_score", pd.NA)

    # Individual factor scores (0-100, higher = better)
    fx = row.get("fx_score_mom", pd.NA)
    infl = row.get("infl_score", pd.NA)
    growth = row.get("growth_score", pd.NA)
    policy = row.get("policy_score", pd.NA)
    risk = row.get("risk_score", pd.NA)

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

    # More nuanced stance buckets
    if agg >= 75:
        stance = "attractive"
        stance_line = "The model is signaling a strong overall setup with multiple supportive drivers."
    elif agg >= 65:
        stance = "constructive"
        stance_line = "The setup is positive on balance, but not without meaningful risks."
    elif agg >= 55:
        stance = "slightly positive"
        stance_line = "There are more tailwinds than headwinds, though the edge is not strong."
    elif agg >= 45:
        stance = "mixed"
        stance_line = "Signals are conflicting; results may depend on which macro driver dominates next."
    elif agg >= 35:
        stance = "cautious"
        stance_line = "Headwinds dominate; a positive outcome likely needs a clear catalyst or improving data."
    else:
        stance = "unfavorable"
        stance_line = "The model flags weak conditions and elevated downside risk."

    # Biggest contributors (weight * (score-50))
    pos, neg = driver_summary(row, weights)

    def _fmt_factor_list(items):
        return ", ".join([f"**{lbl}** ({s:.0f})" for (lbl, _, s) in items])

    tailwinds_txt = _fmt_factor_list(pos) if pos else "none stand out"
    headwinds_txt = _fmt_factor_list(neg) if neg else "none stand out"

    # Helper: interpret a score into plain-English
    def _interp(label: str, val):
        if pd.isna(val):
            return None
        v = float(val)
        if v >= 70:
            return f"{label} is a clear tailwind"
        if v >= 55:
            return f"{label} is mildly supportive"
        if v >= 45:
            return f"{label} is broadly neutral"
        if v >= 30:
            return f"{label} is a headwind"
        return f"{label} is a strong headwind"

    # Build a "what to watch" list based on weakest factors
    factor_map = [
        ("FX momentum", fx, "price trend / positioning"),
        ("Inflation momentum", infl, "next CPI prints / inflation surprises"),
        ("Growth momentum", growth, "PMIs / GDP releases"),
        ("Policy / real rates & liquidity", policy, "central bank stance / real yields / liquidity"),
        ("Risk / governance & geopolitics", risk, "political risk / geopolitics / policy stability"),
    ]

    weak = []
    for lbl, val, watch in factor_map:
        if pd.isna(val):
            continue
        weak.append((lbl, float(val), watch))
    weak = sorted(weak, key=lambda x: x[1])[:2]  # bottom 2

    watch_lines = []
    for lbl, v, watch in weak:
        if v < 50:
            watch_lines.append(f"- **{lbl}** is weak ({v:.0f}): watch **{watch}**")
    if not watch_lines:
        watch_lines = ["- No single weak factor dominates; watch for **trend changes** in the main drivers above."]

    # Core narrative paragraph (systematic, but reads like a paragraph)
    para1 = (
        f"**{name} ({curr}) screens _{stance}_ (aggregate **{agg:.1f}/100**).** {stance_line} "
        f"On the current weights, the main **tailwinds** are {tailwinds_txt}, while the key **headwinds** are {headwinds_txt}."
    )

    # Add a second paragraph tying “investment decision” framing to the score mix
    para2 = (
        "From an investment decision perspective, this setup is most compelling when the **top drivers stay stable** "
        "for several months and the **weakest drivers begin improving** (rather than deteriorating). "
        "If headwinds intensify (especially risk, inflation, or policy credibility), FX upside can be capped even when momentum looks strong."
    )

    # Summary bullets (kept, but not the whole note)
    fx_txt = "N/A" if pd.isna(fx) else f"{float(fx):.1f}"
    infl_txt = "N/A" if pd.isna(infl) else f"{float(infl):.1f}"
    growth_txt = "N/A" if pd.isna(growth) else f"{float(growth):.1f}"
    policy_txt = "N/A" if pd.isna(policy) else f"{float(policy):.1f}"
    risk_txt = "N/A" if pd.isna(risk) else f"{float(risk):.1f}"

    lines = []
    lines.append(para1)
    lines.append("")
    lines.append(para2)
    lines.append("")
    lines.append("**Factor read-through:**")
    for lbl, val, _ in factor_map:
        msg = _interp(lbl, val)
        if msg:
            lines.append(f"- {msg}.")
    lines.append("")
    lines.append("**What to watch next:**")
    lines.extend(watch_lines)
    lines.append("")
    lines.append("**Scores:**")
    lines.append(f"- Aggregate (0–100): **{agg:.1f}**")
    lines.append(f"- FX: **{fx_txt}** | Inflation: **{infl_txt}** | Growth: **{growth_txt}** | Policy: **{policy_txt}** | Risk: **{risk_txt}**")

    if fill_missing:
        lines.append("")
        lines.append("*(Missing factors are currently filled with a neutral 50 to color every country.)*")

    lines.append("")
    lines.append("*Systematic informational analysis — not financial advice.*")

    return "\n".join(lines)

def compute_rank(gdf: gpd.GeoDataFrame, col: str, value: float) -> tuple[int, int] | tuple[None, int]:
    """Return (rank, N) where rank=1 is best (highest). If value is missing, rank is None."""
    s = pd.to_numeric(gdf[col], errors="coerce")
    s = s.dropna()
    N = int(s.shape[0])
    if N == 0 or pd.isna(value):
        return None, N
    # rank 1 = highest value
    rank = int((s > float(value)).sum() + 1)
    return rank, N


def factor_breakdown_df(row: pd.Series, weights: dict, fill_missing: bool) -> pd.DataFrame:
    """Build a table showing score, weight, and contribution to aggregate."""
    factor_map = [
        ("FX", "fx_score_mom", float(weights["fx"])),
        ("Inflation", "infl_score", float(weights["infl"])),
        ("Growth", "growth_score", float(weights["growth"])),
        ("Policy", "policy_score", float(weights["policy"])),
        ("Risk", "risk_score", float(weights["risk"])),
    ]

    rows = []
    for label, col, w in factor_map:
        score = pd.to_numeric(row.get(col, pd.NA), errors="coerce")
        if fill_missing and pd.isna(score):
            score = 50.0
        rows.append({"Factor": label, "Score": score, "Weight": w})

    df = pd.DataFrame(rows)
    # Use only weights for non-missing scores
    mask = ~df["Score"].isna()
    wsum = df.loc[mask, "Weight"].sum()

    if wsum == 0:
        df["Weight (norm)"] = pd.NA
        df["Contribution to Aggregate"] = pd.NA
        df["Tilt vs Neutral (50)"] = pd.NA
        return df

    df["Weight (norm)"] = df["Weight"] / wsum
    df["Contribution to Aggregate"] = df["Weight (norm)"] * df["Score"]
    df["Tilt vs Neutral (50)"] = df["Weight (norm)"] * (df["Score"] - 50.0)
    return df


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

    # ✅ Remove +/- buttons
    zoom_control=False,

    # ✅ Prevent scroll-wheel from zooming when user tries to scroll the page
    scrollWheelZoom=False,

    # Optional (nice polish): disable double-click zoom too
    doubleClickZoom=False,
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


    # ------------------------------------------------------------
    # Legend (colorbar) styling: bottom-center + white text
    # ------------------------------------------------------------
    legend_css = """
    <style>
    .leaflet-control-container .legend {
        position: fixed !important;

        /* bottom-center */
        left: 50% !important;
        right: auto !important;
        top: auto !important;
        bottom: 20px !important;
        transform: translateX(-50%) !important;

        z-index: 9999 !important;
        background: rgba(0,0,0,0.55) !important;
        padding: 10px 12px !important;
        border-radius: 10px !important;

        /* prevents legend from blocking map clicks */
        pointer-events: none !important;
    }

    .leaflet-control-container .legend .caption {
        color: white !important;
        font-size: 14px !important;
        font-weight: 600 !important;
    }

    .leaflet-control-container .legend text {
        fill: white !important;
    }

    .leaflet-control-container .legend line,
    .leaflet-control-container .legend path {
        stroke: rgba(255,255,255,0.55) !important;
    }
    </style>
    """
    m.get_root().header.add_child(folium.Element(legend_css))

    m.fit_bounds(bounds)
    m.options["maxBounds"] = bounds
    m.options["maxBoundsViscosity"] = 1.0

    # --- Color scale / legend ---
    colormap = cm.linear.RdYlGn_11.scale(0, 100)
    colormap.caption = title
    colormap.add_to(m)

    # Move legend to bottom-center + make legend text white
    legend_css = """
    <style>
    .leaflet-control-container .legend {
        position: fixed !important;
        left: 50% !important;
        right: auto !important;
        top: auto !important;
        bottom: 20px !important;
        transform: translateX(-50%) !important;

        z-index: 9999 !important;
        background: rgba(0,0,0,0.55) !important;
        padding: 10px 12px !important;
        border-radius: 10px !important;

        /* prevents legend from blocking map clicks */
        pointer-events: none !important;
    }

    .leaflet-control-container .legend .caption {
        color: white !important;
        font-size: 14px !important;
        font-weight: 600 !important;
    }

    .leaflet-control-container .legend text {
        fill: white !important;
    }

    .leaflet-control-container .legend line,
    .leaflet-control-container .legend path {
        stroke: rgba(255,255,255,0.55) !important;
    }
    </style>
    """

    
    m.get_root().header.add_child(folium.Element(legend_css))


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
# Leaders & Laggards (Top/Bottom table)
# ============================================================

st.markdown("---")
st.subheader("Leaders & Laggards")

rank_col = metric  # whatever you are coloring the map by
rank_title = metric_title

rank_df = gdf[["display_name", "primary_currency", rank_col]].copy()
rank_df[rank_col] = pd.to_numeric(rank_df[rank_col], errors="coerce")
rank_df = rank_df.dropna(subset=[rank_col])

n_show = st.slider("How many countries?", 5, 25, 10, 1)

if rank_df.empty:
    st.info("No ranking available for this metric (all values are missing).")
else:
    top_df = rank_df.sort_values(rank_col, ascending=False).head(n_show)
    bottom_df = rank_df.sort_values(rank_col, ascending=True).head(n_show)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"#### Top {n_show} — {rank_title}")
        st.dataframe(
            top_df.rename(columns={rank_col: rank_title}),
            use_container_width=True,
        )
    with c2:
        st.markdown(f"#### Bottom {n_show} — {rank_title}")
        st.dataframe(
            bottom_df.rename(columns={rank_col: rank_title}),
            use_container_width=True,
        )

# ============================================================
# Country details (MAIN PAGE, not sidebar)
# ============================================================

st.markdown("---")
st.subheader("Country details")

names = sorted(gdf["display_name"].dropna().unique().tolist())


options = ["(none)"] + names

# Persist selection across reruns (and avoid errors when options list changes)
if "selected_country" not in st.session_state:
    st.session_state["selected_country"] = "(none)"
if st.session_state["selected_country"] not in options:
    st.session_state["selected_country"] = "(none)"

selected_name = st.selectbox(
    "Jump to a country (optional)",
    options,
    key="selected_country",
)


selected_row = None

# 1) Dropdown selection
if selected_name != "(none)":
    hit = gdf.loc[gdf["display_name"] == selected_name]
    if not hit.empty:
        selected_row = hit.iloc[0]

# 2) Map click selection
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
            # Sync dropdown selection with map clicks
            st.session_state["selected_country"] = selected_row.get("display_name", "(none)")


if selected_row is None:
    pass
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
