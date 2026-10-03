"""
Edge-AI Meteorological Station Dashboard
Streamlit + Plotly implementation following ISA-101 / Tufte principles.

Data source: CSV file with weather readings.

Run:  streamlit run dashboard.py
Deps: pip install streamlit pandas plotly
"""

import math
import os
import threading
import time
from datetime import datetime

import openmeteo_requests
import pandas as pd
import plotly.graph_objects as go
import requests_cache
from retry_requests import retry
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Edge-AI Meteorological Station",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

CSV_FILE    = "meteorological_data.csv"

COLUMNS = [
    "timestamp", "temp_c", "humidity_pct", "pressure_hpa",
    "pressure_trend", "wind_kmh", "light_pct", "cloud_pct", "rain_prob_pct",
]

# ISA-101 compliant palette — grayscale dominant, color only for alarms + XAI
P = {
    "bg":        "#0F172A",  # slate-900  · app canvas
    "panel":     "#1E293B",  # slate-800  · card background
    "border":    "#334155",  # slate-700  · dividers
    "txt_pri":   "#F1F5F9",  # slate-100  · primary telemetry values
    "txt_sec":   "#94A3B8",  # slate-400  · labels, axis text
    "line":      "#64748B",  # slate-500  · primary data line
    "grid":      "#1E293B",  # slate-800  · gridlines (very subtle)
    "xai_pos":   "#818CF8",  # indigo-400 · rain-driving features
    "xai_neg":   "#FBBF24",  # amber-400  · clear-driving features
    "ok":        "#34D399",  # emerald-400· nominal / connected
    "alarm":     "#F43F5E",  # rose-500   · critical
    "warn":      "#FBBF24",  # amber-400  · warning / stale
    "rain_fill": "rgba(129,140,248,0.15)",
    "cloud_fill":"rgba(148,163,184,0.12)",
    "pres_fill": "rgba(100,116,139,0.12)",
    "dew_line":  "#2DD4BF",  # teal-400   · dew point
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. ML MODEL WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
MODEL_INTERCEPT = -0.002658

# key → (mean, std, coefficient, display_label)
FEATURES = {
    "temp":  (24.192821, 12.307345,  2.638816, "Temperature"),
    "hum":   (84.474717, 11.095475,  0.976659, "Humidity"),
    "dpd":   ( 2.961701,  2.433860, -1.499602, "Dew Pt. Depression"),
    "pres":  ( 0.000884,  0.608765, -0.175996, "Pressure Trend"),
    "wind":  ( 3.908563,  1.697854,  0.908026, "Wind Speed"),
    "cloud": (89.869282, 21.332968,  0.310442, "Cloud Cover"),
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. CACHED RESOURCES  — survive Streamlit reruns
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_csv_lock() -> threading.Lock:
    """Singleton lock shared across all reruns — prevents CSV race conditions."""
    return threading.Lock()


class WeatherDataSource:
    """Abstraction for weather data sources. Dashboard calls this, never Open-Meteo directly."""
    def get_data(self):
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# DATA SOURCE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
OPEN_METEO_LAT  = 51.5074
OPEN_METEO_LON  = -0.1278
OPEN_METEO_URL  = "https://api.open-meteo.com/v1/forecast"


class OpenMeteoWeatherSource(WeatherDataSource):
    """Fetches live weather from Open-Meteo API, returns normalized dict."""

    _REQUIRED_VARIABLES = [
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "wind_speed_10m",
        "cloud_cover",
        "precipitation_probability",
        "shortwave_radiation",
        "dew_point_2m",
    ]

    def __init__(self, lat=None, lon=None):
        cache_dir = os.path.join(os.path.dirname(__file__), ".openmeteo_cache")
        os.makedirs(cache_dir, exist_ok=True)
        self._cache_session = requests_cache.CachedSession(
            os.path.join(cache_dir, "openmeteo"),
            expire_after=300,
        )
        retry_session = retry(
            self._cache_session, retries=3, backoff_factor=1,
            status_to_retry=(500, 502, 504),
        )
        self._client = openmeteo_requests.Client(session=retry_session)
        self.lat = lat or OPEN_METEO_LAT
        self.lon = lon or OPEN_METEO_LON

    def get_data(self):
        params = {
            "latitude":   self.lat,
            "longitude":  self.lon,
            "current":    self._REQUIRED_VARIABLES,
            "hourly":     ["surface_pressure"],
            "timezone":   "auto",
        }
        responses = self._client.weather_api(OPEN_METEO_URL, params=params)
        response = responses[0]
        current = response.Current()

        raw = {}
        for i in range(current.VariablesLength()):
            var = current.Variables(i)
            val = var.Value()
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                name_int = var.Variable()
                mapping = {
                    47: ("temp_c",         round(val, 2)),
                    29: ("humidity_pct",   round(val, 1)),
                    45: ("pressure_hpa",   round(val, 2)),
                    59: ("wind_kmh",       round(max(0, val), 1)),
                    3:  ("cloud_pct",      round(val, 1)),
                    26: ("rain_prob_pct",  round(val, 1)),
                    32: ("shortwave_raw",  float(val)),
                    8:  ("dew_point_c",    round(val, 1)),
                }
                entry = mapping.get(name_int)
                if entry:
                    raw[entry[0]] = entry[1]

        temp      = raw.get("temp_c", 0)
        humidity  = raw.get("humidity_pct", 0)
        cloud_pct = raw.get("cloud_pct", 0)
        pressure  = raw.get("pressure_hpa", 1013.25)
        wind      = max(0, raw.get("wind_kmh", 0))

        # Pressure trend (hPa/hour) from hourly surface_pressure series
        hourly      = response.Hourly()
        hourly_vals = hourly.Variables(0).ValuesAsNumpy()
        idx         = int((current.Time() - hourly.Time()) // hourly.Interval())
        idx         = max(1, min(idx, len(hourly_vals) - 1))
        pressure_trend = round(float(hourly_vals[idx] - hourly_vals[idx - 1]), 2)

        # light_pct from shortwave radiation (W/m² → 0-100 %, ~1000 W/m² = 100 %)
        shortwave = raw.get("shortwave_raw", 0.0)
        light_pct = int(min(100, max(0, round(shortwave / 10.0))))

        # Preserve the actual API observation timestamp (station-local time)
        api_ts = pd.to_datetime(
            current.Time() + response.UtcOffsetSeconds(), unit="s"
        ).strftime("%Y-%m-%d %H:%M:%S")

        return {
            "timestamp":      api_ts,
            "temp_c":         round(temp, 2),
            "humidity_pct":   round(humidity, 1),
            "pressure_hpa":   round(pressure, 2),
            "pressure_trend": pressure_trend,
            "wind_kmh":       round(wind, 1),
            "light_pct":      light_pct,
            "cloud_pct":      round(cloud_pct, 1),
            "rain_prob_pct":  round(raw.get("rain_prob_pct", 0), 1),
            "dew_point_c":    round(raw.get("dew_point_c", temp - (100 - humidity) / 5.0), 1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. DATA SOURCE INSTANCE
# ─────────────────────────────────────────────────────────────────────────────
data_source = OpenMeteoWeatherSource()


def init_csv_data():
    """Create an empty CSV with the correct columns if none exists."""
    if not os.path.exists(CSV_FILE):
        pd.DataFrame(columns=COLUMNS).to_csv(CSV_FILE, index=False)


init_csv_data()
_csv_lock = get_csv_lock()

# ─────────────────────────────────────────────────────────────────────────────
# 6. HELPER: FEATURE CONTRIBUTIONS (XAI)
# ─────────────────────────────────────────────────────────────────────────────
def compute_contributions(temp_c, humidity_pct, pressure_trend, wind_kmh, cloud_pct):
    """
    For Logistic Regression: local feature contribution = coefficient × z_score.
    This is the mathematically exact local explanation — no approximation library
    (SHAP, LIME) required. Returns list sorted by |contribution| descending.
    """
    dew_point = temp_c - ((100.0 - humidity_pct) / 5.0)
    dpd       = temp_c - dew_point

    raw_values = {
        "temp":  temp_c,
        "hum":   humidity_pct,
        "dpd":   dpd,
        "pres":  pressure_trend,
        "wind":  wind_kmh,
        "cloud": cloud_pct,
    }

    results = []
    for key, raw_val in raw_values.items():
        mean, std, coef, label = FEATURES[key]
        z      = (raw_val - mean) / std
        contrib = coef * z
        results.append({
            "key":         key,
            "label":       label,
            "raw":         raw_val,
            "z_score":     round(z, 3),
            "coef":        coef,
            "contribution":round(contrib, 4),
        })

    results.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return results


def generate_insight(contributions: list, rain_prob: float) -> str:
    """Plain-English explanation from the two highest-magnitude contributors."""
    top    = contributions[0]
    second = contributions[1]

    _desc = {
        "temp":  lambda c: "warm air temperatures"      if c > 0 else "cool conditions suppressing convection",
        "hum":   lambda c: "high atmospheric moisture"  if c > 0 else "dry air mass",
        "dpd":   lambda c: "air approaching saturation" if c > 0 else "significant dew-point gap",
        "pres":  lambda c: "rapidly falling pressure"   if c > 0 else "stable or rising pressure",
        "wind":  lambda c: "elevated wind activity"     if c > 0 else "calm wind conditions",
        "cloud": lambda c: "heavy cloud cover"          if c > 0 else "clear skies reducing moisture",
    }

    top_desc    = _desc[top["key"]](top["contribution"])
    second_desc = _desc[second["key"]](second["contribution"])

    if abs(rain_prob - 50) < 10:
        return (
            f"⚖ Conditions are borderline — **{top_desc}** and **{second_desc}** "
            f"are partially offsetting each other."
        )

    direction = "elevated" if rain_prob > 50 else "suppressed"
    return f"Rain probability {direction} by **{top_desc}** and **{second_desc}**."

# ─────────────────────────────────────────────────────────────────────────────
# 7. HELPER: SHARED PLOTLY CHART THEME
# ─────────────────────────────────────────────────────────────────────────────
def base_layout(title: str = "", height: int = 220) -> dict:
    """
    ISA-101 / Tufte compliant base layout for all Plotly charts.
    - No top/right border axes
    - No vertical gridlines
    - Horizontal gridlines at very low opacity
    - Monospace font for axis labels
    """
    return dict(
        title=dict(
            text=title,
            font=dict(size=10, color=P["txt_sec"], family="Inter, system-ui"),
            x=0, pad=dict(l=0)
        ),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=P["panel"],
        font=dict(family="'JetBrains Mono', monospace", color=P["txt_sec"], size=9),
        margin=dict(l=44, r=12, t=28, b=24),
        xaxis=dict(
            showgrid=False,
            color=P["txt_sec"],
            tickfont=dict(size=8),
            linecolor=P["border"],
            showline=True,
            mirror=False,
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=P["grid"],
            gridwidth=1,
            color=P["txt_sec"],
            tickfont=dict(size=8),
            linecolor=P["border"],
            showline=True,
            mirror=False,
            zeroline=False,
        ),
        showlegend=False,
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=P["panel"],
            bordercolor=P["border"],
            font=dict(color=P["txt_pri"], size=10),
        ),
    )

# ─────────────────────────────────────────────────────────────────────────────
# 8. CUSTOM CSS  — ISA-101 palette + JetBrains Mono for telemetry values
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', system-ui, sans-serif;
    background-color: {P["bg"]};
    color: {P["txt_pri"]};
}}
/* Kill Streamlit's own white backgrounds */
.main .block-container {{ background: {P["bg"]}; padding-top: 1.5rem; }}
section[data-testid="stSidebar"] {{ background: {P["panel"]}; }}

/* Metric card overrides */
[data-testid="stMetric"] {{
    background: {P["panel"]};
    border: 1px solid {P["border"]};
    border-radius: 6px;
    padding: 12px 16px;
}}
[data-testid="stMetricLabel"] {{
    color: {P["txt_sec"]} !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}}
[data-testid="stMetricValue"] {{
    color: {P["txt_pri"]} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.4rem !important;
}}
[data-testid="stMetricDelta"] {{
    font-size: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
}}

/* Section labels */
.sec-label {{
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: {P["txt_sec"]};
    margin: 12px 0 6px 0;
    border-bottom: 1px solid {P["border"]};
    padding-bottom: 4px;
}}

/* Rain hero card */
.rain-hero {{
    background: {P["panel"]};
    border: 1px solid {P["border"]};
    border-radius: 8px;
    padding: 24px;
    text-align: center;
}}
.rain-hero .label {{
    font-family: 'Inter', sans-serif;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: {P["txt_sec"]};
    margin-bottom: 4px;
}}
.rain-hero .value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 80px;
    font-weight: 600;
    line-height: 1;
}}
.rain-hero .condition {{
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 8px;
}}

/* NLG insight box */
.insight-box {{
    background: {P["panel"]};
    border-left: 3px solid {P["xai_pos"]};
    border-radius: 4px;
    padding: 12px 16px;
    font-size: 13px;
    line-height: 1.6;
    color: {P["txt_pri"]};
}}

/* Status pills */
.pill-ok    {{ display:inline-flex;align-items:center;gap:5px;background:#064E3B;color:#34D399;border-radius:99px;padding:3px 12px;font-size:11px;font-weight:600; }}
.pill-warn  {{ display:inline-flex;align-items:center;gap:5px;background:#451A03;color:#FBBF24;border-radius:99px;padding:3px 12px;font-size:11px;font-weight:600; }}
.pill-error {{ display:inline-flex;align-items:center;gap:5px;background:#4C0519;color:#F43F5E;border-radius:99px;padding:3px 12px;font-size:11px;font-weight:600; }}

/* Alarm banner */
.alarm-crit {{
    background: rgba(244,63,94,0.1);
    border: 1px solid {P["alarm"]};
    border-radius: 6px;
    padding: 8px 14px;
    color: {P["alarm"]};
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 8px;
}}
.alarm-warn {{
    background: rgba(251,191,36,0.08);
    border: 1px solid {P["warn"]};
    border-radius: 6px;
    padding: 8px 14px;
    color: {P["warn"]};
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 8px;
}}

/* Dataframe overrides */
[data-testid="stDataFrame"] {{ background: {P["panel"]}; }}
hr {{ border-color: {P["border"]} !important; margin: 12px 0 !important; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 9. LOAD DATA  (thread-safe read)
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# 10. FETCH FRESH DATA THROUGH DATA SOURCE ABSTRACTION
# ─────────────────────────────────────────────────────────────────────────────
_fetch_ok    = False
_fetch_error = ""
try:
    row = data_source.get_data()
    with _csv_lock:
        # Dedup — append only when the API observation timestamp is newer
        # than the latest row already stored in the CSV.
        try:
            existing = pd.read_csv(CSV_FILE, parse_dates=["timestamp"])
        except Exception:
            existing = pd.DataFrame(columns=COLUMNS)
        last_ts = existing["timestamp"].iloc[-1] if not existing.empty else None
        if last_ts is None or pd.to_datetime(row["timestamp"]) > last_ts:
            ordered_row = {col: row[col] for col in COLUMNS}
            pd.DataFrame([ordered_row])[COLUMNS].to_csv(
                CSV_FILE, mode="a", header=False, index=False
            )
    _fetch_ok = True
except Exception as e:
    _fetch_error = f"{type(e).__name__}: {e}"

with _csv_lock:
    try:
        df = pd.read_csv(CSV_FILE, parse_dates=["timestamp"])
    except Exception:
        df = pd.DataFrame(columns=COLUMNS)

if not df.empty:
    df = df.tail(7200).reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# 10. HEADER
# ─────────────────────────────────────────────────────────────────────────────
h_left, h_right = st.columns([3, 1])

with h_left:
    st.markdown(
        "<h2 style='margin:0;font-family:Inter,sans-serif;font-weight:600;"
        f"color:{P['txt_pri']}'>⬡ Weather Data Pipeline</h2>"
        f"<p style='margin:2px 0 0 0;font-size:11px;color:{P['txt_sec']}'>"
        "Real-time meteorological monitoring & rain probability model</p>",
        unsafe_allow_html=True,
    )

with h_right:
    # Status pill — reflects whether fresh API data was successfully received
    if _fetch_ok:
        st.markdown('<span class="pill-ok">● LIVE</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill-error">✕ API ERROR — STALE DATA</span>',
                    unsafe_allow_html=True)
        if _fetch_error:
            st.caption(f"`{_fetch_error}`")

    st.markdown(
        f"<p style='font-size:11px;color:{P['txt_sec']};margin:4px 0 0 0'>"
        f"🕒 {datetime.now().strftime('%H:%M:%S')}</p>",
        unsafe_allow_html=True,
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 11. NO DATA STATE
# ─────────────────────────────────────────────────────────────────────────────
if df.empty:
    st.info("⏳ Awaiting weather data...")
    st.stop()

latest = df.iloc[-1].copy()
prev   = df.iloc[-2].copy() if len(df) >= 2 else latest

# ─────────────────────────────────────────────────────────────────────────────
# 12. DERIVED VALUES
# ─────────────────────────────────────────────────────────────────────────────
dew_point_c = float(latest["temp_c"]) - ((100.0 - float(latest["humidity_pct"])) / 5.0)
dpd_c       = float(latest["temp_c"]) - dew_point_c

# Plot window: last 60 min (~1200 rows)
plot_df = df.tail(1200).copy()
plot_df["dew_point_c"] = plot_df["temp_c"] - ((100.0 - plot_df["humidity_pct"]) / 5.0)

# Feature contributions for current reading
contributions = compute_contributions(
    float(latest["temp_c"]),
    float(latest["humidity_pct"]),
    float(latest["pressure_trend"]),
    float(latest["wind_kmh"]),
    float(latest["cloud_pct"]),
)
insight = generate_insight(contributions, float(latest["rain_prob_pct"]))

# Deltas vs previous reading
d_temp = round(float(latest["temp_c"])       - float(prev["temp_c"]),       2)
d_hum  = round(float(latest["humidity_pct"]) - float(prev["humidity_pct"]), 1)
d_pres = round(float(latest["pressure_hpa"]) - float(prev["pressure_hpa"]), 2)

# ─────────────────────────────────────────────────────────────────────────────
# 13. ALARM EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
pressure_trend_val = float(latest["pressure_trend"])
rain_prob_val      = float(latest["rain_prob_pct"])

if pressure_trend_val < -2.0:
    st.markdown(
        f'<div class="alarm-crit">'
        f'⚠ CRITICAL — Severe pressure drop ({pressure_trend_val:.1f} hPa/hr). '
        f'Storm risk elevated. Check shelter status.'
        f'</div>',
        unsafe_allow_html=True,
    )
elif pressure_trend_val < -1.0:
    st.markdown(
        f'<div class="alarm-warn">'
        f'⚠ WARNING — Pressure falling ({pressure_trend_val:.1f} hPa/hr). '
        f'Monitor trend.'
        f'</div>',
        unsafe_allow_html=True,
    )

if rain_prob_val > 75:
    st.markdown(
        f'<div class="alarm-warn">'
        f'🌧 Rain probability at {rain_prob_val:.0f}% — high confidence event.'
        f'</div>',
        unsafe_allow_html=True,
    )

if dpd_c < 2.0:
    st.markdown(
        f'<div class="alarm-warn">'
        f'💧 Dew point depression only {dpd_c:.1f} °C — fog or condensation risk.'
        f'</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 14. LEVEL 1 — PRIMARY INSIGHT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-label">Primary Insight</div>', unsafe_allow_html=True)

hero_col, insight_col = st.columns([1, 2])

with hero_col:
    rain = float(latest["rain_prob_pct"])

    if rain >= 75:
        cond_label, hero_color = "RAIN LIKELY",   P["xai_pos"]
    elif rain >= 50:
        cond_label, hero_color = "POSSIBLE RAIN", P["txt_sec"]
    elif rain >= 25:
        cond_label, hero_color = "MOSTLY CLEAR",  P["txt_sec"]
    else:
        cond_label, hero_color = "CLEAR",         P["txt_sec"]

        st.markdown(
            f'<div class="rain-hero">'
            f'  <div class="label">RAIN PROBABILITY</div>'
            f'  <div class="value" style="color:{hero_color}">{rain:.0f}%</div>'
            f'  <div class="condition" style="color:{hero_color}">{cond_label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

with insight_col:
    st.markdown(
        f'<div class="insight-box">{insight}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Temperature", f"{latest['temp_c']:.1f} °C",       delta=f"{d_temp:+.2f}°")
    m2.metric("Humidity",    f"{latest['humidity_pct']:.1f} %",   delta=f"{d_hum:+.1f}%",  delta_color="off")
    m3.metric("Pressure",    f"{latest['pressure_hpa']:.1f} hPa", delta=f"{d_pres:+.2f}")
    m4.metric("Wind",        f"{latest['wind_kmh']:.1f} km/h",    delta=None)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 15. LEVEL 2 — ENVIRONMENTAL METRICS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-label">Environmental Metrics</div>', unsafe_allow_html=True)

row1_l, row1_r = st.columns(2)

# ── Temperature + Dew Point overlay ─────────────────────────────────────────
with row1_l:
    fig_temp = go.Figure()

    # Dew point (lower, dashed teal line) — plotted first for fill-between
    fig_temp.add_trace(go.Scatter(
        x=plot_df["timestamp"], y=plot_df["dew_point_c"],
        mode="lines",
        line=dict(color=P["dew_line"], width=1.5, dash="dot"),
        name="Dew Point",
        hovertemplate="Dew Point: %{y:.1f} °C",
    ))
    # Temperature (solid line) with fill between the two
    fig_temp.add_trace(go.Scatter(
        x=plot_df["timestamp"], y=plot_df["temp_c"],
        mode="lines", fill="tonexty",
        fillcolor="rgba(45,212,191,0.07)",
        line=dict(color=P["line"], width=2),
        name="Temperature",
        hovertemplate="Temperature: %{y:.1f} °C",
    ))

    layout = base_layout("TEMPERATURE  &  DEW POINT", height=220)
    layout["yaxis"]["title"] = dict(text="°C", font=dict(size=9))
    layout["showlegend"]  = True
    layout["legend"] = dict(
        font=dict(size=8, color=P["txt_sec"]),
        bgcolor="rgba(0,0,0,0)",
        x=0.02, y=0.98,
    )
    fig_temp.update_layout(**layout)
    st.plotly_chart(fig_temp, use_container_width=True, config={"displayModeBar": False})

# ── Pressure (range-framed — NOT zero-origin) ────────────────────────────────
with row1_r:
    p_data = plot_df["pressure_hpa"]
    p_min  = p_data.min()
    p_max  = p_data.max()
    # Tight range framing — a 2 hPa variation must be visible, not flattened
    p_pad  = max((p_max - p_min) * 0.2, 0.8)

    trend_arrow = "▼" if pressure_trend_val < -0.5 else ("▲" if pressure_trend_val > 0.5 else "→")
    trend_color = P["xai_pos"] if pressure_trend_val < -1.0 else (
                  P["xai_neg"] if pressure_trend_val > 1.0 else P["txt_sec"])

    fig_pres = go.Figure()
    fig_pres.add_trace(go.Scatter(
        x=plot_df["timestamp"], y=p_data,
        mode="lines", fill="tozeroy",
        fillcolor=P["pres_fill"],
        line=dict(color=P["line"], width=2),
        hovertemplate="Pressure: %{y:.2f} hPa",
    ))
    # Session average reference line
    fig_pres.add_hline(
        y=p_data.mean(),
        line=dict(color=P["border"], width=1, dash="dot"),
        annotation_text=f"avg {p_data.mean():.1f}",
        annotation_font=dict(size=8, color=P["txt_sec"]),
        annotation_position="top right",
    )

    pres_layout = base_layout(
        f"ATMOSPHERIC PRESSURE   {trend_arrow} {pressure_trend_val:+.1f} hPa/hr",
        height=220,
    )
    pres_layout["title"]["font"]["color"] = trend_color
    pres_layout["yaxis"].update(
        range=[p_min - p_pad, p_max + p_pad],
        title=dict(text="hPa", font=dict(size=9)),
    )
    fig_pres.update_layout(**pres_layout)
    st.plotly_chart(fig_pres, use_container_width=True, config={"displayModeBar": False})

row2_l, row2_r = st.columns(2)

# ── Wind speed — step-line (instantaneous, not interpolated) ─────────────────
with row2_l:
    fig_wind = go.Figure()
    fig_wind.add_trace(go.Scatter(
        x=plot_df["timestamp"], y=plot_df["wind_kmh"],
        mode="lines", line_shape="hv",  # step-after
        line=dict(color=P["line"], width=2),
        fill="tozeroy", fillcolor="rgba(100,116,139,0.08)",
        hovertemplate="Wind: %{y:.1f} km/h",
    ))
    # Beaufort scale reference lines (calm / breeze / fresh thresholds)
    for kmh, label in [(1.5, "Calm"), (19.8, "Fresh Breeze"), (38.9, "Strong Breeze")]:
        fig_wind.add_hline(
            y=kmh,
            line=dict(color=P["border"], width=1, dash="dot"),
            annotation_text=label,
            annotation_font=dict(size=7, color=P["txt_sec"]),
            annotation_position="top right",
        )

    w_layout = base_layout("WIND SPEED", height=200)
    w_layout["yaxis"]["title"] = dict(text="km/h", font=dict(size=9))
    fig_wind.update_layout(**w_layout)
    st.plotly_chart(fig_wind, use_container_width=True, config={"displayModeBar": False})

# ── Cloud cover ONLY (light_pct is its inverse — plotting both is redundant) ─
with row2_r:
    fig_cloud = go.Figure()
    fig_cloud.add_trace(go.Scatter(
        x=plot_df["timestamp"], y=plot_df["cloud_pct"],
        mode="lines", fill="tozeroy",
        fillcolor=P["cloud_fill"],
        line=dict(color=P["txt_sec"], width=2),
        hovertemplate="Cloud Cover: %{y:.0f}%",
    ))
    c_layout = base_layout("CLOUD COVER", height=200)
    c_layout["yaxis"].update(
        range=[0, 100],
        title=dict(text="%", font=dict(size=9)),
        ticksuffix="%",
    )
    fig_cloud.update_layout(**c_layout)
    st.plotly_chart(fig_cloud, use_container_width=True, config={"displayModeBar": False})

# ── Humidity bullet graph ─────────────────────────────────────────────────────
st.markdown('<div class="sec-label">Humidity</div>', unsafe_allow_html=True)

hum_val = float(latest["humidity_pct"])
hum_avg = float(plot_df["humidity_pct"].mean())

fig_hum = go.Figure()

# Background qualitative bands (shapes, not traces — avoids barmode conflicts)
band_defs = [
    (0,  40,  "rgba(51,65,85,0.5)",    "Dry"),
    (40, 70,  "rgba(71,85,105,0.5)",   "Normal"),
    (70, 90,  "rgba(100,116,139,0.5)", "Humid"),
    (90, 100, "rgba(99,102,241,0.2)",  "Saturated"),
]
for x0, x1, color, band_label in band_defs:
    fig_hum.add_shape(
        type="rect",
        x0=x0, x1=x1, y0=0.1, y1=0.9,
        fillcolor=color, line_width=0, layer="below",
    )
    fig_hum.add_annotation(
        x=(x0 + x1) / 2, y=0.5,
        text=band_label,
        showarrow=False,
        font=dict(size=7, color=P["txt_sec"]),
        yref="y",
    )

# Current value bar (narrow, on top of bands)
fig_hum.add_trace(go.Bar(
    x=[hum_val], y=[0.5],
    orientation="h",
    width=0.25,
    marker_color=P["txt_pri"],
    showlegend=False,
    hovertemplate=f"Current: {hum_val:.1f}%",
))

# Session average marker
fig_hum.add_shape(
    type="line",
    x0=hum_avg, x1=hum_avg, y0=0.0, y1=1.0,
    line=dict(color=P["xai_pos"], width=2, dash="dot"),
)
fig_hum.add_annotation(
    x=hum_avg, y=0.95,
    text=f"avg {hum_avg:.0f}%",
    showarrow=False,
    font=dict(size=8, color=P["xai_pos"]),
    yref="y",
)

hum_layout = base_layout(f"  {hum_val:.1f}%  current   ·   {hum_avg:.1f}%  session avg", height=72)
hum_layout.update({
    "xaxis": {
        "range": [0, 100],
        "showgrid": False,
        "ticksuffix": "%",
        "color": P["txt_sec"],
        "tickfont": dict(size=8),
    },
    "yaxis": {
        "showticklabels": False,
        "showgrid": False,
        "range": [0, 1],
        "zeroline": False,
    },
    "margin": dict(l=10, r=12, t=26, b=20),
    "bargap": 0,
})
fig_hum.update_layout(**hum_layout)
st.plotly_chart(fig_hum, use_container_width=True, config={"displayModeBar": False})

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 16. LEVEL 3 — ENGINEERING DIAGNOSTICS (collapsed by default)
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("⚙  Engineering Diagnostics", expanded=False):

    st.markdown('<div class="sec-label">Model Feature Contributions</div>', unsafe_allow_html=True)

    diag_l, diag_r = st.columns(2)

    # ── XAI Diverging Bar Chart ───────────────────────────────────────────────
    with diag_l:
        labels  = [c["label"]        for c in reversed(contributions)]
        values  = [c["contribution"] for c in reversed(contributions)]
        colors  = [P["xai_pos"] if v > 0 else P["xai_neg"] for v in values]
        v_texts = [f"{v:+.3f}" for v in values]

        fig_xai = go.Figure()
        fig_xai.add_trace(go.Bar(
            y=labels, x=values,
            orientation="h",
            marker_color=colors,
            text=v_texts,
            textposition="outside",
            textfont=dict(size=8, color=P["txt_sec"]),
            hovertemplate="%{y}: %{x:+.4f} log-odds<extra></extra>",
        ))
        # Mandatory zero baseline
        fig_xai.add_vline(x=0, line=dict(color=P["border"], width=1.5))

        xai_layout = base_layout(
            "FEATURE CONTRIBUTIONS    (▶ Rain  ·  ◀ Clear)",
            height=280,
        )
        xai_layout["xaxis"].update(
            title=dict(text="log-odds contribution", font=dict(size=8)),
            zeroline=True, zerolinecolor=P["border"],
            showgrid=True, gridcolor=P["grid"],
        )
        fig_xai.update_layout(**xai_layout)
        st.plotly_chart(fig_xai, use_container_width=True, config={"displayModeBar": False})

    # ── Pressure Rate-of-Change ΔP/Δt ────────────────────────────────────────
    with diag_r:
        rate_df = df.tail(300).copy()
        rate_df["dp_dt"] = rate_df["pressure_hpa"].diff().fillna(0)
        dp_colors = [P["xai_pos"] if v < 0 else P["xai_neg"] for v in rate_df["dp_dt"]]

        fig_dpdt = go.Figure()
        fig_dpdt.add_trace(go.Bar(
            x=rate_df["timestamp"],
            y=rate_df["dp_dt"],
            marker_color=dp_colors,
            showlegend=False,
            hovertemplate="ΔP: %{y:.4f} hPa<extra></extra>",
        ))
        fig_dpdt.add_hline(y=0, line=dict(color=P["border"], width=1))
        # Significant-drop threshold marker
        fig_dpdt.add_hline(
            y=-0.05,
            line=dict(color=P["alarm"], width=1, dash="dot"),
            annotation_text="Significant drop",
            annotation_font=dict(size=7, color=P["alarm"]),
            annotation_position="bottom right",
        )

        dpdt_layout = base_layout("PRESSURE RATE OF CHANGE  (ΔP / Δt)", height=280)
        dpdt_layout["yaxis"].update(
            zeroline=True, zerolinecolor=P["border"],
            title=dict(text="hPa / 3 s", font=dict(size=8)),
        )
        fig_dpdt.update_layout(**dpdt_layout)
        st.plotly_chart(fig_dpdt, use_container_width=True, config={"displayModeBar": False})

    # ── Model Internals Table ─────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Model Internals — Z-Scores & Contributions</div>', unsafe_allow_html=True)

    log_odds_sum = MODEL_INTERCEPT + sum(c["contribution"] for c in contributions)
    sigmoid_out  = 1.0 / (1.0 + math.exp(-log_odds_sum))

    debug_rows = []
    for c in contributions:
        debug_rows.append({
            "Feature":        c["label"],
            "Raw Value":      f"{c['raw']:.3f}",
            "Z-Score":        f"{c['z_score']:+.3f}",
            "Coefficient":    f"{c['coef']:+.6f}",
            "Contribution":   f"{c['contribution']:+.4f}",
            "Direction":      "→ Rain" if c["contribution"] > 0 else "→ Clear",
        })

    st.dataframe(pd.DataFrame(debug_rows), use_container_width=True, hide_index=True)

    st.caption(
        f"Intercept: {MODEL_INTERCEPT}  ·  "
        f"Σ log-odds: {log_odds_sum:.4f}  ·  "
        f"σ(Σ) = {sigmoid_out * 100:.2f}%  "
        f"≈  Model probability: **{latest['rain_prob_pct']:.2f}%**"
    )

    # ── Raw Telemetry Log ─────────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Raw Telemetry Log  (last 20 readings)</div>', unsafe_allow_html=True)

    raw_cols = ["timestamp", "temp_c", "humidity_pct", "pressure_hpa",
                "pressure_trend", "wind_kmh", "cloud_pct", "rain_prob_pct"]
    raw_display = df.tail(20)[raw_cols].copy()[::-1]
    raw_display.columns = ["Time", "Temp °C", "Hum %", "Pres hPa",
                           "Trend hPa/h", "Wind km/h", "Cloud %", "Rain %"]
    st.dataframe(raw_display, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# 17. SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### Controls")
    auto_refresh = st.checkbox("Auto-Refresh (3 s)", value=True)
    st.divider()

    st.markdown("### Derived Values")
    c_pri = P["txt_pri"]
    c_sec = P["txt_sec"]
    st.markdown(
        f"<span style='font-family:JetBrains Mono,monospace;font-size:18px;"
        f"color:{c_pri}'>{dew_point_c:.1f} °C</span>"
        f"<br><span style='font-size:11px;color:{c_sec}'>Dew Point</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<span style='font-family:JetBrains Mono,monospace;font-size:18px;"
        f"color:{c_pri}'>{dpd_c:.1f} °C</span>"
        f"<br><span style='font-size:11px;color:{c_sec}'>Depression</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("### Session Stats")
    st.caption(f"Rows stored: {len(df):,}")
    st.caption(f"Last reading: {df['timestamp'].iloc[-1]}")

    st.divider()
    if st.button("Clear All Data", type="secondary"):
        with _csv_lock:
            pd.DataFrame(columns=COLUMNS).to_csv(CSV_FILE, index=False)
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# 18. AUTO-REFRESH
# ─────────────────────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(3)
    st.rerun()
