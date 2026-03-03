"""
dashboard.py
Chicago Streetlight Outages & Crime — Interactive Streamlit Dashboard

Metrics (symmetric K-day windows, rate = crimes / (K * n_requests)):
  A) pre_rate    — crime rate in the K days BEFORE each complaint
  B) during_rate — crime rate in the first K days AFTER each complaint
  C) diff_rate   — during_rate minus pre_rate  (the key comparison)
  D) avg_outage_days — mean time-to-fix for requests in each tract
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely.geometry
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chicago Streetlight Outages & Crime",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Label lookup ─────────────────────────────────────────────────────────────
METRIC_LABELS = {
    "diff_rate":       "Difference (During − Pre)  [crimes/day/request]",
    "during_rate":     "During-window crime rate  [crimes/day/request]",
    "pre_rate":        "Pre-window crime rate  [crimes/day/request]",
    "avg_outage_days": "Avg outage days",
}

METRIC_SHORT = {
    "diff_rate":       "During − Pre",
    "during_rate":     "During rate",
    "pre_rate":        "Pre rate",
    "avg_outage_days": "Avg outage days",
}

# ─── Data loading (runs once, stays in memory) ────────────────────────────────
@st.cache_resource
def load_data():
    # Crime events: one row = one crime within a streetlight buffer
    gdf = gpd.read_file("data/streetlight_crime_events_with_tracts.geojson")

    # Tract polygon geometry for the choropleth (reproject to WGS84)
    tract_polys = (
        gpd.read_file("data/tract_level_crime_summary.geojson")[["tract_geoid", "geometry"]]
        .to_crs(epsg=4326)
    )

    # Pre-compute WGS84 lon/lat for hex layer (avoids repeated CRS conversion)
    pts_wgs = gdf.to_crs(epsg=4326)
    gdf = gdf.copy()
    gdf["lng"] = pts_wgs.geometry.x
    gdf["lat"]  = pts_wgs.geometry.y

    # Drop geometry column → plain DataFrame for fast pandas groupby
    df = pd.DataFrame(gdf.drop(columns="geometry"))

    crime_types = sorted(df["primary_type"].dropna().unique())

    return df, tract_polys, crime_types


CRIME_TYPES = [
    "ARSON",
    "ASSAULT",
    "BATTERY",
    "BURGLARY",
    "CONCEALED CARRY LICENSE VIOLATION",
    "CRIM SEXUAL ASSAULT",
    "CRIMINAL DAMAGE",
    "CRIMINAL SEXUAL ASSAULT",
    "CRIMINAL TRESPASS",
    "DECEPTIVE PRACTICE",
    "GAMBLING",
    "HOMICIDE",
    "INTERFERENCE WITH PUBLIC OFFICER",
    "INTIMIDATION",
    "KIDNAPPING",
    "LIQUOR LAW VIOLATION",
    "MOTOR VEHICLE THEFT",
    "NARCOTICS",
    "OBSCENITY",
    "OFFENSE INVOLVING CHILDREN",
    "OTHER OFFENSE",
    "PROSTITUTION",
    "PUBLIC PEACE VIOLATION",
    "ROBBERY",
    "SEX OFFENSE",
    "STALKING",
    "THEFT",
    "WEAPONS VIOLATION",
]

with st.spinner("Loading data…"):
    events, tract_polys, _ = load_data()


# ─── Tract metric computation ─────────────────────────────────────────────────
def compute_tract_metrics(df: pd.DataFrame, K: int) -> pd.DataFrame:
    """
    Given a filtered events DataFrame, compute per-tract metrics.

    Returns columns:
        tract_geoid, n_requests, avg_outage_days,
        pre_crimes, during_crimes, pre_rate, during_rate, diff_rate
    """
    # Number of unique requests per tract + mean outage length
    req_meta = (
        df.drop_duplicates(subset=["request_id", "tract_geoid"])
        .groupby("tract_geoid", as_index=False)
        .agg(
            n_requests=("request_id", "nunique"),
            avg_outage_days=("time_to_fix", "mean"),
        )
    )

    # Pre-window crimes: [-K, 0)
    pre_agg = (
        df[(df["days_from_outage_request"] >= -K) & (df["days_from_outage_request"] < 0)]
        .groupby("tract_geoid")
        .size()
        .rename("pre_crimes")
        .reset_index()
    )

    # During-window crimes: [0, K)  — symmetric with pre
    during_agg = (
        df[(df["days_from_outage_request"] >= 0) & (df["days_from_outage_request"] < K)]
        .groupby("tract_geoid")
        .size()
        .rename("during_crimes")
        .reset_index()
    )

    m = req_meta.merge(pre_agg, on="tract_geoid", how="left")
    m = m.merge(during_agg, on="tract_geoid", how="left")
    m[["pre_crimes", "during_crimes"]] = m[["pre_crimes", "during_crimes"]].fillna(0)

    m["pre_rate"]    = m["pre_crimes"]    / (K * m["n_requests"])
    m["during_rate"] = m["during_crimes"] / (K * m["n_requests"])
    m["diff_rate"]   = m["during_rate"]   - m["pre_rate"]

    return m


# ─── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Dashboard Controls")
    st.markdown("---")

    radius = st.radio(
        "Buffer radius (m)",
        [15, 30, 50],
        index=1,
        horizontal=True,
        help="Spatial buffer around each streetlight complaint used to count nearby crimes",
    )

    K = st.radio(
        "Symmetric window K (days)",
        [5, 10],
        index=0,
        horizontal=True,
        help=(
            "Pre-window = K days before complaint date\n"
            "During-window = first K days after complaint date\n"
            "(Max pre-window in data is 10 days)"
        ),
    )

    crime_filter = st.selectbox("Crime type", ["All"] + CRIME_TYPES)

    st.markdown("---")

    metric_choice = st.selectbox(
        "Metric to display on map",
        list(METRIC_LABELS.keys()),
        format_func=METRIC_LABELS.get,
    )

    min_requests = st.slider(
        "Min requests per tract",
        min_value=1,
        max_value=20,
        value=3,
        help="Hides tracts with fewer outage complaints — reduces noise from low-N averages",
    )

    st.markdown("---")
    st.markdown("**Hex overlay (During − Pre weight)**")
    show_hex = st.toggle("Show hex overlay", value=False)
    show_3d  = st.toggle("3D extrusion", value=False, disabled=not show_hex)
    st.caption(
        "Hex color = +1 for during-window crimes, −1 for pre-window crimes. "
        "Red areas = more crime during outage than before."
    )

    st.markdown("---")
    st.caption(
        "Data: Chicago 311 streetlight outages × Chicago crime reports, 2011–2018\n\n"
        f"Buffer: {radius} m | K={K} days | Crime: {crime_filter}"
    )


# ─── Apply filters ────────────────────────────────────────────────────────────
df_filtered = events[events["buffer_radius_m"] == radius].copy()
if crime_filter != "All":
    df_filtered = df_filtered[df_filtered["primary_type"] == crime_filter]

tract_metrics  = compute_tract_metrics(df_filtered, K)
tract_filtered = tract_metrics[tract_metrics["n_requests"] >= min_requests].copy()


# ─── Header ───────────────────────────────────────────────────────────────────
st.title("Chicago Streetlight Outages & Crime")
st.markdown(
    f"Comparing crime rates **{K} days before** vs **{K} days after** each streetlight "
    f"outage complaint | Buffer: **{radius} m** | Crime type: **{crime_filter}**"
)
st.markdown("---")


# ─── KPI tiles ────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

total_reqs = int(df_filtered.drop_duplicates("request_id")["request_id"].nunique())
k1.metric(
    "Total requests",
    f"{total_reqs:,}",
    help="Unique streetlight outage complaints after radius and crime-type filter",
)
k2.metric(
    "Tracts in view",
    f"{len(tract_filtered):,}",
    help=f"Census tracts with >= {min_requests} requests",
)
k3.metric(
    "Mean outage days",
    f"{tract_filtered['avg_outage_days'].mean():.1f}",
    help="Mean days from complaint to resolution, averaged across tracts",
)

mean_diff = tract_filtered["diff_rate"].mean()
k4.metric(
    "Mean during − pre",
    f"{mean_diff:+.4f}",
    delta=f"{mean_diff:+.4f}",
    delta_color="inverse",
    help="Positive = more crime during the outage window than before it",
)

st.markdown("---")


# ─── Map ──────────────────────────────────────────────────────────────────────
st.subheader(f"Census tract map — {METRIC_SHORT[metric_choice]}")

tract_map = tract_polys.merge(tract_filtered, on="tract_geoid", how="inner")

if len(tract_map) == 0:
    st.warning("No tracts match the current filters. Try reducing 'Min requests per tract'.")
else:
    # --- Color scale ---------------------------------------------------------
    col_vals = tract_map[metric_choice].values

    if metric_choice == "diff_rate":
        # Diverging: blue (negative) → white (zero) → red (positive)
        abs_max = max(
            abs(float(np.nanquantile(col_vals, 0.05))),
            abs(float(np.nanquantile(col_vals, 0.95))),
            1e-9,
        )

        def get_fill_color(v: float):
            n = max(-1.0, min(1.0, float(v) / abs_max))
            if n >= 0:
                # white → red
                r, g, b = 220, int(220 * (1 - n)), int(220 * (1 - n))
            else:
                # white → blue
                r, g, b = int(220 * (1 + n)), int(220 * (1 + n)), 220
            return [r, g, b, 175]

    else:
        # Sequential: light grey → dark blue
        vmin = float(np.nanquantile(col_vals, 0.05))
        vmax = float(np.nanquantile(col_vals, 0.95))
        vrange = vmax - vmin if vmax > vmin else 1e-9

        def get_fill_color(v: float):
            n = max(0.0, min(1.0, (float(v) - vmin) / vrange))
            r = int(240 - 200 * n)
            g = int(240 - 200 * n)
            b = int(100 + 155 * n)
            return [r, g, b, 175]

    # --- Build GeoJSON with embedded colors ----------------------------------
    features = []
    for _, row in tract_map.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue
        features.append({
            "type": "Feature",
            "geometry": shapely.geometry.mapping(row.geometry),
            "properties": {
                "tract_geoid":     str(row["tract_geoid"]),
                "n_requests":      int(row["n_requests"]),
                "avg_outage_days": round(float(row["avg_outage_days"]), 1),
                "pre_rate":        round(float(row["pre_rate"]), 5),
                "during_rate":     round(float(row["during_rate"]), 5),
                "diff_rate":       round(float(row["diff_rate"]), 5),
                "fill_color":      get_fill_color(row[metric_choice]),
            },
        })

    geojson_data = {"type": "FeatureCollection", "features": features}

    # --- Layers --------------------------------------------------------------
    layers = [
        pdk.Layer(
            "GeoJsonLayer",
            geojson_data,
            pickable=True,
            stroked=True,
            filled=True,
            get_fill_color="properties.fill_color",
            get_line_color=[80, 80, 80, 120],
            line_width_min_pixels=0.5,
        )
    ]

    if show_hex:
        # Weight crimes: +1 during window, -1 pre window
        during_mask = (
            (df_filtered["days_from_outage_request"] >= 0) &
            (df_filtered["days_from_outage_request"] < K)
        )
        pre_mask = (
            (df_filtered["days_from_outage_request"] >= -K) &
            (df_filtered["days_from_outage_request"] < 0)
        )
        df_hex = df_filtered[during_mask | pre_mask][["lng", "lat", "days_from_outage_request"]].copy()
        df_hex["weight"] = np.where(
            df_hex["days_from_outage_request"] >= 0, 1, -1
        )
        hex_pts = df_hex[["lng", "lat"]].to_dict("records")

        layers.append(
            pdk.Layer(
                "HexagonLayer",
                hex_pts,
                get_position=["lng", "lat"],
                radius=300,
                coverage=0.85,
                extruded=show_3d,
                elevation_scale=60 if show_3d else 0,
                elevation_range=[0, 800],
                pickable=True,
                color_range=[
                    [1,   152, 189],
                    [73,  227, 206],
                    [216, 254, 181],
                    [254, 237, 177],
                    [254, 173, 84],
                    [209, 55,  78],
                ],
                auto_highlight=True,
            )
        )

    view = pdk.ViewState(
        latitude=41.83,
        longitude=-87.68,
        zoom=10,
        pitch=40 if (show_hex and show_3d) else 0,
    )

    tooltip_html = (
        "<b>Tract:</b> {properties.tract_geoid}<br/>"
        "<b>Requests (n):</b> {properties.n_requests}<br/>"
        "<b>Avg outage days:</b> {properties.avg_outage_days}<br/>"
        "<b>Pre rate:</b> {properties.pre_rate} crimes/day/req<br/>"
        "<b>During rate:</b> {properties.during_rate} crimes/day/req<br/>"
        "<b>Diff (During−Pre):</b> {properties.diff_rate}"
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=view,
            tooltip={
                "html": tooltip_html,
                "style": {
                    "backgroundColor": "#1a1a2e",
                    "color": "white",
                    "fontSize": "13px",
                    "padding": "8px",
                },
            },
            map_style="mapbox://styles/mapbox/light-v9",
        ),
        height=520,
    )

    # Legend note
    if metric_choice == "diff_rate":
        st.caption(
            "Color: blue = less crime during outage than before (protective effect) | "
            "red = more crime during outage (harmful effect) | "
            "white = no change"
        )
    else:
        st.caption(
            f"Color: light = low {METRIC_SHORT[metric_choice]}  →  dark blue = high. "
            f"5th–95th percentile color range."
        )

st.markdown("---")


# ─── Supporting charts ────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader(f"Top 10 tracts — {METRIC_SHORT[metric_choice]}")

    top10 = (
        tract_filtered
        .nlargest(10, metric_choice)
        [["tract_geoid", metric_choice, "n_requests"]]
        .copy()
    )
    top10["label"] = top10["tract_geoid"] + "  (n=" + top10["n_requests"].astype(str) + ")"

    color_scale = "RdBu_r" if metric_choice == "diff_rate" else "Blues"

    fig_bar = px.bar(
        top10,
        x=metric_choice,
        y="label",
        orientation="h",
        color=metric_choice,
        color_continuous_scale=color_scale,
        labels={
            metric_choice: METRIC_LABELS[metric_choice],
            "label": "Tract  (n = requests)",
        },
        height=380,
    )
    fig_bar.update_layout(
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
        margin=dict(l=10, r=20, t=20, b=20),
        plot_bgcolor="white",
    )
    if metric_choice == "diff_rate":
        fig_bar.add_vline(x=0, line_color="black", line_width=1)

    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    st.subheader(f"Tract distribution — {METRIC_SHORT[metric_choice]}")

    mean_val   = tract_filtered[metric_choice].mean()
    median_val = tract_filtered[metric_choice].median()

    fig_hist = px.histogram(
        tract_filtered,
        x=metric_choice,
        nbins=40,
        color_discrete_sequence=["#4477AA"],
        labels={metric_choice: METRIC_LABELS[metric_choice]},
        height=380,
    )
    fig_hist.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="#CC3333",
        annotation_text=f"Mean: {mean_val:.4f}",
        annotation_position="top right",
    )
    fig_hist.add_vline(
        x=median_val,
        line_dash="dot",
        line_color="#555555",
        annotation_text=f"Median: {median_val:.4f}",
        annotation_position="top left",
    )
    if metric_choice == "diff_rate":
        fig_hist.add_vline(x=0, line_color="black", line_width=1.5)

    fig_hist.update_layout(
        margin=dict(l=10, r=10, t=20, b=20),
        plot_bgcolor="white",
        yaxis_gridcolor="#eeeeee",
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# ─── Scatter: outage days vs diff_rate ────────────────────────────────────────
st.markdown("---")
st.subheader("Outage duration vs crime rate difference")

fig_scatter = px.scatter(
    tract_filtered,
    x="avg_outage_days",
    y="diff_rate",
    size="n_requests",
    color="diff_rate",
    color_continuous_scale="RdBu_r",
    hover_data={"tract_geoid": True, "n_requests": True, "avg_outage_days": ":.1f", "diff_rate": ":.5f"},
    labels={
        "avg_outage_days": "Avg outage days",
        "diff_rate": "During − Pre  [crimes/day/request]",
        "n_requests": "# Requests",
    },
    opacity=0.7,
    height=400,
)
fig_scatter.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
fig_scatter.update_layout(
    coloraxis_showscale=False,
    plot_bgcolor="white",
    xaxis_gridcolor="#eeeeee",
    yaxis_gridcolor="#eeeeee",
    margin=dict(l=10, r=10, t=20, b=20),
)
st.plotly_chart(fig_scatter, use_container_width=True)
st.caption(
    "Bubble size = number of requests in tract. "
    "Above zero line = more crime during outage than before. "
    "Hover over a bubble to see tract details."
)


# ─── Data table ───────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("View full tract data table"):
    display_cols = [
        "tract_geoid", "n_requests", "avg_outage_days",
        "pre_crimes", "during_crimes",
        "pre_rate", "during_rate", "diff_rate",
    ]
    tbl = (
        tract_filtered[display_cols]
        .sort_values(metric_choice, ascending=False)
        .reset_index(drop=True)
    )
    tbl = tbl.rename(columns={
        "tract_geoid":     "Tract GEOID",
        "n_requests":      "Requests (n)",
        "avg_outage_days": "Avg outage days",
        "pre_crimes":      f"Pre crimes (K={K})",
        "during_crimes":   f"During crimes (K={K})",
        "pre_rate":        "Pre rate",
        "during_rate":     "During rate",
        "diff_rate":       "Diff (During−Pre)",
    })
    st.dataframe(
        tbl.style.format({
            "Avg outage days":    "{:.1f}",
            "Pre rate":           "{:.5f}",
            "During rate":        "{:.5f}",
            "Diff (During−Pre)":  "{:+.5f}",
        }),
        use_container_width=True,
    )
    st.caption(
        f"Rates = crimes / (K={K} days × n_requests). "
        "Showing only tracts with >= min_requests filter."
    )
