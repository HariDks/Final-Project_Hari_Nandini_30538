from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import altair as alt

# -----------------------
# Paths
# -----------------------
data_dir = Path(__file__).parent / "data"
before_file = data_dir / "streetlight_crime_pre_day_buckets_events.geojson"
during_file = data_dir / "streetlight_crime_events.geojson"

# Optional output
out_html = data_dir / "event_time_plot_at_risk_adjusted.html"

# -----------------------
# Load data
# -----------------------
before = gpd.read_file(before_file)
during = gpd.read_file(during_file)

# -----------------------
# Clean / standardize
# -----------------------
for df in [before, during]:
    df["crime_date"] = pd.to_datetime(df["crime_date"], errors="coerce")
    df["creation_date"] = pd.to_datetime(df["creation_date"], errors="coerce")
    df["request_id"] = df["request_id"].astype(str)
    df["buffer_radius_m"] = pd.to_numeric(df["buffer_radius_m"], errors="coerce")

before = before.dropna(subset=["crime_date", "creation_date", "request_id", "buffer_radius_m"]).copy()
during = during.dropna(subset=["crime_date", "creation_date", "request_id", "buffer_radius_m"]).copy()

# completion_date only exists in DURING file
during["completion_date"] = pd.to_datetime(during.get("completion_date"), errors="coerce")

# -----------------------
# Relative day (Day 0 = complaint creation)
# -----------------------
before["relative_day"] = (
    before["crime_date"].dt.floor("D") - before["creation_date"].dt.floor("D")
).dt.days

during["relative_day"] = (
    during["crime_date"].dt.floor("D") - during["creation_date"].dt.floor("D")
).dt.days

# -----------------------
# Window selection
# NOTE: before dataset only supports -5..-1 (because it is a 1–5 day pre bucket file)
# -----------------------
window_min = -5
window_max = 30

before_win = before[(before["relative_day"] >= window_min) & (before["relative_day"] <= -1)].copy()
during_win = during[(during["relative_day"] >= 0) & (during["relative_day"] <= window_max)].copy()

# Combine event rows (crime events)
event_df = pd.concat([before_win, during_win], ignore_index=True)

# -----------------------
# Request-level metadata: days_open per request_id x radius
# days_open = completion_date - creation_date (in days)
# -----------------------
req_meta = (
    during[["request_id", "buffer_radius_m", "creation_date", "completion_date"]]
    .drop_duplicates(subset=["request_id", "buffer_radius_m"])
    .copy()
)

req_meta["creation_day"] = req_meta["creation_date"].dt.floor("D")
req_meta["completion_day"] = req_meta["completion_date"].dt.floor("D")

req_meta["days_open"] = (req_meta["completion_day"] - req_meta["creation_day"]).dt.days

# If completion_date missing, we should NOT pretend it stays open forever.
# Conservative choice: cap at window_max (we only plot up to window_max anyway).
req_meta["days_open_capped"] = req_meta["days_open"]
req_meta.loc[req_meta["days_open_capped"].isna(), "days_open_capped"] = window_max
req_meta["days_open_capped"] = req_meta["days_open_capped"].clip(lower=0, upper=window_max).astype(int)

# -----------------------
# Observed crime counts per request × day
# Use nunique(id) to avoid double-counting the same crime joining multiple times
# -----------------------
observed_counts = (
    event_df
    .groupby(["request_id", "buffer_radius_m", "relative_day"], as_index=False)
    .agg(crime_count=("id", "nunique"))
)

# -----------------------
# Build FULL "at-risk" grid of request × day
# - Pre days: include all requests for days window_min..-1
# - During days: include request only for days 0..min(days_open, window_max)
# -----------------------

# Requests that appear in our analysis universe (paired idea):
# Use requests that exist in req_meta (i.e., have a service window)
requests = req_meta[["request_id", "buffer_radius_m", "days_open_capped"]].drop_duplicates().copy()

# PRE grid (everyone has pre-days)
pre_days = pd.DataFrame({"relative_day": np.arange(window_min, 0)})  # -5..-1
pre_grid = (
    requests[["request_id", "buffer_radius_m"]]
    .assign(key=1)
    .merge(pre_days.assign(key=1), on="key")
    .drop("key", axis=1)
)

# DURING grid (only while at-risk)
# Create rows for each request with relative_day 0..days_open_capped
during_grids = []
for _, row in requests.iterrows():
    rid = row["request_id"]
    rad = row["buffer_radius_m"]
    dmax = int(row["days_open_capped"])
    days = pd.DataFrame({"relative_day": np.arange(0, dmax + 1)})
    days["request_id"] = rid
    days["buffer_radius_m"] = rad
    during_grids.append(days)

during_grid = pd.concat(during_grids, ignore_index=True)

# Full at-risk grid
full_grid = pd.concat([pre_grid, during_grid], ignore_index=True)

# -----------------------
# Merge observed counts and fill zeros
# -----------------------
full_event = full_grid.merge(
    observed_counts,
    on=["request_id", "buffer_radius_m", "relative_day"],
    how="left"
)
full_event["crime_count"] = full_event["crime_count"].fillna(0)

# -----------------------
# Average across AT-RISK requests (correct denominator)
# -----------------------
event_avg = (
    full_event
    .groupby(["buffer_radius_m", "relative_day"], as_index=False)
    .agg(
        avg_crimes_per_request=("crime_count", "mean"),
        n_at_risk=("request_id", "nunique")
    )
)

# -----------------------
# Altair plot
# -----------------------
alt.data_transformers.disable_max_rows()

line = (
    alt.Chart(event_avg)
    .mark_line(point=True)
    .encode(
        x=alt.X("relative_day:Q", title="Days relative to complaint (Day 0 = complaint)"),
        y=alt.Y("avg_crimes_per_request:Q", title="Average crimes per request (per day)"),
        color=alt.Color("buffer_radius_m:N", title="Buffer radius (m)"),
        tooltip=[
            alt.Tooltip("buffer_radius_m:N", title="Radius (m)"),
            alt.Tooltip("relative_day:Q", title="Relative day"),
            alt.Tooltip("avg_crimes_per_request:Q", title="Avg crimes/request-day", format=".4f"),
            alt.Tooltip("n_at_risk:Q", title="number of streetlight requests that are still “active”"),
        ],
    )
    .properties(
        width=750,
        height=420,
        title="Event-Time Plot (At-risk adjusted): Crime intensity around complaint date"
    )
)

rule = (
    alt.Chart(pd.DataFrame({"relative_day": [0]}))
    .mark_rule(strokeDash=[6, 6], color="black")
    .encode(x="relative_day:Q")
)

chart = line + rule
chart.save(out_html)
print(f"Saved: {out_html}")

chart