"""
final_graph.py
Two chosen final graphs — both use the same crime-change metric:

  rate_change = during_rate − before_rate
             = (crimes per outage-day after outage report)
             − (crimes per outage-day before outage report)

Identification: symmetric W=7-day window; restricted to outages with
time_to_fix >= W so both periods have identical length and constant
denominator → no mechanical drop-off.

Graph 1 — Bivariate choropleth  (WHERE does crime rise, and WHO bears it?)
Graph 2 — Binned dose-response  (HOW MUCH does it rise by repair duration?)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextily as ctx
from scipy.stats import sem

data_dir = Path(__file__).parent / "data"
W = 7   # symmetric window in days

# ================================================================
# GRAPH 1 — BIVARIATE CHOROPLETH
# Data source: streetlight_crime_events.geojson
#              + illinois_tract_income shapefile
# ================================================================
print("Graph 1: loading crime events...")

events = gpd.read_file(data_dir / "streetlight_crime_events.geojson")
events["days_from_outage_request"] = pd.to_numeric(events["days_from_outage_request"], errors="coerce")
events["time_to_fix"]              = pd.to_numeric(events["time_to_fix"],              errors="coerce")
events["crime_streetlight_outage"] = pd.to_numeric(events["crime_streetlight_outage"], errors="coerce")
events["buffer_radius_m"]          = pd.to_numeric(events["buffer_radius_m"],          errors="coerce")
events["crime_date"]               = pd.to_datetime(events["crime_date"],               errors="coerce")

# Nighttime only (5 pm onwards), reasonable repair durations
events = events[events["crime_date"].dt.hour >= 17].copy()
events = events[events["time_to_fix"] <= 365].copy()

# 30 m buffer radius
ev30 = events[events["buffer_radius_m"] == 30].copy()

# ----------------------------------------------------------------
# Spatial join to census tract income
# ----------------------------------------------------------------
print("  Joining to tract income shapefile...")
tracts_gdf = gpd.read_file(data_dir / "illinois_tract_income" / "illinois_tract_income.shp")
tracts_gdf = tracts_gdf.to_crs(ev30.crs)

ev30_tracts = gpd.sjoin(
    ev30[["request_id", "crime_streetlight_outage", "time_to_fix",
          "days_from_outage_request", "primary_type", "geometry"]],
    tracts_gdf[["GEOID", "ASQPE001", "geometry"]],
    how="left",
    predicate="within",
).drop(columns=["index_right"], errors="ignore")

# ----------------------------------------------------------------
# Outage-days per tract (winsorised at 95th pct)
# ----------------------------------------------------------------
ttf_cap = ev30_tracts.drop_duplicates(subset=["request_id"])["time_to_fix"].quantile(0.95)
ev30_tracts["time_to_fix_w"] = ev30_tracts["time_to_fix"].clip(upper=ttf_cap)

outage_days_per_tract = (
    ev30_tracts.drop_duplicates(subset=["GEOID", "request_id"])
    .groupby("GEOID")
    .agg(n_requests=("request_id", "nunique"),
         total_outage_days=("time_to_fix_w", "sum"),
         median_income=("ASQPE001", "first"))
    .reset_index()
)

outage_days_base = outage_days_per_tract[
    ["GEOID", "total_outage_days", "median_income", "n_requests"]
].copy()

# ----------------------------------------------------------------
# Before / during rates per tract  (symmetric window)
# ----------------------------------------------------------------
def before_during_tract(ev30_tracts_sub, outage_days_df):
    """Symmetric-window pre vs during rates per tract. Returns one row per tract."""
    sym = ev30_tracts_sub[
        ev30_tracts_sub["days_from_outage_request"].abs() <= ev30_tracts_sub["time_to_fix"]
    ].copy()
    pre = sym[sym["crime_streetlight_outage"] == 0].groupby("GEOID").size().reset_index(name="pre_n")
    dur = sym[sym["crime_streetlight_outage"] == 1].groupby("GEOID").size().reset_index(name="dur_n")
    result = outage_days_df.merge(pre, on="GEOID", how="left").merge(dur, on="GEOID", how="left")
    result["pre_n"]       = result["pre_n"].fillna(0)
    result["dur_n"]       = result["dur_n"].fillna(0)
    result["pre_rate"]    = result["pre_n"] / result["total_outage_days"]
    result["dur_rate"]    = result["dur_n"] / result["total_outage_days"]
    result["rate_change"] = result["dur_rate"] - result["pre_rate"]
    return result[result["median_income"] > 0].copy()

tc_all = before_during_tract(ev30_tracts, outage_days_base)
print(f"  Tracts with rate_change: {len(tc_all)}")

# ----------------------------------------------------------------
# Assign 2×2 quadrant colours
# ----------------------------------------------------------------
inc_med = tc_all["median_income"].median()

QUAD_COLORS = {
    ("low_inc",  "crime_up"):   ("#d7191c", "Low income,  Crime ↑ during outage"),
    ("low_inc",  "crime_down"): ("#fdae61", "Low income,  Crime ↓ during outage"),
    ("high_inc", "crime_down"): ("#2c7bb6", "High income, Crime ↓ during outage"),
    ("high_inc", "crime_up"):   ("#7b3294", "High income, Crime ↑ during outage"),
}

def assign_quad(row):
    inc = "low_inc"  if row["median_income"] < inc_med else "high_inc"
    chg = "crime_up" if row["rate_change"]  >= 0       else "crime_down"
    color, label = QUAD_COLORS[(inc, chg)]
    return pd.Series({"quad_color": color, "quad_label": label})

tc_e = tc_all.copy()
tc_e[["quad_color", "quad_label"]] = tc_e.apply(assign_quad, axis=1)

biv_map = (
    tracts_gdf[["GEOID", "geometry"]]
    .merge(tc_e[["GEOID", "quad_color", "quad_label"]], on="GEOID", how="inner")
    .pipe(lambda df: gpd.GeoDataFrame(df, geometry="geometry", crs=tracts_gdf.crs))
    .to_crs(epsg=3857)
)

# ----------------------------------------------------------------
# Plot
# ----------------------------------------------------------------
print("  Plotting bivariate choropleth...")
fig, ax = plt.subplots(figsize=(11, 14))
for color, group in biv_map.groupby("quad_color"):
    group.plot(ax=ax, color=color, edgecolor="white", linewidth=0.1, alpha=0.75)

ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=11)

legend_patches = [mpatches.Patch(color=v[0], label=v[1]) for v in QUAD_COLORS.values()]
ax.legend(handles=legend_patches, loc="lower left", fontsize=9,
          title="Income × crime change during outage\n(income: city median split; crime: 0 = baseline)",
          title_fontsize=9, framealpha=0.9)
ax.set_title(
    "Bivariate: tract income vs change in nighttime crime during streetlight outages\n"
    "(rate_change = during_rate − before_rate; symmetric 7-day window)",
    fontsize=12,
)
ax.set_axis_off()
plt.tight_layout()
plt.savefig(data_dir / "final_bivariate_choropleth.png", dpi=150)
print("  Saved → final_bivariate_choropleth.png")
plt.show()

# ================================================================
# GRAPH 2 — BINNED DOSE-RESPONSE
# Data source: streetlight_crime_events_with_tracts.geojson
# ================================================================
print("\nGraph 2: loading events with tracts...")

ev2 = gpd.read_file(data_dir / "streetlight_crime_events_with_tracts.geojson")
for c in ["crime_streetlight_outage", "time_to_fix", "days_from_outage_request"]:
    ev2[c] = pd.to_numeric(ev2[c], errors="coerce")

# Qualifying outages: time_to_fix >= W
outage_ttf  = ev2.groupby("service_request_number")["time_to_fix"].first().dropna()
qualifying  = outage_ttf[outage_ttf >= W].index
n_q         = len(qualifying)
print(f"  Qualifying outages (ttf ≥ {W}d): {n_q:,} / {len(outage_ttf):,}")

# Symmetric-window event slice
ev_sym = ev2[
    ev2["service_request_number"].isin(qualifying) &
    (ev2["days_from_outage_request"] >= -W) &
    (ev2["days_from_outage_request"] <   W)
].copy()

# ----------------------------------------------------------------
# Per-outage excess crime rate
# ----------------------------------------------------------------
def per_outage_excess(ev_subset, w=W):
    """For each outage return before_rate, during_rate, excess (during − before)."""
    ev_subset = ev_subset.copy()
    ev_subset["period"] = np.where(ev_subset["days_from_outage_request"] >= 0,
                                   "during", "before")
    counts = (ev_subset.groupby(["service_request_number", "period"])
              .size().unstack(fill_value=0))
    for col in ["before", "during"]:
        if col not in counts.columns:
            counts[col] = 0
    ttf = outage_ttf.reindex(counts.index)
    counts = counts[ttf.notna()].copy()
    counts["before_rate"] = counts["before"] / w
    counts["during_rate"] = counts["during"] / w
    counts["excess"]      = counts["during_rate"] - counts["before_rate"]
    counts["time_to_fix"] = ttf.reindex(counts.index).values
    return counts.dropna(subset=["time_to_fix"])

excess_df = per_outage_excess(ev_sym)

# ----------------------------------------------------------------
# Bin by repair duration (start at 7 to match qualifying threshold;
# cap at 60 days to avoid noisy tail)
# ----------------------------------------------------------------
bins_g   = [7, 14, 30, 60]
labels_g = ["1–2 weeks", "2–4 weeks", "1–2 months"]

po_g = excess_df[excess_df["time_to_fix"] < 60].copy()
po_g["ttf_bin"] = pd.cut(po_g["time_to_fix"], bins=bins_g, labels=labels_g, right=False)

binned = po_g.groupby("ttf_bin", observed=True)["excess"].agg(
    mean="mean", se=sem
).reset_index()

# ----------------------------------------------------------------
# Plot
# ----------------------------------------------------------------
print("  Plotting binned dose-response...")
fig, ax = plt.subplots(figsize=(8, 5))
colors_g = ["#92c5de", "#4393c3", "#2166ac"]

for i, row in binned.iterrows():
    ax.bar(i, row["mean"], color=colors_g[i], edgecolor="white",
           linewidth=0.8, alpha=0.85)
    ax.errorbar(i, row["mean"], yerr=1.96 * row["se"],
                fmt="none", color="#333", linewidth=1.5, capsize=5)

ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5,
           label="Pre-outage baseline")
ax.set_xticks(range(len(binned)))
ax.set_xticklabels(labels_g, fontsize=13)
ax.set_xlabel("How long the streetlight stayed broken", fontsize=12)
ax.set_ylabel("Extra crimes per outage per day\nabove pre-outage baseline", fontsize=12)
ax.set_title(
    "The longer a streetlight stays broken, the more crime rises above baseline\n"
    f"(error bars = 95% CI; symmetric {W}-day before/after window)",
    fontsize=11,
)
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.35)
plt.tight_layout()
plt.savefig(data_dir / "final_binned_dose_response.png", dpi=150)
print("  Saved → final_binned_dose_response.png")
plt.show()

print("\nDone. Both graphs saved to data/")
