from pathlib import Path
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextily as ctx

data_dir = Path(__file__).parent / "data"

# =============================================================
# LOAD & FILTER
# =============================================================
events = gpd.read_file(data_dir / "streetlight_crime_events.geojson")
events["days_from_outage_request"] = pd.to_numeric(events["days_from_outage_request"], errors="coerce")
events["time_to_fix"]              = pd.to_numeric(events["time_to_fix"],              errors="coerce")
events["crime_streetlight_outage"] = pd.to_numeric(events["crime_streetlight_outage"], errors="coerce")
events["buffer_radius_m"]          = pd.to_numeric(events["buffer_radius_m"],          errors="coerce")
events["crime_date"]               = pd.to_datetime(events["crime_date"],               errors="coerce")

events = events[events["crime_date"].dt.hour >= 17].copy()
events = events[events["time_to_fix"] <= 365].copy()
events["crime_outside_outage"] = (events["crime_streetlight_outage"] == 0).astype(int)

type_colors = plt.cm.tab10.colors

# 30m buffer + top 10 crime types
ev30 = events[events["buffer_radius_m"] == 30].copy()
ev30["crime_outside_outage"] = (ev30["crime_streetlight_outage"] == 0).astype(int)
top_types = ev30["primary_type"].value_counts().head(10).index.tolist()
ev30 = ev30[ev30["primary_type"].isin(top_types)]

# =============================================================
# TRACT DATA — computed once, shared by plots D and E
# Metric: crimes_during_outage / total_outage_days
#   = crimes per outage-day, controls for both number AND duration
#     of outages in each tract → fully comparable across tracts
# =============================================================
tracts_gdf = gpd.read_file(data_dir / "illinois_tract_income" / "illinois_tract_income.shp")
tracts_gdf = tracts_gdf.to_crs(ev30.crs)

# Include time_to_fix, days_from_outage_request, primary_type for all downstream comparisons
ev30_tracts = gpd.sjoin(
    ev30[["request_id", "crime_streetlight_outage", "time_to_fix",
          "days_from_outage_request", "primary_type", "geometry"]],
    tracts_gdf[["GEOID", "ASQPE001", "geometry"]],
    how="left",
    predicate="within",
).drop(columns=["index_right"], errors="ignore")

# Total crimes during outage per tract
crimes_per_tract = (
    ev30_tracts.groupby("GEOID")["crime_streetlight_outage"].sum()
    .reset_index(name="crimes_during_outage")
)

# Winsorize time_to_fix at 95th percentile before summing outage-days.
# Prevents one very long outage from dominating a tract's denominator
# and making its crime rate appear artificially low.
ttf_cap = ev30_tracts.drop_duplicates(subset=["request_id"])["time_to_fix"].quantile(0.95)
ev30_tracts["time_to_fix_w"] = ev30_tracts["time_to_fix"].clip(upper=ttf_cap)

# Total outage-days per tract (sum of winsorized time_to_fix across unique requests)
outage_days_per_tract = (
    ev30_tracts.drop_duplicates(subset=["GEOID", "request_id"])
    .groupby("GEOID")
    .agg(n_requests=("request_id", "nunique"),
         total_outage_days=("time_to_fix_w", "sum"),
         median_income=("ASQPE001", "first"))
    .reset_index()
)

tract_agg = crimes_per_tract.merge(outage_days_per_tract, on="GEOID")
tract_agg["crime_per_outage_day"] = (
    tract_agg["crimes_during_outage"] / tract_agg["total_outage_days"]
)
# Drop tracts with missing income (coded 0 in shapefile)
tract_agg = tract_agg[tract_agg["median_income"] > 0].copy()

# =============================================================
# SYMMETRIC WINDOW DATA — for plot C
# =============================================================
max_day  = 10
ev30_days = ev30.dropna(subset=["days_from_outage_request"]).copy()
ev30_days["day_bin"] = ev30_days["days_from_outage_request"].round().astype(int)

ev30_sym = ev30_days[ev30_days["day_bin"].abs() <= ev30_days["time_to_fix"]].copy()
ev30_sym = ev30_sym[(ev30_sym["day_bin"] >= -max_day) & (ev30_sym["day_bin"] <= max_day)]

req_ttf  = ev30.drop_duplicates(subset=["request_id"])[["request_id", "time_to_fix"]]
day_bins = list(range(-max_day, max_day + 1))
n_active = {d: int((req_ttf["time_to_fix"] >= abs(d)).sum()) for d in day_bins}

counts5 = ev30_sym.groupby(["primary_type", "day_bin"]).size().reset_index(name="count")
counts5["n_active"] = counts5["day_bin"].map(n_active)
counts5["rate"]     = counts5["count"] / counts5["n_active"]

pivot5 = (
    counts5.pivot(index="primary_type", columns="day_bin", values="rate")
    .reindex(index=top_types, columns=day_bins)
    .fillna(0)
)

# =============================================================
# PLOT C: Symmetric-window small multiples
# =============================================================
y_max          = pivot5.values.max() * 1.1
n_cols, n_rows = 5, 2

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6), sharey=True)
axes = axes.flatten()

for i, ct in enumerate(top_types):
    ax = axes[i]
    if ct in pivot5.index:
        rates = pivot5.loc[ct, day_bins].values
        ax.plot(day_bins, rates, color=type_colors[i % 10], linewidth=1.4)
        ax.fill_between(day_bins, rates, alpha=0.15, color=type_colors[i % 10])
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
    ax.set_title(ct, fontsize=8, pad=3)
    ax.set_ylim(0, y_max)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(labelsize=7)
    if i % n_cols == 0:
        ax.set_ylabel("Crimes / active req / day", fontsize=7)
    ax.set_xlabel("Day", fontsize=7)

fig.suptitle("Symmetric-window nighttime crime rate by day — 30m buffer (shared y-axis)", fontsize=11)
plt.tight_layout()
plt.savefig(data_dir / "plot_C_symmetric_grid.png", dpi=150)
plt.show()

# =============================================================
# PLOT D: Income quintile bar chart
# Divide tracts into 5 income quintiles; show mean ± SE of
# crime_per_outage_day per quintile — cleaner than a scatter
# =============================================================
scatter_df = tract_agg.dropna(subset=["median_income", "crime_per_outage_day"]).copy()
cap_d      = scatter_df["crime_per_outage_day"].quantile(0.99)
scatter_df = scatter_df[scatter_df["crime_per_outage_day"] <= cap_d]

scatter_df["income_quintile"] = pd.qcut(
    scatter_df["median_income"], q=5,
    labels=["Q1\n(lowest)", "Q2", "Q3", "Q4", "Q5\n(highest)"]
)

quintile_stats = (
    scatter_df.groupby("income_quintile", observed=True)["crime_per_outage_day"]
    .agg(mean="mean", se=lambda x: x.sem())
    .reset_index()
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(
    quintile_stats["income_quintile"].astype(str),
    quintile_stats["mean"],
    yerr=quintile_stats["se"],
    color="#1f77b4", alpha=0.8, capsize=5, width=0.55,
)
ax.set_xlabel("Tract income quintile")
ax.set_ylabel("Avg nighttime crimes per outage-day")
ax.set_title("Nighttime outage crime rate by tract income quintile\n(mean ± SE, capped 99th pct)")
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(data_dir / "plot_D_income_quintile_bar.png", dpi=150)
plt.show()

# =============================================================
# BEFORE / DURING COMPARISON SETUP  (used by plots E, G, H)
# Symmetric window: only events where abs(days_from_outage_request) <= time_to_fix
# Both pre and during normalised by same winsorised total_outage_days → comparable
# =============================================================
outage_days_base = outage_days_per_tract[
    ["GEOID", "total_outage_days", "median_income", "n_requests"]
].copy()

def before_during_tract(ev30_tracts_sub, outage_days_df):
    """Symmetric-window pre vs during rates per tract. Returns one row per tract."""
    sym = ev30_tracts_sub[
        ev30_tracts_sub["days_from_outage_request"].abs() <= ev30_tracts_sub["time_to_fix"]
    ].copy()
    pre = sym[sym["crime_streetlight_outage"] == 0].groupby("GEOID").size().reset_index(name="pre_n")
    dur = sym[sym["crime_streetlight_outage"] == 1].groupby("GEOID").size().reset_index(name="dur_n")
    result = outage_days_df.merge(pre, on="GEOID", how="left").merge(dur, on="GEOID", how="left")
    result["pre_n"]          = result["pre_n"].fillna(0)
    result["dur_n"]          = result["dur_n"].fillna(0)
    result["pre_rate"]       = result["pre_n"] / result["total_outage_days"]
    result["dur_rate"]       = result["dur_n"] / result["total_outage_days"]
    result["rate_change"]    = result["dur_rate"] - result["pre_rate"]
    result["crime_increased"] = result["rate_change"] > 0
    return result[result["median_income"] > 0].copy()

tc_all = before_during_tract(ev30_tracts, outage_days_base)

# =============================================================
# PLOT E: Bivariate choropleth — income × change in crime during outage
# Overlaid on Chicago basemap via contextily
#
# Colour scheme (2×2 quadrant) — same metric as binned dose-response:
#   rate_change = during_rate − before_rate (symmetric window, per outage-day)
#   Income split: median; Crime split: 0 (positive = rose, negative = fell)
#
#   Low income  + Crime ↑ → #d7191c  red    (equity concern)
#   Low income  + Crime ↓ → #fdae61  orange (low income but safer)
#   High income + Crime ↓ → #2c7bb6  blue   (affluent, safer during outage)
#   High income + Crime ↑ → #7b3294  purple (wealthy, still exposed)
# =============================================================
inc_med_e = tc_all["median_income"].median()

QUAD_COLORS = {
    ("low_inc",  "crime_up"):   ("#d7191c", "Low income,  Crime ↑ during outage"),
    ("low_inc",  "crime_down"): ("#fdae61", "Low income,  Crime ↓ during outage"),
    ("high_inc", "crime_down"): ("#2c7bb6", "High income, Crime ↓ during outage"),
    ("high_inc", "crime_up"):   ("#7b3294", "High income, Crime ↑ during outage"),
}

def assign_quad(row):
    inc = "low_inc"  if row["median_income"] < inc_med_e else "high_inc"
    chg = "crime_up" if row["rate_change"]   >= 0        else "crime_down"
    color, label = QUAD_COLORS[(inc, chg)]
    return pd.Series({"quad_color": color, "quad_label": label})

tc_e = tc_all.copy()
tc_e[["quad_color", "quad_label"]] = tc_e.apply(assign_quad, axis=1)

# Merge quadrant labels onto tract polygons; project to Web Mercator for basemap
biv_map = (
    tracts_gdf[["GEOID", "geometry"]]
    .merge(tc_e[["GEOID", "quad_color", "quad_label"]], on="GEOID", how="inner")
    .pipe(lambda df: gpd.GeoDataFrame(df, geometry="geometry", crs=tracts_gdf.crs))
    .to_crs(epsg=3857)
)

fig, ax = plt.subplots(figsize=(11, 14))
for color, group in biv_map.groupby("quad_color"):
    group.plot(ax=ax, color=color, edgecolor="white", linewidth=0.1, alpha=0.75)

ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=11)

legend_patches = [
    mpatches.Patch(color=v[0], label=v[1])
    for v in QUAD_COLORS.values()
]
ax.legend(handles=legend_patches, loc="lower left", fontsize=9,
          title="Income × crime change during outage\n(income: median split; crime: 0 = baseline)",
          title_fontsize=9, framealpha=0.9)
ax.set_title(
    "Bivariate: tract income vs change in nighttime crime during streetlight outages\n"
    "(rate_change = during_rate − before_rate; same metric as binned dose-response)",
    fontsize=12
)
ax.set_axis_off()
plt.tight_layout()
plt.savefig(data_dir / "plot_E_bivariate_choropleth.png", dpi=150)
plt.show()

# =============================================================
# PLOT G: Absolute rate change choropleth — excess crimes per outage-day
# Metric: dur_rate − pre_rate = (dur_n − pre_n) / total_outage_days
# This is the relative change weighted by the pre-outage crime rate,
# so sparse tracts (low pre_n) naturally show near-zero without any
# threshold or shrinkage. All tracts shown.
# =============================================================

# rate_change already computed in before_during_tract()
lo = tc_all["rate_change"].quantile(0.05)
hi = tc_all["rate_change"].quantile(0.95)
abs_bound = max(abs(lo), abs(hi))
tc_all["rate_capped"] = tc_all["rate_change"].clip(-abs_bound, abs_bound)

choro = (
    tracts_gdf[["GEOID", "geometry"]]
    .merge(tc_all[["GEOID", "rate_capped"]], on="GEOID", how="inner")
    .pipe(lambda df: gpd.GeoDataFrame(df, geometry="geometry", crs=tracts_gdf.crs))
    .to_crs(epsg=3857)
)

fig, ax = plt.subplots(figsize=(11, 14))
choro.plot(
    ax=ax,
    column="rate_capped",
    cmap="RdBu_r",
    vmin=-abs_bound, vmax=abs_bound,
    edgecolor="white", linewidth=0.1, alpha=0.85,
    legend=True,
    legend_kwds={
        "label": "Excess nighttime crimes per outage-day (during − before)",
        "orientation": "horizontal",
        "shrink": 0.55,
        "pad": 0.02,
    },
)
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=11)
ax.set_title(
    "Nighttime crime change during streetlight outages — by census tract\n"
    "(dur_rate − pre_rate, symmetric window; red = more crime, blue = less)",
    fontsize=11,
)
ax.set_axis_off()
plt.tight_layout()
plt.savefig(data_dir / "plot_G_direction_map_overall.png", dpi=150)
plt.show()

# =============================================================
# PLOT H: Relative-change lollipop — crime mix shift during outages
# X = % change relative to each type's own pre-outage baseline,
# making types directly comparable regardless of base frequency.
# =============================================================
type_stats = []
for ct in top_types:
    sub      = ev30_tracts[ev30_tracts["primary_type"] == ct]
    tc_ct    = before_during_tract(sub, outage_days_base)
    if len(tc_ct) < 5:
        continue
    pre_total = tc_ct["pre_n"].sum()
    dur_total = tc_ct["dur_n"].sum()
    if pre_total <= 0:
        continue
    type_stats.append({
        "crime_type": ct,
        "rel_change": (dur_total - pre_total) / pre_total * 100,
        "is_overall": False,
    })

type_df = pd.DataFrame(type_stats).sort_values("rel_change").reset_index(drop=True)

# Append overall row at the bottom as a reference benchmark
pre_all = tc_all["pre_n"].sum()
dur_all = tc_all["dur_n"].sum()
if pre_all > 0:
    type_df = pd.concat([
        type_df,
        pd.DataFrame([{
            "crime_type": "ALL CRIMES",
            "rel_change": (dur_all - pre_all) / pre_all * 100,
            "is_overall": True,
        }])
    ], ignore_index=True)

x_range = type_df["rel_change"].abs().max()
lbl_off = x_range * 0.05

fig, ax = plt.subplots(figsize=(9, 7))

for i, row in type_df.iterrows():
    if row["is_overall"]:
        color = "#333333"
    else:
        color = "#d7191c" if row["rel_change"] > 0 else "#2c7bb6"
    ax.plot([0, row["rel_change"]], [i, i], color=color, linewidth=2.2, alpha=0.65)
    ax.scatter(row["rel_change"], i, color=color, s=90, zorder=5)
    ha     = "left"  if row["rel_change"] >= 0 else "right"
    offset = lbl_off if row["rel_change"] >= 0 else -lbl_off
    sign   = "+" if row["rel_change"] > 0 else ""
    ax.text(row["rel_change"] + offset, i,
            f"{sign}{row['rel_change']:.1f}%",
            va="center", ha=ha, fontsize=9, color=color, fontweight="bold")

# Separator line between per-type rows and overall
if type_df["is_overall"].any():
    ax.axhline(len(type_df) - 1.5, color="#aaaaaa", linewidth=0.8, linestyle="-")

ax.set_yticks(type_df.index.tolist())
ax.set_yticklabels(type_df["crime_type"].tolist(), fontsize=10)
ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
ax.set_xlabel(
    "% change in city-wide crime count during outage vs before\n"
    "(symmetric window — each type compared to its own pre-outage total)",
    fontsize=10
)
ax.set_title(
    "Outages shift the crime mix, not just overall crime\n"
    "(30 m buffer, nighttime, symmetric window, winsorised outage-days)",
    fontsize=11
)
ax.legend(
    handles=[
        mpatches.Patch(color="#d7191c", label="Crime ↑ during outage"),
        mpatches.Patch(color="#2c7bb6", label="Crime ↓ during outage"),
        mpatches.Patch(color="#333333", label="All crimes (overall)"),
    ],
    fontsize=9
)
ax.grid(axis="x", linestyle="--", alpha=0.3)
ax.set_xlim(
    type_df["rel_change"].min() - x_range * 0.35,
    type_df["rel_change"].max() + x_range * 0.35
)
plt.tight_layout()
plt.savefig(data_dir / "plot_H_crime_type_relative_change.png", dpi=150)
plt.show()
