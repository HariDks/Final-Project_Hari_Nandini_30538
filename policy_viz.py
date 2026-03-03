"""
policy_viz.py  —  Policy visualizations: streetlight outages × crime, Chicago 2011–2018

Figures:
  A. Equity scatter      — income (p10–p90) vs outage-crime share per tract
  B. Equity choropleth   — per-tract rate change (during − before), symmetric window
  C. Dose-response       — time_to_fix vs excess crime rate per outage (during−before/day)
  D. Event study grid    — symmetric event study overall + by income quintile (6 panels)
  F. Seasonal heatmap    — (during − before) rate change by month × hour
  G. Binned dose-response— excess crime rate by time_to_fix bin (bar + SE), no regression
  H. LOESS scatter       — time_to_fix vs excess crime rate + LOESS smoother

Identification note for C/D/G/H:
  Symmetric W=7-day window; restricted to outages with time_to_fix ≥ W.
  Denominator constant throughout → no mechanical drop-off.
  Excess = during_rate − before_rate (both measured in crimes per outage per day).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import contextily as ctx
from scipy import stats
from scipy.stats import sem
from statsmodels.nonparametric.smoothers_lowess import lowess

data_dir = Path(__file__).parent / "data"
W = 7   # symmetric window in days (1-week SLA threshold)

# ================================================================
# LOAD DATA
# ================================================================
print("Loading data...")

# Events WITH tract income (use the pre-joined file)
events = gpd.read_file(data_dir / "streetlight_crime_events_with_tracts.geojson")
for c in ["crime_streetlight_outage", "time_to_fix", "days_from_outage_request", "year",
          "median_income_estimate"]:
    events[c] = pd.to_numeric(events[c], errors="coerce")
events["crime_date"] = pd.to_datetime(events["crime_date"], errors="coerce")
events["month"] = events["crime_date"].dt.month
events["hour"]  = events["crime_date"].dt.hour

# Tract-level summaries (for A and B base geometries)
tracts = gpd.read_file(data_dir / "tract_level_crime_summary.geojson")
tracts["median_income_estimate"] = pd.to_numeric(tracts["median_income_estimate"],
                                                  errors="coerce")
tracts["outage_share"] = tracts["crimes_during_outage"] / tracts["total_crime_events"]

# ----------------------------------------------------------------
# Qualifying outages: time_to_fix >= W (symmetric window is valid)
# ----------------------------------------------------------------
outage_ttf = events.groupby("service_request_number")["time_to_fix"].first().dropna()
qualifying  = outage_ttf[outage_ttf >= W].index
n_q = len(qualifying)
print(f"  Events: {len(events):,} | Qualifying outages (ttf≥{W}d): {n_q:,}/{len(outage_ttf):,}")

# Symmetric-window event slice
ev_sym = events[
    events["service_request_number"].isin(qualifying) &
    (events["days_from_outage_request"] >= -W) &
    (events["days_from_outage_request"] <  W)
].copy()

# ----------------------------------------------------------------
# Per-outage excess crime rate  (used by C, G, H)
# ----------------------------------------------------------------
def per_outage_excess(ev_subset, w=W):
    """For each outage in ev_subset return before_rate, during_rate, excess."""
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

# ================================================================
# VIZ A — Equity scatter  (income p10–p90, no outliers)
# ================================================================
print("Viz A …")

tr_a = tracts.dropna(subset=["median_income_estimate", "outage_share"])
tr_a = tr_a[tr_a["median_income_estimate"] > 0]
p10, p90 = tr_a["median_income_estimate"].quantile([0.10, 0.90])
tr_a = tr_a[(tr_a["median_income_estimate"] >= p10) &
            (tr_a["median_income_estimate"] <= p90)].copy()

x_a = tr_a["median_income_estimate"] / 1000
y_a = tr_a["outage_share"] * 100
slope, intercept, r, p, _ = stats.linregress(x_a, y_a)
xl = np.linspace(x_a.min(), x_a.max(), 200)

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(x_a, y_a, s=tr_a["total_crime_events"] / 2,
           color="#4393c3", alpha=0.55, edgecolors="white", linewidths=0.4, zorder=2)
ax.plot(xl, intercept + slope * xl, color="#b2182b", linewidth=2,
        label=f"OLS  r = {r:.2f},  p = {p:.3f}")
for sz, lbl in [(25, "100 events"), (75, "300 events"), (150, "600 events")]:
    ax.scatter([], [], s=sz, color="#4393c3", alpha=0.55,
               edgecolors="white", linewidths=0.4, label=lbl)
ax.set_xlabel("Tract median household income ($000s)", fontsize=12)
ax.set_ylabel("% of crimes occurring during active outage", fontsize=12)
ax.set_title("Equity: lower-income tracts bear a greater share of outage-linked crime\n"
             f"(p10–p90 income range; dot size ∝ total crime events; n = {len(tr_a)} tracts)",
             fontsize=11)
ax.legend(fontsize=9, loc="upper right", framealpha=0.85)
ax.grid(linestyle="--", alpha=0.35)
plt.tight_layout()
plt.savefig(data_dir / "policy_A_equity_scatter.png", dpi=150)
plt.close()

# ================================================================
# VIZ B — Choropleth: per-tract rate change (during − before)
# ================================================================
print("Viz B …")

# Compute per-tract before/during crime rates from symmetric window
tract_rates = (
    ev_sym.groupby(["tract_geoid", "crime_streetlight_outage"])
    .size().unstack(fill_value=0)
    .rename(columns={0.0: "before", 1.0: "before_err",  # will fix below
                     **{k: "before" if k == 0 else "during"
                        for k in [0, 1]}})
)
# Cleaner approach: groupby period label
ev_sym_b = ev_sym.copy()
ev_sym_b["period"] = np.where(ev_sym_b["days_from_outage_request"] >= 0, "during", "before")
tract_period = (ev_sym_b.groupby(["tract_geoid", "period"]).size()
                .unstack(fill_value=0))
for col in ["before", "during"]:
    if col not in tract_period.columns:
        tract_period[col] = 0

# Outages per tract (for denominator)
outage_tract = (ev_sym_b[ev_sym_b["service_request_number"].isin(qualifying)]
                .groupby("tract_geoid")["service_request_number"].nunique()
                .rename("n_outages"))
tract_period = tract_period.join(outage_tract, how="left")
tract_period["n_outages"] = tract_period["n_outages"].fillna(1).clip(lower=1)

tract_period["before_rate"] = tract_period["before"] / (tract_period["n_outages"] * W)
tract_period["during_rate"] = tract_period["during"] / (tract_period["n_outages"] * W)
tract_period["rate_change"] = tract_period["during_rate"] - tract_period["before_rate"]

tracts_b = tracts.merge(tract_period[["rate_change"]].reset_index(),
                         on="tract_geoid", how="left")
tracts_b = tracts_b.to_crs(epsg=3857)

vabs = tracts_b["rate_change"].abs().quantile(0.95)
divnorm = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)

fig, ax = plt.subplots(figsize=(10, 13))
tracts_b.plot(column="rate_change", ax=ax, cmap="RdBu_r", norm=divnorm,
               edgecolor="white", linewidth=0.3, legend=True,
               legend_kwds={"label": "Δ crime rate (during − before), per outage-day",
                            "shrink": 0.55})
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=11)
ax.set_axis_off()
ax.set_title("Chicago — Change in crime rate during vs before streetlight outage\n"
             f"(red = more crime during outage; symmetric {W}-day window, "
             f"n = {n_q:,} qualifying outages)",
             fontsize=11)
plt.tight_layout()
plt.savefig(data_dir / "policy_B_equity_choropleth.png", dpi=150)
plt.close()

# ================================================================
# VIZ C — Dose-response: time_to_fix vs excess crime rate + LOESS
# ================================================================
print("Viz C …")

cap = excess_df["time_to_fix"].quantile(0.97)
po_c = excess_df[excess_df["time_to_fix"] <= cap].copy()

# LOESS smoother
loess_out = lowess(po_c["excess"], po_c["time_to_fix"], frac=0.4, return_sorted=True)

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(po_c["time_to_fix"], po_c["excess"],
           color="#4393c3", s=10, alpha=0.35, zorder=2, label="One outage")
ax.plot(loess_out[:, 0], loess_out[:, 1], color="#b2182b", linewidth=2.2,
        label="LOESS smoother")
ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
ax.axvline(W, color="#f4a582", linewidth=1.5, linestyle="--",
           label=f"{W}-day repair target")
ax.set_xlabel("Outage duration — days until repaired (time to fix)", fontsize=12)
ax.set_ylabel("Excess crime rate\n(crimes/day during − crimes/day before)", fontsize=12)
ax.set_title("Dose-response: longer outages generate more excess crime per day\n"
             "(each dot = one outage; excess = during rate − before rate; "
             "top 3% durations removed)",
             fontsize=11)
ax.legend(fontsize=10)
ax.grid(linestyle="--", alpha=0.35)
plt.tight_layout()
plt.savefig(data_dir / "policy_C_dose_response.png", dpi=150)
plt.close()

# ================================================================
# VIZ D — Event study: overall + 5 income quintile grid
# ================================================================
print("Viz D …")

# Assign income quintile to each outage via the events file
outage_income = (events[events["service_request_number"].isin(qualifying)]
                 .groupby("service_request_number")["median_income_estimate"]
                 .first().dropna())
quintile_edges = outage_income.quantile([0, .2, .4, .6, .8, 1.0]).values
quintile_labels = ["Q1\n(lowest income)", "Q2", "Q3", "Q4", "Q5\n(highest income)"]

def get_quintile(inc):
    for i, (lo, hi) in enumerate(zip(quintile_edges[:-1], quintile_edges[1:])):
        if lo <= inc <= hi:
            return i
    return np.nan

outage_income_q = outage_income.copy()
quintiles_series = outage_income.apply(get_quintile)

def event_study_panel(ax, outage_subset, title, n_label):
    ev_panel = ev_sym[ev_sym["service_request_number"].isin(outage_subset)].copy()
    n_panel  = len(outage_subset)
    ev_panel["day_bin"] = ev_panel["days_from_outage_request"].apply(
        lambda x: int(np.floor(x))).clip(-W, W - 1)
    rate = ev_panel.groupby("day_bin").size() / n_panel
    rate = rate.reindex(range(-W, W), fill_value=0)

    pre_avg    = rate[rate.index < 0].mean()
    during_avg = rate[rate.index >= 0].mean()
    pct        = (during_avg - pre_avg) / pre_avg * 100 if pre_avg > 0 else 0

    colors = ["#92c5de" if d < 0 else "#d6604d" for d in rate.index]
    ax.bar(rate.index, rate.values, color=colors, alpha=0.75, width=0.8)
    ax.axvline(-0.5, color="black", linewidth=1.2, linestyle="--")
    ax.axhline(pre_avg,    color="#4393c3", linewidth=1.2, linestyle=":",
               label=f"Before {pre_avg:.3f}")
    ax.axhline(during_avg, color="#b2182b", linewidth=1.2, linestyle=":",
               label=f"During {during_avg:.3f} ({pct:+.1f}%)")
    ax.set_title(f"{title}\n(n = {n_panel:,} outages)", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_ylim(bottom=0)

fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=False)
axes_flat = axes.flatten()

# Overall (top-left)
event_study_panel(axes_flat[0], qualifying, "Overall", n_q)

# Quintile panels
for qi in range(5):
    q_outages = quintiles_series[quintiles_series == qi].index
    q_outages = q_outages.intersection(qualifying)
    event_study_panel(axes_flat[qi + 1], q_outages, quintile_labels[qi], len(q_outages))

for ax in axes_flat:
    ax.set_xlabel("Days rel. to outage report", fontsize=8)
    ax.set_ylabel("Crimes / outage / day", fontsize=8)

fig.suptitle(
    f"Event study: crime rate {W} days before vs {W} days after outage report\n"
    f"Overall and by tract median income quintile  "
    f"(symmetric window, qualifying outages only)",
    fontsize=12, y=1.01,
)
plt.tight_layout()
plt.savefig(data_dir / "policy_D_event_study.png", dpi=150, bbox_inches="tight")
plt.close()

# ================================================================
# VIZ F — Seasonal heatmap: Δ crime rate by month × hour
# ================================================================
print("Viz F …")

ev_sym_f = ev_sym.copy()
ev_sym_f["period"] = np.where(ev_sym_f["days_from_outage_request"] >= 0, "during", "before")

def month_hour_rate(period_label):
    sub = ev_sym_f[ev_sym_f["period"] == period_label]
    return (sub.groupby(["month", "hour"]).size()
              .unstack(fill_value=0)
              .reindex(index=range(1, 13), columns=range(24), fill_value=0)
            / n_q)   # crimes per outage (constant denom)

before_mh = month_hour_rate("before")
during_mh = month_hour_rate("during")
delta_mh  = during_mh - before_mh   # Δ rate per outage per (month, hour) cell

vabs_f = np.abs(delta_mh.values).max()
month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(delta_mh.values, aspect="auto", cmap="RdBu_r",
               vmin=-vabs_f, vmax=vabs_f, interpolation="nearest", origin="upper")
ax.set_xticks(range(24))
ax.set_xticklabels([f"{h:02d}h" for h in range(24)], rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(12))
ax.set_yticklabels(month_labels, fontsize=10)
ax.axvspan(16.5, 23.5, color="#333", alpha=0.07, label="Night 5pm–5am")
ax.axvspan(-0.5, 4.5, color="#333", alpha=0.07)
cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Δ crimes per outage  (during − before)", fontsize=10)
ax.set_xlabel("Hour of day", fontsize=12)
ax.set_ylabel("Month", fontsize=12)
ax.set_title(
    f"Seasonal pattern: when is crime rate elevated during outages? (2011–2018)\n"
    f"(red = more crime during outage than before; symmetric {W}-day window, "
    f"n = {n_q:,} outages)",
    fontsize=11,
)
ax.legend(fontsize=9, loc="upper left")
plt.tight_layout()
plt.savefig(data_dir / "policy_F_seasonal_heatmap.png", dpi=150)
plt.close()

# ================================================================
# VIZ G — Binned dose-response (bar + SE, no regression)
# ================================================================
print("Viz G …")

# Bins start at 7 (matching qualifying threshold); drop 60d+ (too few, noisy)
bins_g   = [7, 14, 30, 60]
labels_g = ["1–2 weeks", "2–4 weeks", "1–2 months"]
po_g = excess_df[excess_df["time_to_fix"] < 60].copy()
po_g["ttf_bin"] = pd.cut(po_g["time_to_fix"], bins=bins_g, labels=labels_g, right=False)

binned = po_g.groupby("ttf_bin", observed=True)["excess"].agg(
    mean="mean", se=sem
).reset_index()

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
    "(error bars = 95% CI; each bar = outages of that duration; "
    f"symmetric {W}-day before/after window)",
    fontsize=11,
)
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.35)
plt.tight_layout()
plt.savefig(data_dir / "policy_G_binned_dose_response.png", dpi=150)
plt.close()

# ================================================================
# VIZ H — Scatter + LOESS: within-outage excess crime rate
# ================================================================
print("Viz H …")

# (Same as C but larger, more annotated — standalone figure)
loess_h = lowess(po_c["excess"], po_c["time_to_fix"], frac=0.35, return_sorted=True)
overall_mean = po_c["excess"].mean()

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(po_c["time_to_fix"], po_c["excess"],
           color="#4393c3", s=12, alpha=0.3, zorder=2, label="One outage event")
ax.plot(loess_h[:, 0], loess_h[:, 1], color="#b2182b", linewidth=2.5,
        label="LOESS smoother")
ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5, label="No effect")
ax.axhline(overall_mean, color="#f4a582", linewidth=1.5, linestyle="-.",
           label=f"Overall mean excess ({overall_mean:.3f})")
ax.axvline(W, color="#888", linewidth=1.2, linestyle=":",
           label=f"{W}-day repair target")
ax.set_xlabel("Outage duration — days until repaired", fontsize=12)
ax.set_ylabel("Excess crime rate per day\n(during − before, within same outage event)", fontsize=12)
ax.set_title(
    "LOESS: relationship between outage duration and excess crime rate\n"
    "(non-parametric — reveals whether effect is linear, threshold, or flat)",
    fontsize=11,
)
ax.legend(fontsize=9)
ax.grid(linestyle="--", alpha=0.35)
plt.tight_layout()
plt.savefig(data_dir / "policy_H_loess_scatter.png", dpi=150)
plt.close()

# ================================================================
# VIZ I + J — Δ crime rate by type × income group
#   I: Diverging heatmap  (rows = crime type, cols = income group)
#   J: Slope graph        (x = income group, one line per crime type)
# ================================================================
print("Viz I + J …")

# ── income quintile assignment (5 bins) ──────────────────────────
quintile_labels_ij = ["Q1\n(lowest)", "Q2", "Q3", "Q4", "Q5\n(highest)"]
outage_income_group = pd.qcut(
    outage_income, q=5, labels=quintile_labels_ij, duplicates="drop"
)
income_order = quintile_labels_ij[:len(outage_income_group.cat.categories)]

# counts of qualifying outages per quintile (constant denominator)
n_req_grp = outage_income_group.value_counts().sort_index()

# ── crime type grouping ──────────────────────────────────────────
top8 = events["primary_type"].value_counts().head(8).index.tolist()

ev_ij = ev_sym.copy()
ev_ij["income_group"] = ev_ij["service_request_number"].map(outage_income_group)
ev_ij["crime_grp"]    = ev_ij["primary_type"].where(ev_ij["primary_type"].isin(top8), "Other")
ev_ij["period"]       = np.where(ev_ij["days_from_outage_request"] >= 0, "post", "pre")
ev_ij = ev_ij.dropna(subset=["income_group"])
ev_ij["income_group"] = ev_ij["income_group"].astype(str)

# ── rates per (income_group, crime_grp, period) ──────────────────
counts_ij = (
    ev_ij.groupby(["income_group", "crime_grp", "period"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)
for col in ["pre", "post"]:
    if col not in counts_ij.columns:
        counts_ij[col] = 0

counts_ij["n_req"]      = counts_ij["income_group"].map(n_req_grp)
counts_ij["pre_rate"]   = counts_ij["pre"]  / (W * counts_ij["n_req"])
counts_ij["post_rate"]  = counts_ij["post"] / (W * counts_ij["n_req"])
counts_ij["delta_rate"] = counts_ij["post_rate"] - counts_ij["pre_rate"]

pivot_ij = (
    counts_ij.pivot(index="crime_grp", columns="income_group", values="delta_rate")
    .reindex(columns=income_order)
)
# sort rows: largest average Δ at top
pivot_ij = pivot_ij.loc[pivot_ij.mean(axis=1).sort_values(ascending=False).index]
n_dec = len(income_order)

# ── VIZ I: Faceted bar chart — one panel per quintile ────────────
# Sort crime types by overall mean delta (same order across all panels)
type_order = pivot_ij.mean(axis=1).sort_values(ascending=True).index.tolist()
vabs       = np.nanmax(np.abs(pivot_ij.values))
bar_colors = ["#d6604d" if v >= 0 else "#4393c3" for v in
              pivot_ij.reindex(type_order).mean(axis=1)]

fig, axes = plt.subplots(1, n_dec, figsize=(16, 6), sharey=True, sharex=True)

for j, (qname, ax) in enumerate(zip(income_order, axes)):
    vals   = pivot_ij.reindex(type_order)[qname].values
    colors = ["#d6604d" if v >= 0 else "#4393c3" for v in vals]
    ax.barh(range(len(type_order)), vals, color=colors, edgecolor="white",
            linewidth=0.5, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title(qname, fontsize=11, fontweight="bold")
    ax.set_xlim(-vabs * 1.15, vabs * 1.15)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.tick_params(axis="x", labelsize=7)
    if j == 0:
        ax.set_yticks(range(len(type_order)))
        ax.set_yticklabels(type_order, fontsize=10)

# Shared x-label via figure text
fig.text(0.5, 0.01, "Δ crime rate  (post − pre, crimes / outage-day)",
         ha="center", fontsize=11)

# Shared legend patches
red_p  = mpatches.Patch(color="#d6604d", alpha=0.85, label="Rate rises during outage")
blue_p = mpatches.Patch(color="#4393c3", alpha=0.85, label="Rate falls during outage")
fig.legend(handles=[red_p, blue_p], loc="lower center", ncol=2,
           fontsize=10, bbox_to_anchor=(0.5, 0.06))

fig.suptitle(
    f"Crime type shifts during streetlight outages — by income quintile\n"
    f"(Q1 = lowest income, Q5 = highest; symmetric {W}-day window, "
    f"n = {n_q:,} qualifying outages)",
    fontsize=12, y=1.01,
)
plt.tight_layout(rect=[0, 0.10, 1, 1])
plt.savefig(data_dir / "policy_I_bar_quintile.png", dpi=150, bbox_inches="tight")
plt.close()

# ── VIZ K: Spatial map of income quintiles ───────────────────────
print("Viz K (spatial quintile map) …")

tr_k = tracts.dropna(subset=["median_income_estimate"])
tr_k = tr_k[tr_k["median_income_estimate"] > 0].copy()
tr_k["quintile"] = pd.qcut(
    tr_k["median_income_estimate"], q=5,
    labels=["Q1\n(lowest)", "Q2", "Q3", "Q4", "Q5\n(highest)"],
    duplicates="drop",
)
tr_k = tr_k.to_crs(epsg=3857)

q_colors = ["#b2182b", "#ef8a62", "#fddbc7", "#67a9cf", "#2166ac"]
q_cats   = tr_k["quintile"].cat.categories.tolist()

fig, ax = plt.subplots(figsize=(10, 13))
legend_handles = []
for qcat, qcol in zip(q_cats, q_colors):
    sub = tr_k[tr_k["quintile"] == qcat]
    sub.plot(ax=ax, color=qcol, edgecolor="white", linewidth=0.3, alpha=0.85)
    legend_handles.append(
        mpatches.Patch(color=qcol, alpha=0.85, label=qcat.replace("\n", " "))
    )

ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=11)
ax.set_axis_off()
ax.set_title(
    "Chicago census tracts by median household income quintile\n"
    "(red = lowest income, blue = highest income)",
    fontsize=12,
)
ax.legend(handles=legend_handles, title="Income quintile", fontsize=10,
          title_fontsize=10, loc="lower right", framealpha=0.9)

plt.tight_layout()
plt.savefig(data_dir / "policy_K_quintile_map.png", dpi=150)
plt.close()

print("\nAll done. Saved:")
for fn in sorted(data_dir.glob("policy_*.png")):
    print(f"  {fn.name}")
