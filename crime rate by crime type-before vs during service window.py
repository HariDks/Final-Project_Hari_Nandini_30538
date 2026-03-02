from pathlib import Path
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# -----------------------
# Paths
# -----------------------
data_dir = Path("/Users/haridharshinik.s/Final-Project_Hari_Nandini_30538/data")
before_file = data_dir / "streetlight_crime_pre_day_buckets_events.geojson"
during_file = data_dir / "streetlight_crime_events.geojson"
during_sl_csv = data_dir / "streetlight_level_crime_during_service.csv"

# Save outputs (optional)
save_plots = True

# -----------------------
# Load data
# -----------------------
before = gpd.read_file(before_file)
during = gpd.read_file(during_file)
sl = pd.read_csv(during_sl_csv)

# -----------------------
# Basic cleaning
# -----------------------
before["request_id"] = before["request_id"].astype(str)
during["request_id"] = during["request_id"].astype(str)
sl["request_id"] = sl["request_id"].astype(str)

before["buffer_radius_m"] = pd.to_numeric(before["buffer_radius_m"], errors="coerce")
during["buffer_radius_m"] = pd.to_numeric(during["buffer_radius_m"], errors="coerce")
sl["buffer_radius_m"] = pd.to_numeric(sl["buffer_radius_m"], errors="coerce")
sl["days_open"] = pd.to_numeric(sl["days_open"], errors="coerce")

before = before.dropna(subset=["request_id", "buffer_radius_m", "days_before_request", "id", "primary_type"]).copy()
during = during.dropna(subset=["request_id", "buffer_radius_m", "id", "primary_type"]).copy()
sl = sl.dropna(subset=["request_id", "buffer_radius_m", "days_open"]).copy()
sl = sl[sl["days_open"] > 0].copy()

# Keep only days 1..5 for BEFORE
before["days_before_request"] = pd.to_numeric(before["days_before_request"], errors="coerce")
before = before[(before["days_before_request"] >= 1) & (before["days_before_request"] <= 5)].copy()

# -----------------------
# BEFORE: request-level type counts -> per-day rate
# One row per request_id x radius x type
# -----------------------
before_req_type = (
    before
    .groupby(["request_id", "buffer_radius_m", "primary_type"], as_index=False)
    .agg(n_before=("id", "nunique"))
)
before_req_type["rate_before"] = before_req_type["n_before"] / 5.0

# -----------------------
# DURING: merge days_open -> request-level type counts -> per-day rate
# -----------------------
during2 = during.merge(
    sl[["request_id", "buffer_radius_m", "days_open"]],
    on=["request_id", "buffer_radius_m"],
    how="inner"
)

during_req_type = (
    during2
    .groupby(["request_id", "buffer_radius_m", "primary_type"], as_index=False)
    .agg(n_during=("id", "nunique"), days_open=("days_open", "first"))
)
during_req_type["rate_during"] = during_req_type["n_during"] / during_req_type["days_open"]

# -----------------------
# Merge BEFORE and DURING at request-level type
# (fill missing with 0: if a request had none of that type in a window)
# -----------------------
req_type = pd.merge(
    before_req_type[["request_id", "buffer_radius_m", "primary_type", "rate_before"]],
    during_req_type[["request_id", "buffer_radius_m", "primary_type", "rate_during"]],
    on=["request_id", "buffer_radius_m", "primary_type"],
    how="outer"
)

req_type["rate_before"] = req_type["rate_before"].fillna(0.0)
req_type["rate_during"] = req_type["rate_during"].fillna(0.0)

# -----------------------
# Choose top K types by total volume (before + during counts)
# Use counts, not rates, to pick meaningful types
# -----------------------
# Recompute totals using counts (more stable than rates)
before_counts_total = (
    before
    .groupby("primary_type")["id"].nunique()
)
during_counts_total = (
    during
    .groupby("primary_type")["id"].nunique()
)

type_totals = (before_counts_total.add(during_counts_total, fill_value=0)).sort_values(ascending=False)

K = 8
top_types = type_totals.head(K).index.tolist()
print("Top types:", top_types)

req_type_plot = req_type[req_type["primary_type"].isin(top_types)].copy()

# -----------------------
# Summarize: average rate across requests, by radius and type
# -----------------------
summary = (
    req_type_plot
    .groupby(["buffer_radius_m", "primary_type"], as_index=False)
    .agg(
        avg_rate_before=("rate_before", "mean"),
        avg_rate_during=("rate_during", "mean"),
        n_requests=("request_id", "nunique")
    )
)

# -----------------------
# Plot: one figure per radius
# -----------------------
for radius in sorted(summary["buffer_radius_m"].unique()):
    temp = summary[summary["buffer_radius_m"] == radius].copy()

    # Keep consistent type ordering
    temp["primary_type"] = pd.Categorical(temp["primary_type"], categories=top_types, ordered=True)
    temp = temp.sort_values("primary_type")

    x = range(len(temp["primary_type"]))

    plt.figure(figsize=(10, 4))
    plt.plot(x, temp["avg_rate_before"], marker="o", label="Before (1–5 days pre)")
    plt.plot(x, temp["avg_rate_during"], marker="o", label="During (service window)")

    plt.xticks(x, temp["primary_type"], rotation=45, ha="right")
    plt.xlabel("Crime type")
    plt.ylabel("Average crimes per day (per request)")
    plt.title(f"Crime rate by type: Before vs During (radius {int(radius)} m)")
    plt.legend()
    plt.tight_layout()

    if save_plots:
        out_file = data_dir / f"crime_rate_by_type_before_vs_during_{int(radius)}m.png"
        plt.savefig(out_file, dpi=200)
        print("Saved:", out_file)

    plt.show()