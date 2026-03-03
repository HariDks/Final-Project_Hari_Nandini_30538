from pathlib import Path
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# -----------------------
# Paths
# -----------------------
data_dir = Path(__file__).parent / "data"

before_file = data_dir / "streetlight_crime_pre_day_buckets_events.geojson"
during_csv  = data_dir / "streetlight_level_crime_during_service.csv"

out_png4 = data_dir / "crime_rate_before_vs_during_by_radius.png"
out_png5 = data_dir / "crime_rate_diff_during_minus_before_by_radius.png"

# -----------------------
# Load BEFORE (crime-level, bucketed 1..5 days before)
# -----------------------
before = gpd.read_file(before_file)

needed_before = {"request_id", "buffer_radius_m", "days_before_request", "id"}
missing_before = needed_before - set(before.columns)
if missing_before:
    raise ValueError(f"Missing BEFORE columns: {missing_before}")

# Ensure correct types
before["request_id"] = before["request_id"].astype(str)
before["buffer_radius_m"] = pd.to_numeric(before["buffer_radius_m"], errors="coerce")
before["days_before_request"] = pd.to_numeric(before["days_before_request"], errors="coerce")

before = before.dropna(subset=["buffer_radius_m", "days_before_request"]).copy()

# Keep only days 1..5 (should already be true, but safe)
before = before[(before["days_before_request"] >= 1) & (before["days_before_request"] <= 5)].copy()

# Streetlight-level BEFORE crimes (unique crimes across the 5-day window)
before_sl = (
    before
    .groupby(["request_id", "buffer_radius_m"], as_index=False)
    .agg(unique_crimes_before=("id", "nunique"))
)

# Convert to rate per day (window length = 5 days)
before_sl["crimes_per_day_before"] = before_sl["unique_crimes_before"] / 5.0

# -----------------------
# Load DURING (streetlight-level, already computed)
# -----------------------
during = pd.read_csv(during_csv)

needed_during = {"request_id", "buffer_radius_m", "crimes_per_day", "days_open"}
missing_during = needed_during - set(during.columns)
if missing_during:
    raise ValueError(f"Missing DURING columns: {missing_during}")

during["request_id"] = during["request_id"].astype(str)
during["buffer_radius_m"] = pd.to_numeric(during["buffer_radius_m"], errors="coerce")
during["crimes_per_day"] = pd.to_numeric(during["crimes_per_day"], errors="coerce")
during["days_open"] = pd.to_numeric(during["days_open"], errors="coerce")

during = during.dropna(subset=["buffer_radius_m", "crimes_per_day", "days_open"]).copy()
during = during[during["days_open"] > 0].copy()

# Rename for clarity
during_sl = during.rename(columns={"crimes_per_day": "crimes_per_day_during"})

# -----------------------
# Merge BEFORE vs DURING (paired at request_id x radius)
# -----------------------
paired = pd.merge(
    during_sl[["request_id", "buffer_radius_m", "crimes_per_day_during", "days_open"]],
    before_sl[["request_id", "buffer_radius_m", "crimes_per_day_before"]],
    on=["request_id", "buffer_radius_m"],
    how="inner"
)

# Optional: if you want to avoid very tiny denominators (super short open durations)
# paired = paired[paired["days_open"] >= 1].copy()

paired["diff_during_minus_before"] = paired["crimes_per_day_during"] - paired["crimes_per_day_before"]

print("Paired rows:", len(paired))
print(paired.head())

# -----------------------
# Summarize by radius
# -----------------------
summary = (
    paired
    .groupby("buffer_radius_m", as_index=False)
    .agg(
        avg_before=("crimes_per_day_before", "mean"),
        avg_during=("crimes_per_day_during", "mean"),
        avg_diff=("diff_during_minus_before", "mean"),
        n_requests=("request_id", "nunique")
    )
).sort_values("buffer_radius_m")

print("\nSummary by radius:")
print(summary)

# -----------------------
# Plot 1: BEFORE vs DURING (two lines)
# -----------------------
plt.figure(figsize=(7, 4))

x = summary["buffer_radius_m"].astype(int)

plt.plot(x, summary["avg_before"], marker="o", label="Before (1–5 days pre)")
plt.plot(x, summary["avg_during"], marker="o", label="During (service window)")

plt.xticks(x, [f"{r} m" for r in x])
plt.xlabel("Buffer radius")
plt.ylabel("Average crimes per day")
plt.title("Crime rate before vs during service window (paired requests)")
plt.legend()
plt.tight_layout()
plt.savefig(out_png4, dpi=200)
plt.show()

print(f"Saved: {out_png4}")

# -----------------------
# Plot 2: Difference (During - Before)
# -----------------------
plt.figure(figsize=(7, 4))

plt.plot(x, summary["avg_diff"], marker="o")
plt.axhline(0)  # reference line

plt.xticks(x, [f"{r} m" for r in x])
plt.xlabel("Buffer radius")
plt.ylabel("Average (during - before) crimes/day")
plt.title("Change in crime rate during service window vs pre-window")
plt.tight_layout()
plt.savefig(out_png5, dpi=200)
plt.show()

print(f"Saved: {out_png5}")

# -----------------------
# Optional: sanity check sample sizes
# -----------------------
print("\nN requests used (paired) by radius:")
print(summary[["buffer_radius_m", "n_requests"]])
