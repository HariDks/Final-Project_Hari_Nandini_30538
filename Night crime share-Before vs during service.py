from pathlib import Path
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# -----------------------
# Paths
# -----------------------
data_dir = Path("/Users/haridharshinik.s/Final-Project_Hari_Nandini_30538/data")

during_file = data_dir / "streetlight_crime_events.geojson"
before_file = data_dir / "streetlight_crime_pre_day_buckets_events.geojson"

out_png3 = data_dir / "night_share_before_vs_during.png"

# -----------------------
# Load datasets
# -----------------------
during = gpd.read_file(during_file)
before = gpd.read_file(before_file)

# Ensure datetime
during["crime_date"] = pd.to_datetime(during["crime_date"], errors="coerce")
before["crime_date"] = pd.to_datetime(before["crime_date"], errors="coerce")

# -----------------------
# Create night indicator
# -----------------------
for df in [during, before]:
    df["hour"] = df["crime_date"].dt.hour
    df["is_night"] = (df["hour"] >= 18) | (df["hour"] < 6)

# -----------------------
# Aggregate to streetlight level
# -----------------------
during_sl = (
    during
    .groupby(["request_id", "buffer_radius_m"], as_index=False)
    .agg(
        total_during=("id", "nunique"),
        night_during=("is_night", "sum")
    )
)

during_sl = during_sl[during_sl["total_during"] > 0].copy()
during_sl["night_share_during"] = (
    during_sl["night_during"] / during_sl["total_during"]
)

before_sl = (
    before
    .groupby(["request_id", "buffer_radius_m"], as_index=False)
    .agg(
        total_before=("id", "nunique"),
        night_before=("is_night", "sum")
    )
)

before_sl = before_sl[before_sl["total_before"] > 0].copy()
before_sl["night_share_before"] = (
    before_sl["night_before"] / before_sl["total_before"]
)

# -----------------------
# Merge
# -----------------------
comparison = pd.merge(
    during_sl,
    before_sl,
    on=["request_id", "buffer_radius_m"],
    how="inner"
)

# -----------------------
# Average comparison
# -----------------------
summary = (
    comparison
    .groupby("buffer_radius_m", as_index=False)
    .agg(
        avg_night_before=("night_share_before", "mean"),
        avg_night_during=("night_share_during", "mean"),
        n_requests=("request_id", "nunique")
    )
).sort_values("buffer_radius_m")

print(summary)

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(6,4))

x = summary["buffer_radius_m"].astype(int).astype(str) + " m"

plt.plot(x, summary["avg_night_before"], marker="o", label="Before")
plt.plot(x, summary["avg_night_during"], marker="o", label="During")

plt.xlabel("Buffer radius")
plt.ylabel("Average night crime share")
plt.title("Night crime share: Before vs During service window")
plt.legend()
plt.tight_layout()
plt.savefig(out_png3, dpi=200)
plt.show()

print(f"Saved: {out_png3}")