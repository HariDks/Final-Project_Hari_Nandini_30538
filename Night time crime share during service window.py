from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Paths
# -----------------------
data_dir = Path("/Users/haridharshinik.s/Final-Project_Hari_Nandini_30538/data")
input_geojson = data_dir / "streetlight_crime_events.geojson"
out_png2 = data_dir / "day_vs_night_crime_during_service.png"

# -----------------------
# Load joined (crime-level) dataset
# -----------------------
import geopandas as gpd
joined = gpd.read_file(input_geojson)

# Ensure datetime
joined["crime_date"] = pd.to_datetime(joined["crime_date"], errors="coerce")

# -----------------------
# Create day/night variable
# -----------------------
joined["hour"] = joined["crime_date"].dt.hour
joined["is_night"] = (joined["hour"] >= 18) | (joined["hour"] < 6)

# -----------------------
# Streetlight-level aggregation
# -----------------------
streetlight_dn = (
    joined
    .groupby(["request_id", "buffer_radius_m"], as_index=False)
    .agg(
        total_crimes=("id", "nunique"),
        night_crimes=("is_night", "sum"),
    )
)

streetlight_dn["day_crimes"] = (
    streetlight_dn["total_crimes"] - streetlight_dn["night_crimes"]
)

# Avoid divide by zero
streetlight_dn = streetlight_dn[streetlight_dn["total_crimes"] > 0].copy()

# Proportion night per request
streetlight_dn["prop_night"] = (
    streetlight_dn["night_crimes"] / streetlight_dn["total_crimes"]
)

# -----------------------
# Aggregate across requests
# -----------------------
summary = (
    streetlight_dn
    .groupby("buffer_radius_m", as_index=False)
    .agg(
        avg_prop_night=("prop_night", "mean"),
        avg_night_crimes=("night_crimes", "mean"),
        avg_day_crimes=("day_crimes", "mean"),
    )
)

# -----------------------
# Plot: proportion night (cleanest figure)
# -----------------------
plt.figure(figsize=(6, 4))

plt.bar(
    summary["buffer_radius_m"].astype(str) + " m",
    summary["avg_prop_night"]
)

plt.xlabel("Buffer radius")
plt.ylabel("Average proportion of crimes at night")
plt.title("Night-time crime share during service window")
plt.tight_layout()
plt.savefig(out_png2, dpi=200)
plt.show()

print(f"Saved: {out_png2}")
print("\nSummary table:")
print(summary)
