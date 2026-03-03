"""
test_crime_map.py
Scatter plot — one dot per day of 2015
  X = number of active streetlight outage requests that day
  Y = number of nighttime crimes that day (hour >= 17, i.e. 5pm onwards)
  + OLS regression line
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy import stats

data_dir = Path(__file__).parent / "data"
YEAR = 2015

# ---------------------------------------------------------------
# 1. Nighttime crimes per day
# ---------------------------------------------------------------
print("Loading crimes CSV...")
crimes = pd.read_csv(data_dir / "crimes_2011_2018.csv")
crimes["dt"]   = pd.to_datetime(crimes["date"], errors="coerce")
crimes         = crimes.dropna(subset=["dt"])
crimes["year"] = crimes["dt"].dt.year
crimes["date"] = crimes["dt"].dt.normalize()
crimes["hour"] = crimes["dt"].dt.hour

cr15 = crimes[crimes["year"] == YEAR].copy()
night_crimes = cr15[(cr15["hour"] >= 17) | (cr15["hour"] < 5)].groupby("date").size().rename("night_crimes")
print(f"  Days with nighttime crime data: {len(night_crimes)}")

# ---------------------------------------------------------------
# 2. Active outage requests per day (30 m buffers)
# ---------------------------------------------------------------
print("Loading buffers...")
buf = gpd.read_file(data_dir / "streetlights_buffers.geojson")
buf["creation_date"]   = pd.to_datetime(buf["creation_date"],   errors="coerce")
buf["completion_date"] = pd.to_datetime(buf["completion_date"], errors="coerce")
buf30 = buf[buf["buffer_radius_m"] == 30].dropna(
    subset=["creation_date", "completion_date"]
).copy()

yr_start = pd.Timestamp(f"{YEAR}-01-01")
yr_end   = pd.Timestamp(f"{YEAR}-12-31")
buf30 = buf30[
    (buf30["creation_date"]   <= yr_end) &
    (buf30["completion_date"] >= yr_start)
].copy()

days           = pd.date_range(yr_start, yr_end, freq="D")
creation_arr   = buf30["creation_date"].values.astype("datetime64[D]")
completion_arr = buf30["completion_date"].values.astype("datetime64[D]")

print(f"  Counting active outages per day ({len(days)} days)...")
n_active = np.array([
    int(((creation_arr <= np.datetime64(d, "D")) &
         (completion_arr >= np.datetime64(d, "D"))).sum())
    for d in days
])
outages_per_day = pd.Series(n_active, index=days, name="n_outages")

# ---------------------------------------------------------------
# 3. Merge into one row per day
# ---------------------------------------------------------------
df = pd.DataFrame({"n_outages": outages_per_day}).join(
    pd.DataFrame({"night_crimes": night_crimes}), how="left"
)
df["night_crimes"] = df["night_crimes"].fillna(0)
df = df[df["n_outages"] > 0].copy()

# ---------------------------------------------------------------
# 4. Scatter + OLS regression line
# ---------------------------------------------------------------
print("Plotting...")
slope, intercept, r, p, _ = stats.linregress(df["n_outages"], df["night_crimes"])
x_line = np.linspace(df["n_outages"].min(), df["n_outages"].max(), 200)
y_line = intercept + slope * x_line

fig, ax = plt.subplots(figsize=(9, 6))

ax.scatter(df["n_outages"], df["night_crimes"],
           color="#4393c3", s=28, alpha=0.65, zorder=2, label="One day")

ax.plot(x_line, y_line, color="#b2182b", linewidth=2,
        label=f"OLS  y = {slope:.3f}x + {intercept:.1f}\n"
              f"r = {r:.3f},  p = {p:.4f}")

ax.set_xlabel("Active streetlight outage requests that day", fontsize=12)
ax.set_ylabel("Nighttime crimes that day  (5 pm – 5 am)", fontsize=12)
ax.set_title(
    f"Chicago {YEAR} — Streetlight outages vs nighttime crime\n"
    "(each dot = one calendar day; night = 5pm–5am; 30 m outage buffers)",
    fontsize=13,
)
ax.legend(fontsize=10, loc="upper left")
ax.grid(linestyle="--", alpha=0.35)

plt.tight_layout()
out_path = data_dir / f"scatter_outages_vs_crime_{YEAR}.png"
plt.savefig(out_path, dpi=150)
print(f"Saved → {out_path}")
plt.show()
