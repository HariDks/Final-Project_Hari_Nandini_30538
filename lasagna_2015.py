"""
lasagna_2015.py
2-row lasagna (heatmap) for 2015:
  Row 0: nighttime crimes (5pm–5am, i.e. hour>=17 or hour<5) — green gradient
  Row 1: active streetlight outages                           — red gradient

X = every day of 2015 (365 columns)
Each row independently normalised to its own full colour range.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase

data_dir = Path(__file__).parent / "data"
YEAR = 2015

# ---------------------------------------------------------------
# 1. Nighttime crimes per day  (5pm–5am: hour>=17 or hour<5)
# ---------------------------------------------------------------
print("Loading crimes CSV...")
crimes = pd.read_csv(data_dir / "crimes_2011_2018.csv")
crimes["dt"]   = pd.to_datetime(crimes["date"], errors="coerce")
crimes         = crimes.dropna(subset=["dt"])
crimes["year"] = crimes["dt"].dt.year
crimes["date"] = crimes["dt"].dt.normalize()
crimes["hour"] = crimes["dt"].dt.hour

cr15  = crimes[crimes["year"] == YEAR].copy()
night = cr15[(cr15["hour"] >= 17) | (cr15["hour"] < 5)]
night_crimes = night.groupby("date").size()
print(f"  2015 nighttime crimes: {night_crimes.sum():,}")

# ---------------------------------------------------------------
# 2. Active outage requests per day (30 m buffers, clipped to 2015)
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
outages_per_day = pd.Series(n_active, index=days)

# ---------------------------------------------------------------
# 3. Align to same 365-day index, normalise each row independently
# ---------------------------------------------------------------
night_row = night_crimes.reindex(days).fillna(0).values.astype(float)
out_row   = outages_per_day.values.astype(float)

def norm(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)

# ---------------------------------------------------------------
# 4. Lasagna plot
# ---------------------------------------------------------------
print("Plotting...")

crime_cmap  = plt.cm.Greens
outage_cmap = plt.cm.Reds

rgba = np.zeros((2, len(days), 4))
rgba[0] = crime_cmap(norm(night_row))
rgba[1] = outage_cmap(norm(out_row))

fig = plt.figure(figsize=(18, 3.5))
ax  = fig.add_axes([0.06, 0.30, 0.80, 0.52])

ax.imshow(rgba, aspect="auto", interpolation="nearest", origin="upper")

ax.set_yticks([0, 1])
ax.set_yticklabels(["Nighttime crimes\n(5pm – 5am)", "Streetlights out"], fontsize=11)

month_starts   = pd.date_range(f"{YEAR}-01-01", f"{YEAR}-12-01", freq="MS")
tick_positions = [(m - yr_start).days for m in month_starts]
tick_labels    = [m.strftime("%b") for m in month_starts]
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, fontsize=9)
ax.set_xlabel("Day of 2015", fontsize=11, labelpad=6)

ax.axhline(0.5, color="white", linewidth=2)

ax.set_title(
    f"Chicago {YEAR} — Nighttime crime counts and active streetlight outages\n"
    "(each row independently colour-scaled; green = crime count, red = active outages)",
    fontsize=12, pad=10,
)

ax_cb1 = fig.add_axes([0.06, 0.10, 0.37, 0.07])
ax_cb2 = fig.add_axes([0.50, 0.10, 0.37, 0.07])

ColorbarBase(ax_cb1, cmap=crime_cmap,
             norm=mcolors.Normalize(vmin=night_row.min(), vmax=night_row.max()),
             orientation="horizontal").set_label("Nighttime crimes per day", fontsize=9)

ColorbarBase(ax_cb2, cmap=outage_cmap,
             norm=mcolors.Normalize(vmin=out_row.min(), vmax=out_row.max()),
             orientation="horizontal").set_label("Active 30 m outage requests", fontsize=9)

out_path = data_dir / f"lasagna_{YEAR}.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.show()
