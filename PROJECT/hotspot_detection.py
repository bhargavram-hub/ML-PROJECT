"""
Accident Hotspot Detection Using DBSCAN-Based Anomaly Detection
-----------------------------------------------------------------
Domain: Traffic Congestion

Dataset: US FARS (Fatality Analysis Reporting System) 2016 Accident Data
Source : NHTSA / FARS, hosted at
         https://raw.githubusercontent.com/DS202-at-ISU/labs/master/data/fars2016/accident.csv
Scope  : Focused on Texas (STATE code 48) — the state with the highest
         accident-record count in the file — to produce a dense, legible
         set of spatial clusters for demonstration.

Pipeline:
1. Load & clean data (drop invalid/sentinel GPS coordinates)
2. Feature prep for spatial clustering (radians, haversine metric)
3. k-distance elbow plot to justify the DBSCAN eps parameter
4. Run DBSCAN -> spatial clusters (hotspots) + noise points
5. Cluster-level statistics -> flag anomalous (statistically unusual) hotspots
6. Visualizations + exported summary CSV
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

RAW_PATH = "accident.csv"
OUT_DIR = "outputs"
import os
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------
# 1. LOAD & CLEAN
# ---------------------------------------------------------------
df = pd.read_csv(RAW_PATH)
print(f"Raw records: {len(df)}, Raw columns: {df.shape[1]}")

# FARS uses sentinel values (99.9999 / 999.9999) for unknown/unreported GPS coords
df = df[(df["LATITUDE"] < 90) & (df["LONGITUD"].abs() < 180)].copy()

# Focus on Texas (STATE == 48), the highest-volume state in this extract,
# to keep the spatial clustering meaningful at a single-state scale.
tx = df[df["STATE"] == 48].copy()
tx = tx.rename(columns={"LONGITUD": "LONGITUDE"})
tx = tx.reset_index(drop=True)

print(f"Texas records after cleaning: {len(tx)}")
tx.to_csv(f"{OUT_DIR}/texas_cleaned.csv", index=False)

# ---------------------------------------------------------------
# 2. PREP FOR HAVERSINE-BASED DBSCAN
# ---------------------------------------------------------------
coords = tx[["LATITUDE", "LONGITUDE"]].to_numpy()
coords_rad = np.radians(coords)

EARTH_RADIUS_KM = 6371.0088

# ---------------------------------------------------------------
# 3. K-DISTANCE ELBOW PLOT (to choose eps)
# ---------------------------------------------------------------
MIN_SAMPLES = 5
nbrs = NearestNeighbors(n_neighbors=MIN_SAMPLES, metric="haversine").fit(coords_rad)
distances, _ = nbrs.kneighbors(coords_rad)
k_dist = np.sort(distances[:, -1]) * EARTH_RADIUS_KM  # convert radians -> km
k_dist_m = k_dist * 1000

plt.figure(figsize=(7, 5))
plt.plot(k_dist_m)
plt.xlabel("Points sorted by distance")
plt.ylabel(f"{MIN_SAMPLES}-th nearest neighbor distance (meters)")
plt.title("K-Distance Elbow Plot for DBSCAN eps Selection")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/k_distance_elbow.png", dpi=150)
plt.close()

# Elbow observed around ~5km for this statewide data (visually inspected via
# the k-distance plot / percentile analysis); documented in report.
EPS_KM = 5.0
eps_rad = EPS_KM / EARTH_RADIUS_KM

# ---------------------------------------------------------------
# 4. DBSCAN CLUSTERING
# ---------------------------------------------------------------
db = DBSCAN(eps=eps_rad, min_samples=MIN_SAMPLES, metric="haversine")
tx["cluster"] = db.fit_predict(coords_rad)

n_clusters = len(set(tx["cluster"])) - (1 if -1 in tx["cluster"].values else 0)
n_noise = (tx["cluster"] == -1).sum()
print(f"Clusters found: {n_clusters}")
print(f"Noise points (non-hotspot, isolated accidents): {n_noise}")

# ---------------------------------------------------------------
# 5. CLUSTER-LEVEL STATS & ANOMALY (UNUSUAL HOTSPOT) FLAGGING
# ---------------------------------------------------------------
clustered = tx[tx["cluster"] != -1].copy()

summary = clustered.groupby("cluster").agg(
    accident_count=("cluster", "size"),
    avg_fatals=("FATALS", "mean"),
    total_fatals=("FATALS", "sum"),
    center_lat=("LATITUDE", "mean"),
    center_lon=("LONGITUDE", "mean"),
).reset_index()

# Statistical anomaly flag: a hotspot is "anomalous" if its accident_count
# or total_fatals is a strong outlier relative to other hotspots (z-score > 2)
for col in ["accident_count", "total_fatals"]:
    mu, sigma = summary[col].mean(), summary[col].std(ddof=0)
    summary[f"{col}_z"] = (summary[col] - mu) / (sigma if sigma > 0 else 1)

summary["is_anomalous_hotspot"] = (
    (summary["accident_count_z"] > 2) | (summary["total_fatals_z"] > 2)
)

summary = summary.sort_values("accident_count", ascending=False)
summary.to_csv(f"{OUT_DIR}/cluster_summary.csv", index=False)

print("\nTop 10 hotspots by accident count:")
print(summary.head(10).to_string(index=False))

print(f"\nAnomalous hotspots flagged: {summary['is_anomalous_hotspot'].sum()}")

# ---------------------------------------------------------------
# 6. VISUALIZATIONS
# ---------------------------------------------------------------
plt.figure(figsize=(8, 7))
noise_pts = tx[tx["cluster"] == -1]
plt.scatter(noise_pts["LONGITUDE"], noise_pts["LATITUDE"], s=6, c="lightgray", label="Noise (isolated)")
sc = plt.scatter(
    clustered["LONGITUDE"], clustered["LATITUDE"],
    s=8, c=clustered["cluster"], cmap="tab20", label="Clustered (hotspot)"
)
anomalous_clusters = summary[summary["is_anomalous_hotspot"]]["cluster"]
anom_pts = clustered[clustered["cluster"].isin(anomalous_clusters)]
plt.scatter(
    anom_pts["LONGITUDE"], anom_pts["LATITUDE"],
    s=40, facecolors="none", edgecolors="red", linewidths=1.2, label="Anomalous hotspot"
)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Texas Accident Hotspots via DBSCAN (2016 FARS data)")
plt.legend(loc="lower left", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/hotspot_map.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 5))
plt.bar(summary["cluster"].astype(str), summary["accident_count"],
        color=["red" if a else "steelblue" for a in summary["is_anomalous_hotspot"]])
plt.xlabel("Cluster ID")
plt.ylabel("Accident count")
plt.title("Accident Count per Hotspot Cluster (red = anomalous)")
plt.xticks(rotation=90, fontsize=6)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/cluster_sizes.png", dpi=150)
plt.close()

print("\nAll outputs written to ./outputs/")
