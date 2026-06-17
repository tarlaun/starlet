#!/usr/bin/env python3
"""Generate experiment-section figures for the Starlet robust version.

Reads the v2 tile+MVT CSV; serving p50s and the tippecanoe baseline are inlined
from the campaign logs (sources noted). Writes PDF (vector, for LaTeX) + PNG to
bench/results_v2/figs/.
"""
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CSV = HERE / "results_v2" / "starlet_bench_v2.csv"
OUT = HERE / "results_v2" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "font.family": "serif",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.bbox": "tight",
})
C = {"tile": "#1f6f8b", "mvt": "#c1502e", "total": "#2d2d2d",
     "tip": "#6a8d3a", "v0": "#9e9e9e", "v1": "#e08a1e", "v2": "#2e7d4f"}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("wrote", OUT / f"{name}.{{pdf,png}}")


# ---- load tile+MVT sweep ----
rows = {}
with open(CSV) as f:
    for r in csv.DictReader(f):
        rows.setdefault(r["label"], {})[r["phase"]] = r
labels = ["osm_parks_25", "osm_parks_50", "osm_parks_75", "osm_parks_100"]
feat = np.array([int(rows[l]["tile"]["input_rows"]) / 1e6 for l in labels])
tile = np.array([float(rows[l]["tile"]["wall_s"]) for l in labels])
mvt = np.array([float(rows[l]["mvt"]["wall_s"]) for l in labels])
total = tile + mvt
tile_rss = np.array([int(rows[l]["tile"]["max_rss_kb"]) / 1e6 for l in labels])   # GB
mvt_rss = np.array([int(rows[l]["mvt"]["max_rss_kb"]) / 1e6 for l in labels])

# tippecanoe baseline (baselines.csv): full 9.96M, z0-7, 8 threads
TIP_FEAT, TIP_WALL, TIP_RSS = 9.961884, 235.442, 1020648 / 1e6  # s, GB

# ============================ Fig 1: runtime scalability ============================
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.plot(feat, total, "o-", color=C["total"], lw=2, label="Total (tile + MVT)")
ax.plot(feat, tile, "s--", color=C["tile"], lw=1.6, label="Tiling")
ax.plot(feat, mvt, "^--", color=C["mvt"], lw=1.6, label="MVT generation")
for fx, fy in zip(feat, total):
    pass
ax.annotate(f"{total[-1]:.0f} s\n({total[-1]/3600:.1f} h)", (feat[-1], total[-1]),
            textcoords="offset points", xytext=(-10, -2), color=C["total"], fontsize=8.5, ha="right", va="top")
ax.set_xlabel("Input size (million polygons)")
ax.set_ylabel("Wall-clock time (s)")
ax.set_title("Pipeline scalability (OSM-parks, 16-core server)")
ax.legend(frameon=False, fontsize=9.5, loc="upper left")
ax.set_ylim(0, total[-1] * 1.12)
save(fig, "fig1_runtime_scalability")

# ============================ Fig 2: memory scalability ============================
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.plot(feat, tile_rss, "s-", color=C["tile"], lw=1.8, label="Tiling")
ax.plot(feat, mvt_rss, "^-", color=C["mvt"], lw=1.8, label="MVT generation")
ax.axhline(24, color="#999", ls="--", lw=1, alpha=0.7)
ax.text(feat[0], 24.3, "single-machine RAM budget (24 GB)", fontsize=8, color="#666")
ax.set_xlabel("Input size (million polygons)")
ax.set_ylabel("Peak RSS (GB)")
ax.set_ylim(0, 26)
ax.set_title("Memory scalability (bounded)")
ax.legend(frameon=False, fontsize=9.5, loc="center left")
save(fig, "fig2_memory_scalability")

# ============================ Fig 3: serving latency by regime ============================
# osm_parks_100, v2 (v2_serving_100.log), p50 ms
regimes = ["warm-mem\n(LRU)", "warm-disk\n(pyramid)", "on-the-fly\nz13", "on-the-fly\nz10", "on-the-fly\nz8"]
p50 = [1.7, 2.2, 103.98, 133.96, 231.74]
colors = ["#2e7d4f", "#3a8fb0", "#e08a1e", "#e08a1e", "#e08a1e"]
fig, ax = plt.subplots(figsize=(6.8, 3.7))
b = ax.bar(regimes, p50, color=colors, width=0.6)
ax.tick_params(axis="x", labelsize=9)
ax.set_yscale("log")
ax.set_ylabel("p50 latency (ms, log scale)")
ax.set_title("Tile-serving latency by regime (OSM-parks 100%)")
for rect, v in zip(b, p50):
    ax.annotate(f"{v:.1f} ms" if v >= 10 else f"{v:.1f} ms",
                (rect.get_x() + rect.get_width() / 2, v), textcoords="offset points",
                xytext=(0, 3), ha="center", fontsize=8.5)
ax.set_ylim(1, 600)
save(fig, "fig3_serving_latency_regimes")

# ============================ Fig 4: on-the-fly improvement ============================
# p50 ms per zoom: original (whole-partition, from per-tile verify medians, range 17-39s),
# v1 prefilter (v1 serving run), v2 row-group skipping (this campaign).
zooms = ["z8", "z10", "z13"]
v1 = [3279.52, 2453.21, 2245.87]   # prefilter only
v2 = [231.74, 133.96, 103.98]      # + bbox columns + row-group skipping
x = np.arange(len(zooms)); w = 0.36
from matplotlib.patches import Patch
fig, ax = plt.subplots(figsize=(5.6, 3.7))
# original whole-partition band (measured per-tile 17-39 s)
ax.axhspan(17000, 39000, color=C["v0"], alpha=0.22)
b1 = ax.bar(x - w/2, v1, w, color=C["v1"])
b2 = ax.bar(x + w/2, v2, w, color=C["v2"])
ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels(zooms)
ax.set_xlabel("Tile zoom (beyond pre-generated pyramid)")
ax.set_ylabel("Cold on-the-fly p50 (ms, log scale)")
ax.set_title("On-demand generation: structural-cure impact")
handles = [Patch(facecolor=C["v0"], alpha=0.5, label="original: whole-partition (17–39 s)"),
           Patch(facecolor=C["v1"], label="+ bbox pre-filter"),
           Patch(facecolor=C["v2"], label="+ row-group skipping")]
ax.legend(handles=handles, fontsize=8.3, loc="upper center",
          bbox_to_anchor=(0.5, 0.80), frameon=True, framealpha=0.95, edgecolor="none")
for r in b2:  # label the row-group-skipping (v2) bars — the result of interest
    ax.annotate(f"{r.get_height():.0f} ms", (r.get_x() + r.get_width()/2, r.get_height()),
                textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8.5,
                color=C["v2"], fontweight="bold")
ax.set_ylim(40, 60000)
save(fig, "fig4_ondemand_improvement")

# ============================ Fig 5: read-time pruning effectiveness ============================
prune_csv = HERE / "results_v2" / "pruning_effectiveness.csv"
if prune_csv.exists():
    pz, pbytes, prows = [], [], []
    with open(prune_csv) as f:
        for r in csv.DictReader(f):
            pz.append(int(r["zoom"]))
            pbytes.append(float(r["bytes_read_frac"]) * 100)
            prows.append(float(r["rows_read_frac"]) * 100)
    xx = np.arange(len(pz))
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.axhline(100, color="#9e9e9e", ls="--", lw=1.2)
    ax.text(-0.4, 103, "original: whole-partition read (100%)", fontsize=8.3, color="#666")
    ax.bar(xx, pbytes, color="#3a8fb0", width=0.6, label="partition bytes read")
    for i, v in enumerate(pbytes):
        ax.annotate(f"{v:.0f}%", (xx[i], v), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8)
    ax.set_xticks(xx); ax.set_xticklabels([f"z{z}" for z in pz])
    ax.set_ylim(0, 118)
    ax.set_xlabel("On-the-fly tile zoom (beyond pyramid)")
    ax.set_ylabel("% of intersecting partition read")
    ax.set_title("Read-time pruning: data touched per on-demand tile")
    ax.text(len(xx) / 2 - 0.5, 50,
            "geometries actually decoded: <0.1% of partition\n"
            "(row-group statistics skip ~90%, then a row filter)",
            fontsize=8.3, color="#333", ha="center",
            bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    save(fig, "fig5_pruning_effectiveness")

# ============================ Fig 6: queryable-store latency ============================
q_csv = HERE / "results_v2" / "queryable.csv"
if q_csv.exists():
    qscale, qlat, qfeat = [], [], []
    with open(q_csv) as f:
        for r in csv.DictReader(f):
            qscale.append(f"{r['scale']}\n({r['box_deg']}°)")
            qlat.append(float(r["lat_ms"]))
            qfeat.append(int(r["features"]))
    xx = np.arange(len(qscale))
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    bars = ax.bar(xx, qlat, color="#3a8fb0", width=0.6)
    ax.set_yscale("log")
    ax.axhline(2.1, color="#2e7d4f", ls=":", lw=1.6)
    ax.text(xx[-1] + 0.1, 2.3, "attribute stats: 2.1 ms", fontsize=8, color="#2e7d4f", ha="right")
    for r, lt, nf in zip(bars, qlat, qfeat):
        ax.annotate(f"{lt:.0f} ms\n{nf:,} feat", (r.get_x() + r.get_width()/2, lt),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    ax.set_xticks(xx); ax.set_xticklabels(qscale)
    ax.set_ylim(1, 20000)
    ax.set_xlabel("Spatial query box size (degrees)")
    ax.set_ylabel("Query latency (ms, log scale)")
    ax.set_title("Queryable store: spatial feature-query latency")
    ax.text(0.4, 9000, "latency ∝ result size\n(before fix: 0 features in 30–85 s)",
            fontsize=8.2, color="#444", ha="left",
            bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
    save(fig, "fig6_queryable_store")

print("\nAll figures in", OUT)
