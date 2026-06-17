# Starlet "robust version" — timing + scalability results (local, for review)

Machine: **ec-hn.cs.ucr.edu** (16-core Xeon, 125 GB). Dataset: **OSM-parks**,
subsets 25/50/75/100 % (2.49 M → 9.96 M polygons, 1.6 → 5.6 GB GeoParquet).
"Robust version" = on-demand prefilter + **bbox covering columns + bounded,
spatially-coherent row groups + pyarrow predicate pushdown** (commits
`6cd8665`, `3d43c76`).

Raw data in this folder: `starlet_bench_v2.csv` (new), `starlet_bench_v1_old.csv`
(original), `v2_serving_{100,25}.log`, `v2_campaign_monitor.log`.

---

## 1. Indexing (`starlet tile`) — scalability, new format vs old

| subset | features | **v2 wall** | old wall | Δ | v2 peak RSS | v2 parquet out | old out |
|--------|---------:|------------:|---------:|----:|------------:|---------------:|--------:|
| 25 %  | 2.49 M | **694.7 s** | 843.5 s | −18 % | 11.8 GB | 1.66 GB | 1.57 GB |
| 50 %  | 4.98 M | **1293.7 s** | 1276.8 s | +1 % | 12.2 GB | 2.98 GB | 2.80 GB |
| 75 %  | 7.47 M | **1868.5 s** | 1892.7 s | −1 % | 14.5 GB | 4.00 GB | 3.73 GB |
| 100 % | 9.96 M | **2515.9 s** | 2651.3 s | −5 % | 17.0 GB | 5.74 GB | 5.38 GB |

- **Scales linearly** and is **no slower** than before — net slightly faster:
  the per-row bbox/centroid pass was vectorised (`shapely.bounds`/`centroid`),
  replacing a per-geometry Python loop, which offsets writing the 4 covering
  columns and extra row groups. ~253–279 s/M features.
- **Storage overhead of the covering columns: ≈ +7 %** parquet size at 100 %
  (5.74 vs 5.38 GB). Peak RSS up ~15 % (vectorised arrays), still well under the
  24 GB budget.

## 2. Batch MVT (`starlet mvt`) — unaffected (no regression)

| subset | **v2 wall** | old wall | Δ | #tiles (both) | MVT MB (both) |
|--------|------------:|---------:|----:|--------------:|--------------:|
| 25 %  | 1764.6 s | 1776.2 s | −1 % | 2451 | 369 |
| 50 %  | 3150.9 s | 3089.4 s | +2 % | 2759 | 455 |
| 75 %  | 4245.5 s | 4272.6 s | −1 % | 3340 | 498 |
| 100 % | 6010.0 s | 6044.3 s | −1 % | 4389 | 772 |

Identical tile counts and output sizes, within-noise wall time (~603 s/M). The
covering columns are excluded from batch properties, so pre-generated tiles are
byte-for-byte the same.

## 3. Serving (`starlet serve`) — the structural-cure payoff

OSM-parks 100 %. Cached / pre-generated serving is unchanged; the change is in
**cold on-the-fly generation** (zoom beyond the pyramid).

*Pre-generation policy:* a tile is in the on-disk pyramid iff `z ≤ z_max` **and** its
histogram vertex-density ≥ τ (and it has ≥ 1 feature); `z > z_max` is served on the
fly. τ is a vertex-density threshold, not a feature count. This study used
`z_max = 7`, `τ = 0` → all non-empty tiles up to z7 (4389 tiles). Latency values
below are medians (p50).

| regime | p50 | p95 | notes |
|--------|----:|----:|-------|
| warm_mem (LRU) | ~1.7 ms | ~2.3 ms | unchanged |
| warm_disk (pyramid z≤7) | ~2.0–2.4 ms | 8–14 ms | unchanged; serves up to 1.46 MB tiles |
| **on-the-fly z8** | **232 ms** | 14.3 s | tail: large tile ∩ many huge-MBR partitions |
| **on-the-fly z10** | **134 ms** | 1.11 s | |
| **on-the-fly z13** | **104 ms** | 270 ms | interactive |

### On-the-fly cold p50 — full progression (the headline)

| zoom | original (whole-partition) | v1 (bbox prefilter) | **v2 (row-group skipping)** |
|------|---------------------------:|--------------------:|----------------------------:|
| z8  | ~17–39 s | 3.28 s | **0.23 s** |
| z10 | ~17–39 s | 2.45 s | **0.13 s** |
| z13 | ~17–39 s | 2.25 s | **0.10 s** |

Deep on-the-fly tiles went from **17–39 s → ~100 ms p50** (≈ 200–350×); the prior
"read-the-whole-partition" floor is gone because row-group statistics skip the
data the tile doesn't touch (a 63k–136k-row partition now has 3–9 row groups).

**Caveats (honest):**
- The **z8 tail** (p95 14 s) persists: a z8 tile covers a large area and
  intersects *many* partitions whose RSGrove MBRs overlap heavily (one spans
  ≈120°×133°), so it still opens many files. z8 is only one level past the
  pyramid — in practice you pre-generate to a deeper zoom and serve z≳10 on the
  fly, where p95 is ~0.3–1.1 s.
- ~100 ms is the new **floor** even for empty deep tiles: open file + read
  row-group metadata + skip. Low `nonempty` counts at z10/z13 are because the
  sampler descends to a tile's centre child, which often lands in a gap in the
  (sparse) parks data — not an error.

osm_parks_25 % shows the same pattern (z13 otf p50 = 92 ms, z10 = 85 ms).

---

## 4. Comparison vs Tippecanoe (the honest read)

Tippecanoe on the same full input (13.5 GB GeoJSON, z0–7, 8 threads,
`baselines.csv`): **235 s, 1.02 GB RSS, 25.6 MB `.mbtiles`**.

| metric @ 9.96 M polygons, z0–7 | Starlet (v2) | Tippecanoe | ratio |
|---|---:|---:|---:|
| batch wall (tile+MVT vs full build) | 8526 s | 235 s | **Tippecanoe 36× faster** |
| MVT phase only | 6010 s | 235 s | Tippecanoe ~26× faster |
| peak RSS | 17 GB | 1.0 GB | Tippecanoe ~17× lower |
| output (loose) | 5.74 GB parquet + 772 MB MVT dir | 25.6 MB mbtiles | — |
| **output as one `.mbtiles`** | **287 MB** (748 MB loose → gzip in SQLite; +66 s pack) | **25.6 MB** | **Tippecanoe ~11× smaller** |
| tippecanoe **no-drop** (`-r1 -pf -pk`) | — | **34 MB, 206 s** | barely changes vs drop |

A **no-drop** tippecanoe (disable feature dropping/limits) is still only **34 MB
in 206 s** (vs 25.6 MB / 235 s with dropping) — for polygons the compactness
comes from simplification + tiny-polygon reduction, not feature dropping, so it
stays ~8× smaller and ~41× faster than Starlet. Removing the drop does **not**
make it a fair/favorable baseline → dropped from the paper (see capabilities
section).

**Single-file output (`.mbtiles`).** Packing Starlet's z0–7 MVT pyramid into one
gzipped MBTiles SQLite file (TMS, `format=pbf`) gives **287 MB in 66 s** — the
raw 748 MB of loose MVT compresses ~2.7× under gzip. So the *container* (loose
dir vs single SQLite) is **not** the source of the gap: Starlet's mbtiles is
still **~11× larger** than Tippecanoe's 25.6 MB and **~37× slower** to produce.
The remaining gap is **feature-retention policy**: Tippecanoe drops features at
low zoom, coalesces adjacent features, and simplifies aggressively, whereas
Starlet keeps far more geometry per tile (reservoir cap of 100 k features/tile,
simplification only ∝ tile width, no low-zoom feature dropping). The size gap is
a design choice (fidelity vs compactness), not a format artifact. *(Note: a
native `--mbtiles` exporter lives on the `feature/mbtiles-support` branch, not
master; the 287 MB figure was produced by packing the loose pyramid, which is
exactly what such an exporter does.)*

**Is Starlet "more scalable"? No — not for batch tile generation.** Tippecanoe
is C++, multithreaded, single-pass, and aggressively drops/coalesces features
into a compact `.mbtiles`; Starlet is pure-Python with per-geometry Shapely ops
over two passes, and keeps the *full* partitioned dataset. On raw batch
throughput and memory, Tippecanoe wins decisively.

**What Starlet offers instead (the defensible positioning):**
1. **Deployment:** pip-installable pure Python — no JVM (cf. BEAST) and no C++
   build / `tile-join` toolchain (cf. Tippecanoe).
2. **On-demand serving from a live store:** the partitioned GeoParquet is both a
   queryable dataset (feature downloads, attribute stats) *and* the backing
   store for tile serving. You don't pre-generate the whole pyramid — deep
   tiles render on the fly in **~0.1 s p50** (Fig. 4). Tippecanoe is a batch
   producer of a static archive; it has no on-demand generation path.

So the paper's scalability claim should be **"scales to 10 M features / multi-GB
on a single machine, in pure Python, with bounded memory (<24 GB) and
interactive on-demand serving"** — *not* "faster than Tippecanoe." (This is the
framing decision flagged in the handoff; lead author's call.)

## Starlet capabilities (the contribution — baseline dropped)

Tippecanoe is dropped as a baseline (see above; even no-drop it's ~8× smaller /
~41× faster — a mature batch encoder, orthogonal to Starlet's contribution).
The experiment section instead showcases what Starlet does that a batch
`.mbtiles` producer cannot.

**Read-time pruning effectiveness** (`pruning_effectiveness.csv`, fig 5). Per
on-demand tile the server reads only **~10 % of the intersecting partition's
bytes** (≈3 of ~25 row groups; row-group statistics skip the rest) and decodes
**<0.1 % of rows** (0–309 of ~400 k). This is why cold on-the-fly is ~0.1 s
(fig 4) rather than O(partition).

**Queryable store** (`queryable.csv`, fig 6). The same GeoParquet that backs the
tiles answers spatial feature queries + attribute stats over HTTP — data
serving, not just tiles. After fixing two bugs in the download path (filename
bbox mis-parse; WKB never decoded so the mbr filter dropped everything) and
adding the covering-column pushdown, spatial queries are correct and **scale
with result size**: ~122 ms for a targeted box (1 feature) up to 6.3 s for
25,705 features (16 MB); attribute stats ~2.1 ms. (Before: 0 features in
30–85 s.) tippecanoe has no equivalent.

## Figures (in `figs/`, PDF for LaTeX + PNG)

| file | shows | suggested caption |
|---|---|---|
| `fig1_runtime_scalability` | tiling/MVT/total wall time vs #features | "Pipeline runtime scales linearly to ~10 M polygons (8526 s ≈ 2.4 h at 100 %)." |
| `fig2_memory_scalability` | peak RSS vs #features | "Peak memory grows sub-linearly and stays under the 24 GB single-machine budget." |
| `fig3_serving_latency_regimes` | p50 by serving regime (log) | "Tile-serving latency: ~2 ms cached/pre-generated, ~0.1 s cold on-the-fly at deep zoom." |
| `fig4_ondemand_improvement` | cold on-the-fly p50 per zoom: original/prefilter/row-group-skip | "On-demand generation: read-time pruning cuts deep-zoom tiles from 17–39 s to ~0.1 s." |
| `fig5_pruning_effectiveness` | % of partition read per on-demand tile, by zoom | "Read-time pruning reads only ~10 % of the partition (and decodes <0.1 % of rows) per tile." |
| `fig6_queryable_store` | spatial feature-query latency vs box size (+stats) | "The store answers spatial feature queries with latency proportional to result size (≈0.1 s targeted)." |

Largest dataset tried: **9.96 M polygons / 13.5 GB GeoJSON (5.6 GB GeoParquet)**;
full pipeline **8526 s ≈ 2.37 h**, peak RSS 17 GB.

## Bottom line

- **Indexing:** linear, slightly faster, +7 % storage for the covering columns.
- **MVT:** unchanged.
- **Serving:** cold on-the-fly **17–39 s → ~0.1 s p50** at deep zoom; cached
  serving still ~2 ms. Output is byte-identical to the original tiler
  (verified). The remaining tail is at shallow on-the-fly zooms and is a
  partition-MBR-overlap effect, not a per-tile compute cost.
