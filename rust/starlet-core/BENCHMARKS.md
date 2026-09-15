# starlet-core benchmarks

All numbers measured on an 11-core Apple laptop (18 GB RAM), starlet on the
`rust` branch, extension built with `maturin develop --release`, Python 3.12.
The machine was otherwise idle; times are medians of 5 unless noted.

## Datasets

| dataset | features | geometry | partitions | bbox columns |
|---|---:|---|---:|---|
| `osm_parks_25` | 2,490,306 | polygons (OSM parks, 25% sample, 1.6 GB GeoParquet) | 17 | yes (`starlet tile` default) |
| `states_provinces` | 4,594 | polygons (Natural Earth admin-1) | 12 | yes |
| `TIGER2018_COUNTY` | 3,233 | polygons (US counties) | 58 | **no** (legacy build) |
| `riverside_vegetation_types` | 17,294 | polygons | 56 | **no** |

## 1. On-the-fly tile latency (the interactive path)

Per-tile generation time for a tile that is *not* pre-generated, Python
pipeline vs Rust engine. "steady state" is a repeated request (the Rust
row-group LRU is warm; Python re-reads the parquet each time — it has no
equivalent cache, and its warm and cold numbers differ little). The Rust
"first touch" number is the one-off cost of opening the dataset and decoding
the first row group that a tile needs; every later tile that touches the
same row group pays the steady-state cost.

### `osm_parks_25` — 2.5M polygons, bbox columns (Tokyo)

| zoom | Python | Rust steady state | speedup | Rust first touch | tile bytes |
|---:|---:|---:|---:|---:|---:|
| 8  | 909 ms | 23.3 ms | **39×** | 171 ms | 399 KB |
| 10 | 623 ms | 9.1 ms | **69×** | 126 ms | 277 KB |
| 12 | 119 ms | 1.85 ms | **64×** | 124 ms | 40 KB |
| 14 | 50 ms | 0.85 ms | **58×** | 119 ms | 6.3 KB |
| 16 | 39 ms | 0.71 ms | **54×** | 120 ms | 707 B |
| 18 | 38 ms | 0.69 ms | **55×** | 118 ms | 325 B |
| 20 | 38 ms | 0.68 ms | **55×** | 123 ms | 108 B |

Python's ~38 ms floor at z14–z20 is fixed per-request overhead (index setup,
filtered parquet read, pyproj), paid on every tile regardless of content.

### `states_provinces` — bbox columns (Europe)

| zoom | Python | Rust steady state | speedup |
|---:|---:|---:|---:|
| 4  | 391 ms | 18.6 ms | **21×** |
| 8  | 40 ms | 0.18 ms | **227×** |
| 12 | 40 ms | 0.18 ms | **219×** |
| 16 | 40 ms | 0.17 ms | **237×** |
| 20 | 41 ms | 0.18 ms | **231×** |

### Datasets **without** bbox columns (WKB-bbox fallback)

| dataset / zoom | Python | Rust steady state | speedup |
|---|---:|---:|---:|
| TIGER z5 (wide tile, 102 counties) | 113 ms | 10.7 ms | 10.5× |
| TIGER z8 / z12 / z20 | 6.5 / 2.6 / 2.7 ms | 0.95 / 0.53 / 0.63 ms | 4–7× |
| riverside z9 (9,994 features) | 1,638 ms | 46.9 ms | **35×** |
| riverside z12 / z15 / z20 | 64 / 16 / 2.1 ms | 1.84 / 0.57 / 0.40 ms | 5–35× |

Without bbox columns the engine computes per-row bboxes with a
zero-allocation WKB scan on first touch and caches them with the row group;
the first tile over a large partition pays that scan (TIGER z5: 256 ms cold).
Re-tiling with `--covering-bbox` (the default) gives the fast path above.

### End to end through `starlet serve`

`starlet serve` on `osm_parks_25`, first request after start, then z14–z20
tiles no pre-generated pyramid covers (server log `method=generated`):

| request | server-side generation | HTTP round trip (curl, Flask dev server) |
|---|---:|---:|
| z12 (first request: opens dataset) | 145 ms | 158 ms |
| z14 | 1 ms | 17.5 ms |
| z16 / z18 / z20 | 1 ms | 14 ms |

Before, the same z14–z20 requests spent 40–60 ms in generation alone.

## 2. Batch pyramid generation (preprocessing)

`starlet mvt --threshold 0`, Python map/reduce (default) vs Rust pull-based
pyramid (`STARLET_ENGINE=rust`), wall clock and peak RSS from
`/usr/bin/time -l`.

### `osm_parks_25` — 2.5M polygons

| max zoom | Python map/reduce | Rust | tiles | output |
|---:|---|---|---:|---:|
| 6 | **79.4 s**, 0.96 GB RSS, **6.7 GB** peak temp files | **5.1 s** (15.7×), 2.7 GB RSS, no temp files | 830 (identical per-zoom counts) | 88 MB |
| 8 | **killed** — spilled **9.5 GB** of temporary intermediate files and exhausted the disk before writing a tile | **7.7 s**, 2.6 GB RSS, no temp files | 6,630 | 380 MB |
| 10 | **killed** — >10 GB of temporary files, 0 tiles written after several minutes | **15.3 s**, 2.6 GB RSS | 47,626 | 994 MB |
| 12 | not attempted (would need more scratch disk than the laptop has) | **51.7 s**, 3.1 GB RSS | 344,722 | 2.5 GB |

The Python map stage materialises every feature into an intermediate tile
file per zoom before the reduce stage merges them, so its temporary footprint
grows with features × zoom levels; the Rust path writes only the output.
These rows were measured with the first (pull-mode) Rust batch path, whose
RSS came from pinning decoded row groups per tile; the push-mode path that
replaced it (next section) needs a fraction of that. Note also that the
Python RSS column is `/usr/bin/time` on the parent process only — it does
not sum the multiprocessing workers. (TileAQP's cluster measurement of the
same two designs on 10M parks at z12: starlet 1 h 48 min with 86 GB of
temporaries vs 3 min, 36× — consistent with this.)

### Push-mode pyramid (bounded memory) — `osm_parks_full`, 9.96M polygons, ec-hn

The first Rust batch path *pulled* row groups per tile, so a z0 tile pinned
every decoded row group at once and parallel tiles multiplied it: **33–35 GB
RSS** on the full parks dataset. `write_pyramid` now streams the row groups
twice (per-tile top-k *references* first, then one decode of each winning
row), writing a tile the moment its last winner lands. Measured on ec-hn
(16-core Xeon, 125 GB), `feature_capacity` 25,000, output identical per tile:

| zoom range | pull-mode (per-tile) | push-mode (streaming) | tiles | output |
|---|---|---|---:|---:|
| z0–9 | 171 s, **34.7 GB** RSS | **42 s, 4.95 GB** RSS | 35,112 (same set; 63/63 sampled tiles identical) | 2.6 GB |
| z0–12 | not run (memory) | **227 s, 7.0 GB** RSS | 743,927 | 8.4 GB |
| TIGER z0–7 (laptop) | 1.63 s | **0.78 s**, 0.5 GB RSS | 548 (identical) | — |

For reference, starlet's Python map/reduce needed 34 min (v0.4.0, z0–19,
threshold 50k) on the same input, and its earlier version 5 h 43 min at z7.

### `TIGER2018_COUNTY` — 3,233 counties, z0–7 (both complete)

| | Python map/reduce | Rust |
|---|---:|---:|
| wall clock | 8.86 s | **1.63 s** (5.4×) |
| per-zoom tiles | [1, 3, 5, 10, 22, 49, 119, 339] | identical |
| tile set | 548 tiles | identical set |
| feature sets, 40 random tiles | — | 40/40 identical |

## 3. Output parity

`tests/test_mvt/test_rust_engine.py` checks Rust against the Python reference
on a synthetic dataset (selection, attribute typing, buffers, capacity, batch
writes, kill switch). On the real datasets above the selected-feature *sets*
match on the vast majority of tiles; the remainder differ by 0.1–0.3% of
features at the clip edge in both directions (shapely's `clip_by_rect`
returns `GeometryCollection`s whose sliver parts Python emits as duplicate
features; sub-pixel slivers < 0.5 tile units² are dropped here). Tile byte
sizes agree to within ~1%.

Two Python-side issues surfaced while validating and were fixed on this
branch: the legacy (no-bbox) on-demand path stringified integer attributes
(`numpy.int64` is not a Python `int`), which broke numeric styling on
on-demand tiles; and its priority hashes the *reprojected* WKB rather than the
source WKB, unlike the batch pipeline — documented, not changed.

## Reproducing

```bash
cd rust/starlet-core && maturin develop --release && cd ../..
starlet tile --input osm_parks_25.parquet --outdir /tmp/osm_parks_25      # ~52 s, 1.4 GB RSS
STARLET_ENGINE=rust   starlet mvt --dir /tmp/osm_parks_25 --zoom 12       # Rust pyramid
STARLET_ENGINE=python starlet mvt --dir /tmp/osm_parks_25 --zoom 6        # Python pyramid (watch your disk)
starlet serve --dir /tmp --port 8765 --log-level INFO                     # on-demand tiles via Rust
```
