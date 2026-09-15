# starlet-core — Rust acceleration for starlet

An optional native extension (`starlet_core`) that generates Mapbox Vector
Tiles straight from a starlet dataset's `parquet_tiles/`, with starlet's own
selection semantics, using all cores and releasing the GIL. starlet detects it
at import time and uses it for:

- **on-the-fly tiles** served past the pre-generated zoom levels
  (`generate_single_mvt_tile`, hence `starlet serve`), and
- **batch pyramid generation** (`starlet mvt` / `generate_mvt`) when opted in.

Without it, starlet runs its pure-Python pipeline unchanged.

## Build

```bash
# Rust toolchain (once): https://rustup.rs
pip install maturin                 # into the same venv as starlet
cd rust/starlet-core
maturin develop --release           # builds and installs `starlet_core` into the active venv
cargo test --release                # Rust unit tests
```

The wheel is `abi3` (Python ≥ 3.10), so one build serves every interpreter.
Verify:

```bash
python -c "import starlet_core; print(starlet_core.__version__, starlet_core.num_threads())"
```

## Engine selection

`STARLET_ENGINE` controls use of the extension:

| value | on-the-fly tiles | batch pyramid (`starlet mvt`) |
|---|---|---|
| `auto` (default) | Rust when available | Python map/reduce |
| `rust` | Rust | **Rust** (pull-based, parallel) |
| `python` | Python | Python |

## How it works

**On-the-fly tile** (`Dataset.generate_tile(z, x, y)`): partitions are pruned by
their filename bbox, row groups by the `_bbox_*` column statistics (written by
`starlet tile --covering-bbox`, the default), and rows by bbox overlap.
Selection is *raster-consistent*: a candidate whose bbox fits inside one
display pixel (`extent / PIXEL_GRID` tile units, PIXEL_GRID = 256 like a
256 px raster tile) competes only with the other sub-pixel candidates of the
same pixel cell and the one with the highest `crc32(source WKB)` priority is
kept, so the tile shows every occupied pixel with a bounded feature count;
larger candidates are ranked by the same priority and the top
`feature_capacity` kept. The priority is geometry-intrinsic, so adjacent
on-demand and pre-generated tiles agree on what they keep. Winners go through
starlet's pipeline: affine to tile units; a sub-pixel polygon / line becomes a
one-pixel square / segment of its own type (a *dot*, no attributes); larger
shapes get Douglas-Peucker at 1 tile unit (>10 vertices), clipping to
extent+buffer, and their attributes (subject to the tile-attributes policy);
encode. Decoded row groups live in an LRU shared across tiles.

Datasets **without** bbox columns still work: per-row bboxes are computed by a
zero-allocation WKB scan on first touch and cached with the row group. That path
is slower on wide tiles over big partitions — re-tile with `--covering-bbox`
for the fast path.

**Batch pyramid** (`Dataset.write_pyramid(outdir, tiles)`): the tile set is the
same one the Python map/reduce produces (tiles passing the histogram/threshold
filter that end up non-empty), for every requested zoom in one call. It runs
in *push mode* with bounded memory (`src/pyramid.rs`): pass 1 streams every
row group once (geometry + bbox columns only) and pushes a 12-byte
`(crc32, row group, row)` reference into the bounded top-k heap of each tile
whose buffered bounds the row's bbox intersects, at every zoom; pass 2
streams the row groups that hold winners, decodes each winning row once and
appends it to its tiles' layer builders, writing a tile to `<z>/<x>/<y>.mvt`
the moment its last winner arrives. Each row group is decoded by one worker
and dropped, so memory is the references plus the tiles still being filled —
never the decoded dataset. No intermediate files, no reduce stage, no temp
disk. (`write_tiles` / `generate_tiles`, the per-tile pull-mode batch, remain
available; they pin every row group a tile touches and are only suitable for
small tile sets.)

## Output parity with the Python pipeline

Validated on TIGER counties, Natural Earth provinces, Riverside vegetation and
a 2.5M-feature OSM parks set: identical tile sets and per-zoom counts for batch
pyramids, and identical selected-feature sets on the vast majority of tiles.
The remaining differences are at the clip edge (≈0.1–0.3% of features on dense
tiles, in both directions): shapely's `clip_by_rect` can return a
`GeometryCollection` whose sliver parts Python emits as duplicate features,
and sub-pixel slivers (< 0.5 tile units²) are dropped here. Geometries are
otherwise encoded to the same coordinates (byte sizes match to within ~1%).

## Limitations

- Partitions are assumed to be EPSG:4326 (starlet checks the GeoParquet CRS and
  falls back to Python for anything else).
- Only the `layer0` layer name is produced (starlet's default).
- Python's legacy no-bbox path ranks by the *reprojected* WKB, not the source
  WKB; on such datasets at feature capacity the two engines can pick different
  survivors. The bbox path and the batch pipeline use the source WKB, as here.
