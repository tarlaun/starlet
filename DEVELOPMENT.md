# Developing Starlet

This guide is for working on Starlet itself — running it from a clone, running
the tests, understanding the codebase, and cutting releases. If you just want to
*use* Starlet, see [README.md](README.md) (`pip install starlet`).

## Setup (from source)

```bash
git clone https://github.com/ucr-bdlab/starlet.git
cd starlet

python -m venv .venv
source .venv/bin/activate

# editable install with the dev/test extras
pip install -e ".[dev]"
```

The editable install exposes the `starlet` console script and points it at your
working tree, so code changes take effect without reinstalling.

```bash
starlet --help          # sanity check
```

## Running the tests

```bash
pytest tests/ -v
```

A coverage run (matches CI):

```bash
pytest tests/ --cov=starlet --cov-report=term
```

Tests are pure `pytest` and need no external services. To run a single test:

```bash
pytest tests/test_mvt/test_intermediate_tile.py -v
pytest tests/test_mvt/test_intermediate_tile.py::test_merge_is_order_independent
```

## How the code is organized

The public API surface is deliberately small. Everything user-facing is
re-exported from `starlet/__init__.py` (`tile`, `generate_mvt`, `build`,
`export_pmtiles`, `create_app`) and `starlet/_types.py` (`TileResult`,
`MVTResult`, `Dataset`). Everything under `starlet/_internal/` is private and
may change without notice.

```
starlet/
  __init__.py            # public API (lazy imports keep CLI startup fast)
  _types.py              # frozen result/dataset dataclasses
  _cli.py                # Click CLI (one subcommand per public function)
  _internal/
    config.py            # TOML config + defaults + CLI/file/default precedence
    tiling/              # partitioning: datasource, RSGrove partitioner,
                         # two-stage orchestrator, writer pool
    histogram/           # density-histogram pyramid
    mvt/                 # streaming map/reduce MVT generation
    pmtiles/             # PMTiles export
    server/              # Flask tile server, on-demand tiler, web viewer
    stats/               # attribute-statistics sketches
```

Each CLI subcommand in `_cli.py` maps one-to-one to a public function in
`__init__.py`. The CLI's only job is to resolve options (CLI flag → config file
→ built-in default, via `_internal/config.py`) and call that function.

## The pipeline

A *dataset* on disk is a directory; the stages communicate through it rather
than through in-memory objects:

```
datasets/<name>/
  parquet_tiles/         # spatially-partitioned GeoParquet (one file per tile)
  histograms/            # density histograms (global.npy, global_prefix.npy)
  stats/attributes.json  # per-attribute statistics incl. geometry MBR
  mvt/<z>/<x>/<y>.mvt    # pre-generated vector tiles (optional)
  tiles.pmtiles          # single-file archive (optional, with --pmtiles)
```

1. **Tiling** (`tiling/`, driven by `starlet.tile()`). A `DataSource`
   (`GeoParquetSource` / `GeoJSONSource` / CSV) streams Arrow tables. An
   **assigner** (default `RSGrove`, which builds balanced partitions by
   reservoir-sampling centroids) maps each row to a spatial tile. A two-stage
   orchestrator scatters rows to per-tile buffers via a `WriterPool` and merges
   them into final GeoParquet files; rows are optionally sorted within a tile by
   a space-filling curve (`--sort zorder|hilbert`). Attribute statistics are
   collected in the same pass, then the histogram pyramid is built.

2. **MVT generation** (`mvt/`, driven by `starlet.generate_mvt()`). A map/reduce
   design keeps memory bounded at high zoom: **map** workers stream Parquet row
   groups and sample features per output tile into a bounded
   `IntermediateVectorTile`, spilling partials to Arrow IPC; **reduce** workers
   merge the partials for each tile and encode the `.mvt` protobuf. Sampling is
   deterministic and *seam-consistent* — a geometry's keep/drop priority is
   `crc32(WKB)`, so a feature spanning several tiles is kept or dropped
   consistently in all of them (no visible seams between adjacent tiles).

3. **Serving** (`server/`, driven by `starlet.create_app()`). A Flask app with a
   four-tier tile lookup: in-memory LRU cache → pre-generated PMTiles archive →
   pre-generated `.mvt` on disk → generated on the fly from the intersecting
   Parquet tiles. On-the-fly tiles are promoted into the cache. The app also
   serves the web viewer, dataset metadata, and feature downloads.

## Conventions

- Source data is EPSG:4326 (lon/lat); MVT/tile math is EPSG:3857 (Web Mercator).
  Reprojection happens in the streaming/rendering stages.
- The internal tile-partition column is `geo_parquet_tile_num`.
- Result objects (`TileResult`, `MVTResult`) are frozen dataclasses. When adding
  a field, append it with a default so positional construction stays compatible.
- Keep new public surface minimal — add to `__init__.py`/`_types.py` only what
  callers need; treat `_internal/` as private.
- Imports inside the public functions are **lazy** (done in the function body) so
  `import starlet` and `starlet --help` stay fast and don't drag in the geo stack.

## Adding a CLI flag or config key

Because CLI options resolve through the config system, a new tunable touches
three places:

1. Add the option to the relevant command in `_cli.py` (default `None` so the
   config layer can supply it).
2. Add its default to `DEFAULT_CONFIG` in `_internal/config.py`, and thread it
   through with `resolve_command_value(...)`.
3. Add the matching keyword argument to the public function in `__init__.py`.

Document it in the README's option table (if user-facing) and in
[docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## Packaging

`pyproject.toml` uses setuptools. Package data includes the server templates and
the web viewer under `starlet/_internal/server/` — when you add a new template
or static asset, register it under `[tool.setuptools.package-data]` or it won't
ship in the wheel.

## Continuous integration

`.github/workflows/publish.yml` runs on every `v*.*.*` tag push:

1. **test** — `pytest` on Python 3.10 / 3.11 / 3.12 (gates everything below).
2. **build** — builds the sdist/wheel.
3. **benchmark** — runs `.github/scripts/run_benchmark.py` on a 1 GB dataset and
   attaches the results to the GitHub Release. (A benchmark failure does **not**
   block the PyPI publish.)
4. **publish** — uploads to PyPI via trusted publishing (OIDC, `pypi`
   environment).

## Cutting a release

See [RELEASE.md](RELEASE.md) for the full process. In short:

1. Bump `version` in `pyproject.toml`.
2. Commit and push to `master`.
3. Tag and push: `git tag vX.Y.Z && git push origin vX.Y.Z` — the tag push runs
   the workflow above and publishes to PyPI.

> PyPI rejects re-uploading an existing version, so make sure `pyproject.toml`'s
> version is bumped before tagging.

## Deployment

For standing up a live tile server (including a no-root Apache/CGI recipe), see
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).
