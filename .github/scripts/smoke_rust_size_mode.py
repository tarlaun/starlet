"""End-to-end check of a starlet-core wheel: build a small dataset with the
Rust engine in size-threshold mode and export PMTiles.

Run by .github/workflows/publish-core.yml after installing a freshly built
wheel next to starlet; usable locally too:

    STARLET_ENGINE=rust python .github/scripts/smoke_rust_size_mode.py

Kept in a file (not a heredoc) because the sampling stage uses a process pool
and spawned workers re-import ``__main__``.
"""
from __future__ import annotations

import json
import os
import pathlib
import tempfile


def main() -> None:
    os.environ.setdefault("STARLET_ENGINE", "rust")
    import starlet
    from starlet._internal.mvt import rust_engine

    starlet_core = rust_engine._load()
    assert starlet_core is not None, "starlet.starlet_core (the Rust extension) is not importable"

    print("starlet", starlet.__version__, "starlet_core", starlet_core.__version__, starlet_core.num_threads(), "threads")
    assert rust_engine.available(), "starlet_core not picked up by starlet"
    assert rust_engine.batch_enabled(), "STARLET_ENGINE=rust not honoured"

    root = pathlib.Path(tempfile.mkdtemp(prefix="starlet_core_smoke_"))
    features = [
        {
            "type": "Feature",
            "properties": {"id": i},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[x, y], [x + 0.5, y], [x + 0.5, y + 0.5], [x, y + 0.5], [x, y]]],
            },
        }
        for i, (x, y) in enumerate((x, y) for x in range(-10, 10) for y in range(-10, 10))
    ]
    (root / "grid.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))

    tile_result, mvt_result, pmtiles_path = starlet.build(
        input=str(root / "grid.geojson"),
        outdir=str(root / "ds"),
        min_tile_bytes="2kb",
        pmtiles=True,
        parallelism=2,
    )
    print("rows", tile_result.total_rows, "tiles by zoom", mvt_result.tile_counts_by_zoom, "pmtiles", pmtiles_path)
    assert tile_result.total_rows == 400, tile_result.total_rows
    assert mvt_result.tile_count > 1, mvt_result.tile_counts_by_zoom
    assert pmtiles_path and pathlib.Path(pmtiles_path).stat().st_size > 0

    # The Rust single-tile path must agree with the Python reference on tile
    # content: same features, same attributes, equivalent geometry (the two
    # encoders may start a clipped ring at a different vertex).
    import mapbox_vector_tile
    from shapely.geometry import shape
    from starlet._internal.mvt.mvt_generator import _generate_single_mvt_tile_python, generate_single_mvt_tile

    def decoded(data: bytes):
        layers = mapbox_vector_tile.decode(data)
        return sorted(
            (shape(f["geometry"]).normalize().wkt, json.dumps(f["properties"], sort_keys=True))
            for layer in layers.values() for f in layer["features"]
        )

    dataset = str(root / "ds")
    for z, x, y in ((0, 0, 0), (2, 1, 1), (4, 7, 7)):
        rust_features = decoded(generate_single_mvt_tile(dataset, (z, x, y)))
        python_features = decoded(_generate_single_mvt_tile_python(dataset, (z, x, y)))
        assert rust_features, f"tile {z}/{x}/{y} is empty"
        assert rust_features == python_features, f"tile {z}/{x}/{y}: {len(rust_features)} Rust vs {len(python_features)} Python features"
        print(f"tile {z}/{x}/{y}: {len(rust_features)} features, engines agree")
    print("ok")


if __name__ == "__main__":
    main()
