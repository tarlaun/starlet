"""Tests for the optional Rust acceleration core (``starlet_core``).

Skipped entirely when the extension is not installed (build it with
``maturin develop --release`` in ``rust/starlet-core``). Every test compares
the Rust engine against the pure-Python reference on a small dataset built
in ``tmp_path``.
"""
from __future__ import annotations

import json

import mapbox_vector_tile
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from shapely import wkb
from shapely.geometry import LineString, Point, Polygon

starlet_core = pytest.importorskip("starlet_core")

from starlet._internal.mvt import rust_engine  # noqa: E402
from starlet._internal.mvt.mvt_generator import (  # noqa: E402
    _generate_single_mvt_tile_python,
    generate_single_mvt_tile,
)

GEO = {
    "version": "1.1.0",
    "primary_column": "geometry",
    "columns": {"geometry": {"encoding": "WKB", "crs": "EPSG:4326"}},
}


def _features():
    """A few features in the NW quadrant (tile 1/0/0) plus one in the SE."""
    geoms = [
        Point(-100.0, 80.0),
        Polygon([(-60, 30), (-40, 30), (-40, 50), (-60, 50), (-60, 30)]),
        LineString([(-120, 10), (-80, 20), (-30, 60)]),
        Point(100.0, -80.0),  # SE quadrant: must not appear in 1/0/0
    ]
    return geoms


def _write_dataset(root, *, with_bbox: bool):
    parquet_dir = root / "parquet_tiles"
    parquet_dir.mkdir(parents=True)
    geoms = _features()
    cols = {
        "geometry": [wkb.dumps(g) for g in geoms],
        "id": pa.array([1, 2, 3, 4], pa.int64()),
        "area_m2": pa.array([1.5, 12495083015.0, 0.25, 7.0], pa.float64()),
        "name": ["a", "b", "c", "d"],
        "flag": [True, False, True, False],
    }
    if with_bbox:
        b = [g.bounds for g in geoms]
        cols["_bbox_xmin"] = [x[0] for x in b]
        cols["_bbox_ymin"] = [x[1] for x in b]
        cols["_bbox_xmax"] = [x[2] for x in b]
        cols["_bbox_ymax"] = [x[3] for x in b]
    table = pa.table(cols).replace_schema_metadata({b"geo": json.dumps(GEO).encode("utf-8")})
    pq.write_table(table, parquet_dir / "tile_000000__-180_0_-90_0_180_0_90_0.parquet")
    return root


def _props(tile_bytes):
    decoded = mapbox_vector_tile.decode(tile_bytes)
    out = set()
    for layer in decoded.values():
        for f in layer["features"]:
            out.add(tuple(sorted(f["properties"].items())))
    return out


@pytest.mark.parametrize("with_bbox", [True, False], ids=["bbox-cols", "wkb-bbox-fallback"])
def test_rust_matches_python_feature_selection_and_properties(tmp_path, with_bbox):
    ds = _write_dataset(tmp_path / "ds", with_bbox=with_bbox)
    rust_engine.invalidate()
    py = _generate_single_mvt_tile_python(str(ds), (1, 0, 0), feature_capacity=10)
    rs = rust_engine.generate_tile(str(ds), 1, 0, 0, feature_capacity=10, extent=4096, buffer=256)

    py_props, rs_props = _props(py), _props(rs)
    assert rs_props == py_props
    ids = {dict(p)["id"] for p in rs_props}
    assert ids == {1, 2, 3}  # the SE point (id 4) is outside tile 1/0/0
    # attribute typing survives (int stays int, float stays float, bool stays bool)
    big = next(dict(p) for p in rs_props if dict(p)["id"] == 2)
    assert big["area_m2"] == pytest.approx(12495083015.0)
    assert isinstance(big["id"], int) and isinstance(big["flag"], bool)


def test_dispatcher_uses_rust_and_kill_switch_falls_back(tmp_path, monkeypatch):
    ds = _write_dataset(tmp_path / "ds", with_bbox=True)
    rust_engine.invalidate()
    monkeypatch.delenv(rust_engine.ENV_VAR, raising=False)
    assert rust_engine.available()
    via_dispatch = generate_single_mvt_tile(str(ds), (1, 0, 0), feature_capacity=10)
    assert _props(via_dispatch) == _props(
        rust_engine.generate_tile(str(ds), 1, 0, 0, feature_capacity=10, extent=4096, buffer=256)
    )
    monkeypatch.setenv(rust_engine.ENV_VAR, "python")
    assert not rust_engine.available()
    assert not rust_engine.batch_enabled()
    assert _props(generate_single_mvt_tile(str(ds), (1, 0, 0), feature_capacity=10)) == _props(via_dispatch)


def test_batch_write_tiles_matches_python_per_tile(tmp_path):
    """write_tiles writes exactly the tiles the Python reference finds non-empty,
    with the same features (tile buffers included: a feature near an edge shows
    up in the neighbouring tile's buffer zone in both engines)."""
    ds = _write_dataset(tmp_path / "ds", with_bbox=True)
    rust_engine.invalidate()
    out = tmp_path / "mvt"
    handle = rust_engine.dataset(ds)
    tiles = [(1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)]
    expected = {
        t: _props(_generate_single_mvt_tile_python(str(ds), t, feature_capacity=10)) for t in tiles
    }
    non_empty = [t for t, p in expected.items() if p]
    assert (1, 0, 0) in non_empty and (1, 1, 1) in non_empty and len(non_empty) < len(tiles)

    written = handle.write_tiles(str(out), tiles, feature_capacity=10, extent=4096, buffer=256)
    assert written == len(non_empty)
    for (z, x, y), props in expected.items():
        path = out / str(z) / str(x) / f"{y}.mvt"
        assert path.exists() == bool(props)
        if props:
            assert _props(path.read_bytes()) == props

    gen = handle.generate_tiles(tiles, feature_capacity=10, extent=4096, buffer=256)
    assert [g is None for g in gen] == [not expected[t] for t in tiles]
    assert _props(gen[0]) == expected[(1, 0, 0)]


def test_feature_capacity_keeps_top_k_consistently_with_python(tmp_path):
    ds = _write_dataset(tmp_path / "ds", with_bbox=True)
    rust_engine.invalidate()
    # tile 1/0/0 holds one point (sub-pixel: kept per pixel cell, never
    # capped) and two larger-than-a-pixel features competing for capacity 1
    py = _generate_single_mvt_tile_python(str(ds), (1, 0, 0), feature_capacity=1)
    rs = rust_engine.generate_tile(str(ds), 1, 0, 0, feature_capacity=1, extent=4096, buffer=256)
    # point + one full feature (attributes) + one demoted dot (no attributes);
    # with 2 dots > capacity 1 the sub-pixel point is bare too
    feats = [f for l in mapbox_vector_tile.decode(rs).values() for f in l["features"]]
    assert len(feats) == 3 and sum(1 for f in feats if f["properties"]) == 1
    assert _decoded(rs) == _decoded(py)  # same (size, crc32) priority => same survivors


def _decoded(tile_bytes):
    """Set of (properties, geometry) pairs for cross-engine comparison."""
    dec = mapbox_vector_tile.decode(tile_bytes)
    out = set()
    for layer in dec.values():
        for f in layer["features"]:
            out.add((json.dumps(f["properties"], sort_keys=True), json.dumps(f["geometry"], sort_keys=True)))
    return out


@pytest.mark.parametrize("with_bbox", [True, False])
def test_write_pyramid_matches_per_tile_generation(tmp_path, with_bbox):
    """The push-mode pyramid (two streaming passes, bounded memory) must
    produce exactly the tiles and features the per-tile generator does,
    across several zooms in one call, including the empty tiles it skips."""
    ds = _write_dataset(tmp_path / "ds", with_bbox=with_bbox)
    rust_engine.invalidate()
    handle = rust_engine.dataset(ds)
    tiles = [(0, 0, 0)] + [(1, x, y) for x in range(2) for y in range(2)] + [(2, x, y) for x in range(4) for y in range(4)]
    out = tmp_path / "pyr"
    st = handle.write_pyramid(str(out), tiles, feature_capacity=10, extent=4096, buffer=256)
    assert st["tiles_requested"] == len(tiles)

    expected_written = 0
    for (z, x, y) in tiles:
        ref = handle.generate_tile(z, x, y, feature_capacity=10, extent=4096, buffer=256)
        path = out / str(z) / str(x) / f"{y}.mvt"
        if _props(ref):
            expected_written += 1
            assert path.exists(), f"{z}/{x}/{y} missing"
            assert _decoded(path.read_bytes()) == _decoded(ref), f"{z}/{x}/{y} differs"
        else:
            assert not path.exists(), f"{z}/{x}/{y} should be empty"
    assert st["tiles_written"] == expected_written
    # capacity binds at z0 (2 points in their own pixel cells + 2 larger
    # features, cap 1 -> one in full, one demoted to a dot): same as pull mode
    st2 = handle.write_pyramid(str(tmp_path / "pyr2"), [(0, 0, 0)], feature_capacity=1, extent=4096, buffer=256)
    assert st2["features"] == 4
    ref2 = handle.generate_tile(0, 0, 0, feature_capacity=1, extent=4096, buffer=256)
    assert _decoded((tmp_path / "pyr2" / "0" / "0" / "0.mvt").read_bytes()) == _decoded(ref2)


def test_tile_attributes_policy_matches_python(tmp_path):
    """"none" strips every attribute, a column list keeps only those — in the
    Rust engine and the Python reference alike."""
    ds = _write_dataset(tmp_path / "ds", with_bbox=True)
    rust_engine.invalidate()
    for policy, expected_keys in (([], set()), (["name", "flag"], {"name", "flag"})):
        rs = rust_engine.generate_tile(str(ds), 1, 0, 0, feature_capacity=10, extent=4096, buffer=256,
                                       tile_attributes=policy)
        py = _generate_single_mvt_tile_python(str(ds), (1, 0, 0), feature_capacity=10, tile_attributes=policy)
        for blob in (rs, py):
            feats = [f for l in mapbox_vector_tile.decode(blob).values() for f in l["features"]]
            assert len(feats) == 3
            assert all(set(f["properties"]) == expected_keys for f in feats), policy
        assert _props(rs) == _props(py)
