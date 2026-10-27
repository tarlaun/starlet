"""min_tile_bytes: size-threshold pyramid generation."""
import json

import numpy as np
import pytest

from starlet._internal.mvt.mvt_generator import DatasetMVTGenerator


def _dataset(tmp_path, hist, cover=None, attributes=None):
    root = tmp_path / "ds"
    (root / "parquet_tiles").mkdir(parents=True)
    (root / "parquet_tiles" / "tile_000000.parquet").write_bytes(b"x")
    (root / "histograms").mkdir()
    hist = np.asarray(hist, dtype=np.float64)
    np.save(root / "histograms" / "global_prefix.npy", hist.cumsum(axis=0).cumsum(axis=1))
    if cover is None:
        cover = (hist > 0).astype(np.uint32)
    np.save(root / "histograms" / "global_cover.npy", np.asarray(cover, dtype=np.uint32))
    if attributes is not None:
        (root / "stats").mkdir()
        (root / "stats" / "attributes.json").write_text(json.dumps({"attributes": attributes}))
    return root


def _generator(root, **kw):
    return DatasetMVTGenerator(str(root), num_zoom_levels=kw.pop("num_zoom_levels", 4), threshold=0, **kw)


class _Writer:
    """write_zoom stub: writes files whose size is ``size_of(z, x, y, vertices)``."""

    def __init__(self, generator, size_of):
        self.generator = generator
        self.size_of = size_of
        self.requests = []
        self.partitioner = generator._build_partitioner()

    def __call__(self, z, tiles):
        self.requests.append((z, list(tiles)))
        for (_, x, y) in tiles:
            size = int(self.size_of(z, x, y, self.partitioner.histogram_value(z, x, y)))
            if size <= 0:
                continue
            d = self.generator.outdir / str(z) / str(x)
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{y}.mvt").write_bytes(b"\0" * size)

    def requested(self, z):
        return [t for (zz, tiles) in self.requests if zz == z for t in tiles]

    def written(self, z):
        zoom_dir = self.generator.outdir / str(z)
        return sorted((z, int(p.parent.name), int(p.stem)) for p in zoom_dir.rglob("*.mvt")) if zoom_dir.exists() else []


def test_size_mode_writes_tiles_estimated_above_threshold_and_prunes_by_bound(tmp_path):
    # 4x4 histogram (hist zoom 2): 1000 vertices in cell (row 0, col 0), 10 in (row 3, col 3).
    hist = np.zeros((4, 4)); hist[0, 0] = 1000; hist[3, 3] = 10
    gen = _generator(_dataset(tmp_path, hist), min_tile_bytes=400)
    gen.outdir.mkdir(parents=True)
    writer = _Writer(gen, lambda z, x, y, v: v * 1.0)  # 1 byte per vertex, at every zoom

    gen._run_size_mode(writer)

    assert writer.requested(0) == [(0, 0, 0)]
    # z1: both children pass the histogram filter; the bound (8 bytes/vertex + per-feature
    # payload) keeps (1,1,1) explorable, but the probe calibration (1 byte/vertex) says 10 < 400.
    assert writer.written(1) == [(1, 0, 0)]
    assert writer.written(2) == [(2, 0, 0)]
    # Below the histogram resolution the children are measured: 250 bytes < 400 at z3.
    assert writer.written(3) == []
    assert gen.size_mode_stats[3]["probes"] == 4
    assert gen.size_mode_stats[1]["kept"] == 1
    assert gen.size_mode_stats[2]["kept"] == 1
    assert gen.size_mode_stats[3]["kept"] == 0


def test_size_mode_stops_when_bound_rules_out_every_child(tmp_path):
    hist = np.zeros((4, 4)); hist[1, 1] = 5
    gen = _generator(_dataset(tmp_path, hist), min_tile_bytes=10_000)
    gen.outdir.mkdir(parents=True)
    writer = _Writer(gen, lambda z, x, y, v: v)

    gen._run_size_mode(writer)

    assert [z for z, _ in writer.requests] == [0]
    assert gen.size_mode_stats == {0: {"candidates": 1, "probes": 0, "kept": 1}}


def test_size_mode_keeps_exploring_when_shallow_tiles_collapse(tmp_path):
    """Point data: shallow tiles deduplicate to a few dots (tiny), deep tiles are large.
    The upper bound must keep the walk alive until the real sizes show up."""
    hist = np.zeros((4, 4)); hist[2, 1] = 100_000  # a dense city in one histogram cell
    gen = _generator(_dataset(tmp_path, hist), min_tile_bytes=50_000, num_zoom_levels=6)
    gen.outdir.mkdir(parents=True)

    def size_of(z, x, y, v):
        return 40 if z < 3 else min(v * 2.0, 400_000)  # dots until z3, then 2 bytes per vertex

    writer = _Writer(gen, size_of)
    gen._run_size_mode(writer)

    assert writer.written(1) == [] and writer.written(2) == []  # collapsed tiles not kept
    # z3 is below the histogram resolution: all explored children are generated and
    # measured; each holds an estimated 25k vertices -> 50 KB, exactly the threshold.
    assert len(writer.written(3)) == 4
    assert writer.written(4) == []                   # 6.25k * 2 = 12.5 KB, removed again
    assert gen.size_mode_stats[1]["probes"] == 1 and gen.size_mode_stats[1]["kept"] == 0
    assert gen.size_mode_stats[3]["probes"] == 4 and gen.size_mode_stats[3]["kept"] == 4
    assert gen.size_mode_stats[4]["kept"] == 0


def test_size_mode_removes_probes_below_threshold(tmp_path):
    hist = np.zeros((4, 4)); hist[0, 0] = 1000; hist[0, 1] = 900
    gen = _generator(_dataset(tmp_path, hist), min_tile_bytes=950, num_zoom_levels=3)
    gen.outdir.mkdir(parents=True)
    writer = _Writer(gen, lambda z, x, y, v: v)  # tile (2,1,0) has 900 bytes: probed, then removed

    gen._run_size_mode(writer)

    assert (2, 1, 0) in writer.requested(2)
    assert writer.written(2) == [(2, 0, 0)]
    assert gen.size_mode_stats[2] == {"candidates": 2, "probes": 2, "kept": 1}


def test_size_mode_interior_tiles_are_not_written(tmp_path):
    hist = np.zeros((4, 4))
    for r, c in ((0, 0), (0, 3), (3, 0), (3, 3)):
        hist[r, c] = 500
    gen = _generator(_dataset(tmp_path, hist, cover=np.ones((4, 4))), min_tile_bytes=100)
    gen.outdir.mkdir(parents=True)
    writer = _Writer(gen, lambda z, x, y, v: v if v > 0 else 60)  # interior fill tiles are 60 bytes

    gen._run_size_mode(writer)

    assert set(writer.written(2)) == {(2, 0, 0), (2, 3, 0), (2, 0, 3), (2, 3, 3)}
    assert gen.size_mode_stats[2]["candidates"] == 16  # covered interior tiles are explored...
    assert gen.size_mode_stats[2]["kept"] == 4         # ...but their vertex estimate is 0


def test_bytes_per_feature_from_stats(tmp_path):
    root = _dataset(tmp_path, np.zeros((4, 4)), attributes=[
        {"name": "name", "stats": {"avg_length": 10.0, "top_k": []}},
        {"name": "pop", "stats": {"min": 1, "max": 2, "mean": 1.5}},
        {"name": "geometry", "stats": {"mbr": [0, 0, 1, 1]}},
    ])
    gen = _generator(root, min_tile_bytes=1)
    assert gen._size_mode_bytes_per_feature() == pytest.approx(12 + (2 + 15) + (2 + 8))
    assert _generator(root, min_tile_bytes=1, tile_attributes="none")._size_mode_bytes_per_feature() == 12
    assert _generator(root, min_tile_bytes=1, tile_attributes=["pop"])._size_mode_bytes_per_feature() == pytest.approx(22)


def test_size_mode_curve_interpolates_probes_with_safety_margin(tmp_path):
    gen = _generator(_dataset(tmp_path, np.zeros((4, 4))), min_tile_bytes=1)
    # Perfectly linear probes: the curve passes through them, scaled by the safety margin.
    assert gen.size_mode_safety == 2.5
    probes = [(3, 0, 0), (3, 1, 0), (3, 2, 0)]
    vertices = {probes[0]: 10.0, probes[1]: 100.0, probes[2]: 1000.0}
    sizes = {(0, 0): 100, (1, 0): 1000, (2, 0): 10000}
    estimate = gen._size_mode_curve(probes, vertices, sizes)
    assert estimate(100.0) == pytest.approx(1000 * gen.size_mode_safety)
    assert estimate(316.0) == pytest.approx(3160 * gen.size_mode_safety, rel=0.01)
    assert estimate(5000.0) == pytest.approx(10000 * gen.size_mode_safety)  # clamped to probe range
    assert estimate(0.0) == 0.0
    # No usable probes -> nothing is estimated as large.
    assert gen._size_mode_curve([], vertices, sizes)(1000.0) == 0.0


def test_generate_mvt_resolves_size_mode_zoom_and_size_strings(monkeypatch, tmp_path):
    import starlet
    from starlet._internal.config import _reset_loaded_config_for_tests

    _reset_loaded_config_for_tests()
    captured = {}

    class FakeGenerator:
        def __init__(self, tile_dir, **kwargs):
            captured.update(kwargs)

        def run(self):
            from types import SimpleNamespace
            return SimpleNamespace(tile_count=0, tile_counts_by_zoom=[], zoom_levels=[], pmtiles_path=None)

    monkeypatch.setattr("starlet._internal.mvt.mvt_generator.DatasetMVTGenerator", FakeGenerator)

    starlet.generate_mvt(str(tmp_path), min_tile_bytes="50kb")
    assert captured["min_tile_bytes"] == 51200
    assert captured["num_zoom_levels"] == 23  # size_mode_max_zoom (22) + 1

    starlet.generate_mvt(str(tmp_path), min_tile_bytes=1000, zoom=5)
    assert captured["num_zoom_levels"] == 6

    starlet.generate_mvt(str(tmp_path))
    assert captured["min_tile_bytes"] == 0 and captured["num_zoom_levels"] == 8

    with pytest.raises(ValueError):
        starlet.generate_mvt(str(tmp_path), min_tile_bytes=-1)


def test_generator_rejects_negative_min_tile_bytes(tmp_path):
    with pytest.raises(ValueError, match="min_tile_bytes"):
        _generator(_dataset(tmp_path, np.zeros((4, 4))), min_tile_bytes=-5)
