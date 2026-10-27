"""min_tile_bytes: size-threshold pyramid generation."""
import numpy as np
import pytest

from starlet._internal.mvt.mvt_generator import DatasetMVTGenerator


def _dataset(tmp_path, hist, cover=None):
    root = tmp_path / "ds"
    (root / "parquet_tiles").mkdir(parents=True)
    (root / "parquet_tiles" / "tile_000000.parquet").write_bytes(b"x")
    (root / "histograms").mkdir()
    hist = np.asarray(hist, dtype=np.float64)
    np.save(root / "histograms" / "global_prefix.npy", hist.cumsum(axis=0).cumsum(axis=1))
    if cover is not None:
        np.save(root / "histograms" / "global_cover.npy", np.asarray(cover, dtype=np.uint32))
    return root


def _generator(root, **kw):
    return DatasetMVTGenerator(str(root), num_zoom_levels=kw.pop("num_zoom_levels", 4), threshold=0, **kw)


def _recording_writer(generator, bytes_per_vertex):
    """write_zoom stub that writes files sized bytes_per_vertex[z] * vertices."""
    requested = {}

    def write_zoom(z, tiles):
        requested[z] = list(tiles)
        partitioner = generator._build_partitioner()
        for (_, x, y) in tiles:
            vertices = partitioner.histogram_value(z, x, y)
            size = int(vertices * bytes_per_vertex[z]) if vertices > 0 else 50
            d = generator.outdir / str(z) / str(x)
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{y}.mvt").write_bytes(b"\0" * max(size, 1))
    return requested, write_zoom


def test_size_mode_keeps_only_tiles_estimated_above_threshold(tmp_path):
    # 4x4 histogram (hist zoom 2): 1000 vertices in cell (row 0, col 0), 10 in (row 3, col 3).
    hist = np.zeros((4, 4)); hist[0, 0] = 1000; hist[3, 3] = 10
    gen = _generator(_dataset(tmp_path, hist), min_tile_bytes=400)
    gen.outdir.mkdir(parents=True)
    requested, write_zoom = _recording_writer(gen, {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0})

    gen._run_size_mode(write_zoom)

    assert requested[0] == [(0, 0, 0)]
    # z1: estimate = vertices * p90(bytes/vertex at z0) * prior growth 3 -> (1,0,0) has 1000*3 >= 400,
    # (1,1,1) has 10*3 < 400.
    assert requested[1] == [(1, 0, 0)]
    assert requested[2] == [(2, 0, 0)]
    # below the histogram resolution the estimate quarters per level: 1000/4 * ratio(1)*growth(1) = 250 < 400
    assert 3 not in requested
    assert gen.size_mode_stats[1] == {"candidates": 2, "kept": 1}
    assert gen.size_mode_stats[3] == {"candidates": 4, "kept": 0}


def test_size_mode_stops_when_nothing_is_kept_and_writes_root_always(tmp_path):
    hist = np.zeros((4, 4)); hist[1, 1] = 5
    gen = _generator(_dataset(tmp_path, hist), min_tile_bytes=10_000)
    gen.outdir.mkdir(parents=True)
    requested, write_zoom = _recording_writer(gen, {z: 1.0 for z in range(4)})

    gen._run_size_mode(write_zoom)

    assert list(requested) == [0]
    assert gen.size_mode_stats == {0: {"candidates": 1, "kept": 1}, 1: {"candidates": 1, "kept": 0}}


def test_size_mode_interior_tiles_are_not_pregenerated(tmp_path):
    # Vertices only in the corner cells; the interior cell is covered (inside a polygon).
    hist = np.zeros((4, 4))
    for r, c in ((0, 0), (0, 3), (3, 0), (3, 3)):
        hist[r, c] = 500
    cover = np.ones((4, 4))
    gen = _generator(_dataset(tmp_path, hist, cover), min_tile_bytes=100)
    gen.outdir.mkdir(parents=True)
    requested, write_zoom = _recording_writer(gen, {z: 1.0 for z in range(4)})

    gen._run_size_mode(write_zoom)

    assert set(requested[2]) == {(2, 0, 0), (2, 3, 0), (2, 0, 3), (2, 3, 3)}
    assert (2, 1, 1) not in requested[2]
    assert gen.size_mode_stats[2]["candidates"] == 16  # all covered cells are candidates


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
