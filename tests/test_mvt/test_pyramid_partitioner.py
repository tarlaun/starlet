"""Tests for the standalone MVT pyramid partitioner helper."""

import numpy as np
import pytest
from shapely.geometry import box

from starlet._internal.mvt.pyramid_partitioner import PyramidPartitioner


WORLD = (0.0, 0.0, 4.0, 4.0)


def test_tile_id_encoding_round_trips():
    tile_id = PyramidPartitioner.encode_tile_id(12, 12345, 67890)

    assert isinstance(tile_id, int)
    assert PyramidPartitioner.decode_tile_id(tile_id) == (12, 12345, 67890)


def test_returns_overlapping_tiles_from_deepest_zoom_to_root():
    partitioner = PyramidPartitioner(WORLD, num_zoom_levels=3)

    tiles = partitioner.overlapping_tiles((1.1, 1.1, 1.9, 1.9))

    assert tiles == [
        (2, 1, 2),
        (1, 0, 1),
        (0, 0, 0),
    ]


def test_accepts_shapely_geometry_bounds():
    partitioner = PyramidPartitioner(WORLD, num_zoom_levels=2)

    tiles = partitioner.overlapping_tiles(box(2.1, 2.1, 2.9, 2.9))

    assert tiles == [
        (1, 1, 0),
        (0, 0, 0),
    ]


def test_accepts_tuple_bounds():
    partitioner = PyramidPartitioner(WORLD, num_zoom_levels=2)

    assert partitioner.overlapping_tiles((2.1, 2.1, 2.9, 2.9)) == [
        (1, 1, 0),
        (0, 0, 0),
    ]


def test_large_geometry_overlaps_multiple_deep_tiles_then_parents():
    partitioner = PyramidPartitioner(WORLD, num_zoom_levels=3)

    tiles = partitioner.overlapping_tiles((0.5, 0.5, 2.5, 2.5))

    assert (2, 0, 1) in tiles
    assert (2, 2, 3) in tiles
    assert (1, 0, 0) in tiles
    assert (1, 1, 1) in tiles
    assert tiles[-1] == (0, 0, 0)


def test_histogram_filters_tiles_at_or_below_threshold():
    hist = np.zeros((4, 4), dtype=float)
    hist[2, 1] = 11.0
    hist[1, 1] = 10.0
    prefix = hist.cumsum(axis=0).cumsum(axis=1)
    partitioner = PyramidPartitioner(
        WORLD,
        num_zoom_levels=3,
        prefix_histogram=prefix,
        size_threshold=10.0,
    )

    tiles = partitioner.overlapping_tiles((1.1, 1.1, 1.9, 2.9))

    assert (2, 1, 2) in tiles
    assert (2, 1, 1) not in tiles


def test_histogram_aggregates_parent_tiles():
    hist = np.zeros((4, 4), dtype=float)
    hist[0:2, 0:2] = 3.0
    prefix = hist.cumsum(axis=0).cumsum(axis=1)
    partitioner = PyramidPartitioner(
        WORLD,
        num_zoom_levels=2,
        prefix_histogram=prefix,
        size_threshold=10.0,
    )

    tiles = partitioner.overlapping_tiles((0.1, 2.1, 1.9, 3.9))

    assert tiles == [(1, 0, 0), (0, 0, 0)]


def test_estimate_deepest_level_extends_beyond_histogram():
    hist = np.zeros((4, 4), dtype=float)
    hist[2, 1] = 100.0
    prefix = hist.cumsum(axis=0).cumsum(axis=1)

    assert PyramidPartitioner.estimate_deepest_level(prefix, 10.0) == 3


def test_estimate_deepest_level_uses_strict_threshold():
    hist = np.zeros((4, 4), dtype=float)
    hist[2, 1] = 40.0
    prefix = hist.cumsum(axis=0).cumsum(axis=1)

    assert PyramidPartitioner.estimate_deepest_level(prefix, 10.0) == 2


def test_estimate_deepest_level_walks_up_then_one_level_down():
    hist = np.zeros((4, 4), dtype=float)
    hist[0:2, 0:2] = 3.0
    prefix = hist.cumsum(axis=0).cumsum(axis=1)

    assert PyramidPartitioner.estimate_deepest_level(prefix, 10.0) == 2


def test_estimate_deepest_level_uses_max_cell_parent_chain():
    hist = np.zeros((8, 8), dtype=float)
    hist[6, 6] = 8.0
    hist[6, 7] = 4.0
    prefix = hist.cumsum(axis=0).cumsum(axis=1)

    assert PyramidPartitioner.estimate_deepest_level(prefix, 10.0) == 3


def test_estimate_deepest_level_returns_zero_when_total_is_below_threshold():
    hist = np.ones((4, 4), dtype=float)
    prefix = hist.cumsum(axis=0).cumsum(axis=1)

    assert PyramidPartitioner.estimate_deepest_level(prefix, 100.0) == 0


def test_histogram_estimate_caps_constructor_max_zoom():
    hist = np.ones((4, 4), dtype=float)
    prefix = hist.cumsum(axis=0).cumsum(axis=1)

    partitioner = PyramidPartitioner(
        WORLD,
        num_zoom_levels=8,
        prefix_histogram=prefix,
        size_threshold=100.0,
    )

    assert partitioner.max_zoom == 0
    assert partitioner.num_zoom_levels == 1


def test_buffer_expands_assignment_to_nearby_tiles():
    without_buffer = PyramidPartitioner(WORLD, num_zoom_levels=3)
    with_buffer = PyramidPartitioner(WORLD, num_zoom_levels=3, buffer=0.2)

    raw_tiles = without_buffer.overlapping_tiles((0.9, 0.9, 0.95, 0.95))
    buffered_tiles = with_buffer.overlapping_tiles((0.9, 0.9, 0.95, 0.95))

    assert (2, 0, 3) in raw_tiles
    assert (2, 1, 3) not in raw_tiles
    assert (2, 1, 3) in buffered_tiles
    assert (2, 0, 2) in buffered_tiles


def test_buffer_is_ratio_of_each_level_tile_size():
    partitioner = PyramidPartitioner(WORLD, num_zoom_levels=3, buffer=0.1)

    tiles = partitioner.overlapping_tiles((2.15, 2.15, 2.2, 2.2))

    assert (2, 1, 1) not in tiles
    assert (2, 2, 1) in tiles
    assert (1, 0, 0) in tiles
    assert (1, 1, 0) in tiles


def test_lru_cache_keeps_only_recent_accepted_tiles():
    partitioner = PyramidPartitioner(WORLD, num_zoom_levels=2, cache_size=2)

    partitioner.overlapping_tile_ids((0.1, 3.1, 0.9, 3.9))
    partitioner.overlapping_tile_ids((2.1, 3.1, 2.9, 3.9))

    cached = {
        PyramidPartitioner.decode_tile_id(tile_id)
        for tile_id in partitioner.cached_tile_ids
    }

    assert len(cached) == 2
    assert (1, 0, 0) not in cached
    assert (1, 1, 0) in cached
    assert (0, 0, 0) in cached


def test_out_of_bounds_geometry_returns_no_tiles():
    partitioner = PyramidPartitioner(WORLD, num_zoom_levels=3)

    assert partitioner.overlapping_tiles((10.0, 10.0, 11.0, 11.0)) == []


@pytest.mark.parametrize("num_zoom_levels", [0, -1, 33])
def test_rejects_invalid_zoom_level_counts(num_zoom_levels):
    with pytest.raises(ValueError):
        PyramidPartitioner(WORLD, num_zoom_levels=num_zoom_levels)


def _prefix(raw):
    return np.asarray(raw, dtype=np.float64).cumsum(axis=0).cumsum(axis=1)


def test_tile_inside_polygon_passes_with_coverage_but_not_vertices_alone():
    # 4x4 histogram (hist_zoom=2). A square polygon whose four vertices sit in
    # the corner cells (0,0), (0,3), (3,0), (3,3) covers every cell, but the
    # vertex histogram is zero in the interior cells.
    vertices = np.zeros((4, 4))
    for r, c in ((0, 0), (0, 3), (3, 0), (3, 3)):
        vertices[r, c] = 1
    coverage = np.ones((4, 4))

    vertex_only = PyramidPartitioner(WORLD, num_zoom_levels=3, prefix_histogram=_prefix(vertices))
    with_cover = PyramidPartitioner(
        WORLD, num_zoom_levels=3, prefix_histogram=_prefix(vertices), coverage_prefix=_prefix(coverage)
    )

    interior = PyramidPartitioner.encode_tile_id(2, 1, 1)
    assert vertex_only._tile_passes_filter(interior, 2, 1, 1) is False
    assert with_cover._tile_passes_filter(interior, 2, 1, 1) is True
    assert with_cover.coverage_value(2, 1, 1) == 1.0
    assert with_cover.histogram_value(2, 1, 1) == 0.0


def test_coverage_is_inherited_below_histogram_resolution():
    coverage = np.zeros((4, 4))
    coverage[2, 1] = 3  # only this cell is covered
    partitioner = PyramidPartitioner(WORLD, num_zoom_levels=5, coverage_prefix=_prefix(coverage))

    # zoom 4 children of hist cell (x=1, y=2) inherit the full value ...
    assert partitioner.coverage_value(4, 4, 8) == 3.0
    assert partitioner.coverage_value(4, 7, 11) == 3.0
    # ... and tiles outside it stay empty.
    assert partitioner.coverage_value(4, 0, 0) == 0.0
    assert partitioner._tile_passes_filter(PyramidPartitioner.encode_tile_id(4, 0, 0), 4, 0, 0) is False
    assert partitioner._tile_passes_filter(PyramidPartitioner.encode_tile_id(4, 5, 9), 4, 5, 9) is True


def test_threshold_applies_to_max_of_vertices_and_coverage():
    vertices = np.zeros((4, 4))
    vertices[0, 0] = 10
    coverage = np.zeros((4, 4))
    coverage[3, 3] = 2
    partitioner = PyramidPartitioner(
        WORLD, num_zoom_levels=3, prefix_histogram=_prefix(vertices),
        coverage_prefix=_prefix(coverage), size_threshold=5,
    )
    assert partitioner._tile_passes_filter(PyramidPartitioner.encode_tile_id(2, 0, 0), 2, 0, 0) is True
    assert partitioner._tile_passes_filter(PyramidPartitioner.encode_tile_id(2, 3, 3), 2, 3, 3) is False


def test_coverage_shape_must_match_histogram():
    with pytest.raises(ValueError):
        PyramidPartitioner(
            WORLD, num_zoom_levels=2,
            prefix_histogram=_prefix(np.zeros((4, 4))), coverage_prefix=_prefix(np.zeros((8, 8))),
        )
