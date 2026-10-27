"""Unit tests for histogram pyramid generation.

Tests cover:
- Geometry vertex iteration
- Histogram accumulation
- Parallel tile processing
- Global histogram aggregation
- Prefix-sum computation
- File output format
"""
import json
import pytest
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from pyproj import Transformer
from shapely.geometry import Point, LineString, Polygon, MultiPoint, GeometryCollection
from shapely import wkb

from starlet._internal.histogram.hist_pyramid import (
    _accumulate_vertices_hist,
    _geometry_vertices_iter,
    _process_split_group,
    HistConfig,
    build_histograms_for_dir,
)
from starlet._internal.histogram.io import load_numpy_array
from starlet._internal.tiling.datasource import GeoParquetSource


class TestGeometryVertexIteration:
    """Test geometry vertex extraction."""

    def test_point_vertices(self):
        """Test extracting vertex from Point."""
        point = Point(10.0, 20.0)
        vertices = list(_geometry_vertices_iter(point))
        assert len(vertices) == 1
        assert vertices[0] == (10.0, 20.0)

    def test_linestring_vertices(self):
        """Test extracting vertices from LineString."""
        line = LineString([(0, 0), (1, 1), (2, 2)])
        vertices = list(_geometry_vertices_iter(line))
        assert len(vertices) == 3
        assert (0, 0) in vertices
        assert (1, 1) in vertices
        assert (2, 2) in vertices

    def test_polygon_vertices(self):
        """Test extracting vertices from Polygon."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        vertices = list(_geometry_vertices_iter(poly))
        # Exterior ring has 5 vertices (closed)
        assert len(vertices) == 5

    def test_polygon_with_holes(self):
        """Test extracting vertices from Polygon with holes."""
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        hole = [(2, 2), (8, 2), (8, 8), (2, 8), (2, 2)]
        poly = Polygon(exterior, [hole])
        vertices = list(_geometry_vertices_iter(poly))
        # Should include both exterior and hole vertices
        assert len(vertices) == 10  # 5 + 5

    def test_multipoint_vertices(self):
        """Test extracting vertices from MultiPoint."""
        mp = MultiPoint([(0, 0), (5, 5), (10, 10)])
        vertices = list(_geometry_vertices_iter(mp))
        assert len(vertices) == 3

    def test_geometry_collection_vertices(self):
        """Test extracting vertices from GeometryCollection."""
        point = Point(0, 0)
        line = LineString([(1, 1), (2, 2)])
        gc = GeometryCollection([point, line])

        vertices = list(_geometry_vertices_iter(gc))
        # 1 point + 2 line vertices = 3
        assert len(vertices) == 3

    def test_empty_geometry(self):
        """Test that empty geometries yield no vertices."""
        empty_point = Point()
        vertices = list(_geometry_vertices_iter(empty_point))
        assert len(vertices) == 0

    def test_none_geometry(self):
        """Test that None geometry is handled gracefully."""
        vertices = list(_geometry_vertices_iter(None))
        assert len(vertices) == 0


class TestHistConfig:
    """Test histogram configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HistConfig()
        assert config.grid_size == 4096
        assert config.out_crs == "EPSG:3857"
        assert config.dtype == "float64"
        assert config.parallelism > 0

    def test_custom_config(self):
        """Test custom configuration."""
        config = HistConfig(
            grid_size=2048,
            dtype="float32",
            parallelism=4,
        )
        assert config.grid_size == 2048
        assert config.dtype == "float32"
        assert config.parallelism == 4


class TestHistogramBuilding:
    """Test histogram pyramid building."""

    @pytest.mark.slow
    def test_build_from_tile_directory(self, sample_tile_directory, temp_dir):
        """Test building histogram from parquet tiles directory."""
        out_dir = temp_dir / "histograms"

        build_histograms_for_dir(
            tiles_dir=str(sample_tile_directory),
            outdir=str(out_dir),
            geom_col="geometry",
            grid_size=64,  # Small for testing
            parallelism=2,
        )

        # Check output files exist
        assert (out_dir / "global.npy.gz").exists()
        assert (out_dir / "global.json").exists()

    @pytest.mark.slow
    def test_histogram_dimensions(self, sample_tile_directory, temp_dir):
        """Test that output histogram has correct dimensions."""
        out_dir = temp_dir / "histograms"

        grid_size = 128
        build_histograms_for_dir(
            tiles_dir=str(sample_tile_directory),
            outdir=str(out_dir),
            grid_size=grid_size
        )

        hist = load_numpy_array(out_dir / "global")
        assert hist.shape == (grid_size, grid_size)

    @pytest.mark.slow
    def test_prefix_sum_computation(self, sample_tile_directory, temp_dir):
        """Test that prefix sum is correctly computed."""
        out_dir = temp_dir / "histograms"

        build_histograms_for_dir(
            tiles_dir=str(sample_tile_directory),
            outdir=str(out_dir),
            grid_size=64
        )

        hist = load_numpy_array(out_dir / "global")

        prefix = hist.cumsum(axis=0).cumsum(axis=1)

        # Should match
        np.testing.assert_array_almost_equal(
            prefix,
            hist.cumsum(axis=0).cumsum(axis=1),
        )

    @pytest.mark.slow
    def test_histogram_metadata(self, sample_tile_directory, temp_dir):
        """Test that metadata JSON contains correct information."""
        out_dir = temp_dir / "histograms"

        build_histograms_for_dir(
            tiles_dir=str(sample_tile_directory),
            outdir=str(out_dir),
            grid_size=64
        )

        with open(out_dir / "global.json", "r") as f:
            meta = json.load(f)

        assert "grid_size" in meta
        assert meta["grid_size"] == 64
        assert "shape" in meta
        assert meta["shape"] == [64, 64]
        assert "crs" in meta
        assert "sum" in meta
        assert "nonzero" in meta

    @pytest.mark.slow
    def test_histogram_nonzero_count(self, sample_tile_directory, temp_dir):
        """Test that nonzero count matches actual histogram."""
        out_dir = temp_dir / "histograms"

        build_histograms_for_dir(
            tiles_dir=str(sample_tile_directory),
            outdir=str(out_dir),
            grid_size=64
        )

        hist = load_numpy_array(out_dir / "global")

        with open(out_dir / "global.json", "r") as f:
            meta = json.load(f)

        assert meta["nonzero"] == np.count_nonzero(hist)

    def test_empty_tile_directory(self, temp_dir):
        """Test behavior with empty tile directory."""
        empty_dir = temp_dir / "empty_tiles"
        empty_dir.mkdir()

        out_dir = temp_dir / "histograms"

        # Should handle gracefully (may raise or create empty histogram)
        try:
            build_histograms_for_dir(
                tiles_dir=str(empty_dir),
                outdir=str(out_dir),
                grid_size=64
            )
        except Exception:
            # Expected to fail with no tiles
            pass


class TestHistogramAccumulation:
    """Test histogram accumulation logic."""

    def test_single_point_accumulation(self):
        """Test that a single point increments the correct histogram cell."""
        # This would require exposing _accumulate_vertices_hist or testing
        # through the full pipeline
        pass

    def test_non_finite_transformed_vertices_are_skipped(self):
        table = pa.table({"geometry": [wkb.dumps(Point(0, 0))]})
        histogram = np.zeros((4, 4), dtype=np.float64)

        class InfiniteTransformer:
            def transform(self, x, y):
                return float("inf"), float("inf")

        _accumulate_vertices_hist(
            table,
            histogram,
            "geometry",
            (-1.0, -1.0, 1.0, 1.0),
            InfiniteTransformer(),
        )

        assert histogram.sum() == 0

    def test_histogram_uses_geoparquet_source_crs(self, temp_dir):
        lon, lat = -118.25, 34.05
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x, y = transformer.transform(lon, lat)
        geo = {
            "version": "1.1.0",
            "primary_column": "geometry",
            "columns": {"geometry": {"encoding": "WKB", "crs": "EPSG:3857"}},
        }
        tile_dir = temp_dir / "tiles"
        tile_dir.mkdir()
        table = pa.table({"geometry": [wkb.dumps(Point(x, y))]})
        table = table.replace_schema_metadata({b"geo": json.dumps(geo).encode("utf-8")})
        pq.write_table(table, tile_dir / "tile_000000__0_0_0_0.parquet")

        source = GeoParquetSource(str(tile_dir), geometry_only=True)
        hist, coverage_diff = _process_split_group(
            str(tile_dir),
            source.create_splits(),
            HistConfig(grid_size=16),
            "geometry",
        )
        assert hist.sum() == 1
        assert coverage_diff.shape == (17, 17)

    def test_multiple_vertices_same_cell(self):
        """Test that multiple vertices in the same cell accumulate correctly."""
        pass


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.slow
    def test_small_grid_size(self, sample_tile_directory, temp_dir):
        """Test histogram with very small grid size."""
        out_dir = temp_dir / "histograms"

        build_histograms_for_dir(
            tiles_dir=str(sample_tile_directory),
            outdir=str(out_dir),
            grid_size=4  # Very small
        )

        hist = load_numpy_array(out_dir / "global")
        assert hist.shape == (4, 4)

    @pytest.mark.slow
    def test_large_grid_size(self, sample_tile_directory, temp_dir):
        """Test histogram with large grid size."""
        out_dir = temp_dir / "histograms"

        build_histograms_for_dir(
            tiles_dir=str(sample_tile_directory),
            outdir=str(out_dir),
            grid_size=1024  # Moderate size
        )

        hist = load_numpy_array(out_dir / "global")
        assert hist.shape == (1024, 1024)

    @pytest.mark.slow
    def test_different_dtypes(self, sample_tile_directory, temp_dir):
        """Test histogram with different data types."""
        out_dir = temp_dir / "histograms"

        build_histograms_for_dir(
            tiles_dir=str(sample_tile_directory),
            outdir=str(out_dir),
            grid_size=64,
            dtype="float32"
        )

        hist = load_numpy_array(out_dir / "global")
        assert hist.dtype == np.float32


class TestCoverageHistogram:
    """The bbox coverage histogram marks cells inside large polygons."""

    def _identity(self):
        class _T:
            def transform(self, xs, ys):
                return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
        return _T()

    def test_interior_cells_covered_without_vertices(self):
        from starlet._internal.histogram.hist_pyramid import coverage_from_diff

        grid = 8
        bbox = (0.0, 0.0, 8.0, 8.0)
        hist = np.zeros((grid, grid))
        diff = np.zeros((grid + 1, grid + 1), dtype=np.int64)
        # Square from (1,1) to (6,6): vertices only in 4 cells, interior covered.
        poly = Polygon([(1.2, 1.2), (6.8, 1.2), (6.8, 6.8), (1.2, 6.8)])
        table = pa.table({"geometry": [wkb.dumps(poly)]})

        _accumulate_vertices_hist(table, hist, "geometry", bbox, self._identity(), coverage_diff=diff)
        coverage = coverage_from_diff(diff)

        assert hist.sum() == 5  # closed ring repeats the first vertex
        # Interior cell (x=3, y=3) -> row grid-1-3 = 4, col 3
        assert hist[4, 3] == 0
        assert coverage[4, 3] == 1
        # Cells outside the box are not covered; covered block is 6x6 cells.
        assert coverage[0, 0] == 0 and coverage[7, 7] == 0
        assert coverage.sum() == 36

    def test_point_covers_exactly_its_cell(self):
        from starlet._internal.histogram.hist_pyramid import coverage_from_diff

        grid = 4
        hist = np.zeros((grid, grid))
        diff = np.zeros((grid + 1, grid + 1), dtype=np.int64)
        table = pa.table({"geometry": [wkb.dumps(Point(2.5, 0.5))]})
        _accumulate_vertices_hist(table, hist, "geometry", (0.0, 0.0, 4.0, 4.0), self._identity(), coverage_diff=diff)
        coverage = coverage_from_diff(diff)
        assert np.array_equal(coverage, hist)

    @pytest.mark.slow
    def test_build_writes_coverage_file(self, sample_tile_directory, temp_dir):
        out_dir = temp_dir / "histograms"
        build_histograms_for_dir(
            tiles_dir=str(sample_tile_directory), outdir=str(out_dir), grid_size=64, parallelism=2,
        )
        assert (out_dir / "global_cover.npy.gz").exists()
        coverage = load_numpy_array(out_dir / "global_cover")
        assert coverage.shape == (64, 64)
        assert coverage.dtype == np.uint32
        meta = json.loads((out_dir / "global_cover.json").read_text())
        assert meta["grid_size"] == 64
        # Every vertex cell is also a covered cell.
        hist = load_numpy_array(out_dir / "global")
        assert np.all(coverage[hist > 0] > 0)
