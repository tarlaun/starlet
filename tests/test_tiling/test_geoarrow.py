"""Native GeoArrow GeoParquet geometry columns are read as WKB."""
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box

from starlet._internal.tiling.datasource import read_spatial_sample
from starlet._internal.tiling.geoarrow import geoarrow_to_wkb, native_geometry_type, wkb_schema
from starlet._internal.tiling.geoparquet_source import GeoParquetSource


def _struct_coords(coords: np.ndarray) -> pa.Array:
    return pa.StructArray.from_arrays(
        [pa.array(coords[:, 0], pa.float64()), pa.array(coords[:, 1], pa.float64())], ["x", "y"]
    )


def _native_array(geoms, interleaved=False) -> tuple[pa.Array, str]:
    """Build a GeoArrow native array (struct or interleaved coords) from shapely geometries."""
    geometry_type, coords, offsets = shapely.to_ragged_array(geoms)
    coords = np.asarray(coords, dtype=np.float64)
    if interleaved:
        values = pa.FixedSizeListArray.from_arrays(pa.array(coords.ravel(), pa.float64()), 2)
    else:
        values = _struct_coords(coords)
    array = values
    for level in offsets:  # innermost first
        array = pa.ListArray.from_arrays(pa.array(np.asarray(level, dtype=np.int32)), array)
    name = {
        shapely.GeometryType.POINT: "geoarrow.point",
        shapely.GeometryType.LINESTRING: "geoarrow.linestring",
        shapely.GeometryType.POLYGON: "geoarrow.polygon",
        shapely.GeometryType.MULTIPOINT: "geoarrow.multipoint",
        shapely.GeometryType.MULTILINESTRING: "geoarrow.multilinestring",
        shapely.GeometryType.MULTIPOLYGON: "geoarrow.multipolygon",
    }[geometry_type]
    return array, name


def _write_native(path, geoms, *, interleaved=False, crs="EPSG:4326", geo_metadata=True, col="geometry"):
    array, ext_name = _native_array(geoms, interleaved=interleaved)
    field = pa.field(col, array.type, metadata={
        "ARROW:extension:name": ext_name,
        "ARROW:extension:metadata": json.dumps({"crs": crs}),
    })
    ids = pa.array(list(range(len(geoms))), pa.int64())
    schema = pa.schema([pa.field("id", pa.int64()), field])
    if geo_metadata:
        schema = schema.with_metadata({"geo": json.dumps({
            "version": "1.1.0", "primary_column": col,
            "columns": {col: {"encoding": ext_name.split(".")[1], "crs": crs}},
        })})
    table = pa.Table.from_arrays([ids, array], schema=schema)
    pq.write_table(table, path)
    return table


def test_geoarrow_to_wkb_round_trips_each_geometry_type():
    cases = [
        [Point(1, 2), Point(3, 4)],
        [LineString([(0, 0), (1, 1), (2, 0)]), LineString([(5, 5), (6, 6)])],
        [box(0, 0, 1, 1), Polygon([(0, 0), (4, 0), (4, 4), (0, 4)], holes=[[(1, 1), (2, 1), (2, 2), (1, 2)]])],
        [shapely.multipoints([Point(0, 0), Point(1, 1)]), shapely.multipoints([Point(2, 2)])],
        [shapely.multilinestrings([LineString([(0, 0), (1, 1)]), LineString([(2, 2), (3, 3)])])],
        [MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)]), MultiPolygon([box(5, 5, 6, 6)])],
    ]
    for geoms in cases:
        for interleaved in (False, True):
            array, ext_name = _native_array(geoms, interleaved=interleaved)
            field = pa.field("g", array.type, metadata={"ARROW:extension:name": ext_name})
            geometry_type = native_geometry_type(field)
            assert geometry_type is not None
            wkb = geoarrow_to_wkb(array, geometry_type)
            assert wkb.type == pa.binary()
            assert [g.equals(e) for g, e in zip(shapely.from_wkb(wkb.to_pylist()), geoms)] == [True] * len(geoms)


def test_geoarrow_to_wkb_keeps_nulls():
    array, ext_name = _native_array([Point(1, 2), Point(3, 4)])
    array = pa.array([array[0].as_py(), None], type=array.type)
    wkb = geoarrow_to_wkb(array, native_geometry_type(pa.field("g", array.type, metadata={"ARROW:extension:name": ext_name})))
    assert wkb[1].as_py() is None
    assert shapely.from_wkb(wkb[0].as_py()).equals(Point(1, 2))


def test_geoparquet_source_converts_native_multipolygons(tmp_path):
    geoms = [MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)]), MultiPolygon([box(-10, -10, -5, -5)])]
    path = tmp_path / "native.parquet"
    _write_native(path, geoms)

    source = GeoParquetSource(str(path))
    schema = source.schema()
    assert source.geom_col == "geometry"
    assert schema.field("geometry").type == pa.binary()
    geo = json.loads(schema.metadata[b"geo"])
    assert geo["columns"]["geometry"]["encoding"] == "WKB"
    assert geo["columns"]["geometry"]["crs"] == "EPSG:4326"

    tables = list(source.iter_tables())
    assert sum(t.num_rows for t in tables) == 2
    assert tables[0].schema.field("geometry").type == pa.binary()
    decoded = shapely.from_wkb(tables[0]["geometry"].to_pylist())
    assert all(d.equals(g) for d, g in zip(decoded, geoms))
    assert tables[0]["id"].to_pylist() == [0, 1]


def test_geoparquet_source_detects_native_column_without_geo_metadata(tmp_path):
    path = tmp_path / "native_points.parquet"
    _write_native(path, [Point(1, 2), Point(3, 4)], interleaved=True, geo_metadata=False, col="geom")

    source = GeoParquetSource(str(path), geom_col="geometry")
    assert source.geom_col == "geom"
    table = next(iter(source.iter_tables()))
    assert shapely.from_wkb(table["geom"][1].as_py()).equals(Point(3, 4))


def test_spatial_sample_from_native_geoparquet(tmp_path):
    path = tmp_path / "native_sample.parquet"
    _write_native(path, [box(0, 0, 1, 1), box(10, 10, 11, 11)])

    sample = read_spatial_sample(str(path), sample_ratio=1.0, sample_cap=None, seed=1, geoparquet_workers=1)

    assert sample.total_seen == 2
    assert tuple(sample.mbr.mins) == (0.0, 0.0)
    assert tuple(sample.mbr.maxs) == (11.0, 11.0)


def test_wkb_schema_replaces_field_and_metadata():
    array, ext_name = _native_array([Point(0, 0)])
    schema = pa.schema([pa.field("geometry", array.type, metadata={
        "ARROW:extension:name": ext_name, "ARROW:extension:metadata": json.dumps({"crs": "EPSG:3857"}),
    })])
    converted = wkb_schema(schema, "geometry")
    assert converted.field("geometry").type == pa.binary()
    geo = json.loads(converted.metadata[b"geo"])
    assert geo["primary_column"] == "geometry"
    assert geo["columns"]["geometry"] == {"encoding": "WKB", "crs": "EPSG:3857"}
