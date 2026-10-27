import pyarrow as pa
import shapely
from shapely.geometry import GeometryCollection, LineString, Point, Polygon

from starlet._internal.tiling.geometry_collections import explode_geometry_collections, geometry_collection_mask


def _table(geoms, ids=None):
    ids = ids or list(range(len(geoms)))
    return pa.table({"id": ids, "geometry": [shapely.to_wkb(g) if g is not None else None for g in geoms]})


def test_mask_detects_collections_in_both_byte_orders():
    gc = GeometryCollection([Point(0, 0)])
    big_endian = shapely.to_wkb(gc, byte_order=0)
    table = pa.table({"geometry": [shapely.to_wkb(Point(1, 1)), shapely.to_wkb(gc), big_endian, None]})
    assert geometry_collection_mask(table["geometry"]).tolist() == [False, True, True, False]


def test_collection_rows_are_exploded_by_dimension_with_attributes_copied():
    gc = GeometryCollection([Point(0, 0), LineString([(0, 0), (1, 1)]), Point(2, 2),
                             Polygon([(0, 0), (1, 0), (1, 1)])])
    out = explode_geometry_collections(_table([Point(5, 5), gc, Point(6, 6)], ids=[10, 20, 30]), "geometry")

    assert out["id"].to_pylist() == [10, 20, 20, 20, 30]
    types = [g.geom_type for g in shapely.from_wkb(out["geometry"].to_pylist())]
    assert types == ["Point", "MultiPoint", "LineString", "Polygon", "Point"]


def test_nested_collections_and_empty_collections():
    nested = GeometryCollection([GeometryCollection([Point(0, 0), Point(1, 1)]), LineString([(0, 0), (2, 2)])])
    out = explode_geometry_collections(_table([nested, GeometryCollection(), Point(9, 9)]), "geometry")
    assert out["id"].to_pylist() == [0, 0, 2]
    types = [g.geom_type for g in shapely.from_wkb(out["geometry"].to_pylist())]
    assert types == ["MultiPoint", "LineString", "Point"]


def test_tables_without_collections_are_returned_unchanged():
    table = _table([Point(0, 0), None])
    assert explode_geometry_collections(table, "geometry") is table
    assert explode_geometry_collections(table.slice(0, 0), "geometry").num_rows == 0
