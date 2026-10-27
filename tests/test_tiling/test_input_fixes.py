"""End-to-end ingest fixes: GeoJSON crs, empty input, reserved columns,
compressed inputs and single-file OGR formats."""
import gzip
import json

import geopandas as gpd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import shapely
from shapely.geometry import Point, box

import starlet
from starlet._internal.tiling.compressed import decompress_to, materialized_input
from starlet._internal.tiling.crs import geoparquet_crs
from starlet._internal.tiling.datasource import _source_kind, read_spatial_sample, source_for_path
from starlet._internal.tiling.geojson_source import GeoJSONSource, _extract_feature_collection_crs_hint
from starlet._internal.tiling.two_stage_orchestrator import rename_reserved_columns, reserved_column_renames
from starlet._internal.tiling.vector_source import VectorFileSource


def _feature_collection(features, crs_name=None, crs_first=False):
    doc = {"type": "FeatureCollection"}
    crs = {"crs": {"type": "name", "properties": {"name": crs_name}}} if crs_name else {}
    if crs_first:
        doc.update(crs)
    doc["features"] = features
    if not crs_first:
        doc.update(crs)
    return json.dumps(doc)


def _point_feature(x, y, n=1):
    return {"type": "Feature", "properties": {"n": n}, "geometry": {"type": "Point", "coordinates": [x, y]}}


# --------------------------------------------------------------------------- GeoJSON crs

def test_crs_hint_found_before_or_after_features():
    before = _feature_collection([_point_feature(0, 0)], "urn:ogc:def:crs:EPSG::3857", crs_first=True)
    after = _feature_collection([_point_feature(0, 0)], "urn:ogc:def:crs:EPSG::3857")
    assert _extract_feature_collection_crs_hint(before) == "urn:ogc:def:crs:EPSG::3857"
    assert _extract_feature_collection_crs_hint(after) == "urn:ogc:def:crs:EPSG::3857"
    assert _extract_feature_collection_crs_hint(_feature_collection([_point_feature(0, 0)])) is None
    # a property called "crs" that is not an object does not confuse the scan
    text = '{"type":"FeatureCollection","features":[{"type":"Feature","properties":{"crs":"x"},"geometry":null}],"crs":{"type":"name","properties":{"name":"EPSG:3857"}}}'
    assert _extract_feature_collection_crs_hint(text) == "EPSG:3857"


def test_geojson_source_applies_declared_crs(tmp_path):
    path = tmp_path / "mercator.geojson"
    path.write_text(_feature_collection([_point_feature(-13070000.0, 4010000.0), _point_feature(-13060000.0, 4020000.0, 2)],
                                        "urn:ogc:def:crs:EPSG::3857"))
    source = GeoJSONSource(str(path))
    assert source._crs_hint == "urn:ogc:def:crs:EPSG::3857"
    source.set_schema(source.infer_schema())
    table = next(iter(source.iter_tables(*source.create_splits())))
    assert geoparquet_crs(table.schema, "geometry") == "urn:ogc:def:crs:EPSG::3857"

    sample = read_spatial_sample(str(path), sample_cap=None, seed=1, geojson_workers=1)
    assert sample.sample_points[0].min() == pytest.approx(-13070000.0)


def test_geojson_source_ignores_crs84_and_bad_crs(tmp_path):
    ok = tmp_path / "crs84.geojson"
    ok.write_text(_feature_collection([_point_feature(1, 2)], "urn:ogc:def:crs:OGC:1.3:CRS84"))
    assert GeoJSONSource(str(ok))._crs_hint is None
    bad = tmp_path / "bad.geojson"
    bad.write_text(_feature_collection([_point_feature(1, 2)], "not-a-crs"))
    assert GeoJSONSource(str(bad))._crs_hint is None


# --------------------------------------------------------------------------- empty input

def test_tile_and_mvt_on_empty_feature_collection(tmp_path):
    path = tmp_path / "empty.geojson"
    path.write_text(_feature_collection([]))
    outdir = tmp_path / "ds"

    result = starlet.tile(str(path), str(outdir), parallelism=1)
    assert (result.num_files, result.total_rows) == (0, 0)
    assert (outdir / "parquet_tiles").is_dir()

    mvt = starlet.generate_mvt(str(outdir), zoom=2, parallelism=1)
    assert mvt.tile_count == 0 and mvt.zoom_levels == []


# --------------------------------------------------------------------------- reserved columns

def test_reserved_column_renames_avoid_existing_names():
    renames = reserved_column_renames(["_tile_id", "_bbox_xmin", "_tile_id_src", "name"])
    assert renames == {"_tile_id": "_tile_id_src2", "_bbox_xmin": "_bbox_xmin_src"}
    table = pa.table({"_tile_id": [1], "name": ["a"]})
    assert rename_reserved_columns(table, reserved_column_renames(table.column_names)).column_names == ["_tile_id_src", "name"]


def test_tile_input_with_internal_column_names(tmp_path):
    table = pa.table({
        "_tile_id": [7, 8],
        "_bbox_xmin": [0.0, 1.0],
        "name": ["a", "b"],
        "geometry": [shapely.to_wkb(Point(1, 1)), shapely.to_wkb(Point(2, 2))],
    })
    path = tmp_path / "clash.parquet"
    pq.write_table(table, path)

    result = starlet.tile(str(path), str(tmp_path / "ds"), parallelism=1)
    assert result.total_rows == 2
    out = pq.read_table(next((tmp_path / "ds" / "parquet_tiles").glob("*.parquet")))
    assert "_tile_id_src" in out.column_names and "_bbox_xmin_src" in out.column_names
    assert sorted(out["_tile_id_src"].to_pylist()) == [7, 8]
    assert out["_bbox_xmin"].to_pylist() == pytest.approx(sorted([1.0, 2.0])) or set(out["_bbox_xmin"].to_pylist()) == {1.0, 2.0}


# --------------------------------------------------------------------------- compressed

def test_decompress_and_materialize_gzip_geojson(tmp_path):
    raw = _feature_collection([_point_feature(1, 2)]).encode()
    gz = tmp_path / "points.geojson.gz"
    gz.write_bytes(gzip.compress(raw))

    assert decompress_to(gz, tmp_path).read_bytes() == raw
    with materialized_input(str(gz), str(tmp_path / "tmp")) as path:
        assert path.endswith("points.geojson")
        assert _source_kind(path) == "geojson"
    with materialized_input(str(tmp_path / "points.geojson"), None) as path:
        assert path == str(tmp_path / "points.geojson")
    with pytest.raises(ValueError, match="decompressed"):
        _source_kind(str(gz))


def test_tile_reads_gzipped_geojson(tmp_path):
    gz = tmp_path / "points.geojson.gz"
    gz.write_bytes(gzip.compress(_feature_collection([_point_feature(1, 2), _point_feature(3, 4, 2)]).encode()))
    result = starlet.tile(str(gz), str(tmp_path / "ds"), parallelism=1, temp_dir=str(tmp_path / "tmp"))
    assert result.total_rows == 2
    assert result.bbox == pytest.approx((1.0, 2.0, 3.0, 4.0))


# --------------------------------------------------------------------------- OGR single files

@pytest.mark.parametrize("suffix,driver", [(".gpkg", "GPKG"), (".fgb", "FlatGeobuf")])
def test_vector_file_formats_are_read(tmp_path, suffix, driver):
    gdf = gpd.GeoDataFrame({"name": ["a", "b"]}, geometry=[box(0, 0, 1, 1), box(5, 5, 6, 6)], crs="EPSG:4326")
    path = tmp_path / f"boxes{suffix}"
    gdf.to_file(path, driver=driver, engine="pyogrio")

    assert _source_kind(str(path)) == "vector"
    source = source_for_path(str(path))
    assert isinstance(source, VectorFileSource)
    tables = list(source.iter_tables())
    assert sum(t.num_rows for t in tables) == 2
    assert set(tables[0]["name"].to_pylist()) == {"a", "b"}
    assert tables[0].schema.field("geometry").type == pa.binary()

    sample = read_spatial_sample(str(path), sample_cap=None, seed=1, source_workers=1)
    assert sample.total_seen == 2
    # sample points are centroids: the second box spans 5..6
    assert tuple(sample.sample_points.max(axis=1)) == (5.5, 5.5)
