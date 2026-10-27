"""Dataset.bbox is lon/lat regardless of CRS and geometry column name."""
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from starlet._types import Dataset


def _dataset(tmp_path, *, geom_col="geometry", crs=None, mbr=(-117.5, 33.5, -117.0, 34.0)):
    root = tmp_path / "ds"
    (root / "parquet_tiles").mkdir(parents=True)
    (root / "stats").mkdir()
    (root / "stats" / "attributes.json").write_text(json.dumps({"attributes": [
        {"name": "n", "stats": {"min": 1, "max": 1}},
        {"name": geom_col, "stats": {"mbr": list(mbr), "geom_types": {"Point": 1}}},
    ]}))
    geo = {"version": "1.1.0", "primary_column": geom_col, "columns": {geom_col: {"encoding": "WKB"}}}
    if crs is not None:
        geo["columns"][geom_col]["crs"] = crs
    table = pa.table({"n": [1], geom_col: [b"\x00"]}).replace_schema_metadata({b"geo": json.dumps(geo).encode()})
    pq.write_table(table, root / "parquet_tiles" / "tile_000000.parquet")
    return Dataset(str(root))


def test_bbox_from_lonlat_dataset(tmp_path):
    assert _dataset(tmp_path).bbox == (-117.5, 33.5, -117.0, 34.0)


def test_bbox_uses_non_default_geometry_column_name(tmp_path):
    ds = _dataset(tmp_path, geom_col="shape")
    assert ds.source_bbox == (-117.5, 33.5, -117.0, 34.0)
    assert ds.bbox == (-117.5, 33.5, -117.0, 34.0)


def test_bbox_is_reprojected_from_projected_crs(tmp_path):
    from pyproj import CRS, Transformer

    to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    mbr = to_mercator.transform_bounds(-117.5, 33.5, -117.0, 34.0)
    ds = _dataset(tmp_path, crs=json.loads(CRS.from_epsg(3857).to_json()), mbr=mbr)

    assert ds.source_bbox == mbr
    west, south, east, north = ds.bbox
    assert west == pytest.approx(-117.5, abs=0.01)
    assert south == pytest.approx(33.5, abs=0.01)
    assert east == pytest.approx(-117.0, abs=0.01)
    assert north == pytest.approx(34.0, abs=0.01)


def test_bbox_accepts_dict_style_stats(tmp_path):
    root = tmp_path / "ds"
    (root / "stats").mkdir(parents=True)
    (root / "stats" / "attributes.json").write_text(json.dumps({"attributes": {
        "geometry": {"type": "geometry", "mbr": [0.0, 0.0, 90.0, 10.0]}
    }}))
    assert Dataset(str(root)).bbox == (0.0, 0.0, 90.0, 10.0)
