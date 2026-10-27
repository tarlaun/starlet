"""PMTiles archives carry TileJSON metadata and real header bounds."""
import gzip
import json

from pmtiles.reader import MmapSource, Reader

from starlet._internal.pmtiles.exporter import export_to_pmtiles
from starlet._internal.pmtiles.metadata import center_zoom_for_bounds, layer_fields, pmtiles_export_options


def _write_stats(dataset_dir, attributes):
    stats_dir = dataset_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    (stats_dir / "attributes.json").write_text(json.dumps({"attributes": attributes}))


def test_layer_fields_types_from_stats(tmp_path):
    _write_stats(tmp_path, [
        {"name": "population", "stats": {"non_null_count": 3, "min": 1, "max": 9, "mean": 4.0}},
        {"name": "name", "stats": {"non_null_count": 3, "avg_length": 4.0, "top_k": [{"value": "a", "count": 1}]}},
        {"name": "flag", "stats": {"non_null_count": 3, "top_k": [{"value": True, "count": 2}, {"value": False, "count": 1}]}},
        {"name": "geometry", "stats": {"mbr": [0, 0, 1, 1], "geom_types": {"Point": 3}}},
        {"name": "_tile_id", "stats": {"min": 0, "max": 4}},
    ])

    assert layer_fields(tmp_path) == {"population": "Number", "name": "String", "flag": "Boolean"}
    assert layer_fields(tmp_path, ["name"]) == {"name": "String"}
    assert layer_fields(tmp_path, "none") == {}


def test_center_zoom_fits_bounds():
    assert center_zoom_for_bounds((-180, -90, 180, 90), 0, 7) == 0
    assert center_zoom_for_bounds((-117.5, 33.5, -117.0, 34.0), 0, 12) == 8
    assert center_zoom_for_bounds((-117.5, 33.5, -117.0, 34.0), 0, 4) == 4


def test_export_options_use_dataset_bbox_and_zooms(tmp_path):
    dataset = tmp_path / "ds"
    _write_stats(dataset, [
        {"name": "n", "stats": {"min": 1, "max": 2, "mean": 1.5}},
        {"name": "geometry", "stats": {"mbr": [-117.5, 33.5, -117.0, 34.0]}},
    ])

    options = pmtiles_export_options(dataset, [1, 0, 4])

    assert options["bounds"] == (-117.5, 33.5, -117.0, 34.0)
    assert options["center"][:2] == (-117.25, 33.75)
    meta = options["metadata"]
    assert meta["name"] == "ds"
    assert meta["format"] == "pbf"
    assert meta["minzoom"] == "0" and meta["maxzoom"] == "2"
    assert meta["bounds"] == "-117.500000,33.500000,-117.000000,34.000000"
    assert meta["vector_layers"] == [
        {"id": "layer0", "description": "", "minzoom": 0, "maxzoom": 2, "fields": {"n": "Number"}}
    ]


def test_export_options_without_stats(tmp_path):
    options = pmtiles_export_options(tmp_path, [1])
    assert options["bounds"] is None and options["center"] is None
    assert "bounds" not in options["metadata"]
    assert options["metadata"]["vector_layers"][0]["fields"] == {}


def test_export_writes_metadata_and_header_bounds(tmp_path):
    mvt_dir = tmp_path / "mvt"
    (mvt_dir / "0" / "0").mkdir(parents=True)
    (mvt_dir / "0" / "0" / "0.mvt").write_bytes(b"tile-bytes")
    out = tmp_path / "tiles.pmtiles"

    export_to_pmtiles(
        str(mvt_dir), str(out), "mvt", "gzip",
        metadata={"name": "demo", "vector_layers": []},
        bounds=(-117.5, 33.5, -117.0, 34.0),
        center=(-117.25, 33.75, 9),
    )

    with out.open("rb") as handle:
        reader = Reader(MmapSource(handle))
        header = reader.header()
        assert reader.metadata() == {"name": "demo", "vector_layers": []}
        assert (header["min_lon_e7"], header["min_lat_e7"]) == (-1175000000, 335000000)
        assert (header["max_lon_e7"], header["max_lat_e7"]) == (-1170000000, 340000000)
        assert (header["center_lon_e7"], header["center_lat_e7"], header["center_zoom"]) == (-1172500000, 337500000, 9)
        assert gzip.decompress(reader.get(0, 0, 0)) == b"tile-bytes"
