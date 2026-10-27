"""Native GeoArrow geometry columns -> WKB.

GeoParquet 1.1 allows geometry columns stored as native GeoArrow arrays
(``geoarrow.point`` ... ``geoarrow.multipolygon``: nested lists of
``struct<x, y>`` or ``fixed_size_list<double>`` coordinates) instead of WKB.
Starlet's pipeline works on WKB, so such columns are converted on read.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

import numpy as np
import pyarrow as pa
import shapely
from shapely import GeometryType

logger = logging.getLogger(__name__)

NATIVE_ENCODINGS = {
    "point": GeometryType.POINT,
    "linestring": GeometryType.LINESTRING,
    "polygon": GeometryType.POLYGON,
    "multipoint": GeometryType.MULTIPOINT,
    "multilinestring": GeometryType.MULTILINESTRING,
    "multipolygon": GeometryType.MULTIPOLYGON,
}
_LIST_DEPTH = {
    GeometryType.POINT: 0,
    GeometryType.LINESTRING: 1,
    GeometryType.POLYGON: 2,
    GeometryType.MULTIPOINT: 1,
    GeometryType.MULTILINESTRING: 2,
    GeometryType.MULTIPOLYGON: 3,
}


def extension_name(field: pa.Field) -> Optional[str]:
    meta = field.metadata or {}
    name = meta.get(b"ARROW:extension:name")
    if name is None:
        ext = getattr(field.type, "extension_name", None)
        return str(ext) if ext else None
    return name.decode("utf-8")


def extension_crs(field: pa.Field) -> Any:
    """CRS recorded in the field's GeoArrow extension metadata, if any."""
    meta = field.metadata or {}
    raw = meta.get(b"ARROW:extension:metadata")
    if not raw:
        return None
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    return parsed.get("crs") if isinstance(parsed, dict) else None


def is_wkb_type(dtype: pa.DataType) -> bool:
    return pa.types.is_binary(dtype) or pa.types.is_large_binary(dtype)


def native_geometry_type(field: pa.Field, geo_encoding: Optional[str] = None) -> Optional[GeometryType]:
    """GeometryType of a native GeoArrow field, or ``None`` for WKB / unknown."""
    if is_wkb_type(field.type):
        return None
    name = extension_name(field)
    key = None
    if name and name.startswith("geoarrow."):
        key = name[len("geoarrow."):]
    elif geo_encoding:
        key = str(geo_encoding).lower()
    if key == "wkt":
        return GeometryType.GEOMETRYCOLLECTION  # marker: handled by _wkt_to_wkb
    if key in NATIVE_ENCODINGS:
        return NATIVE_ENCODINGS[key]
    # Fall back on structure: nested lists ending in struct<x,y> / fixed_size_list.
    depth, inner = 0, field.type
    while pa.types.is_list(inner) or pa.types.is_large_list(inner):
        inner = inner.value_type
        depth += 1
    if pa.types.is_struct(inner) or pa.types.is_fixed_size_list(inner):
        for geometry_type, list_depth in _LIST_DEPTH.items():
            if list_depth == depth and geometry_type in (
                GeometryType.POINT, GeometryType.LINESTRING, GeometryType.POLYGON, GeometryType.MULTIPOLYGON
            ):
                logger.warning(
                    "Geometry column %r has no GeoArrow extension name; assuming %s from its nesting depth",
                    field.name, geometry_type.name,
                )
                return geometry_type
    return None


def _coords(array: pa.Array) -> np.ndarray:
    if pa.types.is_struct(array.type):
        fields = [array.field(i) for i in range(array.type.num_fields)]
        return np.column_stack([np.asarray(f.to_numpy(zero_copy_only=False), dtype=np.float64) for f in fields])
    if pa.types.is_fixed_size_list(array.type):
        dim = array.type.list_size
        flat = np.asarray(array.flatten().to_numpy(zero_copy_only=False), dtype=np.float64)
        return flat.reshape(-1, dim)
    raise TypeError(f"Unsupported GeoArrow coordinate type: {array.type}")


def _ragged(array: pa.Array, depth: int) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Coordinates and shapely-style offsets (innermost first) of a nested list array."""
    offsets_outer_to_inner: list[np.ndarray] = []
    current = array
    for _ in range(depth):
        if not (pa.types.is_list(current.type) or pa.types.is_large_list(current.type)):
            raise TypeError(f"Expected a list array at depth {len(offsets_outer_to_inner)}, got {current.type}")
        offsets = np.asarray(current.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
        offsets = offsets - offsets[0]
        offsets_outer_to_inner.append(offsets)
        current = current.flatten()
    coords = _coords(current)
    return coords, tuple(reversed(offsets_outer_to_inner))


def geoarrow_to_wkb(column: pa.ChunkedArray | pa.Array, geometry_type: GeometryType) -> pa.Array:
    """Convert a native GeoArrow column to a WKB ``binary`` array (nulls kept)."""
    array = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    if geometry_type == GeometryType.GEOMETRYCOLLECTION:  # WKT marker
        values = array.to_pylist()
        geoms = shapely.from_wkt([v if v is not None else None for v in values])
        wkb = shapely.to_wkb(geoms)
        return pa.array([None if v is None else w for v, w in zip(values, wkb)], type=pa.binary())

    null_mask = np.asarray(array.is_null().to_numpy(zero_copy_only=False), dtype=bool)
    if np.any(null_mask):
        # Nulls have empty list slots in GeoArrow; fill them so offsets stay valid.
        array = array.fill_null(_empty_value(array.type)) if hasattr(array, "fill_null") else array
    depth = _LIST_DEPTH[geometry_type]
    coords, offsets = _ragged(array, depth)
    if geometry_type == GeometryType.POINT:
        geoms = shapely.from_ragged_array(geometry_type, coords, None)
    else:
        geoms = shapely.from_ragged_array(geometry_type, coords, offsets)
    if len(geoms) != len(array):
        raise ValueError(f"GeoArrow conversion produced {len(geoms)} geometries for {len(array)} rows")
    wkb = shapely.to_wkb(geoms)
    if np.any(null_mask):
        return pa.array([None if is_null else w for is_null, w in zip(null_mask, wkb)], type=pa.binary())
    return pa.array(list(wkb), type=pa.binary())


def _empty_value(dtype: pa.DataType):
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        return pa.scalar([], type=dtype)
    if pa.types.is_struct(dtype):
        return pa.scalar({f.name: float("nan") for f in dtype}, type=dtype)
    if pa.types.is_fixed_size_list(dtype):
        return pa.scalar([float("nan")] * dtype.list_size, type=dtype)
    return None


def wkb_schema(schema: pa.Schema, geom_col: str, crs: Any = None) -> pa.Schema:
    """``schema`` with ``geom_col`` as a WKB binary field and GeoParquet
    ``geo`` metadata describing it (encoding WKB, optional CRS)."""
    index = schema.get_field_index(geom_col)
    field = schema.field(index)
    if crs is None:
        crs = extension_crs(field)
    new_field = pa.field(geom_col, pa.binary(), nullable=True)
    fields = list(schema)
    fields[index] = new_field
    metadata = dict(schema.metadata or {})
    geo: dict = {}
    if b"geo" in metadata:
        try:
            geo = json.loads(metadata[b"geo"].decode("utf-8"))
        except Exception:
            geo = {}
    geo.setdefault("version", "1.1.0")
    geo["primary_column"] = geom_col
    columns = geo.setdefault("columns", {})
    column_meta = dict(columns.get(geom_col) or {})
    column_meta["encoding"] = "WKB"
    if crs is not None:
        column_meta["crs"] = crs
    columns[geom_col] = column_meta
    metadata[b"geo"] = json.dumps(geo).encode("utf-8")
    return pa.schema(fields, metadata=metadata)
