"""Explode GeometryCollection rows into homogeneous Multi* rows.

Mapbox Vector Tiles have no collection type, so both tile engines dropped
GeometryCollection features. At ingest a collection row becomes up to three
rows (points -> MultiPoint, lines -> MultiLineString, polygons ->
MultiPolygon) carrying the same attributes, so nothing is lost.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import shapely
from shapely import GeometryType

logger = logging.getLogger(__name__)

_GC_TYPE_CODES = (7, 1007, 2007, 3007)
_GC_HEADERS = tuple(
    bytes([order]) + code.to_bytes(4, "little" if order == 1 else "big")
    for order in (0, 1)
    for code in _GC_TYPE_CODES
)
_DIMENSION = {0: 0, 4: 0, 1: 1, 5: 1, 2: 1, 3: 2, 6: 2}  # shapely type id -> dimension
_MULTI_FOR_DIMENSION = {0: GeometryType.MULTIPOINT, 1: GeometryType.MULTILINESTRING, 2: GeometryType.MULTIPOLYGON}


def geometry_collection_mask(column: pa.ChunkedArray | pa.Array) -> np.ndarray:
    """Boolean mask of rows whose WKB encodes a GeometryCollection (2D/Z/M/ZM)."""
    if len(column) == 0:
        return np.zeros(0, dtype=bool)
    head = pc.binary_slice(column, 0, 5)
    mask = np.zeros(len(column), dtype=bool)
    for header in _GC_HEADERS:
        mask |= np.asarray(pc.fill_null(pc.equal(head, header), False).to_numpy(zero_copy_only=False), dtype=bool)
    return mask


def _flatten_parts(geom) -> List:
    parts: List = []
    for part in shapely.get_parts(geom):
        if shapely.get_type_id(part) == 7:
            parts.extend(_flatten_parts(part))
        elif not shapely.is_empty(part):
            parts.append(part)
    return parts


def _homogeneous_geometries(geom) -> List:
    """One Multi* (or single) geometry per dimension present in a collection."""
    by_dimension: dict[int, List] = {}
    for part in _flatten_parts(geom):
        type_id = int(shapely.get_type_id(part))
        dimension = _DIMENSION.get(type_id)
        if dimension is None:
            continue
        # Multi* parts contribute their members.
        members = list(shapely.get_parts(part)) if type_id in (4, 5, 6) else [part]
        by_dimension.setdefault(dimension, []).extend(members)
    out = []
    for dimension in sorted(by_dimension):
        members = by_dimension[dimension]
        if len(members) == 1:
            out.append(members[0])
        else:
            out.append(shapely.multipoints(members) if dimension == 0
                       else shapely.multilinestrings(members) if dimension == 1
                       else shapely.multipolygons(members))
    return out


def explode_geometry_collections(table: pa.Table, geom_col: str) -> pa.Table:
    """Return ``table`` with GeometryCollection rows replaced by one row per
    homogeneous part group (same attributes). Other rows are untouched."""
    if geom_col not in table.column_names or table.num_rows == 0:
        return table
    column = table[geom_col]
    if not is_wkb_column(column):
        return table
    mask = geometry_collection_mask(column)
    if not mask.any():
        return table

    gc_rows = np.flatnonzero(mask)
    wkb_values = column.take(pa.array(gc_rows)).to_pylist()
    geoms = shapely.from_wkb(wkb_values)
    take_indices: List[int] = []
    new_wkb: List[bytes | None] = []
    kept_rows = np.flatnonzero(~mask)
    for row in kept_rows:
        take_indices.append(int(row))
        new_wkb.append(None)  # placeholder, filled from the original column
    exploded = 0
    for row, geom in zip(gc_rows, geoms):
        for part in _homogeneous_geometries(geom):
            take_indices.append(int(row))
            new_wkb.append(shapely.to_wkb(part))
            exploded += 1
    order = np.argsort(np.asarray(take_indices), kind="stable")
    take_indices = [take_indices[i] for i in order]
    new_wkb = [new_wkb[i] for i in order]
    out = table.take(pa.array(take_indices, type=pa.int64()))
    original = out[geom_col].to_pylist()
    merged = [w if w is not None else o for w, o in zip(new_wkb, original)]
    geometry = pa.array(merged, type=column.type)
    out = out.set_column(out.schema.get_field_index(geom_col), out.schema.field(geom_col), geometry)
    logger.info(
        "Exploded %d GeometryCollection rows into %d homogeneous rows", int(mask.sum()), exploded
    )
    return out


def is_wkb_column(column: pa.ChunkedArray | pa.Array) -> bool:
    return pa.types.is_binary(column.type) or pa.types.is_large_binary(column.type)
