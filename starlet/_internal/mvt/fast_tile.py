"""Vectorised pure-Python single-tile generation.

The same algorithm as the Rust core (``rust/starlet-core/src/tiler.rs``)
expressed with numpy and shapely array functions, so that a tile costs a
handful of array operations instead of a Python loop per feature:

1. partitions pruned by filename bbox, rows by the ``_bbox_*`` columns
   (Parquet predicate push-down) — or, for legacy datasets without them, by
   the bounds of the decoded geometries;
2. **raster-consistent selection**: priority ``(size, crc32(source WKB))``;
   the ``feature_capacity`` best features larger than a display pixel kept
   in full, every other feature reduced to one dot per pixel cell — all as
   array sorts / group-bys;
3. winners only are decoded: projected to y-down tile units, simplified
   (Douglas-Peucker), clipped, sub-pixel slivers dropped;
4. encoded with :mod:`starlet._internal.mvt.mvt_encoder` (numpy varints),
   dots packed into one MultiPolygon / MultiLineString feature.

Output is byte-compatible with the Rust engine for EPSG:4326 datasets.
"""
from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import shapely

from starlet._internal.mvt.helpers import mercator_tile_bounds
from starlet._internal.mvt.intermediate_tile import (
    PIXEL_GRID,
    normalize_simplify_tolerance,
    normalize_tile_attributes,
)
from starlet._internal.mvt.mvt_encoder import (
    GEOM_LINE,
    GEOM_POLYGON,
    LayerEncoder,
    geometry_commands,
    packed_segment_dots,
    packed_square_dots,
)
from starlet._internal.server.tiler.parquet_index import INTERNAL_COLS, ParquetIndex

WORLD = 20037508.342789244
LAT_MAX = 85.05112878
SIMPLIFY_MIN_COORDS = 10
MIN_RING_AREA = 0.5
BBOX_COLS = ("_bbox_xmin", "_bbox_ymin", "_bbox_xmax", "_bbox_ymax")


def lonlat_to_merc(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.clip(lon, -180.0, 180.0) * (WORLD / 180.0)
    la = np.clip(lat, -LAT_MAX, LAT_MAX)
    y = np.log(np.tan((90.0 + la) * (np.pi / 360.0))) / np.pi * WORLD
    return x, y


def merc_to_lonlat(x: float, y: float) -> tuple[float, float]:
    lon = x / WORLD * 180.0
    lat = (2.0 * np.arctan(np.exp(y / WORLD * np.pi)) - np.pi / 2.0) * 180.0 / np.pi
    return float(lon), float(lat)


class TileFrame:
    """Tile geometry: Mercator bounds, buffered query box, and the affine map
    to y-down tile units (identical to the Rust ``TileTransform``)."""

    def __init__(self, z: int, x: int, y: int, extent: int, buffer: int) -> None:
        minx, miny, maxx, maxy = mercator_tile_bounds(int(z), int(x), int(y))
        self.z, self.x, self.y = int(z), int(x), int(y)
        self.extent = int(extent)
        self.buffer = int(buffer)
        self.minx = minx
        self.maxy = maxy
        self.scale = self.extent / (maxx - minx)
        self.cell = self.extent / PIXEL_GRID
        pad = self.buffer / self.scale
        self.query_merc = (minx - pad, miny - pad, maxx + pad, maxy + pad)
        x0, y0 = merc_to_lonlat(self.query_merc[0], self.query_merc[1])
        x1, y1 = merc_to_lonlat(self.query_merc[2], self.query_merc[3])
        self.query_lonlat = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    def to_tile(self, xm: np.ndarray, ym: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (xm - self.minx) * self.scale, (self.maxy - ym) * self.scale

    def lonlat_to_tile_coords(self, coords: np.ndarray) -> np.ndarray:
        xm, ym = lonlat_to_merc(coords[:, 0], coords[:, 1])
        tx, ty = self.to_tile(xm, ym)
        return np.column_stack((tx, ty))


def _crc32_array(wkb_list: list) -> np.ndarray:
    crc = zlib.crc32
    return np.fromiter((crc(w) if w is not None else 0 for w in wkb_list), dtype=np.uint32, count=len(wkb_list))


def select(
    frame: TileFrame,
    lon0: np.ndarray,
    lat0: np.ndarray,
    lon1: np.ndarray,
    lat1: np.ndarray,
    crc: np.ndarray,
    feature_capacity: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Raster-consistent selection over candidate bboxes (EPSG:4326).

    Returns ``(full_idx, dot_idx, dot_cx, dot_cy)``: indices of the features
    kept in full, indices of the features kept as dots, and the dots' tile-
    unit centres. ``dot_idx`` is one per pixel cell (best priority).
    """
    k = max(1, int(feature_capacity))
    ax, ay = lonlat_to_merc(lon0, lat0)
    bx, by = lonlat_to_merc(lon1, lat1)
    x0, y0 = frame.to_tile(ax, ay)
    x1, y1 = frame.to_tile(bx, by)
    w = np.abs(x1 - x0)
    h = np.abs(y1 - y0)
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    cell = frame.cell
    small = (w <= cell) & (h <= cell)
    size16 = np.minimum(np.maximum(w, h) * 16.0, 2.0**31 - 1).astype(np.int64)
    prio = (size16 << 32) | crc.astype(np.int64)
    cellx = np.floor(cx / cell).astype(np.int64)
    celly = np.floor(cy / cell).astype(np.int64)
    cellkey = ((celly + (1 << 20)) << 21) + (cellx + (1 << 20))

    large = np.flatnonzero(~small)
    if len(large) > k:
        order = np.argsort(-prio[large], kind="stable")
        full = large[order[:k]]
        demoted = large[order[k:]]
    else:
        full = large
        demoted = np.zeros(0, dtype=np.int64)
    dot_cand = np.concatenate((np.flatnonzero(small), demoted))
    if len(dot_cand):
        keys = cellkey[dot_cand]
        # best priority per cell; ties -> earliest offered (stable)
        order = np.lexsort((dot_cand, -prio[dot_cand], keys))
        skeys = keys[order]
        first = np.ones(len(order), dtype=bool)
        first[1:] = skeys[1:] != skeys[:-1]
        dots = dot_cand[order[first]]
    else:
        dots = np.zeros(0, dtype=np.int64)
    return np.sort(full), dots, cx[dots], cy[dots]


def _tile_geometries(frame: TileFrame, geoms: np.ndarray, tolerance: float) -> np.ndarray:
    """Project lon/lat geometries to tile units, simplify, clip, drop slivers.
    Returns an object array (None where the feature vanished)."""
    if len(geoms) == 0:
        return geoms
    g = shapely.transform(geoms, frame.lonlat_to_tile_coords)
    tid = shapely.get_type_id(g)
    is_point = (tid == 0) | (tid == 4)
    if tolerance > 0:
        many = shapely.get_num_coordinates(g) > SIMPLIFY_MIN_COORDS
        m = many & ~is_point
        if m.any():
            g[m] = shapely.simplify(g[m], tolerance, preserve_topology=False)
    b = frame.buffer
    e = frame.extent
    clip_mask = ~is_point
    if clip_mask.any():
        g[clip_mask] = shapely.clip_by_rect(g[clip_mask], -b, -b, e + b, e + b)
    # sub-pixel slivers (clipping artefacts) are dropped like the Rust core
    tid = shapely.get_type_id(g)
    poly = (tid == 3) | (tid == 6)
    if poly.any():
        idx = np.flatnonzero(poly)
        parts, part_of = shapely.get_parts(g[idx], return_index=True)
        keep = shapely.area(parts) >= MIN_RING_AREA
        if not keep.all():
            parts = parts[keep]
            part_of = part_of[keep]
            rebuilt = np.full(len(idx), None, dtype=object)
            if len(parts):
                multi = shapely.multipolygons(parts, indices=part_of)
                # multipolygons() returns one geometry per distinct index in order
                rebuilt[np.unique(part_of)] = multi
            g[idx] = rebuilt
    # GeometryCollections (mixed clip results): keep the polygon / line parts
    tid = shapely.get_type_id(g)
    coll = tid == 7
    if coll.any():
        for i in np.flatnonzero(coll):
            parts = shapely.get_parts(g[i])
            ptid = shapely.get_type_id(parts)
            polys = parts[(ptid == 3) | (ptid == 6)]
            lines = parts[(ptid == 1) | (ptid == 5)]
            if len(polys):
                g[i] = shapely.multipolygons(shapely.get_parts(polys)) if len(polys) > 1 else polys[0]
            elif len(lines):
                g[i] = shapely.multilinestrings(shapely.get_parts(lines)) if len(lines) > 1 else lines[0]
            else:
                g[i] = None
    empty = shapely.is_empty(g) | shapely.is_missing(g)
    g[empty] = None
    return g


def _partition_arrays(index: ParquetIndex, path: Path, query_lonlat, geom_col: str, has_bbox: bool, columns):
    """Read one partition's candidates: the table plus lon/lat bbox arrays."""
    minx, miny, maxx, maxy = query_lonlat
    if has_bbox:
        flt = (
            (pc.field("_bbox_xmax") >= minx)
            & (pc.field("_bbox_xmin") <= maxx)
            & (pc.field("_bbox_ymax") >= miny)
            & (pc.field("_bbox_ymin") <= maxy)
        )
        table = pq.read_table(path, columns=columns, filters=flt)
        if table.num_rows == 0:
            return None
        bb = [table[c].to_numpy(zero_copy_only=False).astype(np.float64) for c in BBOX_COLS]
        return table, bb[0], bb[1], bb[2], bb[3]
    table = pq.read_table(path, columns=columns)
    if table.num_rows == 0:
        return None
    geoms = shapely.from_wkb(table[geom_col].to_pylist(), on_invalid="ignore")
    bounds = shapely.bounds(geoms)
    ok = ~np.isnan(bounds[:, 0])
    ok &= (bounds[:, 2] >= minx) & (bounds[:, 0] <= maxx) & (bounds[:, 3] >= miny) & (bounds[:, 1] <= maxy)
    if not ok.any():
        return None
    sel = np.flatnonzero(ok)
    table = table.take(pa.array(sel))
    bounds = bounds[sel]
    return table, bounds[:, 0], bounds[:, 1], bounds[:, 2], bounds[:, 3]


def generate_tile(
    dataset_path: str | Path,
    tile_id: tuple[int, int, int],
    *,
    feature_capacity: int,
    extent: int,
    buffer: int,
    tile_attributes: Any = None,
    simplify_tolerance: Any = None,
    layer_name: str = "layer0",
    index: ParquetIndex | None = None,
) -> bytes:
    """One MVT tile from a starlet dataset directory (EPSG:4326 partitions)."""
    z, x, y = tile_id
    frame = TileFrame(z, x, y, extent, buffer)
    attrs_policy = normalize_tile_attributes(tile_attributes)
    tolerance = normalize_simplify_tolerance(simplify_tolerance, frame.cell)
    parquet_dir = Path(dataset_path) / "parquet_tiles"
    if index is None:
        index = ParquetIndex(parquet_dir)

    tables: list[pa.Table] = []
    geom_cols: list[str] = []
    attr_cols: list[list[str]] = []
    lon0s, lat0s, lon1s, lat1s, crcs, part_of, row_of = [], [], [], [], [], [], []
    for pi, path in enumerate(index.find_intersecting_files(frame.query_lonlat)):
        names, geom_col, has_bbox, _crs = index._schema_info(path)
        attrs = [n for n in names if n != geom_col and n not in INTERNAL_COLS]
        if attrs_policy is not None:
            attrs = [n for n in attrs if n in attrs_policy]
        columns = [geom_col] + (list(BBOX_COLS) if has_bbox else []) + attrs
        res = _partition_arrays(index, path, frame.query_lonlat, geom_col, has_bbox, columns)
        if res is None:
            continue
        table, a, b, c, d = res
        wkb = table[geom_col].to_pylist()
        n = table.num_rows
        tables.append(table)
        geom_cols.append(geom_col)
        attr_cols.append(attrs)
        lon0s.append(a)
        lat0s.append(b)
        lon1s.append(c)
        lat1s.append(d)
        crcs.append(_crc32_array(wkb))
        part_of.append(np.full(n, len(tables) - 1, dtype=np.int64))
        row_of.append(np.arange(n, dtype=np.int64))

    layer = LayerEncoder(layer_name, extent)
    if not tables:
        return layer.encode_tile()

    lon0 = np.concatenate(lon0s)
    lat0 = np.concatenate(lat0s)
    lon1 = np.concatenate(lon1s)
    lat1 = np.concatenate(lat1s)
    crc = np.concatenate(crcs)
    part_of_arr = np.concatenate(part_of)
    row_of_arr = np.concatenate(row_of)
    full, dots, dot_cx, dot_cy = select(frame, lon0, lat0, lon1, lat1, crc, feature_capacity)
    strip_point_attrs = len(dots) > max(1, int(feature_capacity))

    # ---- decode winners --------------------------------------------------
    def wkb_of(idx: np.ndarray) -> list:
        out = []
        for pi in np.unique(part_of_arr[idx]):
            m = part_of_arr[idx] == pi
            rows = row_of_arr[idx[m]]
            col = tables[pi][geom_cols[pi]].take(pa.array(rows)).to_pylist()
            out.append((idx[m], col))
        # restore idx order
        merged = [None] * len(idx)
        pos = {int(i): j for j, i in enumerate(idx)}
        for sub_idx, col in out:
            for i, w in zip(sub_idx.tolist(), col):
                merged[pos[i]] = w
        return merged

    def props_of(idx: np.ndarray) -> list:
        out = [None] * len(idx)
        pos = {int(i): j for j, i in enumerate(idx)}
        for pi in np.unique(part_of_arr[idx]):
            m = part_of_arr[idx] == pi
            sub = idx[m]
            rows = pa.array(row_of_arr[sub])
            cols = {name: tables[pi][name].take(rows).to_pylist() for name in attr_cols[pi]}
            for j, i in enumerate(sub.tolist()):
                out[pos[i]] = [(name, cols[name][j]) for name in attr_cols[pi]]
        return out

    # dots: kind from the geometry type, centre from the bbox
    dot_geoms = shapely.from_wkb(wkb_of(dots), on_invalid="ignore") if len(dots) else np.zeros(0, dtype=object)
    dot_tid = shapely.get_type_id(dot_geoms) if len(dots) else np.zeros(0, dtype=np.int64)
    lo, hi = -frame.buffer, frame.extent + frame.buffer
    inside = (dot_cx >= lo) & (dot_cy >= lo) & (dot_cx <= hi) & (dot_cy <= hi)
    poly_dots = ((dot_tid == 3) | (dot_tid == 6) | (dot_tid == 7)) & inside
    line_dots = ((dot_tid == 1) | (dot_tid == 5) | (dot_tid == 2)) & inside
    point_dots = ((dot_tid == 0) | (dot_tid == 4)) & inside

    # features encoded individually, in offer (file) order: full winners and native points
    indiv = np.sort(np.concatenate((full, dots[point_dots])))
    is_full = np.isin(indiv, full)
    geoms = shapely.from_wkb(wkb_of(indiv), on_invalid="ignore") if len(indiv) else np.zeros(0, dtype=object)
    tgeoms = _tile_geometries(frame, geoms, tolerance) if len(indiv) else geoms
    # native points that turned out sub-pixel keep their point form (no simplification / clipping)
    cmds, types = geometry_commands(tgeoms)
    need_props = np.ones(len(indiv), dtype=bool)
    if strip_point_attrs:
        need_props &= is_full | (types != 1)
    props = props_of(indiv[need_props]) if need_props.any() else []
    pj = 0
    for j in range(len(indiv)):
        if cmds[j] is None:
            if need_props[j]:
                pj += 1
            continue
        p = None
        if need_props[j]:
            p = props[pj]
            pj += 1
        layer.add(cmds[j], int(types[j]), p)

    half = frame.cell * 0.5
    sq = packed_square_dots(np.column_stack((dot_cx[poly_dots], dot_cy[poly_dots])), half)
    if sq:
        layer.add(sq, GEOM_POLYGON, None)
    seg = packed_segment_dots(np.column_stack((dot_cx[line_dots], dot_cy[line_dots])), half)
    if seg:
        layer.add(seg, GEOM_LINE, None)
    return layer.encode_tile()
