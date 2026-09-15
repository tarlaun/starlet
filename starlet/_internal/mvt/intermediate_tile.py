"""Intermediate vector tile used by the map/reduce MVT pipeline.

Provides a small tile object that collects Web Mercator geometries,
retains a bounded uniform sample of them, merges with partial tiles from
other mappers, and simplifies the retained features into tile pixel
coordinates only when encoding MVT bytes.

Sampling is **raster-consistent and priority-based**. Every feature
carries a priority (by default ``crc32`` of its WKB — geometry-intrinsic and
deterministic; the batch pipeline passes the crc32 of the *source* WKB bytes
computed before decode). A feature whose bbox fits inside one display pixel
(``extent / PIXEL_GRID`` tile units per side) competes only with the other
sub-pixel features of the *same pixel cell*: the highest priority one is
kept, so the tile shows every occupied pixel — like a rasterised plot —
with a bounded number of features. Features larger than a pixel are ranked
by the same priority and the ``feature_capacity`` best are kept. Because
the same geometry has the same priority in every tile (and zoom level) it
touches, adjacent tiles make consistent keep/drop decisions — no seam
popping — and merging partial tiles from different mappers is a
deterministic union instead of a statistical resample.

Sub-pixel polygons and lines are encoded as a one-pixel square / segment of
their own geometry type (so they are styled like their full-size siblings)
and carry no attributes — those *dots* are looked up on demand.
"""
from __future__ import annotations

import heapq
import json
import math
import random
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mapbox_vector_tile
import pyarrow as pa
import shapely
from shapely.affinity import affine_transform
from shapely.geometry import LineString, Point, box

from starlet._internal.mvt.pyramid_partitioner import PyramidPartitioner

from .helpers import EXTENT, explode_geom, mercator_tile_bounds


DEFAULT_FEATURE_CAPACITY = 10_000
_FEATURES_SEEN_HEADER = struct.Struct("<Q")
_FEATURES_SEEN_PADDING = 0

# Display pixels per tile side: a feature whose transformed bbox fits inside
# ``extent / PIXEL_GRID`` tile units in BOTH dimensions is a sub-pixel "dot".
# Judged per-dimension, not by bbox area: a long straight line has bbox area
# 0 but is not a dot.
PIXEL_GRID = 256


def feature_priority(wkb_bytes: bytes) -> int:
    """Deterministic, geometry-intrinsic sampling priority for a feature."""
    return zlib.crc32(wkb_bytes)


@dataclass(frozen=True)
class _TileFeature:
    geometry: Any
    properties: dict[str, Any]


class IntermediateVectorTile:
    """Collect sampled Web Mercator geometries before final MVT encoding."""

    def __init__(
        self,
        z: int,
        x: int,
        y: int,
        *,
        feature_capacity: int = DEFAULT_FEATURE_CAPACITY,
        extent: int = EXTENT,
        buffer: int = 256,
        rng: random.Random | None = None,
        tile_attributes: Any = None,
    ) -> None:
        self.z = int(z)
        # None = every attribute; a list = only those columns ([] = none)
        self.tile_attributes = normalize_tile_attributes(tile_attributes)
        self.x = int(x)
        self.y = int(y)
        self.feature_capacity = max(1, int(feature_capacity))
        self.extent = int(extent)
        self.buffer = int(buffer)
        # Accepted for backward compatibility; sampling is deterministic
        # (priority top-k) and no longer draws from an RNG.
        self.rng = rng or random.Random()

        minx, miny, maxx, maxy = mercator_tile_bounds(self.z, self.x, self.y)
        width = maxx - minx
        height = maxy - miny
        x_scale = self.extent / width if width != 0 else 0.0
        y_scale = self.extent / height if height != 0 else 0.0
        self.affine_params = (
            x_scale,
            0.0,
            0.0,
            y_scale,
            -minx * x_scale,
            -miny * y_scale,
        )

        # Min-heap of (priority, seq, _TileFeature); holds the top-k larger-
        # than-a-pixel features by priority. seq is an insertion tiebreaker so
        # heap comparisons never fall through to comparing feature objects.
        self._heap: list[tuple[int, int, _TileFeature]] = []
        # Sub-pixel features: best (priority, seq, feature) per pixel cell.
        self._cells: dict[tuple[int, int], tuple[int, int, _TileFeature]] = {}
        self.cell = self.extent / PIXEL_GRID
        self._seq = 0
        self._features_seen = 0

    def _tile_bbox(self, bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Web Mercator bounds -> tile-unit bounds."""
        minx, miny, maxx, maxy = bounds
        x_scale, _, _, y_scale, xoff, yoff = self.affine_params
        return (minx * x_scale + xoff, miny * y_scale + yoff, maxx * x_scale + xoff, maxy * y_scale + yoff)

    def dot_cell(self, bounds: tuple[float, float, float, float]) -> tuple[int, int] | None:
        """Pixel cell of a sub-pixel feature (Web Mercator ``bounds``), else None."""
        x0, y0, x1, y1 = self._tile_bbox(bounds)
        if (x1 - x0) <= self.cell and (y1 - y0) <= self.cell:
            return (math.floor((x0 + x1) * 0.5 / self.cell), math.floor((y0 + y1) * 0.5 / self.cell))
        return None

    @property
    def tile_id(self) -> int:
        """Unique tile ID for this z/x/y."""
        return PyramidPartitioner.encode_tile_id(self.z, self.x, self.y)

    @property
    def feature_count(self) -> int:
        """Number of retained raw features."""
        return len(self._heap) + len(self._cells)

    def _entries(self) -> list[tuple[int, int, _TileFeature]]:
        """All retained (priority, seq, feature) entries in offer order."""
        return sorted(list(self._heap) + list(self._cells.values()), key=lambda e: e[1])

    @property
    def _features(self) -> list[_TileFeature]:
        """Retained features (offer order); kept for introspection."""
        return [entry[2] for entry in self._entries()]

    def add_feature(
        self,
        geometry: Any,
        properties: dict[str, Any] | None = None,
        priority: int | None = None,
    ) -> bool:
        """Offer a Web Mercator geometry; keep it if it ranks in the top-k.

        ``priority`` should be :func:`feature_priority` of the feature's
        canonical (source) WKB bytes so that every tile the feature touches
        ranks it identically. When omitted it is derived from the current
        geometry's WKB.
        """
        if geometry is None or geometry.is_empty:
            return False

        self._features_seen += 1

        if priority is None:
            priority = feature_priority(shapely.to_wkb(geometry))
        priority = int(priority)

        cell = self.dot_cell(geometry.bounds)
        if cell is not None:
            current = self._cells.get(cell)
            if current is not None and priority <= current[0]:
                return False
        elif len(self._heap) >= self.feature_capacity and priority <= self._heap[0][0]:
            return False

        clean_properties = {
            key: value
            for key, value in (properties or {}).items()
            if value is not None
        }
        entry = (priority, self._seq, _TileFeature(geometry, clean_properties))
        self._seq += 1

        if cell is not None:
            self._cells[cell] = entry
        elif len(self._heap) < self.feature_capacity:
            heapq.heappush(self._heap, entry)
        else:
            heapq.heapreplace(self._heap, entry)
        return True

    def simplify_geometry(self, geometry: Any) -> list[Any]:
        """Return simplified tile-pixel geometries ready for MVT encoding."""
        return self._tile_geometry(geometry)[0]

    def _tile_geometry(self, geometry: Any) -> tuple[list[Any], bool]:
        """Tile-pixel geometries plus whether the feature became a *dot*."""
        geometry = affine_transform(
            geometry,
            (
                self.affine_params[0],
                0.0,
                0.0,
                self.affine_params[3],
                self.affine_params[4],
                self.affine_params[5],
            ),
        )

        minx, miny, maxx, maxy = geometry.bounds
        if (maxx - minx) <= self.cell and (maxy - miny) <= self.cell:
            # A sub-pixel feature becomes a one-pixel shape of its own kind.
            cx, cy = (minx + maxx) * 0.5, (miny + maxy) * 0.5
            lo, hi = -self.buffer, self.extent + self.buffer
            if cx < lo or cy < lo or cx > hi or cy > hi:
                return [], True
            kind = _base_kind(geometry)
            h = self.cell * 0.5
            if kind == "Polygon":
                return [box(cx - h, cy - h, cx + h, cy + h)], True
            if kind == "LineString":
                return [LineString([(cx - h, cy), (cx + h, cy)])], True
            return [Point(cx, cy)], False

        # Simplify the geometry to reduce the number of coordinates. Use tolerance of one pixel.
        if shapely.count_coordinates(geometry) > 10:
            geometry = shapely.simplify(geometry, 1.0, preserve_topology=False)
        if geometry.geom_type not in {"Point", "MultiPoint"}:
            geometry = shapely.clip_by_rect(
                geometry,
                -self.buffer, -self.buffer, self.extent + self.buffer, self.extent + self.buffer,
            )

        out = []
        for part in explode_geom(geometry):
            if not part.is_empty:
                out.append(part)
        return out, False

    def merge(self, other: "IntermediateVectorTile") -> None:
        """Merge another partial tile for the same z/x/y: top-k union.

        Every entry keeps the priority it was offered with, so the merged
        result is exactly the top ``feature_capacity`` features by priority
        across both partials — deterministic and independent of merge order.
        """
        if (self.z, self.x, self.y) != (other.z, other.x, other.y):
            raise ValueError("Cannot merge intermediate tiles with different tile IDs")

        combined = [(p, s) for (p, _, s) in self._heap]
        combined.extend((p, s) for (p, _, s) in other._heap)
        # Sort by priority (stable: self's entries win ties deterministically).
        combined.sort(key=lambda item: item[0], reverse=True)
        kept = combined[: self.feature_capacity]

        self._heap = []
        self._seq = 0
        for priority, feature in kept:
            heapq.heappush(self._heap, (priority, self._seq, feature))
            self._seq += 1
        # Pixel cells: the higher priority wins; self keeps ties.
        for cell, (priority, _, feature) in other._cells.items():
            current = self._cells.get(cell)
            if current is None or priority > current[0]:
                self._cells[cell] = (priority, self._seq, feature)
                self._seq += 1
        self._features_seen += other._features_seen

    def write_features(self, path) -> None:
        """Write retained features, priorities, and seen count to disk."""
        entries = self._entries()
        table = pa.table(
            {
                "geometry": pa.array(
                    [entry[2].geometry.wkb for entry in entries],
                    type=pa.binary(),
                ),
                "properties": pa.array(
                    [
                        json.dumps(entry[2].properties, separators=(",", ":"))
                        for entry in entries
                    ],
                    type=pa.string(),
                ),
                "priority": pa.array(
                    [entry[0] for entry in entries],
                    type=pa.uint64(),
                ),
            }
        )
        payload_sink = pa.BufferOutputStream()
        with pa.ipc.new_file(payload_sink, table.schema) as writer:
            writer.write_table(table)
        payload = payload_sink.getvalue().to_pybytes()
        with pa.OSFile(str(path), "wb") as sink:
            sink.write(_FEATURES_SEEN_HEADER.pack(self._features_seen))
            sink.write(b"\x00" * _FEATURES_SEEN_PADDING)
            sink.write(payload)

    def load_features(self, path) -> None:
        """Load retained features (with their priorities) from disk."""
        data = Path(path).read_bytes()
        self._features_seen = _FEATURES_SEEN_HEADER.unpack(data[:_FEATURES_SEEN_HEADER.size])[0]
        payload_offset = _FEATURES_SEEN_HEADER.size + _FEATURES_SEEN_PADDING
        table = pa.ipc.open_file(pa.BufferReader(data[payload_offset:])).read_all()

        geometries = table["geometry"].to_pylist()
        properties = table["properties"].to_pylist()
        if "priority" in table.column_names:
            priorities = table["priority"].to_pylist()
        else:
            # Files written before the priority column: recompute from WKB.
            priorities = [feature_priority(geometry_bytes) for geometry_bytes in geometries]

        seen = self._features_seen
        for geometry_bytes, property_json, priority in zip(geometries, properties, priorities):
            # Re-offer: classification (pixel cell vs top-k) is recomputed.
            self.add_feature(shapely.from_wkb(geometry_bytes), json.loads(property_json), priority=int(priority))
        self._features_seen = seen

    def encode(self, layer_name: str = "layer0") -> bytes:
        """Encode the retained features as an MVT binary payload."""
        layer = {
            "name": layer_name,
            "features": self._mvt_features(),
            "extent": self.extent,
        }
        result = mapbox_vector_tile.encode(
            [layer],
            default_options={"extents": self.extent},
        )
        return result

    def _mvt_features(self) -> list[dict[str, Any]]:
        # Sub-pixel *points* keep their attributes only while the tile has at
        # most ``feature_capacity`` sub-pixel features; denser tiles carry
        # them as bare dots (attributes stay reachable through the lookup).
        strip_point_attrs = len(self._cells) > self.feature_capacity
        out = []
        for _, _, feature in self._entries():
            geometries, is_dot = self._tile_geometry(feature.geometry)
            for geometry in geometries:
                bare = is_dot or (strip_point_attrs and geometry.geom_type == "Point")
                out.append(
                    {
                        "geometry": geometry,
                        "properties": {} if bare else self._select_properties(feature.properties),
                    }
                )
        return out

    def _select_properties(self, properties: dict[str, Any]) -> dict[str, Any]:
        if self.tile_attributes is None:
            return dict(properties)
        return {k: v for k, v in properties.items() if k in self.tile_attributes}


def normalize_tile_attributes(value: Any) -> list[str] | None:
    """Tile-attribute policy -> ``None`` (all columns) or a list of columns.

    Accepts ``None``/``"all"`` (everything), ``"none"``/``""``/``[]`` (no
    attributes in tiles — fetch them with the record lookup), a comma-separated
    string, or a sequence of column names.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in ("", "all", "*"):
            return None
        if text.lower() == "none":
            return []
        return [part.strip() for part in text.split(",") if part.strip()]
    return [str(v) for v in value]


def _base_kind(geometry: Any) -> str:
    """'Polygon' | 'LineString' | 'Point' for a (possibly multi/collection) geometry."""
    kinds = {part.geom_type for part in explode_geom(geometry) if not part.is_empty}
    if "Polygon" in kinds:
        return "Polygon"
    if "LineString" in kinds or "LinearRing" in kinds:
        return "LineString"
    return "Point"
