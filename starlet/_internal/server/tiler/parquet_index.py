"""Filename-based spatial index for GeoParquet tiles.

Parquet tiles are named ``tile_XXXXXX__minx_miny_maxx_maxy.parquet``.  The
bounding box is parsed from the filename to enable fast MBR intersection
filtering without reading file metadata.

On-the-fly tile generation only needs the geometries that fall inside the
requested tile.  This module therefore:

  1. parses every partition's filename bbox once, at construction time;
  2. caches decoded partitions (in their native CRS) in a small LRU so that
     consecutive tile requests over the same area avoid re-reading the file;
  3. spatially pre-filters a partition to the tile's bbox *before* the
     expensive reprojection / validity-repair steps run.

That last step turns per-tile cost from O(partition size) into
O(geometries actually inside the tile), which is the difference between
multi-second and millisecond on-demand tiles for dense datasets.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd

BBox = Tuple[float, float, float, float]


def parse_parquet_bbox(fname: str) -> Optional[BBox]:
    """Parse ``(minx, miny, maxx, maxy)`` from a tile filename, or ``None``.

    Expected format: ``tile_XXXXXX__minx_miny_maxx_maxy.parquet`` where each
    coordinate is encoded as an ``int_decimal`` pair (e.g. ``-97_123`` →
    ``-97.123``).
    """
    try:
        coord = fname.replace(".parquet", "").split("__")[1].split("_")
    except IndexError:
        return None
    nums: List[float] = []
    pair: List[str] = []
    for p in coord:
        pair.append(p)
        if len(pair) == 2:
            try:
                nums.append(float(pair[0] + "." + pair[1]))
            except ValueError:
                return None
            pair = []
    if len(nums) != 4:
        return None
    return (nums[0], nums[1], nums[2], nums[3])


def bbox_intersects(a: BBox, b: BBox) -> bool:
    """Whether two ``(minx, miny, maxx, maxy)`` boxes overlap."""
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


class ParquetIndex:
    """Spatial index over GeoParquet tiles with partition-level caching.

    Filename bounding boxes are parsed once at construction.  Decoded
    partitions are kept in a bounded LRU cache (``partition_cache_size``) so
    that panning/zooming within one region does not re-read the same file.
    """

    def __init__(self, folder: Path, partition_cache_size: int = 4) -> None:
        self.folder = Path(folder)
        self._entries: List[Tuple[Path, BBox]] = []
        if self.folder.exists():
            for pf in sorted(self.folder.glob("*.parquet")):
                bbox = parse_parquet_bbox(pf.name)
                if bbox is not None:
                    self._entries.append((pf, bbox))
        self._partition_cache_size = partition_cache_size
        # key -> (native GeoDataFrame, cached per-geometry bounds DataFrame)
        self._partition_cache: "OrderedDict[str, Tuple[gpd.GeoDataFrame, object]]" = OrderedDict()

    # Kept for backward compatibility with callers that used the static helper.
    parse_parquet_bbox = staticmethod(parse_parquet_bbox)

    def find_intersecting_files(self, bbox_4326: BBox) -> List[Path]:
        """Partitions whose filename bbox overlaps ``bbox_4326`` (WGS84)."""
        return [pf for pf, pbbox in self._entries if bbox_intersects(pbbox, bbox_4326)]

    def _read_native(self, path: Path):
        """Read a partition in its native CRS (defaulting to EPSG:4326).

        Returns ``(gdf, bounds_df)`` where ``bounds_df`` is the per-geometry
        envelope used for spatial pre-filtering.  Results are LRU-cached.
        """
        if self._partition_cache_size <= 0:
            gdf = gpd.read_parquet(path)
            if gdf.crs is None:
                gdf = gdf.set_crs(4326)
            return gdf, gdf.geometry.bounds

        key = str(path)
        cached = self._partition_cache.get(key)
        if cached is not None:
            self._partition_cache.move_to_end(key)
            return cached

        gdf = gpd.read_parquet(path)
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        entry = (gdf, gdf.geometry.bounds)
        self._partition_cache[key] = entry
        self._partition_cache.move_to_end(key)
        while len(self._partition_cache) > self._partition_cache_size:
            self._partition_cache.popitem(last=False)
        return entry

    def load_and_reproject(self, path: Path, bbox_4326: Optional[BBox] = None) -> gpd.GeoDataFrame:
        """Load a partition in EPSG:3857, optionally pre-filtered to a tile.

        When ``bbox_4326`` is given, geometries whose envelope does not overlap
        the tile bbox are dropped *before* reprojection, so only the handful of
        geometries inside the tile are ever reprojected and repaired.  The
        result is identical to filtering after reprojection (the downstream
        clip is exact); it is only much cheaper.
        """
        gdf, bounds = self._read_native(path)
        if bbox_4326 is not None and len(gdf):
            minx, miny, maxx, maxy = bbox_4326
            mask = ~(
                (bounds["maxx"] < minx)
                | (bounds["minx"] > maxx)
                | (bounds["maxy"] < miny)
                | (bounds["miny"] > maxy)
            )
            gdf = gdf.loc[mask]
        if len(gdf) and gdf.crs is not None and gdf.crs.to_epsg() != 3857:
            gdf = gdf.to_crs(3857)
        return gdf
