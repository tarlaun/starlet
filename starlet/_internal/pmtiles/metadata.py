"""TileJSON-style metadata and header bounds for PMTiles archives.

MapLibre / PMTiles viewers read the archive's metadata JSON (``name``,
``vector_layers`` with field types, ``bounds``, ``center``, zoom range) and
the header bounds to fit the map and to populate style editors. Starlet used
to write an empty metadata object and world bounds; this module derives both
from the dataset directory (``stats/attributes.json`` and the MVT tree).
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

LAYER_NAME = "layer0"
_INTERNAL_COLUMNS = {"_tile_id", "_bbox_xmin", "_bbox_ymin", "_bbox_xmax", "_bbox_ymax"}


def _field_type(stats: dict) -> str:
    """TileJSON field type ("Number", "Boolean", "String") from a stats sketch."""
    if "mean" in stats or ("min" in stats and isinstance(stats.get("min"), (int, float)) and not isinstance(stats.get("min"), bool)):
        return "Number"
    top_k = stats.get("top_k") or []
    values = [entry.get("value") for entry in top_k if isinstance(entry, dict)]
    if values and all(isinstance(v, bool) for v in values):
        return "Boolean"
    return "String"


def layer_fields(
    dataset_dir: str | Path,
    tile_attributes: Any = "all",
    geom_col: str = "geometry",
) -> dict[str, str]:
    """``{attribute: TileJSON type}`` for the attributes written into tiles."""
    stats_path = Path(dataset_dir) / "stats" / "attributes.json"
    if not stats_path.exists():
        return {}
    try:
        attributes = json.loads(stats_path.read_text()).get("attributes", [])
    except Exception:
        logger.debug("Could not read %s for PMTiles metadata", stats_path, exc_info=True)
        return {}

    selected: Optional[set[str]] = None
    if isinstance(tile_attributes, (list, tuple, set)):
        selected = {str(name) for name in tile_attributes}
    elif isinstance(tile_attributes, str) and tile_attributes.strip().lower() == "none":
        return {}

    fields: dict[str, str] = {}
    for attr in attributes:
        name = attr.get("name")
        stats = attr.get("stats") or {}
        if not isinstance(name, str) or name == geom_col or name in _INTERNAL_COLUMNS or "mbr" in stats:
            continue
        if selected is not None and name not in selected:
            continue
        fields[name] = _field_type(stats)
    return fields


def center_zoom_for_bounds(bounds: Sequence[float], min_zoom: int, max_zoom: int) -> int:
    """Lowest zoom at which the bounds fit in about one tile, clamped to the
    archive's zoom range."""
    west, south, east, north = bounds
    span = max(east - west, (north - south) * 2.0, 1e-9)
    zoom = int(math.floor(math.log2(360.0 / span))) if span < 360.0 else 0
    return max(min_zoom, min(max_zoom, zoom))


def pmtiles_export_options(
    dataset_dir: str | Path,
    tile_counts_by_zoom: Iterable[int],
    *,
    tile_attributes: Any = "all",
    geom_col: str = "geometry",
    layer_name: str = LAYER_NAME,
    name: str | None = None,
) -> dict[str, Any]:
    """Keyword arguments for :func:`export_to_pmtiles`: ``metadata`` (TileJSON
    style), ``bounds`` (lon/lat) and ``center`` (lon, lat, zoom)."""
    from starlet._types import Dataset

    dataset_path = Path(dataset_dir)
    zooms = [z for z, count in enumerate(tile_counts_by_zoom) if count > 0]
    min_zoom, max_zoom = (min(zooms), max(zooms)) if zooms else (0, 0)

    bounds: Optional[tuple[float, float, float, float]] = None
    try:
        bounds = Dataset(str(dataset_path)).bbox
    except Exception:
        logger.debug("Dataset bbox unavailable for PMTiles metadata", exc_info=True)
    if bounds is not None:
        west, south, east, north = (float(v) for v in bounds)
        if not all(math.isfinite(v) for v in (west, south, east, north)) or east < west or north < south:
            bounds = None
        else:
            # Clamp to valid lon/lat; the header stores e7 int32 values.
            bounds = (max(-180.0, west), max(-90.0, south), min(180.0, east), min(90.0, north))

    try:
        from starlet import __version__ as starlet_version
    except Exception:  # pragma: no cover - importlib metadata missing in odd envs
        starlet_version = "unknown"

    metadata: dict[str, Any] = {
        "name": name or dataset_path.name,
        "format": "pbf",
        "type": "overlay",
        "version": "1",
        "generator": f"starlet {starlet_version}",
        "minzoom": str(min_zoom),
        "maxzoom": str(max_zoom),
        "vector_layers": [
            {
                "id": layer_name,
                "description": "",
                "minzoom": min_zoom,
                "maxzoom": max_zoom,
                "fields": layer_fields(dataset_path, tile_attributes, geom_col),
            }
        ],
    }
    center = None
    if bounds is not None:
        west, south, east, north = bounds
        center_zoom = center_zoom_for_bounds(bounds, min_zoom, max_zoom)
        center = ((west + east) / 2.0, (south + north) / 2.0, center_zoom)
        metadata["bounds"] = ",".join(f"{v:.6f}" for v in bounds)
        metadata["center"] = f"{center[0]:.6f},{center[1]:.6f},{center_zoom}"
    return {"metadata": metadata, "bounds": bounds, "center": center}
