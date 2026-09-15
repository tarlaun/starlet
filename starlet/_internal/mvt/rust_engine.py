"""Optional Rust acceleration for tile generation (the ``starlet_core`` extension).

``starlet_core`` (built from ``rust/starlet-core``) generates Mapbox Vector
Tiles directly from a dataset's ``parquet_tiles/`` with starlet's selection
semantics, releasing the GIL and using all cores for batches. It is entirely
optional: when the extension is not installed every call site falls back to
the pure-Python implementation.

Selection is controlled by the ``STARLET_ENGINE`` environment variable:

``auto`` (default)
    Use Rust for on-the-fly single tiles when available; batch pyramid
    generation stays on the Python map/reduce unless explicitly requested.
``rust``
    Use Rust for both on-the-fly tiles and batch pyramid generation.
``python``
    Never use the extension (kill switch).
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Optional, Sequence

ENV_VAR = "STARLET_ENGINE"

_lock = threading.Lock()
_core: Any = None
_import_failed = False
_datasets: dict[str, Any] = {}


def _load() -> Any:
    """Import ``starlet_core`` once; remember a failure so we don't retry."""
    global _core, _import_failed
    if _core is not None or _import_failed:
        return _core
    with _lock:
        if _core is None and not _import_failed:
            try:
                import starlet_core  # type: ignore

                _core = starlet_core
            except Exception:  # ImportError or a broken build
                _import_failed = True
    return _core


def engine_setting() -> str:
    return os.environ.get(ENV_VAR, "auto").strip().lower() or "auto"


def available() -> bool:
    """Rust may be used for on-the-fly tiles."""
    return engine_setting() != "python" and _load() is not None


def batch_enabled() -> bool:
    """Rust is requested for batch pyramid generation (opt-in via ``STARLET_ENGINE=rust``)."""
    return engine_setting() == "rust" and _load() is not None


def version() -> Optional[str]:
    core = _load()
    return getattr(core, "__version__", None) if core is not None else None


def num_threads() -> int:
    core = _load()
    return int(core.num_threads()) if core is not None else 0


def dataset(path: str | os.PathLike, *, rg_cache: int = 256) -> Any:
    """One ``starlet_core.Dataset`` per dataset directory (footers cached, row-group LRU)."""
    core = _load()
    if core is None:
        raise RuntimeError("starlet_core is not available")
    key = str(Path(path).resolve())
    with _lock:
        ds = _datasets.get(key)
        if ds is None:
            ds = core.Dataset(key, rg_cache=rg_cache)
            _datasets[key] = ds
        return ds


def invalidate(path: str | os.PathLike | None = None) -> None:
    """Drop cached dataset handles (after a dataset is rebuilt or deleted)."""
    with _lock:
        if path is None:
            _datasets.clear()
        else:
            _datasets.pop(str(Path(path).resolve()), None)


def generate_tile(
    path: str | os.PathLike,
    z: int,
    x: int,
    y: int,
    *,
    feature_capacity: int,
    extent: int,
    buffer: int,
    tile_attributes: list[str] | None = None,
    simplify_tolerance: float | None = None,
) -> bytes:
    return dataset(path).generate_tile(
        int(z), int(x), int(y),
        feature_capacity=int(feature_capacity), extent=int(extent), buffer=int(buffer),
        tile_attributes=tile_attributes, simplify_tolerance=simplify_tolerance,
    )


def generate_tiles(
    path: str | os.PathLike,
    tiles: Sequence[tuple[int, int, int]],
    *,
    feature_capacity: int,
    extent: int,
    buffer: int,
    tile_attributes: list[str] | None = None,
) -> list[bytes]:
    """Generate many tiles in parallel (GIL released; rayon over all cores)."""
    return dataset(path).generate_tiles(
        [(int(z), int(x), int(y)) for z, x, y in tiles],
        feature_capacity=int(feature_capacity), extent=int(extent), buffer=int(buffer),
        tile_attributes=tile_attributes,
    )


def query(path: str | os.PathLike, mbr: tuple[float, float, float, float], *, limit: int = 50) -> list[dict]:
    """Records intersecting ``mbr`` (lon/lat) with their attributes — the
    "click on a record" lookup, exact after bbox pruning."""
    minx, miny, maxx, maxy = mbr
    return dataset(path).query(float(minx), float(miny), float(maxx), float(maxy), limit=int(limit))
