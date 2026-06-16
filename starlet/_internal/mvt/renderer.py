import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mapbox_vector_tile
from shapely.geometry import box, mapping
from shapely import make_valid
from shapely.ops import transform

from .helpers import EXTENT, mercator_tile_bounds, explode_geom

logger = logging.getLogger(__name__)


def _render_one_tile(task):
    """Render one tile's bucket of ``(geom, attrs)`` to an ``.mvt`` file.

    Self-contained and module-level so it can run in a worker process.
    Returns 1 if a tile was written, 0 if it produced no features.
    """
    z, x, y, geoms, outdir = task
    tb = mercator_tile_bounds(z, x, y)
    width = tb[2] - tb[0]
    height = tb[3] - tb[1]
    pad = width * (256 / EXTENT)
    padded = box(tb[0] - pad, tb[1] - pad, tb[2] + pad, tb[3] + pad)
    simplify_tol = width * 0.0005

    def to_tile(xx, yy, zz=None):
        return (xx - tb[0]) / width * EXTENT, (yy - tb[1]) / height * EXTENT

    features = []
    for (g, attrs) in geoms:
        g2 = g.simplify(simplify_tol, preserve_topology=True)
        if g2.is_empty:
            continue
        # simplify can self-intersect, so repair here; the clip below is an
        # intersection of two valid geometries and therefore stays valid (GEOS
        # OverlayNG), so no second make_valid is needed afterwards.
        g2 = make_valid(g2)

        try:
            clipped = g2.intersection(padded)
        except Exception as e:
            logger.error(f"Error clipping geometry at z={z}, x={x}, y={y}: {e}")
            clipped = g2.buffer(0).intersection(padded)

        if clipped.is_empty:
            continue

        properties = {k: v for k, v in attrs.items() if v is not None}
        for part in explode_geom(clipped):
            transformed = transform(to_tile, part)
            features.append({"geometry": mapping(transformed), "properties": properties})

    if not features:
        return 0

    layer = {"name": "layer0", "features": features, "extent": EXTENT}
    data = mapbox_vector_tile.encode([layer])
    tile_dir = Path(outdir) / str(z) / str(x)
    tile_dir.mkdir(parents=True, exist_ok=True)
    (tile_dir / f"{y}.mvt").write_bytes(data)
    return 1


class TileRenderer:
    def __init__(self, outdir, max_workers=None):
        logger.info(f"Initializing TileRenderer with outdir={outdir}")
        self.outdir = Path(outdir)
        if max_workers is None:
            max_workers = max(1, multiprocessing.cpu_count() - 1)
        self.max_workers = int(max_workers)

    def render(self, buckets):
        # One independent render task per nonempty tile (each writes its own
        # .mvt), so the work parallelizes cleanly across tiles.
        tasks = [
            (z, x, y, geoms, str(self.outdir))
            for z, tiles in buckets.items()
            for (x, y), geoms in tiles.items()
        ]
        logger.info(f"Rendering {len(tasks)} tiles with up to {self.max_workers} workers")
        total = 0

        if self.max_workers <= 1 or len(tasks) <= 1:
            for t in tasks:
                total += _render_one_tile(t)
        else:
            workers = min(self.max_workers, len(tasks))
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for r in ex.map(_render_one_tile, tasks, chunksize=4):
                    total += r

        logger.info(f"Rendering complete. Wrote {total} MVT tiles to {self.outdir}")
