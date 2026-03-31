import logging
from pathlib import Path

import mapbox_vector_tile
from shapely import affinity, make_valid
from shapely.geometry import box, mapping

from .helpers import EXTENT, mercator_tile_bounds, explode_geom

logger = logging.getLogger(__name__)


class TileRenderer:
    def __init__(self, outdir):
        logger.info("Initializing TileRenderer with outdir=%s", outdir)
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

    def render(self, buckets):
        logger.info("Starting tile rendering for %d zoom levels", len(buckets))
        total = 0

        for z, tiles in buckets.items():
            logger.info("Rendering zoom %s: %d tiles", z, len(tiles))

            zoom_dir = self.outdir / str(z)
            zoom_dir.mkdir(parents=True, exist_ok=True)

            for (x, y), geoms in tiles.items():
                if not geoms:
                    continue

                tb_minx, tb_miny, tb_maxx, tb_maxy = mercator_tile_bounds(z, x, y)
                tb_width = tb_maxx - tb_minx
                tb_height = tb_maxy - tb_miny

                # Match previous behavior
                pad = tb_width * (256 / EXTENT)
                padded = box(
                    tb_minx - pad,
                    tb_miny - pad,
                    tb_maxx + pad,
                    tb_maxy + pad,
                )

                simplify_tol = tb_width * 0.0005

                # Affine transform from Mercator tile bounds -> tile coordinate space
                # x' = a*x + xoff
                # y' = e*y + yoff
                x_scale = EXTENT / tb_width if tb_width != 0 else 0.0
                y_scale = EXTENT / tb_height if tb_height != 0 else 0.0
                affine_params = (
                    x_scale,           # a
                    0.0,               # b
                    0.0,               # d
                    y_scale,           # e
                    -tb_minx * x_scale,  # xoff
                    -tb_miny * y_scale,  # yoff
                )

                features = []

                for g, attrs in geoms:
                    if g is None or g.is_empty:
                        continue

                    # Simplify first to reduce later work
                    try:
                        g2 = g.simplify(simplify_tol, preserve_topology=True)
                    except Exception:
                        logger.debug(
                            "Simplify failed for tile z=%s x=%s y=%s; using original geometry",
                            z, x, y,
                            exc_info=True,
                        )
                        g2 = g

                    if g2.is_empty:
                        continue

                    # Only repair invalid geometry when needed
                    if not g2.is_valid:
                        try:
                            g2 = make_valid(g2)
                        except Exception:
                            try:
                                g2 = g2.buffer(0)
                            except Exception:
                                logger.debug(
                                    "Geometry repair failed for tile z=%s x=%s y=%s",
                                    z, x, y,
                                    exc_info=True,
                                )
                                continue

                    try:
                        clipped = g2.intersection(padded)
                    except Exception:
                        try:
                            repaired = g2.buffer(0)
                            clipped = repaired.intersection(padded)
                        except Exception:
                            logger.error(
                                "Error clipping geometry at z=%s, x=%s, y=%s",
                                z, x, y,
                                exc_info=True,
                            )
                            continue

                    if clipped.is_empty:
                        continue

                    if not clipped.is_valid:
                        try:
                            clipped = make_valid(clipped)
                        except Exception:
                            try:
                                clipped = clipped.buffer(0)
                            except Exception:
                                logger.debug(
                                    "Clipped geometry repair failed at z=%s x=%s y=%s",
                                    z, x, y,
                                    exc_info=True,
                                )
                                continue

                    properties = {k: v for k, v in attrs.items() if v is not None}

                    for part in explode_geom(clipped):
                        if part.is_empty:
                            continue
                        try:
                            transformed = affinity.affine_transform(part, affine_params)
                        except Exception:
                            logger.debug(
                                "Transform failed at z=%s x=%s y=%s",
                                z, x, y,
                                exc_info=True,
                            )
                            continue

                        if transformed.is_empty:
                            continue

                        features.append(
                            {
                                "geometry": mapping(transformed),
                                "properties": properties,
                            }
                        )

                if not features:
                    logger.debug("No features for tile z=%s x=%s y=%s, skipping", z, x, y)
                    continue

                layer = {
                    "name": "layer0",
                    "features": features,
                    "extent": EXTENT,
                }
                data = mapbox_vector_tile.encode([layer])

                x_dir = zoom_dir / str(x)
                x_dir.mkdir(parents=True, exist_ok=True)

                tile_file = x_dir / f"{y}.mvt"
                with open(tile_file, "wb") as f:
                    f.write(data)

                total += 1
                logger.debug(
                    "Wrote tile z=%s x=%s y=%s with %d features",
                    z, x, y, len(features),
                )

        logger.info("Rendering complete. Wrote %d MVT tiles to %s", total, self.outdir)