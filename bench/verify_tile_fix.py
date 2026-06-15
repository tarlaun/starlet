#!/usr/bin/env python3
"""Regression + micro-benchmark for the on-the-fly tiler optimization.

For each sampled tile it builds the MVT two ways using the same data:

  * **brute**     — load the whole intersecting partition, reproject & clip all
                    geometries (the original behaviour);
  * **prefilter** — drop geometries outside the tile bbox before reproject/clip
                    (the optimization).

It asserts the two produce **byte-identical** MVTs (correctness) and reports the
wall time of each (speed-up).  Partition caching is disabled here so both paths
pay their own parquet read — i.e. this isolates the compute saved, not the cache.

Usage:
    python bench/verify_tile_fix.py --dataset datasets/riverside_vegetation_types \
        --pyramid-zooms 4,6 --otf-zooms 10,13 --samples 6
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

from shapely.geometry import mapping

from starlet._internal.server.tiler.tiler_bounds import TileBounds
from starlet._internal.server.tiler.parquet_index import ParquetIndex
from starlet._internal.server.tiler.mvt_encoder import MVTEncoder
from starlet._internal.server.tiler.tiler import explode_collections


def build_tile(index: ParquetIndex, z: int, x: int, y: int, use_bbox: bool) -> bytes:
    b = TileBounds(z, x, y)
    enc = MVTEncoder(b.bbox_3857, b.tile_poly_3857)
    inter = index.find_intersecting_files(b.bbox_4326)
    if not inter:
        return enc.empty_tile()
    feats = []
    for pf in inter:
        gdf = index.load_and_reproject(pf, b.bbox_4326 if use_bbox else None)
        clipped = enc.clip_to_tile(gdf)
        if clipped.empty:
            continue
        for _, row in clipped.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            attrs = {k: v for k, v in row.items() if k != "geometry" and v is not None}
            for part in explode_collections(geom):
                if part.is_empty:
                    continue
                scaled = enc.transform_geom(
                    part,
                    lambda xx, yy, zz=None: TileBounds.scale_to_tile_coords(xx, yy, b.bbox_3857),
                )
                if not scaled.is_empty:
                    feats.append({"geometry": mapping(scaled), "properties": attrs})
    return enc.encode(feats) if feats else enc.empty_tile()


def populated_tiles(root: Path, zoom: int):
    d = root / "mvt" / str(zoom)
    out = []
    if d.is_dir():
        for xdir in d.iterdir():
            if xdir.is_dir() and xdir.name.isdigit():
                for f in xdir.glob("*.mvt"):
                    if f.stem.isdigit():
                        out.append((zoom, int(xdir.name), int(f.stem)))
    return out


def deepest_zoom(root: Path):
    d = root / "mvt"
    zs = [int(p.name) for p in d.iterdir() if p.is_dir() and p.name.isdigit()] if d.is_dir() else []
    return max(zs) if zs else None


def otf_coords(root: Path, otf_z: int, n: int, rng: random.Random):
    """Populated coords at a zoom beyond the pyramid, by descending into a
    populated deepest-pyramid tile (centre child)."""
    dz = deepest_zoom(root)
    if dz is None or otf_z <= dz:
        return []
    base = populated_tiles(root, dz)
    rng.shuffle(base)
    f = 2 ** (otf_z - dz)
    return [(otf_z, x * f + f // 2, y * f + f // 2) for (_, x, y) in base[:n]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dataset dir (has parquet_tiles/ and mvt/)")
    ap.add_argument("--pyramid-zooms", default="4,6")
    ap.add_argument("--otf-zooms", default="10,13")
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    root = Path(args.dataset)
    pdir = root / "parquet_tiles"
    rng = random.Random(args.seed)

    tiles = []
    for z in [int(s) for s in args.pyramid_zooms.split(",") if s.strip()]:
        pool = populated_tiles(root, z)
        rng.shuffle(pool)
        tiles += pool[: args.samples]
    for z in [int(s) for s in args.otf_zooms.split(",") if s.strip()]:
        tiles += otf_coords(root, z, args.samples, rng)

    if not tiles:
        print("no tiles discovered (need an mvt/ pyramid in the dataset)")
        return 2

    print(f"{'tile':>16}  {'bytes':>8}  {'brute_s':>9}  {'prefilt_s':>10}  {'speedup':>7}  match")
    print("-" * 70)
    mismatches = 0
    speedups = []
    for (z, x, y) in tiles:
        idx_old = ParquetIndex(pdir, partition_cache_size=0)
        t0 = time.perf_counter()
        old = build_tile(idx_old, z, x, y, use_bbox=False)
        t_old = time.perf_counter() - t0

        idx_new = ParquetIndex(pdir, partition_cache_size=0)
        t0 = time.perf_counter()
        new = build_tile(idx_new, z, x, y, use_bbox=True)
        t_new = time.perf_counter() - t0

        ok = old == new
        mismatches += (not ok)
        sp = (t_old / t_new) if t_new > 0 else float("inf")
        speedups.append(sp)
        print(f"{z}/{x}/{y:>{16 - len(str(z) + str(x)) - 2}}  {len(new):>8}  "
              f"{t_old:>9.3f}  {t_new:>10.3f}  {sp:>6.1f}x  {'OK' if ok else 'MISMATCH!!'}")

    med_sp = sorted(speedups)[len(speedups) // 2]
    print("-" * 70)
    print(f"tiles={len(tiles)}  mismatches={mismatches}  median_speedup={med_sp:.1f}x")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
