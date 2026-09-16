"""Push-mode pass for the low zoom levels of the pure-Python pyramid.

A low-zoom tile touches most of the dataset, so generating such tiles one
by one (pull mode) makes every worker decode everything and starves the
pool. This pass instead streams each row group **once**, in parallel:

1. **select** — a worker classifies the rows of a row group against every
   requested tile they intersect (:func:`fast_tile.classify`) and reduces
   them to that tile's partial winners (:func:`fast_tile.select_from`). The
   parent merges the partials per tile with the same function — a partial's
   dots stay dots, its full features compete once more — which is exact.
2. **render** — a worker decodes each winning row once from its (cached)
   row group, projects it to each tile it won, simplifies / clips and
   returns the encoded geometry (and attributes); the parent frames the
   tiles, packing the dots, and writes them.

Only 20-byte references and encoded winners cross process boundaries; the
row groups themselves are read at most twice and never spilled.
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import shapely

from starlet._internal.mvt import fast_tile
from starlet._internal.mvt.fast_tile import TileFrame, classify, row_group_cache, select_from, tile_geometries
from starlet._internal.mvt.intermediate_tile import normalize_simplify_tolerance, normalize_tile_attributes
from starlet._internal.mvt.mvt_encoder import (
    GEOM_LINE,
    GEOM_POLYGON,
    LayerEncoder,
    geometry_commands,
    packed_segment_dots,
    packed_square_dots,
)
from starlet._internal.server.tiler.parquet_index import ParquetIndex, bbox_intersects

Tile = tuple[int, int, int]


def _row_groups(index: ParquetIndex) -> list[tuple[Path, int, tuple]]:
    """``(path, row group, partition bbox)`` in file order (= offer order)."""
    import pyarrow.parquet as pq

    out = []
    for path, bbox in index._entries:
        n = pq.ParquetFile(path).metadata.num_row_groups
        for rg in range(n):
            out.append((path, rg, bbox))
    return out


# ---------------------------------------------------------------------------
# pass 1: select
# ---------------------------------------------------------------------------

def _select_chunk(dataset_dir: str, rg_ids: list[int], rgs: list[tuple[str, int]], tiles: list[Tile], params: dict):
    """Partial winners of ``rgs`` for every tile. Returns
    ``{slot: (rows, rg_id, prio, cellkey, is_dot, cx, cy)}`` arrays."""
    parquet_dir = Path(dataset_dir) / "parquet_tiles"
    index = ParquetIndex(parquet_dir)
    cache = row_group_cache(parquet_dir, index, params["row_group_cache_size"])
    frames = [TileFrame(z, x, y, params["extent"], params["buffer"]) for (z, x, y) in tiles]
    k = params["feature_capacity"]
    pbboxes = {p: bb for p, bb in index._entries}
    out: dict[int, list] = defaultdict(list)
    for rg_id, (path, rg) in zip(rg_ids, rgs):
        path = Path(path)
        pbbox = pbboxes.get(path)
        table, a, b, c, d, crc = cache.row_group(path, rg)
        for slot, fr in enumerate(frames):
            q = fr.query_lonlat
            if pbbox is not None and not bbox_intersects(pbbox, q):
                continue
            hit = (c >= q[0]) & (a <= q[2]) & (d >= q[1]) & (b <= q[3])
            if not hit.any():
                continue
            rows = np.flatnonzero(hit)
            prio, cellkey, small, cx, cy = classify(fr, a[rows], b[rows], c[rows], d[rows], crc[rows])
            full, dots = select_from(prio, cellkey, small, k)
            keep = np.concatenate((full, dots))
            is_dot = np.concatenate((np.zeros(len(full), dtype=bool), np.ones(len(dots), dtype=bool)))
            out[slot].append((rows[keep], np.full(len(keep), rg_id, dtype=np.int64), prio[keep], cellkey[keep], is_dot, cx[keep], cy[keep]))
    return {slot: tuple(np.concatenate(cols) for cols in zip(*parts)) for slot, parts in out.items()}


# ---------------------------------------------------------------------------
# pass 2: render
# ---------------------------------------------------------------------------

def _render_chunk(dataset_dir: str, rg: tuple[str, int], requests: list[tuple], params: dict):
    """For one row group: ``requests`` = ``[(slot, tile, full_rows, dot_rows,
    want_dot_props)]``. Returns ``{slot: (fulls, dots)}`` where ``fulls`` is a
    list of ``(row, geom_bytes, mvt_type, props)`` and ``dots`` is
    ``(rows, kinds, props_by_row)`` (kinds: shapely type ids)."""
    parquet_dir = Path(dataset_dir) / "parquet_tiles"
    index = ParquetIndex(parquet_dir)
    cache = row_group_cache(parquet_dir, index, params["row_group_cache_size"])
    path = Path(rg[0])
    table, *_ = cache.row_group(path, rg[1])
    _pf, geom_col, _has_bbox, _crs, attrs, _stats = cache.file_info(path)
    policy = normalize_tile_attributes(params["tile_attributes"])
    if policy is not None:
        attrs = [n for n in attrs if n in policy]
    tol_cfg = params["simplify_tolerance"]
    out = {}
    for slot, (z, x, y), full_rows, dot_rows, want_dot_props in requests:
        fr = TileFrame(z, x, y, params["extent"], params["buffer"])
        tol = normalize_simplify_tolerance(tol_cfg, fr.cell)
        fulls = []
        if len(full_rows):
            wkb = table[geom_col].take(pa.array(full_rows)).to_pylist()
            geoms = shapely.from_wkb(wkb, on_invalid="ignore")
            tg = tile_geometries(fr, geoms, tol)
            cmds, types = geometry_commands(tg)
            props = None
            if attrs:
                cols = {n: table[n].take(pa.array(full_rows)).to_pylist() for n in attrs}
                props = [[(n, cols[n][j]) for n in attrs] for j in range(len(full_rows))]
            for j, row in enumerate(full_rows.tolist()):
                if cmds[j] is None:
                    continue
                fulls.append((row, cmds[j], int(types[j]), props[j] if props else None))
        dots = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), {})
        if len(dot_rows):
            wkb = table[geom_col].take(pa.array(dot_rows)).to_pylist()
            kinds = shapely.get_type_id(shapely.from_wkb(wkb, on_invalid="ignore"))
            dprops: dict[int, list] = {}
            if want_dot_props and attrs:
                pts = dot_rows[(kinds == 0) | (kinds == 4)]
                if len(pts):
                    cols = {n: table[n].take(pa.array(pts)).to_pylist() for n in attrs}
                    for j, row in enumerate(pts.tolist()):
                        dprops[row] = [(n, cols[n][j]) for n in attrs]
            dots = (dot_rows, kinds, dprops)
        out[slot] = (fulls, dots)
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run_push(dataset_dir: str, outdir: str, tiles: list[Tile], params: dict, executor: ProcessPoolExecutor, workers: int) -> int:
    """Generate ``tiles`` (any zooms) with the two streaming passes; returns
    the number of non-empty tiles written."""
    parquet_dir = Path(dataset_dir) / "parquet_tiles"
    index = ParquetIndex(parquet_dir)
    rgs = [(str(p), rg) for (p, rg, _b) in _row_groups(index)]
    if not rgs or not tiles:
        return 0
    k = int(params["feature_capacity"])

    # ---- pass 1 ----------------------------------------------------------
    per_task = max(1, len(rgs) // (workers * 4))
    futures = []
    for i in range(0, len(rgs), per_task):
        futures.append(executor.submit(_select_chunk, dataset_dir, list(range(i, min(i + per_task, len(rgs)))), rgs[i:i + per_task], tiles, params))
    partials: dict[int, list] = defaultdict(list)
    for f in futures:
        for slot, arrays in f.result().items():
            partials[slot].append(arrays)

    # merge per tile -> winners grouped by row group for pass 2
    requests: dict[int, list] = defaultdict(list)  # rg index -> [(slot, tile, full_rows, dot_rows, want_dot_props)]
    winners: dict[int, dict] = {}
    for slot, parts in partials.items():
        rows, rg_id, prio, cellkey, is_dot, cx, cy = (np.concatenate(cols) for cols in zip(*parts))
        # offer order = (row group, row): make the stable sorts follow it
        order = np.lexsort((rows, rg_id))
        rows, rg_id, prio, cellkey, is_dot, cx, cy = (v[order] for v in (rows, rg_id, prio, cellkey, is_dot, cx, cy))
        full, dots = select_from(prio, cellkey, is_dot, k)
        strip_point_attrs = len(dots) > k
        winners[slot] = dict(full=full, dots=dots, rows=rows, rg=rg_id, cx=cx, cy=cy, strip=strip_point_attrs)
        for g in np.unique(rg_id[np.concatenate((full, dots))]):
            fm = rg_id[full] == g
            dm = rg_id[dots] == g
            requests[int(g)].append((slot, tiles[slot], rows[full[fm]], rows[dots[dm]], not strip_point_attrs))

    # ---- pass 2 ----------------------------------------------------------
    futures = [executor.submit(_render_chunk, dataset_dir, rgs[g], reqs, params) for g, reqs in requests.items()]
    fulls: dict[int, list] = defaultdict(list)
    dot_kinds: dict[int, dict[int, int]] = defaultdict(dict)  # slot -> {(rg,row) key: kind}
    dot_props: dict[int, dict] = defaultdict(dict)
    for g, f in zip(requests.keys(), futures):
        for slot, (fl, (drows, kinds, dprops)) in f.result().items():
            fulls[slot].extend((g, row, gb, t, p) for (row, gb, t, p) in fl)
            dk = dot_kinds[slot]
            for row, kind in zip(drows.tolist(), kinds.tolist()):
                dk[(g, row)] = kind
            for row, p in dprops.items():
                dot_props[slot][(g, row)] = p

    # ---- assemble --------------------------------------------------------
    out = Path(outdir)
    written = 0
    extent = int(params["extent"])
    for slot, w in winners.items():
        z, x, y = tiles[slot]
        fr = TileFrame(z, x, y, extent, params["buffer"])
        layer = LayerEncoder(params.get("layer_name", "layer0"), extent)
        feats = [(g, row, gb, t, p) for (g, row, gb, t, p) in fulls.get(slot, [])]
        # native points among the dots are individual features (kept in offer order)
        lo, hi = -fr.buffer, fr.extent + fr.buffer
        poly_c, line_c = [], []
        for i in w["dots"].tolist():
            key = (int(w["rg"][i]), int(w["rows"][i]))
            kind = dot_kinds[slot].get(key, -1)
            cx, cy = float(w["cx"][i]), float(w["cy"][i])
            if not (lo <= cx <= hi and lo <= cy <= hi):
                continue
            if kind in (3, 6, 7):
                poly_c.append((cx, cy))
            elif kind in (1, 2, 5):
                line_c.append((cx, cy))
            elif kind in (0, 4):
                cmds, types = geometry_commands(np.array([shapely.Point(cx, cy)], dtype=object))
                if cmds[0]:
                    feats.append((key[0], key[1], cmds[0], int(types[0]), None if w["strip"] else dot_props[slot].get(key)))
        feats.sort(key=lambda f: (f[0], f[1]))
        for _g, _row, gb, t, p in feats:
            layer.add(gb, t, p)
        half = fr.cell * 0.5
        sq = packed_square_dots(np.array(poly_c, dtype=np.float64).reshape(-1, 2), half)
        if sq:
            layer.add(sq, GEOM_POLYGON, None)
        seg = packed_segment_dots(np.array(line_c, dtype=np.float64).reshape(-1, 2), half)
        if seg:
            layer.add(seg, GEOM_LINE, None)
        if layer.feature_count == 0:
            continue
        d = out / str(z) / str(x)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{y}.mvt").write_bytes(layer.encode_tile())
        written += 1
    return written
