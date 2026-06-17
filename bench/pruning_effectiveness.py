#!/usr/bin/env python3
"""Measure read-time pruning: what fraction of a partition the on-demand server
actually reads per tile, by zoom. Uses the v2 (covering-column + row-group)
tiles. Read-only.

Usage: python bench/pruning_effectiveness.py <dataset_dir> [maxzoom]
Prints CSV to stdout.
"""
import os, sys, random, statistics as st
import pyarrow.parquet as pq
import pyarrow.dataset as pds
import pyarrow.compute as pc
from starlet._internal.server.tiler.tiler_bounds import TileBounds
from starlet._internal.server.tiler.parquet_index import ParquetIndex

DS = sys.argv[1]
MAXZ = int(sys.argv[2]) if len(sys.argv) > 2 else 15
pdir = os.path.join(DS, "parquet_tiles")
idx = ParquetIndex(pdir, partition_cache_size=0)
rng = random.Random(7)


def populated(zoom):
    d = os.path.join(DS, "mvt", str(zoom)); out = []
    if os.path.isdir(d):
        for xd in os.listdir(d):
            xp = os.path.join(d, xd)
            if xd.isdigit() and os.path.isdir(xp):
                out += [(zoom, int(xd), int(f[:-4])) for f in os.listdir(xp) if f.endswith(".mvt")]
    return out


def deepest():
    d = os.path.join(DS, "mvt")
    return max(int(p) for p in os.listdir(d) if p.isdigit())


DZ = deepest()
_meta = {}
def fmeta(pf):
    s = str(pf)
    if s not in _meta:
        p = pq.ParquetFile(pf)
        rg_bytes = [p.metadata.row_group(i).total_byte_size for i in range(p.num_row_groups)]
        _meta[s] = (p.num_row_groups, p.metadata.num_rows, rg_bytes)
    return _meta[s]


def coords(z, n):
    base = populated(DZ); rng.shuffle(base); f = 2 ** (z - DZ)
    return [(z, x * f + f // 2, y * f + f // 2) for _, x, y in base[:n]]


print("zoom,n,mean_partitions,rg_total,rg_read,rg_read_frac,bytes_read_frac,rows_total,rows_read,rows_read_frac")
for z in range(DZ + 1, MAXZ + 1):
    rows = []
    for (zz, x, y) in coords(z, 30):
        b = TileBounds(zz, x, y)
        inter = idx.find_intersecting_files(b.bbox_4326)
        if not inter:
            continue
        minx, miny, maxx, maxy = b.bbox_4326
        flt = ((pc.field("_bbox_xmax") >= minx) & (pc.field("_bbox_xmin") <= maxx)
               & (pc.field("_bbox_ymax") >= miny) & (pc.field("_bbox_ymin") <= maxy))
        rgt = rgr = bt = br = rwt = rwr = 0
        for pf in inter:
            tot_rg, tot_rows, rg_bytes = fmeta(pf)
            frag = list(pds.dataset(pf, format="parquet").get_fragments())[0]
            matched = frag.split_by_row_group(flt)
            rd_idx = sorted({rg.id for m in matched for rg in m.row_groups})
            rgt += tot_rg; rgr += len(rd_idx)
            bt += sum(rg_bytes); br += sum(rg_bytes[i] for i in rd_idx)
            rwt += tot_rows
            rwr += pq.read_table(pf, filters=flt, columns=["_bbox_xmin"]).num_rows
        rows.append((len(inter), rgt, rgr, bt, br, rwt, rwr))
    if not rows:
        continue
    m = lambda i: st.mean(r[i] for r in rows)
    print(f"{z},{len(rows)},{m(0):.1f},{m(1):.1f},{m(2):.1f},{m(2)/m(1):.3f},"
          f"{m(4)/m(3):.3f},{m(5):.0f},{m(6):.0f},{m(6)/m(5):.5f}")
