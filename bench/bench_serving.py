#!/usr/bin/env python3
"""Serving-latency benchmark for the Starlet tile server (corrected).

Fixes the methodology of the earlier latency scripts, which requested
``/tiles/{z}/{x}/{y}.mvt`` — but the server route is
``/<dataset>/<z>/<x>/<y>.mvt``, so every request hit a non-existent dataset and
measured the empty-tile fast path (15-byte responses).  This script:

  * launches ``starlet serve --dir <parent>`` (parent of the dataset dir);
  * requests the correct ``/<dataset>/{z}/{x}/{y}.mvt`` URL;
  * measures three regimes:
      - ``warm_disk``  : pyramid tiles already on disk (z within the pyramid);
      - ``warm_mem``   : immediate re-request (LRU hit);
      - ``otf``        : on-the-fly generation at zooms BEYOND the pyramid;
  * **guards against silently measuring empty tiles**: it tracks response sizes
    and fails loudly if a regime returns only 15-byte (empty) MVTs.

Usage:
    python bench/bench_serving.py --dir datasets --dataset riverside_vegetation_types \
        --pyramid-zooms 4,6 --otf-zooms 10,13 --samples 30 --port 5099
"""
from __future__ import annotations

import argparse
import os
import random
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def wait_for_port(host: str, port: int, timeout_s: float = 60.0) -> bool:
    end = time.time() + timeout_s
    while time.time() < end:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.3)
    return False


def spawn_server(starlet_bin: str, parent_dir: Path, port: int, log: Path, cache: int):
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [starlet_bin, "serve", "--dir", str(parent_dir), "--port", str(port),
           "--host", "127.0.0.1", "--cache-size", str(cache)]
    fh = open(log, "w")
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    if not wait_for_port("127.0.0.1", port):
        proc.terminate(); fh.close()
        raise RuntimeError(f"server did not bind :{port} (see {log})")
    return proc


def kill_server(proc) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def hit(host: str, port: int, dataset: str, z: int, x: int, y: int, timeout_s: float = 300.0):
    url = f"http://{host}:{port}/{dataset}/{z}/{x}/{y}.mvt"
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            data = resp.read(); status = resp.status
    except urllib.error.HTTPError as e:
        return (time.perf_counter() - t0) * 1000, e.code, 0
    except (urllib.error.URLError, TimeoutError, socket.timeout):
        return (time.perf_counter() - t0) * 1000, -1, 0
    return (time.perf_counter() - t0) * 1000, status, len(data)


def populated_tiles(root: Path, zoom: int):
    d = root / "mvt" / str(zoom)
    out = []
    if d.is_dir():
        for xd in d.iterdir():
            if xd.is_dir() and xd.name.isdigit():
                for f in xd.glob("*.mvt"):
                    if f.stem.isdigit():
                        out.append((zoom, int(xd.name), int(f.stem)))
    return out


def deepest_zoom(root: Path):
    d = root / "mvt"
    zs = [int(p.name) for p in d.iterdir() if p.is_dir() and p.name.isdigit()] if d.is_dir() else []
    return max(zs) if zs else None


def otf_coords(root: Path, otf_z: int, n: int, rng: random.Random):
    dz = deepest_zoom(root)
    if dz is None or otf_z <= dz:
        return []
    base = populated_tiles(root, dz)
    rng.shuffle(base)
    f = 2 ** (otf_z - dz)
    return [(otf_z, x * f + f // 2, y * f + f // 2) for (_, x, y) in base[:n]]


def summarize(label: str, lats, sizes):
    if not lats:
        print(f"  {label:<22} (no samples)")
        return
    pct = lambda q: sorted(lats)[min(len(lats) - 1, int(q * len(lats)))]
    nonempty = sum(1 for s in sizes if s > 15)
    print(f"  {label:<22} n={len(lats):<3} "
          f"p50={statistics.median(lats):8.2f}ms  p95={pct(0.95):8.2f}ms  "
          f"max={max(lats):8.2f}ms  bytes[min/med/max]={min(sizes)}/{int(statistics.median(sizes))}/{max(sizes)}  "
          f"nonempty={nonempty}/{len(sizes)}")
    return nonempty


def run_regime(host, port, dataset, coords, repeat_for_mem):
    lats, sizes = [], []
    for (z, x, y) in coords:
        lat, st, nb = hit(host, port, dataset, z, x, y)
        lats.append(lat); sizes.append(nb)
    mem_lats, mem_sizes = [], []
    if repeat_for_mem:
        for (z, x, y) in coords:
            lat, st, nb = hit(host, port, dataset, z, x, y)
            mem_lats.append(lat); mem_sizes.append(nb)
    return (lats, sizes), (mem_lats, mem_sizes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="PARENT dir containing the dataset dir")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--starlet-bin", default="starlet")
    ap.add_argument("--pyramid-zooms", default="4,6")
    ap.add_argument("--otf-zooms", default="10,13")
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--port", type=int, default=5099)
    ap.add_argument("--cache-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--log", default="/tmp/bench_serving.log")
    args = ap.parse_args()

    parent = Path(args.dir)
    root = parent / args.dataset
    rng = random.Random(args.seed)
    host = "127.0.0.1"
    failures = 0

    print(f"dataset={args.dataset}  parent={parent}  deepest_pyramid_zoom={deepest_zoom(root)}")

    proc = spawn_server(args.starlet_bin, parent, args.port, Path(args.log), args.cache_size)
    try:
        # sanity: confirm the dataset is actually served (not 'dataset-not-found')
        _, _, nb = hit(host, args.port, args.dataset,
                       *(populated_tiles(root, deepest_zoom(root))[0]))
        if nb <= 15:
            print("!! WARNING: a known-populated pyramid tile served empty — check --dir/--dataset")
            failures += 1

        print("\n== PYRAMID zooms (warm_disk + warm_mem) ==")
        for z in [int(s) for s in args.pyramid_zooms.split(",") if s.strip()]:
            pool = populated_tiles(root, z); rng.shuffle(pool)
            coords = pool[: args.samples]
            (disk_l, disk_s), (mem_l, mem_s) = run_regime(host, args.port, args.dataset, coords, True)
            ne = summarize(f"z{z} warm_disk", disk_l, disk_s)
            summarize(f"z{z} warm_mem", mem_l, mem_s)
            if coords and ne == 0:
                failures += 1

        print("\n== ON-THE-FLY zooms beyond pyramid (otf cold-ish + warm_mem) ==")
        for z in [int(s) for s in args.otf_zooms.split(",") if s.strip()]:
            coords = otf_coords(root, z, args.samples, rng)
            (otf_l, otf_s), (mem_l, mem_s) = run_regime(host, args.port, args.dataset, coords, True)
            ne = summarize(f"z{z} otf", otf_l, otf_s)
            summarize(f"z{z} warm_mem", mem_l, mem_s)
            if coords and ne == 0:
                print(f"  !! z{z} otf returned only empty tiles")
                failures += 1
    finally:
        kill_server(proc)

    print(f"\nDONE. regime_failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
