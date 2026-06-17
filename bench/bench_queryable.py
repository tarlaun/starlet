#!/usr/bin/env python3
"""Queryable-store latency: the partitioned GeoParquet that backs the tiles also
answers spatial feature queries + attribute stats over HTTP — a capability a
batch .mbtiles producer (tippecanoe) does not have.

Measures:
  * GET /api/datasets/<ds>/stats              (attribute statistics)
  * GET /datasets/<ds>/features.csv?mbr=...   (spatial feature download) at
    several bbox scales, reporting latency, #features, response size.

Usage: python bench/bench_queryable.py --dir <parent> --dataset <ds> --starlet-bin <bin> --port N
"""
import argparse, os, signal, socket, statistics, subprocess, sys, time
import urllib.request, urllib.error
from pathlib import Path
from starlet._internal.server.tiler.tiler_bounds import TileBounds


def wait_port(host, port, t=60):
    end = time.time() + t
    while time.time() < end:
        try:
            with socket.create_connection((host, port), 1):
                return True
        except OSError:
            time.sleep(0.3)
    return False


def spawn(binp, parent, port, log):
    fh = open(log, "w")
    p = subprocess.Popen([binp, "serve", "--dir", str(parent), "--port", str(port),
                          "--host", "127.0.0.1"], stdout=fh, stderr=subprocess.STDOUT,
                         preexec_fn=os.setsid)
    if not wait_port("127.0.0.1", port):
        raise RuntimeError("server did not start")
    return p


def kill(p):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM); p.wait(timeout=5)
    except Exception:
        pass


def get(url, timeout=300):
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = r.read(); st = r.status
    except urllib.error.HTTPError as e:
        return (time.perf_counter()-t0)*1000, e.code, b""
    except Exception:
        return (time.perf_counter()-t0)*1000, -1, b""
    return (time.perf_counter()-t0)*1000, st, data


def populated(root, z):
    d = root / "mvt" / str(z); out = []
    if d.is_dir():
        for xd in d.iterdir():
            if xd.is_dir() and xd.name.isdigit():
                out += [(z, int(xd.name), int(f.stem)) for f in xd.glob("*.mvt")]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--dataset", required=True)
    ap.add_argument("--starlet-bin", default="starlet"); ap.add_argument("--port", type=int, default=5095)
    a = ap.parse_args()
    parent = Path(a.dir); root = parent / a.dataset
    host = "127.0.0.1"; base = f"http://{host}:{a.port}"
    p = spawn(a.starlet_bin, parent, a.port, Path("/tmp/bench_queryable.log"))
    try:
        # --- attribute stats ---
        lats = [get(f"{base}/api/datasets/{a.dataset}/stats")[0] for _ in range(7)]
        _, st, data = get(f"{base}/api/datasets/{a.dataset}/stats")
        print(f"STATS  /api/datasets/{a.dataset}/stats  p50={statistics.median(lats):.1f}ms  "
              f"status={st}  bytes={len(data)}")

        # --- spatial feature queries at several box scales ---
        print("\nSPATIAL FEATURE QUERIES (features.csv?mbr=...)")
        print(f"{'scale':>8} {'box_deg':>9} {'p50_ms':>10} {'features':>10} {'resp_KB':>9}")
        import random
        rng = random.Random(3)
        for z, label in [(5, "~11°"), (7, "~2.8°"), (10, "~0.35°"), (13, "~0.04°")]:
            pool = populated(root, min(z, 7))
            if not pool:
                continue
            rng.shuffle(pool)
            # descend to z if beyond pyramid, else use the tile itself
            picks = []
            for (_, x, y) in pool[:4]:
                f = 2 ** (z - min(z, 7))
                picks.append((z, x * f, y * f))
            lats, nfeat, sz = [], [], []
            for (zz, x, y) in picks:
                b = TileBounds(zz, x, y)
                mbr = ",".join(f"{v:.6f}" for v in b.bbox_4326)
                lat, stt, data = get(f"{base}/datasets/{a.dataset}/features.csv?mbr={mbr}", timeout=300)
                if stt != 200:
                    continue
                n = max(0, data.count(b"\n") - 1)
                lats.append(lat); nfeat.append(n); sz.append(len(data))
            if lats:
                print(f"{label:>8} {('z'+str(z)):>9} {statistics.median(lats):>10.1f} "
                      f"{int(statistics.median(nfeat)):>10} {statistics.median(sz)/1024:>9.1f}")
    finally:
        kill(p)


if __name__ == "__main__":
    sys.exit(main())
