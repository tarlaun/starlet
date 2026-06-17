#!/usr/bin/env python3
"""One-shot: spatial feature query over the densest region, plus a few nested
box sizes around it, to get meaningful queryable-store latencies."""
import glob, os, signal, socket, subprocess, sys, time, urllib.request
from starlet._internal.server.tiler.tiler_bounds import TileBounds

ROOT = sys.argv[1]            # results/starlet_v2/osm_parks_100
PARENT = os.path.dirname(ROOT)
DS = os.path.basename(ROOT)
PORT = 5084

# densest z7 tile = densest region
z7 = sorted(glob.glob(f"{ROOT}/mvt/7/*/*.mvt"), key=os.path.getsize, reverse=True)
p = z7[0].split("/"); X, Y = int(p[-2]), int(p[-1][:-4])
print(f"densest z7 tile: {X}/{Y}  ({os.path.getsize(z7[0])/1e6:.1f} MB MVT)")

proc = subprocess.Popen(["venv/bin/starlet", "serve", "--dir", PARENT, "--port", str(PORT),
                         "--host", "127.0.0.1"], stdout=open("/tmp/q.log", "w"),
                        stderr=subprocess.STDOUT, preexec_fn=os.setsid)
for _ in range(80):
    try:
        socket.create_connection(("127.0.0.1", PORT), 1).close(); break
    except OSError: time.sleep(0.3)

def q(mbr):
    t = time.perf_counter()
    d = urllib.request.urlopen(
        f"http://127.0.0.1:{PORT}/datasets/{DS}/features.csv?mbr={mbr}", timeout=900).read()
    return (time.perf_counter() - t) * 1000, max(0, d.count(b"\n") - 1), len(d)

try:
    print(f"{'box':>8} {'deg':>8} {'lat_ms':>9} {'features':>9} {'resp_KB':>9}")
    # nested boxes centered on the dense tile: its z7 bbox, and z9/z11/z13 sub-boxes (centre child)
    for zoom, lab in [(7, "~2.8°"), (9, "~0.7°"), (11, "~0.18°"), (13, "~0.04°")]:
        f = 2 ** (zoom - 7)
        cx, cy = X * f + f // 2, Y * f + f // 2
        b = TileBounds(zoom, cx, cy)
        mbr = ",".join(f"{v:.6f}" for v in b.bbox_4326)
        lat, n, sz = q(mbr)
        print(f"{lab:>8} {('z'+str(zoom)):>8} {lat:>9.0f} {n:>9} {sz/1024:>9.1f}")
finally:
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
