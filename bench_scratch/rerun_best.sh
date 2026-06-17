#!/usr/bin/env bash
# Scalability benchmark of the BEST version (v0.2.3-fast @ 4c0c3db:
# bounded-memory histogram + read-time pruning + low-IPC parallel MVT render
# + make_valid drop + integral image). 4 subsets, smallest first, per-phase
# (tile / mvt) timing + max RSS via Rohan's bench_starlet.py harness.
set -u
BASE=/local_data/scratch/tbaha001/starlet_bench
VENV=/home/tbaha001/starlet_bench/venv
PARQ=$BASE/datasets/parquet
OUT=$BASE/results/best
CSV=$BASE/results/starlet_bench_best.csv
LOG=$BASE/logs/best
mkdir -p "$OUT" "$LOG"
stamp(){ date +%F_%H:%M:%S; }

echo "[$(stamp)] ===== rerun_best START (version $(cd /home/tbaha001/starlet_bench/starlet && git rev-parse --short HEAD)) ====="
for pct in 25 50 75 100; do
  subset="$PARQ/osm_parks_${pct}.parquet"
  run="$OUT/osm_parks_${pct}"
  echo "[$(stamp)] === osm_parks_${pct} : tile+mvt START ==="
  "$VENV/bin/python" "$BASE/bench_starlet.py" \
    --input "$subset" \
    --label "osm_parks_${pct}_best" \
    --outdir "$run" \
    --csv "$CSV" \
    --starlet-bin "$VENV/bin/starlet" \
    --clean-outdir \
    >> "$LOG/osm_parks_${pct}.log" 2>&1
  echo "[$(stamp)] === osm_parks_${pct} : DONE rc=$? ==="
  rm -rf "$run/parquet_tiles" "$run/mvt" 2>/dev/null
done
echo "[$(stamp)] ===== rerun_best ALL DONE ====="
