#!/usr/bin/env bash
# Re-do Rohan's scalability benchmark on ec-hn for two versions, matched same-machine:
#   orig = c96d9b2 (v0.2.3, Rohan's original, no covering bbox)
#   fast = v0.2.3-fast (ours: bounded-memory histogram, covering-bbox OFF by default)
# Interleaved per subset, smallest first, per-phase (tile/mvt) timing via bench_starlet.py.
set -u
BASE=/local_data/scratch/tbaha001/starlet_bench
REPO=/home/tbaha001/starlet_bench/starlet
VENV=/home/tbaha001/starlet_bench/venv
PARQ=$BASE/datasets/parquet
OUT=$BASE/results/v3
CSV=$BASE/results/starlet_bench_v3.csv
LOG=$BASE/logs/v3
mkdir -p "$OUT" "$LOG"

stamp(){ date +%F_%H:%M:%S; }

run_one(){
  local pct="$1" ver="$2" ref="$3"
  local subset="$PARQ/osm_parks_${pct}.parquet"
  local run="$OUT/osm_parks_${pct}_${ver}"
  echo "[$(stamp)] === osm_parks_${pct} [$ver @ $ref] : checkout ==="
  ( cd "$REPO" && git checkout "$ref" >/dev/null 2>&1 ) || { echo "[$(stamp)] checkout $ref FAILED"; return 1; }
  echo "[$(stamp)] === osm_parks_${pct} [$ver] : tile+mvt START ==="
  "$VENV/bin/python" "$BASE/bench_starlet.py" \
    --input "$subset" \
    --label "osm_parks_${pct}_${ver}" \
    --outdir "$run" \
    --csv "$CSV" \
    --starlet-bin "$VENV/bin/starlet" \
    --clean-outdir \
    >> "$LOG/osm_parks_${pct}_${ver}.log" 2>&1
  local rc=$?
  echo "[$(stamp)] === osm_parks_${pct} [$ver] : DONE rc=$rc ==="
  # reclaim disk: keep logs+histograms+stats, drop bulky tile/mvt payloads
  rm -rf "$run/parquet_tiles" "$run/mvt" 2>/dev/null
}

echo "[$(stamp)] ===== rerun_v3 START ====="
for pct in 25 50 75 100; do
  run_one "$pct" orig c96d9b2
  run_one "$pct" fast v0.2.3-fast
done
# restore ec-hn checkout to the fast branch when finished
( cd "$REPO" && git checkout v0.2.3-fast >/dev/null 2>&1 )
echo "[$(stamp)] ===== rerun_v3 ALL DONE ====="
