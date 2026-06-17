#!/bin/bash
# Monitor for the v2 campaign: validate the first new-format tile, poll to
# completion, then print all results. Runs on ec-hn.
W=/local_data/scratch/tbaha001/starlet_bench
cd "$W" || exit 1

echo "[monitor] waiting for first new-format tile (osm_parks_25)..."
for i in $(seq 1 120); do
  f=$(ls results/starlet_v2/osm_parks_25/parquet_tiles/*.parquet 2>/dev/null | head -1)
  if [ -n "$f" ]; then
    echo "[monitor] FIRST TILE: $(basename "$f")"
    venv/bin/python - "$f" <<'PY'
import sys, pyarrow.parquet as pq
pf = pq.ParquetFile(sys.argv[1])
names = pf.schema_arrow.names
print("  columns:", names)
print("  has bbox covering cols:", all(c in names for c in
      ("_bbox_xmin","_bbox_ymin","_bbox_xmax","_bbox_ymax")))
print("  rows:", pf.metadata.num_rows, "row_groups:", pf.num_row_groups)
PY
    break
  fi
  sleep 30
done

echo "[monitor] polling to completion..."
for i in $(seq 1 240); do
  [ -f logs/v2_done.flag ] && break
  echo "[monitor $(date +%H:%M)] $(tail -1 logs/v2_campaign.log) | csv_rows=$(wc -l < results/starlet_bench_v2.csv 2>/dev/null)"
  sleep 300
done

echo "===================== V2 CAMPAIGN RESULTS ====================="
echo "----- starlet_bench_v2.csv (tile + mvt) -----"
cat results/starlet_bench_v2.csv 2>/dev/null
echo; echo "----- serving osm_parks_100 -----"; cat logs/v2_serving_100.log 2>/dev/null
echo; echo "----- serving osm_parks_25 -----";  cat logs/v2_serving_25.log 2>/dev/null
echo "[monitor] exit"
