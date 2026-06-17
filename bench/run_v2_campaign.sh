#!/bin/bash
# Scalability + timing campaign for the "robust" (read-time pruning) version.
# Re-tiles the OSM-parks sweep with the new tiling format (bbox covering columns
# + bounded row groups), regenerates MVTs, then measures serving latency.
# Writes to results/starlet_v2/ so the original results/starlet/ stays intact.
set -u
cd /local_data/scratch/tbaha001/starlet_bench
BIN=venv/bin/starlet
PY=venv/bin/python
CSV=results/starlet_bench_v2.csv
mkdir -p logs results/starlet_v2
rm -f $CSV logs/v2_done.flag

echo "[$(date)] V2 campaign start" > logs/v2_campaign.log

# ---- Stage 1+2: tiling + MVT scalability sweep ----
for s in 25 50 75 100; do
  echo "[$(date)] === osm_parks_${s}: tile + mvt ===" >> logs/v2_campaign.log
  $PY bench_starlet.py \
    --input datasets/parquet/osm_parks_${s}.parquet \
    --label osm_parks_${s} \
    --outdir results/starlet_v2/osm_parks_${s} \
    --csv $CSV --clean-outdir --starlet-bin $BIN \
    > logs/v2_bench_${s}.log 2>&1
  echo "[$(date)] osm_parks_${s} done (rc=$?)" >> logs/v2_campaign.log
done

# ---- Stage 3: serving latency (cold on-the-fly is the structural-cure payoff) ----
for s in 100 25; do
  echo "[$(date)] === serving osm_parks_${s} ===" >> logs/v2_campaign.log
  $PY bench/bench_serving.py --dir results/starlet_v2 --dataset osm_parks_${s} \
    --starlet-bin $BIN --pyramid-zooms 4,7 --otf-zooms 8,10,13 --samples 25 \
    --port 53${s} > logs/v2_serving_${s}.log 2>&1
  # remove on-the-fly tiles the server persisted, so the pyramid stays as-built
  MV=results/starlet_v2/osm_parks_${s}/mvt
  for d in $MV/*/; do z=$(basename "$d"); if [ "$z" -gt 7 ] 2>/dev/null; then rm -rf "$d"; fi; done
done

echo "[$(date)] V2 campaign complete" | tee -a logs/v2_campaign.log > logs/v2_done.flag
