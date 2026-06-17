# Starlet SIGSPATIAL short-paper scalability handoff

**Audience:** the agent / engineer who picks this up next.

**Goal in one paragraph.** Tarlan is lead author on a SIGSPATIAL 2025 short paper
(4-page hard cap including refs) about *Starlet*, a Python library that turns
GeoParquet/GeoJSON into Mapbox Vector Tiles (MVT) and serves them over a Flask
HTTP layer. The paper currently positions Starlet against BEAST / Tippecanoe /
PostGIS+tile-server stacks. **The remaining work is to fold a multi-gigabyte
scalability + serving-interactivity study into §Evaluation, replace the old
laptop-only numbers, and update the figures and tables.** Authors: Tarlan
Bahadori (UCR, lead), Rohan Bennur (UCR), Shaolin Xie (USC), Ibrahim Sabek (USC),
Ahmed Eldawy (UCR, advisor). LaTeX project sits at
`/Users/tarlan/Downloads/Starlet_sigspatial_short/`; the file that compiles
cleanly to 4 pages is `main-fixed.tex`.

User profile / collaboration style (from memory `user_role.md`,
`feedback_*.md`, `latex_environment.md`, `starlet_artifacts.md`,
`scalability_campaign.md` under
`/Users/tarlan/.claude/projects/-Users-tarlan-Downloads-Starlet-sigspatial-short/memory/`)
— read those before doing anything substantive.

---

## 1. State of the experiments (as of the handoff)

### 1.1 Parks pipeline — COMPLETE

Ran overnight on `ec-hn.cs.ucr.edu` (16 c, 125 GB, NFS `/home`, world-writable
local RAID `/local_data/scratch`). 6 h 12 m wall, 23:29 → 05:41. CSVs are
mirrored to the laptop at `bench/results/`.

The starlet pipeline CSV `bench/results/starlet_bench.csv` has 8 rows = 4
subsets (25/50/75/100 %) × 2 phases (tile, mvt). Headline table extracted by
`bench/analyze.py`:

| Subset | Input | Features | Tile (s) | MVT (s) | **Total (s)** | Tile RSS | MVT RSS | #MVT | MVT MB |
|--------|-------|----------|---------:|--------:|--------------:|---------:|---------:|-----:|-------:|
| 25 % | 1.50 GB | 2.49 M | 844 | 1 776 | **2 620** | 9.2 GB | 6.4 GB | 2 451 | 352 |
| 50 % | 2.70 GB | 4.98 M | 1 277 | 3 089 | **4 366** | 11.6 GB | 8.9 GB | 2 759 | 433 |
| 75 % | 3.61 GB | 7.47 M | 1 893 | 4 273 | **6 165** | 13.6 GB | 9.8 GB | 3 340 | 475 |
| 100 % | 5.17 GB | 9.96 M | 2 651 | 6 044 | **8 696** | 16.2 GB | 14.3 GB | 4 389 | 736 |

- **Linear scaling.** MVT phase dominates (~70 % of total).
- **Peak RSS stays below the 24 GB report claim**, even at 100 %.
- **ec-hn is ~5× slower than Rohan's laptop on the tile phase** (single-thread
  Xeon vs M-series), **~1.6× slower on MVT** (parallelizes to ~5 cores).

Scaling plot already rendered at `figs/scaling_parks.png` (matplotlib, 300 dpi).

### 1.2 Tippecanoe baseline — COMPLETE

`bench/results/baselines.csv` (1 row): tippecanoe on the full 13.5 GB GeoJSON,
z=0..7, 8 threads. **235 s wall, 1 GB RSS, 24.5 MB .mbtiles output.** Starlet
MVT at 100 % is 6 044 s — **Tippecanoe is 25× faster on pure batch encoding**.
This is real and embarrassing. The paper has to either (a) acknowledge it
honestly while reframing Starlet's value as integration + serving, or (b) drop
the baseline. **Tarlan has NOT decided yet — see §3.**

### 1.3 Lazy latency — PARTIAL

`bench/results/latency_summary.csv` has 60 rows = 4 subsets × 3 regimes ×
5 zooms. Numbers look uniform across subsets:

| regime | p50 (ms) | what it measures |
|---|---:|---|
| cold | ~6 | histogram prefix-sum says "empty", 15-byte response |
| warm_disk | ~2 | filesystem read of empty 15-byte file |
| warm_mem | ~1.7 | LRU hit on empty body |

**These are the prefix-sum *prune* fast path, NOT the render path.** Random
(x, y) sampling at z ∈ {4, 7, 10, 13} over the global Web-Mercator grid almost
never hits a park-bearing region, so the sampler is measuring "how fast does
Starlet say there's nothing here." Verifiable: bytes column in `latency_long.csv`
is 15 for every sample (= empty MVT magic + version varint).

A **biased latency rerun** is running RIGHT NOW (see §2).

### 1.4 Output compactness — PENDING (Phase 5)

The TIGER-rails 13.83 → 6.69 MB (52 %) PMTiles comparison from the prior session
is already in `main-fixed.tex`. We have NOT measured loose-MVT vs MBTiles vs
PMTiles on the new parks subsets. Feature branches
`feature/mbtiles-support` and `feature/pmtiles-support` on
`https://github.com/rohanbennur43/starlet` have working exporters; the prior
session ran them successfully on TIGER2018_COUNTY. A modest follow-up could
rerun on osm_parks_100 (~30 min) but it's not blocking — Tarlan may decide the
TIGER number suffices for §Output compactness.

---

## 2. What is running RIGHT NOW

A tmux session **`latbiased` on ec-hn** is doing:

1. `starlet mvt --dir results/starlet/osm_parks_100` — regenerating the eager
   MVT pyramid that the broken `bench_lazy_latency.py` had earlier wiped
   (cold regime did `rm -rf mvt/`, then random sampling failed to repopulate
   because all samples hit empty tiles). ETA: ~100 min from launch at
   **05:50 ec-hn local time**.
2. `python bench_lazy_latency_biased.py --dataset results/starlet/osm_parks_100
   --label osm_parks_100 ...` — biased benchmark that **samples only from the
   on-disk MVT pyramid**, i.e. tiles known to contain features. ETA: ~10 min.

Expected total finish around **07:30 ec-hn**. Output goes to
`/local_data/scratch/tbaha001/starlet_bench/results/latency_{long,summary}_biased.csv`.

To check status:

```bash
ssh ec-hn 'tmux ls; tail -20 /local_data/scratch/tbaha001/starlet_bench/logs/mvt_regen_100.log; tail -20 /local_data/scratch/tbaha001/starlet_bench/logs/latency_biased_100.log'
```

When done, pull to laptop:

```bash
scp ec-hn:/local_data/scratch/tbaha001/starlet_bench/results/latency_{long,summary}_biased.csv \
    /Users/tarlan/Downloads/Starlet_sigspatial_short/bench/results/
```

Then rerun `python3 bench/analyze.py` — the script already knows about the
biased CSV and will print the biased table when it's present.

---

## 3. Open decisions for Tarlan (he asked to clarify, no answer yet)

1. **Tippecanoe framing**: Starlet 25× slower than Tippecanoe at 100 % MVT.
   Choices: include with honest narrative ("Starlet trades speed for Python
   integration + serving"), drop the baseline entirely, or put it in
   Limitations as a side reference. **Don't make this call without Tarlan.**
2. **What to do during the 2-hour biased rerun**: write paper edits now with a
   placeholder for the latency cell, or wait until everything is real before
   editing. Tarlan was asked, didn't answer, asked to clarify the questions
   first.

When you re-engage Tarlan, ask the clarifying questions HE wanted to ask
before you reissue the AskUserQuestion. Don't just re-fire the same options.

---

## 4. Concrete next steps once latency biased data lands

1. `scp` biased CSVs to laptop (commands above).
2. `python3 bench/analyze.py` — regenerates `summary_tables.md` and
   `figs/scaling_parks.png`.
3. **Build the latency CDF figure**: x = latency (log ms), y = CDF, three
   curves (cold / warm_disk / warm_mem) per subset, faceted by subset OR just
   showing 100 %. Use `bench/results/latency_long_biased.csv`. Save to
   `figs/latency_cdf.png`.
4. **Edit `main-fixed.tex` §Evaluation**:
   - Replace Figure 2 (current OSM-Parks pgfplots) with the new
     `figs/scaling_parks.png` or rebuild as native pgfplots from the CSV
     (recommended — keeps the paper self-contained).
   - Replace Table 1 (sort × RG) with the **new headline table**: 4 rows
     ×{subset, input, tiling, MVT, total, cold p50, pre-gen p50, mem p50}.
     Tarlan explicitly asked for this column layout — see his "I want all the
     columns reported here in rohan's report but also for the interactivity
     of serving the tiles (pre-generatred as well as on-the-fly)" message.
   - Update the abstract headline numbers: replace "6.2 GB in 19.7 min" with
     the new ec-hn 5.17 GB / 145 min number, and add a one-sentence
     interactivity claim (e.g., "warm-cache tile fetches in <2 ms p50").
   - Hardware paragraph: now ec-hn (16 c / 125 GB, Linux x86_64), NOT a
     laptop. Mention this.
5. Compile locally with the polyfill that `latex_environment.md` describes
   (`\providecommand\IfDocumentMetadataT` + stub `hyperxmp.sty`). Confirm
   exactly 4 pages.

---

## 5. File map

### On the laptop

```
/Users/tarlan/Downloads/Starlet_sigspatial_short/
├── main-fixed.tex            # compiles to 4 pages clean
├── main-fixed.pdf            # 487 KB, 4 pages
├── ref.bib                   # 39 entries
├── Starlet_Report.pdf        # Rohan's June 2026 internal report (source of laptop numbers)
├── figs/
│   ├── density-world.png     # existing figure
│   ├── starlet_overview.png  # 4-stage pipeline diagram (2730×1530, 300 dpi)
│   ├── make_overview.py      # regenerator for overview
│   └── scaling_parks.png     # NEW: scaling plot from analyze.py
└── bench/
    ├── HANDOFF.md            # ← you are here
    ├── analyze.py            # CSV → table + scaling plot
    ├── bench_starlet.py      # phase-split runner (tile + mvt + RSS)
    ├── bench_lazy_latency.py # ORIGINAL — random sampling, EMPTY tiles only
    ├── bench_lazy_latency_biased.py # NEW — samples from on-disk mvt/
    ├── bench_baselines.py    # tippecanoe runner
    ├── convert_geojson_gz.py # streaming GeoJSON.gz → Parquet
    ├── rename_geom_col.py    # wkb_geometry → geometry rewrite
    ├── run_parks_pipeline.sh # master script (use this as a template for new datasets)
    └── results/
        ├── starlet_bench.csv      # 8 rows: 4 subsets × 2 phases
        ├── baselines.csv          # tippecanoe @ 100 %
        ├── latency_long.csv       # 750 rows (empty-tile)
        ├── latency_summary.csv    # 60 rows (empty-tile, by subset/regime/z)
        └── summary_tables.md      # output of analyze.py
```

Memory files at
`/Users/tarlan/.claude/projects/-Users-tarlan-Downloads-Starlet-sigspatial-short/memory/`:
`MEMORY.md`, `user_role.md`, `starlet_artifacts.md`, `latex_environment.md`,
`scalability_campaign.md`. Read `scalability_campaign.md` first — it has the
operational lessons.

### On ec-hn

```
/local_data/scratch/tbaha001/starlet_bench/   (the live workdir — 1.8 TB local RAID)
├── bench_*.py, *.sh                 # copies of the laptop scripts
├── venv -> ~/starlet_bench/venv      # Python 3.12, Starlet 0.2.3 + deps
├── starlet -> ~/starlet_bench/starlet  # master @ c96d9b2 (2026-05-25)
├── baselines/tippecanoe -> ~/starlet_bench/baselines/tippecanoe  # built binary
├── datasets/parquet/
│   ├── osm_parks_full.parquet   # 5.17 GB, 9.96 M rows, col: $1, $2, geometry
│   ├── osm_parks_25.parquet     # 1.50 GB
│   ├── osm_parks_50.parquet     # 2.70 GB
│   ├── osm_parks_75.parquet     # 3.61 GB
│   ├── osm_parks_100.parquet -> osm_parks_full.parquet
│   └── osm_parks_meta.json
├── results/
│   ├── starlet/osm_parks_{25,50,75,100}/   # each has parquet_tiles/, histograms/, stats/
│   │                                       # mvt/ EXISTS only for 100 right now (regenerating)
│   ├── starlet/osm_parks_100/mvt/          # regenerating in tmux latbiased
│   ├── starlet_bench.csv, baselines.csv, latency_*.csv
│   └── tippecanoe/osm_parks_100/tiles.mbtiles  # 24.5 MB
└── logs/
    ├── pipeline_master.log, pipeline.log, parks_convert.log
    ├── starlet_parks_*.log, latency_parks_*.log, tippecanoe_parks_100.log
    └── mvt_regen_100.log, latency_biased_100.log  # for the running tmux session
```

NFS `~/starlet_bench/` still has the raw .gz and the decompressed 13.5 GB
plain `.geojson` (used by tippecanoe). Don't delete those.

---

## 6. Gotchas (these all bit me — please don't repeat)

1. **Starlet's `WriterPool.append` hardcodes column name `geometry`.** The
   `--geom-col` CLI flag is wired to the assigner only, not the writer pool.
   Pyogrio-converted GeoJSON emits `wkb_geometry`. **Always run
   `rename_geom_col.py --input X --in-place` on every parquet before feeding
   Starlet.** Confirm with
   `python -c "import pyarrow.parquet as pq; print(pq.ParquetFile('X').schema_arrow.names)"`.

2. **GDAL's GeoJSON driver does a full-file schema scan on open.** For 3 GB
   `/vsigzip/`, this takes 10+ min and *no batches are yielded until it's done*.
   Decompress to plain `.geojson` first (the converter handles this when given
   a `.gz` input).

3. **NFS `/home` is fine for reads but writes feel slow.** Local-disk
   `/local_data/scratch/<user>/` is world-writable, 1.8 TB. Tested: Starlet
   on local disk is NOT faster than NFS (12 min vs 11.3 min for 25 % tile).
   ec-hn's Xeons are the actual bottleneck. **Use local-disk anyway** to avoid
   NFS-induced flakiness.

4. **Detach background jobs via `tmux new-session -d -s <name>`** — survives
   SSH disconnect, laptop close, ControlMaster reset. `nohup`+`disown`+`setsid`
   together are NOT reliable on this cluster (master died mid-pipeline once).
   Always check `tmux ls` and `tmux capture-pane -t <name> -p` for status.

5. **`pgrep -f "<pattern>"` matches the SSH session's argv** if the pattern
   string appears in the surrounding command. Use `lsof -t -- <file>` instead
   when checking whether a curl is still writing to a file.

6. **`bench_lazy_latency.py` (original) is broken** when samples don't hit
   real data: the cold regime wipes `mvt/`, and warm_disk's "hit each tile
   once" doesn't repopulate empty tiles. Use
   `bench_lazy_latency_biased.py` for any future latency benchmarks. The
   biased script takes a snapshot of `mvt/` before cold and restores it for
   warm_disk.

7. **UCR-STAR returns HTTP 404 for HEAD requests** but GET works. Range
   requests are ignored. URL pattern is
   `https://star.cs.ucr.edu/datasets/<NAME>/features.geojson.gz`. We have the
   parks dataset already downloaded at
   `~/starlet_bench/datasets/raw/parks.geojson.gz` (3.1 GB compressed) and
   decompressed (13.5 GB). Roads + NYCTaxi were canceled mid-download per
   Tarlan's "just parks now".

8. **Don't reintroduce BEAST as a baseline.** Tarlan was explicit: "starlet
   is different, beast is not a baseline right now". Planetiler is also out
   (needs OSM PBF input, not GeoJSON). **Tippecanoe is the only competitor.**

9. **The 4-page hard cap.** Adding new figures means dropping old ones.
   Current `main-fixed.tex` has 2 pgfplots figures + 1 booktabs table +
   1 density-grid PNG. Trade-off plan in `scalability_campaign.md`: drop
   Table 1 (sort × RG, null result), replace Fig 2(b) writer-pool with the
   new latency CDF. Don't add anything else without explicit OK from Tarlan.

10. **LaTeX compile gotcha (local only).** `latex_environment.md` documents
    the polyfills: `\providecommand{\IfDocumentMetadataT}[1]{}` before
    `\documentclass` and a stub `hyperxmp.sty` in `/tmp/stubs/`. Overleaf
    doesn't need either. Don't put those polyfills into the file Tarlan
    submits.

11. **Don't run experiments on the laptop without explicit ask.** Tarlan
    wants ec-hn experiments because Starlet is single-process; the laptop
    runs are reference numbers from Rohan's report. We do NOT have laptop-
    reproduction data and Tarlan didn't ask for it.

---

## 7. Resumable command snippets

**Check live status of ec-hn pipeline:**
```bash
ssh ec-hn 'tmux ls
tail -15 /local_data/scratch/tbaha001/starlet_bench/logs/pipeline.log
wc -l /local_data/scratch/tbaha001/starlet_bench/results/*.csv'
```

**Pull all CSVs to laptop:**
```bash
scp ec-hn:/local_data/scratch/tbaha001/starlet_bench/results/*.csv \
    /Users/tarlan/Downloads/Starlet_sigspatial_short/bench/results/
```

**Regenerate the table + scaling plot:**
```bash
cd /Users/tarlan/Downloads/Starlet_sigspatial_short/bench && python3 analyze.py
```

**Relaunch a Starlet pipeline run on a different dataset:** see
`run_parks_pipeline.sh` for the template — copy it, adjust the four `RAW / PARQ
/ OUT / LOG` paths at the top, then:
```bash
ssh ec-hn 'cd /local_data/scratch/tbaha001/starlet_bench && \
  tmux new-session -d -s starlet "bash run_<dataset>_pipeline.sh > logs/master.log 2>&1; sleep 86400"'
```

**Compile main-fixed.tex locally:**
```bash
cd /Users/tarlan/Downloads/Starlet_sigspatial_short
TEXINPUTS=/tmp/stubs:: pdflatex -interaction=nonstopmode main-fixed.tex
bibtex main-fixed
TEXINPUTS=/tmp/stubs:: pdflatex main-fixed.tex
TEXINPUTS=/tmp/stubs:: pdflatex main-fixed.tex
```

**Generate the overview figure:**
```bash
cd /Users/tarlan/Downloads/Starlet_sigspatial_short/figs && python3 make_overview.py
```

---

## 8. Hardware reference

- **ec-hn.cs.ucr.edu** — 16 c, 125 GB RAM, Linux x86_64 (Rocky / RHEL 8 era),
  Java 11/17/21 available under `/local_data/dblab/pkgs/`, system Python 3.12
  at `/usr/bin/python3.12`. No GDAL, no Tippecanoe, no Planetiler — we
  installed/built them under `~/starlet_bench/`. Spark 3.2 is on the user
  PATH (for the prior BEAST work) but Starlet doesn't need it.
- **Tarlan's laptop** — Apple M-series 10 c, 32 GB DRAM (used by Rohan's
  report).

---

## 9. What "done" looks like

A reviewer reading the final `main-fixed.pdf` should see:
- A scalability claim that holds up to 5+ GB on a server-class machine.
- A phase-split table mirroring Rohan's report layout, extended with the
  cold / pre-gen / memory latency columns Tarlan requested.
- A 4-stage pipeline overview figure (already done — `figs/starlet_overview.png`).
- A multi-panel evaluation figure with: scaling plot, latency CDF, and
  possibly an output-format comparison.
- Limitations paragraph acknowledging the Tippecanoe gap if Tarlan goes for
  the honest framing.

Open work to reach that state: §3 decisions + §4 next steps. Estimated 2–4 h
of writing after the biased latency CSV lands.
