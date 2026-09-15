//! On-the-fly single-tile generation over a starlet dataset directory, with
//! starlet's exact selection semantics:
//!
//! * partitions pruned by their filename bbox, row groups by `_bbox_*`
//!   statistics, rows by bbox overlap (WKB bbox when the columns are absent);
//! * features ranked by `crc32(source WKB)` and the top `feature_capacity`
//!   kept (the same geometry-intrinsic priority the batch pipeline uses, so
//!   adjacent on-demand and pre-generated tiles agree on what they keep);
//! * per-feature pipeline in starlet's order: affine to tile units, collapse
//!   shapes under 5.5 px (per dimension) to a point, Douglas-Peucker at 1 px
//!   when a geometry has more than 10 vertices, then clip to extent+buffer.
//!
//! A row-group LRU makes neighbouring tiles (pans, zooms, batch pyramids)
//! cheap: the decoded Arrow batches are shared across tiles.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use arrow::array::AsArray;
use arrow::datatypes::Float64Type;
use arrow::record_batch::RecordBatch;
use lru::LruCache;
use parking_lot::Mutex;

use crate::geom::clip::{clip, Rect};
use crate::geom::simplify::{signed_area, simplify_dp};
use crate::geom::{merc_bbox_to_lonlat, GeomKind, Geometry, TileId, TileTransform};
use crate::mvt::{LayerBuilder, TileWriter};
use crate::pq::{arrow_value, parse_filename_bbox, wkb_at, PqFile, BBOX_COLS, INTERNAL_COLS};

pub const LAYER_NAME: &str = "layer0";
/// starlet's `_SMALL_GEOMETRY_EXTENT_PX`, scaled by `extent / 4096`.
pub const SMALL_GEOMETRY_EXTENT_PX: f64 = 5.5;
/// starlet simplifies (tolerance 1 tile unit) only when a geometry has more than this many coordinates.
pub const SIMPLIFY_MIN_COORDS: usize = 10;
const BATCH_SIZE: usize = 8192;

#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub feature_capacity: usize,
    pub extent: u32,
    pub buffer: u32,
}

pub struct Partition {
    pub file: PqFile,
    /// bbox from the filename, in the partition's native CRS (EPSG:4326).
    pub bbox: [f64; 4],
    pub geom_col: String,
    pub has_bbox_cols: bool,
    /// attribute columns (everything but geometry and starlet's internal columns)
    pub attr_cols: Vec<String>,
}

/// A decoded row group. For partitions without bbox columns the per-row
/// bboxes are computed once here (a zero-allocation WKB scan) and reused by
/// every tile that touches the row group.
pub struct CachedRg {
    pub batches: Vec<RecordBatch>,
    pub bboxes: Option<Vec<Vec<[f64; 4]>>>,
}

const NO_BBOX: [f64; 4] = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];

pub struct Dataset {
    pub parts: Vec<Partition>,
    rg_cache: Mutex<LruCache<(usize, usize), Arc<CachedRg>>>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub partitions_total: usize,
    pub partitions_read: usize,
    pub row_groups_total: usize,
    pub row_groups_read: usize,
    pub candidates: u64,
    pub retained: u64,
}

impl Dataset {
    /// Open `<dir>/parquet_tiles/*.parquet` (footers only; row groups are read lazily).
    pub fn open(dir: &Path, rg_cache_entries: usize) -> Result<Dataset> {
        let pdir = dir.join("parquet_tiles");
        if !pdir.is_dir() {
            return Err(anyhow!("GeoParquet tile directory not found: {}", pdir.display()));
        }
        let mut paths: Vec<_> = std::fs::read_dir(&pdir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|x| x == "parquet").unwrap_or(false))
            .collect();
        paths.sort();
        // Footers are parsed in parallel: this is the one-time cost of opening
        // a dataset (per process), so it matters for the first request.
        use rayon::prelude::*;
        let opened: Vec<Result<Option<Partition>>> = paths
            .par_iter()
            .map(|p| {
                let fname = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
                let Some(bbox) = parse_filename_bbox(fname) else { return Ok(None) };
                let file = PqFile::open(p)?;
                let schema = file.schema();
                let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
                let geom_col = if names.iter().any(|n| n == "geometry") {
                    "geometry".to_string()
                } else {
                    names.last().cloned().context("empty schema")?
                };
                let has_bbox_cols = BBOX_COLS.iter().all(|c| names.iter().any(|n| n == c));
                let attr_cols = names
                    .iter()
                    .filter(|n| **n != geom_col && !INTERNAL_COLS.contains(&n.as_str()))
                    .cloned()
                    .collect();
                Ok(Some(Partition { file, bbox, geom_col, has_bbox_cols, attr_cols }))
            })
            .collect();
        let mut parts = Vec::with_capacity(paths.len());
        for r in opened {
            if let Some(part) = r? {
                parts.push(part);
            }
        }
        Ok(Dataset {
            parts,
            rg_cache: Mutex::new(LruCache::new(NonZeroUsize::new(rg_cache_entries.max(1)).unwrap())),
        })
    }

    fn read_rg_cached(&self, pi: usize, rg: usize, cols: &[&str]) -> Result<Arc<CachedRg>> {
        if let Some(b) = self.rg_cache.lock().get(&(pi, rg)).cloned() {
            return Ok(b);
        }
        let part = &self.parts[pi];
        let batches = part.file.read(vec![rg], cols, BATCH_SIZE)?;
        let bboxes = if part.has_bbox_cols {
            None
        } else {
            let mut all = Vec::with_capacity(batches.len());
            for b in &batches {
                let gi = b.schema().index_of(&part.geom_col)?;
                all.push(
                    (0..b.num_rows())
                        .map(|i| {
                            wkb_at(b, gi, i)
                                .and_then(|w| Geometry::wkb_bbox(w).ok())
                                .map(|bb| bb.arr())
                                .unwrap_or(NO_BBOX)
                        })
                        .collect::<Vec<_>>(),
                )
            }
            Some(all)
        };
        let entry = Arc::new(CachedRg { batches, bboxes });
        self.rg_cache.lock().put((pi, rg), entry.clone());
        Ok(entry)
    }

    /// Generate one tile (MVT bytes) with starlet's semantics.
    pub fn generate(&self, tile: TileId, p: &Params) -> Result<Vec<u8>> {
        Ok(self.generate_with_stats(tile, p)?.0)
    }

    pub fn generate_with_stats(&self, tile: TileId, p: &Params) -> Result<(Vec<u8>, Stats)> {
        let tt = TileTransform::new(tile, p.extent, p.buffer);
        let q_merc = tt.buffered_bounds();
        let q = merc_bbox_to_lonlat(&q_merc);
        let mut stats = Stats { partitions_total: self.parts.len(), ..Default::default() };

        // ---- candidates: top-k by crc32(wkb) over bbox-pruned rows ----------
        let k = p.feature_capacity.max(1);
        let mut batches: Vec<(Arc<CachedRg>, usize, usize)> = Vec::new(); // (row group, batch idx, partition idx)
        // min-heap on (priority, seq); payload = (batch slot, row)
        let mut heap: BinaryHeap<Reverse<(u32, u64, usize, u32)>> = BinaryHeap::with_capacity(k + 1);
        let mut seq: u64 = 0;

        for (pi, part) in self.parts.iter().enumerate() {
            if !bbox_intersects(&part.bbox, &q) {
                continue;
            }
            stats.partitions_read += 1;
            stats.row_groups_total += part.file.num_row_groups();
            let rgs = if part.has_bbox_cols {
                part.file.prune_bbox(&q)
            } else {
                (0..part.file.num_row_groups()).collect()
            };
            stats.row_groups_read += rgs.len();
            let mut cols: Vec<&str> = vec![part.geom_col.as_str()];
            if part.has_bbox_cols {
                cols.extend(BBOX_COLS.iter());
            }
            cols.extend(part.attr_cols.iter().map(|s| s.as_str()));

            for rg in rgs {
                let rgc = self.read_rg_cached(pi, rg, &cols)?;
                for (bi, b) in rgc.batches.iter().enumerate() {
                    let gi = b.schema().index_of(&part.geom_col)?;
                    let bb_cols = if part.has_bbox_cols {
                        Some([
                            b.column(b.schema().index_of(BBOX_COLS[0])?).as_primitive::<Float64Type>().clone(),
                            b.column(b.schema().index_of(BBOX_COLS[1])?).as_primitive::<Float64Type>().clone(),
                            b.column(b.schema().index_of(BBOX_COLS[2])?).as_primitive::<Float64Type>().clone(),
                            b.column(b.schema().index_of(BBOX_COLS[3])?).as_primitive::<Float64Type>().clone(),
                        ])
                    } else {
                        None
                    };
                    let cached_bb = rgc.bboxes.as_ref().map(|v| &v[bi]);
                    let slot = batches.len();
                    let mut used = false;
                    for i in 0..b.num_rows() {
                        let Some(w) = wkb_at(b, gi, i) else { continue };
                        let rb: [f64; 4] = match (&bb_cols, cached_bb) {
                            (Some(c), _) => [c[0].value(i), c[1].value(i), c[2].value(i), c[3].value(i)],
                            (None, Some(bbs)) => bbs[i],
                            (None, None) => match Geometry::wkb_bbox(w) {
                                Ok(bb) => bb.arr(),
                                Err(_) => continue,
                            },
                        };
                        if !bbox_intersects(&rb, &q) {
                            continue;
                        }
                        stats.candidates += 1;
                        let prio = crc32fast::hash(w);
                        if heap.len() >= k {
                            // starlet: skip unless strictly higher than the current minimum
                            if prio <= heap.peek().unwrap().0 .0 {
                                continue;
                            }
                            heap.pop();
                        }
                        heap.push(Reverse((prio, seq, slot, i as u32)));
                        seq += 1;
                        used = true;
                    }
                    if used {
                        batches.push((rgc.clone(), bi, pi));
                    }
                }
            }
        }

        // ---- winners -> tile features ---------------------------------------
        let mut layer = LayerBuilder::new(LAYER_NAME, p.extent);
        let mut tags: Vec<(u32, u32)> = Vec::new();
        let winners: Vec<(usize, u32)> = heap.into_iter().map(|Reverse((_, _, s, r))| (s, r)).collect();
        for (slot, row) in winners {
            let (rgc, bi, pi) = &batches[slot];
            let b = &rgc.batches[*bi];
            let part = &self.parts[*pi];
            let gi = b.schema().index_of(&part.geom_col)?;
            let Some(w) = wkb_at(b, gi, row as usize) else { continue };
            let Ok(mut g) = Geometry::from_wkb(w) else { continue };
            g.from_lonlat_to_merc();
            let Some(tg) = to_tile_geometry(&g, &tt, p) else { continue };
            tags.clear();
            for name in &part.attr_cols {
                if let Ok(ci) = b.schema().index_of(name) {
                    if let Some(v) = arrow_value(b.column(ci), row as usize) {
                        let ki = layer.key(name);
                        let vi = layer.value(&v);
                        tags.push((ki, vi));
                    }
                }
            }
            layer.add_feature(None, &tg, &tags);
        }
        stats.retained = layer.feature_count as u64;
        let mut w = TileWriter::new();
        w.add_layer(&layer);
        Ok((w.finish(), stats))
    }
}

#[inline]
fn bbox_intersects(a: &[f64; 4], b: &[f64; 4]) -> bool {
    !(a[2] < b[0] || a[0] > b[2] || a[3] < b[1] || a[1] > b[3])
}

/// starlet's `simplify_geometry` pipeline on a Mercator geometry.
pub fn to_tile_geometry(g_merc: &Geometry, tt: &TileTransform, p: &Params) -> Option<Geometry> {
    let mut g = g_merc.clone();
    g.map_coords(|x, y| tt.apply(x, y));
    if g.is_empty() {
        return None;
    }
    // collapse small shapes (per dimension) to their centre
    let bb = g.bbox();
    let thr = SMALL_GEOMETRY_EXTENT_PX * (p.extent as f64 / 4096.0);
    if bb.width() <= thr && bb.height() <= thr {
        let c = bb.center();
        return Some(Geometry { kind: GeomKind::Point, parts: vec![vec![c]], polys: vec![] });
    }
    // simplify at 1 tile unit when the geometry is "big enough"
    if g.vertex_count() > SIMPLIFY_MIN_COORDS && g.kind != GeomKind::Point {
        g = simplify_dp(&g, 1.0)?;
    }
    // clip (points are not clipped by starlet)
    if g.kind != GeomKind::Point {
        let b = p.buffer as f64;
        let e = p.extent as f64;
        g = clip(&g, &Rect::new(-b, -b, e + b, e + b))?;
        g = drop_degenerate(g)?;
    }
    if g.is_empty() {
        None
    } else {
        Some(g)
    }
}

/// Minimum |area| (tile units²) for a ring to survive clipping. Sutherland-
/// Hodgman emits zero-area "bridge" polygons when the clip rectangle lies in
/// a concave notch of a polygon that never actually enters it; shapely's
/// `clip_by_rect` (starlet's Python path) returns empty there.
const MIN_RING_AREA: f64 = 0.5;

fn drop_degenerate(g: Geometry) -> Option<Geometry> {
    match g.kind {
        GeomKind::Point => Some(g),
        GeomKind::Line => {
            let parts: Vec<Vec<[f64; 2]>> = g
                .parts
                .into_iter()
                .filter(|p| p.len() >= 2 && p.windows(2).any(|w| w[0] != w[1]))
                .collect();
            if parts.is_empty() {
                None
            } else {
                Some(Geometry { kind: GeomKind::Line, parts, polys: vec![] })
            }
        }
        GeomKind::Polygon => {
            let mut out = Geometry::empty(GeomKind::Polygon);
            for i in 0..g.n_parts() {
                let rings = g.poly_rings(i);
                if rings.is_empty() || signed_area(&rings[0]).abs() < MIN_RING_AREA {
                    continue;
                }
                let start = out.parts.len();
                out.parts.push(rings[0].clone());
                for hole in &rings[1..] {
                    if signed_area(hole).abs() >= MIN_RING_AREA {
                        out.parts.push(hole.clone());
                    }
                }
                out.polys.push(start);
            }
            if out.polys.is_empty() {
                None
            } else {
                Some(out)
            }
        }
    }
}
