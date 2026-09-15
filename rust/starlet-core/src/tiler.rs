//! On-the-fly single-tile generation over a starlet dataset directory, with
//! starlet's exact selection semantics:
//!
//! * partitions pruned by their filename bbox, row groups by `_bbox_*`
//!   statistics, rows by bbox overlap (WKB bbox when the columns are absent);
//! * **raster-consistent selection**: every candidate gets a priority of
//!   `(size in tile units, crc32(source WKB))` — bigger first, hash as the
//!   tie-break. The `feature_capacity` highest-priority features larger than
//!   a display pixel (`extent / PIXEL_GRID` tile units per side) are kept in
//!   full; *every other* candidate — sub-pixel ones and the larger ones that
//!   did not make the cut — contributes a one-pixel *dot* to the pixel cell
//!   of its bbox centre, one dot per cell (highest priority). So a tile
//!   shows every occupied pixel, like a rasterised plot, and its biggest
//!   shapes in detail, with a bounded number of features. Priorities are
//!   geometry-intrinsic, so adjacent on-demand and pre-generated tiles agree.
//! * per-feature pipeline: affine to tile units; a sub-pixel polygon becomes
//!   a one-pixel square, a sub-pixel line a one-pixel segment (its geometry
//!   type is preserved, so it is styled like its full-size siblings; such
//!   *dots* carry no attributes — fetch them with `query_point`); larger
//!   shapes get Douglas-Peucker at 1 tile unit when they have more than 10
//!   vertices, then are clipped to extent+buffer.
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
use crate::geom::{lonlat_to_merc, merc_bbox_to_lonlat, GeomKind, Geometry, TileId, TileTransform};
use std::collections::HashMap;
use crate::mvt::{LayerBuilder, TileWriter};
use crate::pq::{arrow_value, parse_filename_bbox, wkb_at, PqFile, BBOX_COLS, INTERNAL_COLS};

pub const LAYER_NAME: &str = "layer0";
/// Display pixels per tile side used for the sub-pixel grid: a feature whose
/// bbox fits in `extent / PIXEL_GRID` tile units (per dimension) is a "dot".
pub const PIXEL_GRID: u32 = 256;
/// starlet simplifies (tolerance 1 tile unit) only when a geometry has more than this many coordinates.
pub const SIMPLIFY_MIN_COORDS: usize = 10;
const BATCH_SIZE: usize = 8192;

/// Which attribute columns non-dot features carry.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum AttrPolicy {
    #[default]
    All,
    None,
    Only(Vec<String>),
}

impl AttrPolicy {
    #[inline]
    pub fn allows(&self, name: &str) -> bool {
        match self {
            AttrPolicy::All => true,
            AttrPolicy::None => false,
            AttrPolicy::Only(cols) => cols.iter().any(|c| c == name),
        }
    }
    /// `None` = all, `Some([])` = none, `Some(cols)` = only those.
    pub fn from_option(v: Option<Vec<String>>) -> AttrPolicy {
        match v {
            None => AttrPolicy::All,
            Some(c) if c.is_empty() => AttrPolicy::None,
            Some(c) => AttrPolicy::Only(c),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Params {
    pub feature_capacity: usize,
    pub extent: u32,
    pub buffer: u32,
    pub attrs: AttrPolicy,
    /// Douglas-Peucker tolerance in tile units; `None` = a quarter of a
    /// display pixel (`cell() / 4`, i.e. 4 units at the default extent).
    pub simplify_tolerance: Option<f64>,
}

impl Params {
    /// Side of one display pixel in tile units.
    #[inline]
    pub fn cell(&self) -> f64 {
        self.extent as f64 / PIXEL_GRID as f64
    }
    #[inline]
    pub fn tolerance(&self) -> f64 {
        self.simplify_tolerance.unwrap_or(self.cell() * 0.25).max(0.0)
    }
}

/// How a candidate relates to the tile's pixel grid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placement {
    /// pixel cell of the bbox centre (negative indices occur in the buffer)
    pub cell: (i32, i32),
    /// bbox fits inside one pixel (per dimension)
    pub small: bool,
    /// `max(width, height)` of the bbox in 1/16 tile units, saturated
    pub size16: u32,
}

/// Classify a candidate by its EPSG:4326 bbox `rb` against a tile.
#[inline]
pub fn place(rb: &[f64; 4], tt: &TileTransform, cell: f64) -> Placement {
    let (ax, ay) = lonlat_to_merc(rb[0], rb[1]);
    let (bx, by) = lonlat_to_merc(rb[2], rb[3]);
    let (x0, y0) = tt.apply(ax, ay);
    let (x1, y1) = tt.apply(bx, by);
    let w = (x1 - x0).abs();
    let h = (y1 - y0).abs();
    let cx = (x0 + x1) * 0.5;
    let cy = (y0 + y1) * 0.5;
    Placement {
        cell: ((cx / cell).floor() as i32, (cy / cell).floor() as i32),
        small: w <= cell && h <= cell,
        size16: (w.max(h) * 16.0).min(u32::MAX as f64 / 2.0) as u32,
    }
}

/// Selection priority: size first, `crc32(source WKB)` as the tie-break.
#[inline]
pub fn priority(size16: u32, crc: u32) -> u64 {
    ((size16 as u64) << 32) | crc as u64
}

/// `dot_cell` compatibility helper: the pixel cell when the feature is
/// sub-pixel, else `None`.
#[inline]
pub fn dot_cell(rb: &[f64; 4], tt: &TileTransform, cell: f64) -> Option<(i32, i32)> {
    let pl = place(rb, tt, cell);
    if pl.small {
        Some(pl.cell)
    } else {
        None
    }
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

        // ---- candidates ------------------------------------------------------
        // The top-k (by size, then crc32) features larger than a display
        // pixel are kept in full; every other candidate becomes a dot in the
        // pixel cell of its bbox centre, one dot per cell (raster-consistent).
        let k = p.feature_capacity.max(1);
        let cell = p.cell();
        let mut batches: Vec<(Arc<CachedRg>, usize, usize)> = Vec::new(); // (row group, batch idx, partition idx)
        // min-heap on (priority, seq); payload = (batch slot, row, cell)
        let mut heap: BinaryHeap<Reverse<(u64, u64, usize, u32, (i32, i32))>> = BinaryHeap::with_capacity(k + 1);
        let mut cells: HashMap<(i32, i32), (u64, u64, usize, u32)> = HashMap::new();
        let mut seq: u64 = 0;
        #[inline]
        fn offer_cell(cells: &mut HashMap<(i32, i32), (u64, u64, usize, u32)>, c: (i32, i32), e: (u64, u64, usize, u32)) {
            match cells.get_mut(&c) {
                Some(cur) if e.0 <= cur.0 => {}
                Some(cur) => *cur = e,
                None => {
                    cells.insert(c, e);
                }
            }
        }

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
                        let pl = place(&rb, &tt, cell);
                        let prio = priority(pl.size16, crc32fast::hash(w));
                        let e = (prio, seq, slot, i as u32);
                        if pl.small {
                            offer_cell(&mut cells, pl.cell, e);
                        } else if heap.len() >= k && prio <= heap.peek().unwrap().0 .0 {
                            // did not make the cut: a dot instead
                            offer_cell(&mut cells, pl.cell, e);
                        } else {
                            if heap.len() >= k {
                                let Reverse((ep, eq, es, er, ec)) = heap.pop().unwrap();
                                offer_cell(&mut cells, ec, (ep, eq, es, er));
                            }
                            heap.push(Reverse((prio, seq, slot, i as u32, pl.cell)));
                        }
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
        // Sub-pixel *points* keep their attributes only while the tile has
        // at most `feature_capacity` sub-pixel features; denser tiles carry
        // them as bare dots (attributes stay reachable through `query`).
        let strip_point_attrs = cells.len() > k;
        // (seq, slot, row, as_dot)
        let mut winners: Vec<(u64, usize, u32, bool)> =
            heap.into_iter().map(|Reverse((_, q, s, r, _))| (q, s, r, false)).collect();
        winners.extend(cells.into_values().map(|(_, q, s, r)| (q, s, r, true)));
        winners.sort_unstable(); // deterministic feature order (offer order)
        let mut dots = DotBuffer::default();
        for (_, slot, row, as_dot) in winners {
            let (rgc, bi, pi) = &batches[slot];
            let b = &rgc.batches[*bi];
            let part = &self.parts[*pi];
            let gi = b.schema().index_of(&part.geom_col)?;
            let Some(w) = wkb_at(b, gi, row as usize) else { continue };
            let Ok(mut g) = Geometry::from_wkb(w) else { continue };
            g.from_lonlat_to_merc();
            let tg = match tile_feature(&g, &tt, p, as_dot) {
                None => continue,
                Some(TileFeature::Dot(c, kind)) => {
                    dots.push(c, kind);
                    continue;
                }
                Some(TileFeature::Full(tg)) => tg,
            };
            tags.clear();
            let bare = strip_point_attrs && tg.kind == GeomKind::Point && sub_pixel(&tg, p);
            if !bare {
                for name in part.attr_cols.iter().filter(|n| p.attrs.allows(n)) {
                    if let Ok(ci) = b.schema().index_of(name) {
                        if let Some(v) = arrow_value(b.column(ci), row as usize) {
                            let ki = layer.key(name);
                            let vi = layer.value(&v);
                            tags.push((ki, vi));
                        }
                    }
                }
            }
            layer.add_feature(None, &tg, &tags);
        }
        dots.emit(&mut layer, p);
        stats.retained = layer.feature_count as u64;
        let mut w = TileWriter::new();
        w.add_layer(&layer);
        Ok((w.finish(), stats))
    }
}

/// One record returned by `Dataset::query`.
pub struct Hit {
    pub bbox: [f64; 4],
    pub kind: GeomKind,
    pub attrs: Vec<(String, crate::mvt::Value)>,
}

impl Dataset {
    /// Records whose geometry intersects the lon/lat box `q` (exact test
    /// after bbox pruning), with their attributes — the "click on a record"
    /// lookup. At most `limit` hits, in file order.
    pub fn query(&self, q: &[f64; 4], limit: usize) -> Result<Vec<Hit>> {
        let mut hits = Vec::new();
        for (pi, part) in self.parts.iter().enumerate() {
            if !bbox_intersects(&part.bbox, q) {
                continue;
            }
            let rgs = if part.has_bbox_cols {
                part.file.prune_bbox(q)
            } else {
                (0..part.file.num_row_groups()).collect()
            };
            let mut cols: Vec<&str> = vec![part.geom_col.as_str()];
            if part.has_bbox_cols {
                cols.extend(BBOX_COLS.iter());
            }
            cols.extend(part.attr_cols.iter().map(|s| s.as_str()));
            for rg in rgs {
                let rgc = self.read_rg_cached(pi, rg, &cols)?;
                for (bi, b) in rgc.batches.iter().enumerate() {
                    let gi = b.schema().index_of(&part.geom_col)?;
                    let cached_bb = rgc.bboxes.as_ref().map(|v| &v[bi]);
                    for i in 0..b.num_rows() {
                        let Some(w) = wkb_at(b, gi, i) else { continue };
                        let rb: [f64; 4] = match cached_bb {
                            Some(bbs) => bbs[i],
                            None => {
                                if part.has_bbox_cols {
                                    [
                                        b.column(b.schema().index_of(BBOX_COLS[0])?).as_primitive::<Float64Type>().value(i),
                                        b.column(b.schema().index_of(BBOX_COLS[1])?).as_primitive::<Float64Type>().value(i),
                                        b.column(b.schema().index_of(BBOX_COLS[2])?).as_primitive::<Float64Type>().value(i),
                                        b.column(b.schema().index_of(BBOX_COLS[3])?).as_primitive::<Float64Type>().value(i),
                                    ]
                                } else {
                                    match Geometry::wkb_bbox(w) {
                                        Ok(bb) => bb.arr(),
                                        Err(_) => continue,
                                    }
                                }
                            }
                        };
                        if !bbox_intersects(&rb, q) {
                            continue;
                        }
                        let Ok(g) = Geometry::from_wkb(w) else { continue };
                        if !geom_intersects_rect(&g, q) {
                            continue;
                        }
                        let mut attrs = Vec::with_capacity(part.attr_cols.len());
                        for name in &part.attr_cols {
                            if let Ok(ci) = b.schema().index_of(name) {
                                if let Some(v) = arrow_value(b.column(ci), i) {
                                    attrs.push((name.clone(), v));
                                }
                            }
                        }
                        hits.push(Hit { bbox: rb, kind: g.kind, attrs });
                        if hits.len() >= limit {
                            return Ok(hits);
                        }
                    }
                }
            }
        }
        Ok(hits)
    }
}

/// Exact geometry ∩ axis-aligned rectangle test in the geometry's own CRS.
fn geom_intersects_rect(g: &Geometry, q: &[f64; 4]) -> bool {
    let inside = |p: &[f64; 2]| p[0] >= q[0] && p[0] <= q[2] && p[1] >= q[1] && p[1] <= q[3];
    let seg_hits = |a: &[f64; 2], b: &[f64; 2]| -> bool {
        // Liang-Barsky
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let mut t0 = 0.0f64;
        let mut t1 = 1.0f64;
        for (pk, qk) in [(-dx, a[0] - q[0]), (dx, q[2] - a[0]), (-dy, a[1] - q[1]), (dy, q[3] - a[1])] {
            if pk == 0.0 {
                if qk < 0.0 {
                    return false;
                }
            } else {
                let t = qk / pk;
                if pk < 0.0 {
                    t0 = t0.max(t);
                } else {
                    t1 = t1.min(t);
                }
                if t0 > t1 {
                    return false;
                }
            }
        }
        true
    };
    match g.kind {
        GeomKind::Point => g.parts.iter().flatten().any(inside),
        GeomKind::Line => g.parts.iter().any(|l| l.windows(2).any(|w| seg_hits(&w[0], &w[1]))),
        GeomKind::Polygon => {
            let centre = [(q[0] + q[2]) * 0.5, (q[1] + q[3]) * 0.5];
            for i in 0..g.n_parts() {
                let rings = g.poly_rings(i);
                if rings.is_empty() {
                    continue;
                }
                // boundary crosses the box, or box centre inside the polygon
                if rings.iter().any(|r| r.windows(2).any(|w| seg_hits(&w[0], &w[1]))) {
                    return true;
                }
                if point_in_ring(&centre, &rings[0]) && !rings[1..].iter().any(|h| point_in_ring(&centre, h)) {
                    return true;
                }
            }
            false
        }
    }
}

/// Even-odd ray casting.
fn point_in_ring(p: &[f64; 2], ring: &[[f64; 2]]) -> bool {
    let mut inside = false;
    let n = ring.len();
    if n < 3 {
        return false;
    }
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (ring[i][0], ring[i][1]);
        let (xj, yj) = (ring[j][0], ring[j][1]);
        if (yi > p[1]) != (yj > p[1]) && p[0] < (xj - xi) * (p[1] - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

#[inline]
fn bbox_intersects(a: &[f64; 4], b: &[f64; 4]) -> bool {
    !(a[2] < b[0] || a[0] > b[2] || a[3] < b[1] || a[1] > b[3])
}

/// Whether a tile-unit geometry fits inside one display pixel.
#[inline]
pub fn sub_pixel(g: &Geometry, p: &Params) -> bool {
    let bb = g.bbox();
    bb.width() <= p.cell() && bb.height() <= p.cell()
}

/// A winner ready for encoding: either its full tile-unit geometry (a native
/// point included) or a polygon / line *dot* at a tile-unit centre.
pub enum TileFeature {
    Full(Geometry),
    Dot([f64; 2], GeomKind),
}

/// Classify a winner: dots (sub-pixel polygons / lines, or demoted ones)
/// are returned as centres so the tile can pack them into one feature.
pub fn tile_feature(g_merc: &Geometry, tt: &TileTransform, p: &Params, as_dot: bool) -> Option<TileFeature> {
    let (tg, is_dot) = to_tile_geometry(g_merc, tt, p, as_dot)?;
    if is_dot {
        let c = tg.bbox().center();
        Some(TileFeature::Dot(c, tg.kind))
    } else {
        Some(TileFeature::Full(tg))
    }
}

/// Polygon and line dots of one tile, packed at encode time into a single
/// MultiPolygon / MultiLineString feature each, sorted by position so the
/// delta-encoded coordinates (and gzip) stay small: ~7 bytes per dot
/// instead of ~17 as separate features.
#[derive(Default)]
pub struct DotBuffer {
    pub polys: Vec<[f64; 2]>,
    pub lines: Vec<[f64; 2]>,
}

impl DotBuffer {
    #[inline]
    pub fn push(&mut self, c: [f64; 2], kind: GeomKind) {
        match kind {
            GeomKind::Polygon => self.polys.push(c),
            GeomKind::Line => self.lines.push(c),
            GeomKind::Point => {}
        }
    }
    pub fn is_empty(&self) -> bool {
        self.polys.is_empty() && self.lines.is_empty()
    }
    fn sort(v: &mut [[f64; 2]]) {
        v.sort_by_key(|c| (c[1].round() as i64, c[0].round() as i64));
    }
    pub fn emit(&mut self, layer: &mut LayerBuilder, p: &Params) {
        let h = p.cell() * 0.5;
        if !self.polys.is_empty() {
            Self::sort(&mut self.polys);
            let mut g = Geometry::empty(GeomKind::Polygon);
            for c in &self.polys {
                g.polys.push(g.parts.len());
                g.parts.push(vec![
                    [c[0] - h, c[1] - h],
                    [c[0] + h, c[1] - h],
                    [c[0] + h, c[1] + h],
                    [c[0] - h, c[1] + h],
                    [c[0] - h, c[1] - h],
                ]);
            }
            layer.add_feature(None, &g, &[]);
        }
        if !self.lines.is_empty() {
            Self::sort(&mut self.lines);
            let mut g = Geometry::empty(GeomKind::Line);
            for c in &self.lines {
                g.parts.push(vec![[c[0] - h, c[1]], [c[0] + h, c[1]]]);
            }
            layer.add_feature(None, &g, &[]);
        }
        self.polys.clear();
        self.lines.clear();
    }
}

/// starlet's `simplify_geometry` pipeline on a Mercator geometry. Returns
/// the tile-unit geometry and whether it was reduced to a one-pixel *dot*
/// (a collapsed polygon or line; dots carry no attributes). `as_dot` forces
/// the dot form (a feature that did not make the top-k).
pub fn to_tile_geometry(g_merc: &Geometry, tt: &TileTransform, p: &Params, as_dot: bool) -> Option<(Geometry, bool)> {
    let mut g = g_merc.clone();
    g.map_coords(|x, y| tt.apply(x, y));
    if g.is_empty() {
        return None;
    }
    // sub-pixel shapes (and demoted larger ones) become a one-pixel shape of the same kind
    let bb = g.bbox();
    let cell = p.cell();
    if as_dot || (bb.width() <= cell && bb.height() <= cell) {
        let c = bb.center();
        let h = cell * 0.5;
        // keep dots inside the buffered tile only (a point is never clipped,
        // but a dot outside the buffer is invisible and just costs bytes)
        let b = p.buffer as f64;
        let e = p.extent as f64;
        if c[0] < -b || c[1] < -b || c[0] > e + b || c[1] > e + b {
            return None;
        }
        return Some(match g.kind {
            GeomKind::Point => (Geometry { kind: GeomKind::Point, parts: vec![vec![c]], polys: vec![] }, false),
            GeomKind::Line => (
                Geometry {
                    kind: GeomKind::Line,
                    parts: vec![vec![[c[0] - h, c[1]], [c[0] + h, c[1]]]],
                    polys: vec![],
                },
                true,
            ),
            GeomKind::Polygon => (
                Geometry {
                    kind: GeomKind::Polygon,
                    parts: vec![vec![
                        [c[0] - h, c[1] - h],
                        [c[0] + h, c[1] - h],
                        [c[0] + h, c[1] + h],
                        [c[0] - h, c[1] + h],
                        [c[0] - h, c[1] - h],
                    ]],
                    polys: vec![0],
                },
                true,
            ),
        });
    }
    // simplify when the geometry is "big enough"
    if g.vertex_count() > SIMPLIFY_MIN_COORDS && g.kind != GeomKind::Point {
        g = simplify_dp(&g, p.tolerance())?;
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
        Some((g, false))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sub_pixel_polygon_becomes_one_pixel_square_without_attributes() {
        let p = Params { feature_capacity: 10, extent: 4096, buffer: 256, attrs: AttrPolicy::All, simplify_tolerance: None };
        let tt = TileTransform::new(TileId::new(0, 0, 0), 4096, 256);
        // a 1 m square near the origin: far below one pixel at z0
        let (x, y) = lonlat_to_merc(10.0, 45.0);
        let sq = Geometry {
            kind: GeomKind::Polygon,
            parts: vec![vec![[x, y], [x + 1.0, y], [x + 1.0, y + 1.0], [x, y + 1.0], [x, y]]],
            polys: vec![0],
        };
        let (g, is_dot) = to_tile_geometry(&sq, &tt, &p, false).unwrap();
        assert!(is_dot);
        assert_eq!(g.kind, GeomKind::Polygon);
        let bb = g.bbox();
        assert!((bb.width() - p.cell()).abs() < 1e-9 && (bb.height() - p.cell()).abs() < 1e-9);
        // its lon/lat bbox lands in a pixel cell; a big polygon does not
        assert!(dot_cell(&[10.0, 45.0, 10.00001, 45.00001], &tt, p.cell()).is_some());
        assert!(dot_cell(&[-10.0, 30.0, 40.0, 60.0], &tt, p.cell()).is_none());
    }

    #[test]
    fn native_points_are_never_dots() {
        let p = Params { feature_capacity: 10, extent: 4096, buffer: 256, attrs: AttrPolicy::All, simplify_tolerance: None };
        let tt = TileTransform::new(TileId::new(0, 0, 0), 4096, 256);
        let (x, y) = lonlat_to_merc(10.0, 45.0);
        let pt = Geometry { kind: GeomKind::Point, parts: vec![vec![[x, y]]], polys: vec![] };
        let (g, is_dot) = to_tile_geometry(&pt, &tt, &p, false).unwrap();
        assert!(!is_dot && g.kind == GeomKind::Point);
        // size-first priority: a bigger feature outranks a smaller one whatever the hash
        assert!(priority(100, 0) > priority(3, u32::MAX));
    }

    #[test]
    fn query_rect_tests() {
        let sq = Geometry {
            kind: GeomKind::Polygon,
            parts: vec![vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]]],
            polys: vec![0],
        };
        assert!(geom_intersects_rect(&sq, &[4.0, 4.0, 5.0, 5.0])); // inside
        assert!(geom_intersects_rect(&sq, &[9.5, 4.0, 11.0, 5.0])); // crosses edge
        assert!(!geom_intersects_rect(&sq, &[20.0, 20.0, 21.0, 21.0]));
        let ln = Geometry { kind: GeomKind::Line, parts: vec![vec![[0.0, 0.0], [10.0, 10.0]]], polys: vec![] };
        assert!(geom_intersects_rect(&ln, &[4.0, 4.0, 6.0, 6.0]));
        assert!(!geom_intersects_rect(&ln, &[0.0, 8.0, 2.0, 10.0]));
    }
}
