//! Push-mode pyramid generation with bounded memory.
//!
//! The pull-mode generator (`tiler.rs`) fetches, per tile, every row group
//! the tile touches. That is ideal for single on-demand tiles, but for a
//! pyramid it pins the whole decoded dataset in memory at low zooms (a z0
//! tile references every row group at once, and parallel tiles multiply it).
//!
//! This module does the opposite, in two passes over the row groups, each
//! row group decoded by exactly one worker at a time and then dropped:
//!
//! 1. **select** — stream every row group once (geometry + bbox columns
//!    only). For each row, find the requested tiles whose *buffered* bounds
//!    its bbox intersects (at every requested zoom) and push a 12-byte
//!    reference `(crc32 priority, row group, row)` into that tile's bounded
//!    top-k heap. Heaps are accumulated per worker and merged, so memory is
//!    the references, never the features.
//! 2. **encode** — stream the row groups that hold winners (geometry +
//!    attributes), decode each winning row once, and append it — projected
//!    to each tile it won — to that tile's layer builder. A tile is finished
//!    and written the moment its last winner arrives, so at any time only
//!    tiles still receiving features are held in memory.
//!
//! Selection semantics are the same as `tiler.rs` and starlet's Python
//! pipeline: the buffered-bounds bbox test, `crc32(source WKB)` priority,
//! one sub-pixel feature per pixel cell, strict-greater top-k replacement
//! for larger features, and the per-feature dot / simplify / clip chain.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::hash::{BuildHasherDefault, Hasher};
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use anyhow::Result;
use arrow::array::AsArray;
use arrow::datatypes::Float64Type;
use parking_lot::Mutex;
use rayon::prelude::*;

use crate::geom::mercator::{lonlat_to_merc, tile_width, WORLD_MAX, WORLD_MIN};
use crate::geom::{merc_bbox_to_lonlat, GeomKind, Geometry, TileId, TileTransform};
use crate::mvt::{LayerBuilder, TileWriter};
use crate::pq::{arrow_value, wkb_at, BBOX_COLS};
use crate::tiler::{place, priority, sub_pixel, to_tile_geometry, Dataset, Params, LAYER_NAME};

const BATCH_SIZE: usize = 8192;

/// Minimal Fx-style hasher for the (x, y) -> slot lookup; it is hit for
/// every row × zoom × covered tile and SipHash is measurably slower there.
#[derive(Default)]
struct FxHasher(u64);

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 = (self.0.rotate_left(5) ^ (*b as u64)).wrapping_mul(0x517c_c1b7_2722_0a95);
        }
    }
    #[inline]
    fn write_u64(&mut self, v: u64) {
        self.0 = (self.0.rotate_left(5) ^ v).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

/// `(priority, row group id, row)` — a min-heap on priority keeps the top-k.
type Entry = (u64, u32, u32);
/// heap entries also remember the pixel cell to fall back to when evicted
type HeapEntry = (u64, u32, u32, (i32, i32));
type Heap = BinaryHeap<Reverse<HeapEntry>>;

/// Per-tile selection state: top-k (by size, then hash) of the features
/// larger than a display pixel, and one dot per pixel cell for everything
/// else (sub-pixel features and the larger ones that did not make the cut).
#[derive(Default)]
struct TileSel {
    heap: Heap,
    cells: FxMap<(i32, i32), Entry>,
}

impl TileSel {
    #[inline]
    fn offer_cell(&mut self, c: (i32, i32), e: Entry) {
        match self.cells.get_mut(&c) {
            Some(cur) if e.0 <= cur.0 => {}
            Some(cur) => *cur = e,
            None => {
                self.cells.insert(c, e);
            }
        }
    }
    #[inline]
    fn offer(&mut self, cell: (i32, i32), small: bool, e: Entry, k: usize) {
        if small {
            self.offer_cell(cell, e);
        } else if let Some(evicted) = push_capped(&mut self.heap, (e.0, e.1, e.2, cell), k) {
            self.offer_cell(evicted.3, (evicted.0, evicted.1, evicted.2));
        }
    }
    fn merge_from(&mut self, other: TileSel, k: usize) {
        for Reverse(e) in other.heap.into_iter() {
            self.offer(e.3, false, (e.0, e.1, e.2), k);
        }
        for (c, e) in other.cells {
            self.offer_cell(c, e);
        }
    }
    fn len(&self) -> usize {
        self.heap.len() + self.cells.len()
    }
    /// `(entry, as_dot)`
    fn into_entries(self) -> impl Iterator<Item = (Entry, bool)> {
        self.heap
            .into_iter()
            .map(|Reverse(e)| ((e.0, e.1, e.2), false))
            .chain(self.cells.into_values().map(|e| (e, true)))
    }
}

/// Push with capacity `k`; returns the entry that lost its place (the
/// rejected candidate itself, or the evicted minimum), if any.
#[inline]
fn push_capped(h: &mut Heap, e: HeapEntry, k: usize) -> Option<HeapEntry> {
    if h.len() >= k {
        // starlet: a candidate replaces the current minimum only if strictly higher
        if e.0 <= h.peek().unwrap().0 .0 {
            return Some(e);
        }
        let Reverse(out) = h.pop().unwrap();
        h.push(Reverse(e));
        return Some(out);
    }
    h.push(Reverse(e));
    None
}

#[inline]
fn bbox_intersects(a: &[f64; 4], b: &[f64; 4]) -> bool {
    !(a[2] < b[0] || a[0] > b[2] || a[3] < b[1] || a[1] > b[3])
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PyramidStats {
    pub tiles_requested: usize,
    pub tiles_written: usize,
    pub row_groups: usize,
    pub candidates: u64,
    pub features: u64,
}

/// The requested tiles, indexed for fast membership lookup.
struct Wanted {
    ids: Vec<TileId>,
    /// buffered query bounds per slot, EPSG:4326 (the same test `tiler.rs` uses)
    q: Vec<[f64; 4]>,
    tts: Vec<TileTransform>,
    /// per zoom: (x, y) packed as `x << 32 | y` -> slot
    by_zoom: Vec<Option<FxMap<u64, u32>>>,
    zooms: Vec<u8>,
}

impl Wanted {
    fn new(tiles: &[TileId], p: &Params) -> Wanted {
        let mut by_zoom: Vec<Option<FxMap<u64, u32>>> = Vec::new();
        let mut q = Vec::with_capacity(tiles.len());
        let mut tts = Vec::with_capacity(tiles.len());
        for (slot, t) in tiles.iter().enumerate() {
            let tt = TileTransform::new(*t, p.extent, p.buffer);
            q.push(merc_bbox_to_lonlat(&tt.buffered_bounds()));
            tts.push(tt);
            let z = t.z as usize;
            if by_zoom.len() <= z {
                by_zoom.resize_with(z + 1, || None);
            }
            by_zoom[z]
                .get_or_insert_with(FxMap::default)
                .insert(((t.x as u64) << 32) | t.y as u64, slot as u32);
        }
        let zooms = by_zoom
            .iter()
            .enumerate()
            .filter_map(|(z, m)| m.as_ref().map(|_| z as u8))
            .collect();
        Wanted { ids: tiles.to_vec(), q, tts, by_zoom, zooms }
    }

    /// Slots of requested tiles whose buffered bounds intersect `rb`
    /// (a lon/lat bbox), at every requested zoom.
    #[inline]
    fn slots_for(&self, rb: &[f64; 4], p: &Params, out: &mut Vec<u32>) {
        out.clear();
        let (mx0, my0) = lonlat_to_merc(rb[0], rb[1]);
        let (mx1, my1) = lonlat_to_merc(rb[2], rb[3]);
        let frac = p.buffer as f64 / p.extent as f64;
        for &z in &self.zooms {
            let map = self.by_zoom[z as usize].as_ref().unwrap();
            let w = tile_width(z);
            // buffer pad plus a hair of slack; the exact test below is authoritative
            let pad = w * frac + w * 1e-9;
            let n = (1u64 << z) as f64 - 1.0;
            let tx0 = ((mx0 - pad - WORLD_MIN) / w).floor().clamp(0.0, n) as u64;
            let tx1 = ((mx1 + pad - WORLD_MIN) / w).floor().clamp(0.0, n) as u64;
            let ty0 = ((WORLD_MAX - (my1 + pad)) / w).floor().clamp(0.0, n) as u64;
            let ty1 = ((WORLD_MAX - (my0 - pad)) / w).floor().clamp(0.0, n) as u64;
            for tx in tx0..=tx1 {
                for ty in ty0..=ty1 {
                    if let Some(&slot) = map.get(&((tx << 32) | ty)) {
                        if bbox_intersects(rb, &self.q[slot as usize]) {
                            out.push(slot);
                        }
                    }
                }
            }
        }
    }
}

/// Row-group offsets of the batches `PqFile::read` returns for one row group.
fn batch_offsets(batches: &[arrow::record_batch::RecordBatch]) -> Vec<usize> {
    let mut offs = Vec::with_capacity(batches.len() + 1);
    let mut n = 0;
    for b in batches {
        offs.push(n);
        n += b.num_rows();
    }
    offs.push(n);
    offs
}

impl Dataset {
    /// Generate `tiles` (any mix of zooms) with bounded memory and hand each
    /// non-empty tile to `sink` as soon as it is complete. `sink` is called
    /// from worker threads.
    pub fn generate_pyramid(
        &self,
        tiles: &[TileId],
        p: &Params,
        sink: &(dyn Fn(TileId, Vec<u8>) -> Result<()> + Sync),
    ) -> Result<PyramidStats> {
        let mut stats = PyramidStats { tiles_requested: tiles.len(), ..Default::default() };
        if tiles.is_empty() {
            return Ok(stats);
        }
        let k = p.feature_capacity.max(1);
        let wanted = Wanted::new(tiles, p);

        // Row groups whose partition bbox touches any requested tile at all.
        let mut rgs: Vec<(u32, u32)> = Vec::new();
        for (pi, part) in self.parts.iter().enumerate() {
            let touches = wanted.q.iter().any(|q| bbox_intersects(&part.bbox, q));
            if !touches {
                continue;
            }
            for rg in 0..part.file.num_row_groups() {
                rgs.push((pi as u32, rg as u32));
            }
        }
        stats.row_groups = rgs.len();
        let candidates = AtomicU64::new(0);

        // ---- pass 1: select ------------------------------------------------
        let cell = p.cell();
        let merged: FxMap<u32, TileSel> = rgs
            .par_iter()
            .enumerate()
            .try_fold(
                FxMap::<u32, TileSel>::default,
                |mut acc, (rid, &(pi, rg))| -> Result<FxMap<u32, TileSel>> {
                    let part = &self.parts[pi as usize];
                    let mut cols: Vec<&str> = vec![part.geom_col.as_str()];
                    if part.has_bbox_cols {
                        cols.extend(BBOX_COLS.iter());
                    }
                    let batches = part.file.read(vec![rg as usize], &cols, BATCH_SIZE)?;
                    let offs = batch_offsets(&batches);
                    let mut slots: Vec<u32> = Vec::with_capacity(16);
                    let mut n_cand: u64 = 0;
                    for (bi, b) in batches.iter().enumerate() {
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
                        for i in 0..b.num_rows() {
                            let Some(w) = wkb_at(b, gi, i) else { continue };
                            let rb: [f64; 4] = match &bb_cols {
                                Some(c) => [c[0].value(i), c[1].value(i), c[2].value(i), c[3].value(i)],
                                None => match Geometry::wkb_bbox(w) {
                                    Ok(bb) => bb.arr(),
                                    Err(_) => continue,
                                },
                            };
                            wanted.slots_for(&rb, p, &mut slots);
                            if slots.is_empty() {
                                continue;
                            }
                            n_cand += slots.len() as u64;
                            let crc = crc32fast::hash(w);
                            let row = (offs[bi] + i) as u32;
                            for &s in &slots {
                                let pl = place(&rb, &wanted.tts[s as usize], cell);
                                acc.entry(s)
                                    .or_default()
                                    .offer(pl.cell, pl.small, (priority(pl.size16, crc), rid as u32, row), k);
                            }
                        }
                    }
                    candidates.fetch_add(n_cand, Ordering::Relaxed);
                    Ok(acc)
                },
            )
            .try_reduce(FxMap::default, |a, b| {
                // merge the smaller map into the larger one
                let (mut a, b) = if a.len() >= b.len() { (a, b) } else { (b, a) };
                for (slot, sb) in b {
                    match a.get_mut(&slot) {
                        None => {
                            a.insert(slot, sb);
                        }
                        Some(sa) => sa.merge_from(sb, k),
                    }
                }
                Ok(a)
            })?;
        stats.candidates = candidates.load(Ordering::Relaxed);

        // ---- regroup winners by row group ------------------------------------
        let mut by_rg: Vec<Vec<(u32, u32, bool)>> = vec![Vec::new(); rgs.len()]; // (row, slot, as_dot)
        let mut remaining: Vec<AtomicUsize> = Vec::with_capacity(wanted.ids.len());
        remaining.resize_with(wanted.ids.len(), || AtomicUsize::new(0));
        let mut total: u64 = 0;
        // tiles with more sub-pixel features than the capacity carry their
        // sub-pixel points as bare dots (same rule as `tiler.rs`)
        let mut strip_point_attrs = vec![false; wanted.ids.len()];
        for (slot, sel) in merged {
            remaining[slot as usize].store(sel.len(), Ordering::Relaxed);
            strip_point_attrs[slot as usize] = sel.cells.len() > k;
            total += sel.len() as u64;
            for ((_, rid, row), as_dot) in sel.into_entries() {
                by_rg[rid as usize].push((row, slot, as_dot));
            }
        }
        stats.features = total;
        by_rg.par_iter_mut().for_each(|v| v.sort_unstable());

        // ---- pass 2: encode --------------------------------------------------
        let builders: Vec<Mutex<Option<LayerBuilder>>> = (0..wanted.ids.len())
            .map(|slot| Mutex::new(if remaining[slot].load(Ordering::Relaxed) > 0 {
                Some(LayerBuilder::new(LAYER_NAME, p.extent))
            } else {
                None
            }))
            .collect();
        let written = AtomicUsize::new(0);

        let finish_tile = |slot: usize| -> Result<()> {
            let lb = builders[slot].lock().take();
            if let Some(lb) = lb {
                if lb.feature_count > 0 {
                    let mut w = TileWriter::new();
                    w.add_layer(&lb);
                    sink(wanted.ids[slot], w.finish())?;
                    written.fetch_add(1, Ordering::Relaxed);
                }
            }
            Ok(())
        };

        rgs.par_iter()
            .zip(by_rg.par_iter())
            .try_for_each(|(&(pi, rg), winners)| -> Result<()> {
                if winners.is_empty() {
                    return Ok(());
                }
                let part = &self.parts[pi as usize];
                let mut cols: Vec<&str> = vec![part.geom_col.as_str()];
                cols.extend(part.attr_cols.iter().map(|s| s.as_str()));
                let batches = part.file.read(vec![rg as usize], &cols, BATCH_SIZE)?;
                let offs = batch_offsets(&batches);
                let mut tags: Vec<(u32, u32)> = Vec::new();
                let mut i = 0;
                while i < winners.len() {
                    let row = winners[i].0 as usize;
                    let mut j = i;
                    while j < winners.len() && winners[j].0 as usize == row {
                        j += 1;
                    }
                    // locate the batch holding this row
                    let bi = match offs.binary_search(&row) {
                        Ok(x) => x.min(batches.len() - 1),
                        Err(x) => x - 1,
                    };
                    let b = &batches[bi];
                    let ri = row - offs[bi];
                    let gi = b.schema().index_of(&part.geom_col)?;
                    let geom = wkb_at(b, gi, ri).and_then(|w| Geometry::from_wkb(w).ok()).map(|mut g| {
                        g.from_lonlat_to_merc();
                        g
                    });
                    let attrs: Vec<(&str, crate::mvt::Value)> = part
                        .attr_cols
                        .iter()
                        .filter(|name| p.attrs.allows(name))
                        .filter_map(|name| {
                            let ci = b.schema().index_of(name).ok()?;
                            arrow_value(b.column(ci), ri).map(|v| (name.as_str(), v))
                        })
                        .collect();
                    for &(_, slot, as_dot) in &winners[i..j] {
                        let slot = slot as usize;
                        if let Some(g) = geom.as_ref() {
                            if let Some((tg, is_dot)) = to_tile_geometry(g, &wanted.tts[slot], p, as_dot) {
                                let mut guard = builders[slot].lock();
                                if let Some(lb) = guard.as_mut() {
                                    tags.clear();
                                    let bare = is_dot
                                        || (strip_point_attrs[slot] && tg.kind == GeomKind::Point && sub_pixel(&tg, p));
                                    if !bare {
                                        for (name, v) in &attrs {
                                            let ki = lb.key(name);
                                            let vi = lb.value(v);
                                            tags.push((ki, vi));
                                        }
                                    }
                                    lb.add_feature(None, &tg, &tags);
                                }
                            }
                        }
                        if remaining[slot].fetch_sub(1, Ordering::AcqRel) == 1 {
                            finish_tile(slot)?;
                        }
                    }
                    i = j;
                }
                Ok(())
            })?;

        stats.tiles_written = written.load(Ordering::Relaxed);
        Ok(stats)
    }

    /// `generate_pyramid` writing `<outdir>/<z>/<x>/<y>.mvt` for every
    /// non-empty tile. Returns the stats (incl. the count written).
    pub fn write_pyramid(&self, outdir: &Path, tiles: &[TileId], p: &Params) -> Result<PyramidStats> {
        // Pre-create the <z>/<x> directories once, off the hot path.
        let mut dirs: Vec<(u8, u32)> = tiles.iter().map(|t| (t.z, t.x)).collect();
        dirs.sort_unstable();
        dirs.dedup();
        dirs.par_iter().try_for_each(|(z, x)| {
            std::fs::create_dir_all(outdir.join(z.to_string()).join(x.to_string()))
        })?;
        let sink = |t: TileId, bytes: Vec<u8>| -> Result<()> {
            let path = outdir.join(t.z.to_string()).join(t.x.to_string()).join(format!("{}.mvt", t.y));
            std::fs::write(path, bytes)?;
            Ok(())
        };
        self.generate_pyramid(tiles, p, &sink)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_for_covers_buffer_zone() {
        // two z1 tiles; a point just left of the x=0 meridian must land in
        // the right tile too, because of the buffer.
        let p = Params { feature_capacity: 10, extent: 4096, buffer: 256, attrs: crate::tiler::AttrPolicy::All };
        let tiles = [TileId::new(1, 0, 0), TileId::new(1, 1, 0)];
        let w = Wanted::new(&tiles, &p);
        let mut out = Vec::new();
        w.slots_for(&[-0.5, 40.0, -0.5, 40.0], &p, &mut out);
        out.sort();
        assert_eq!(out, vec![0, 1]);
        w.slots_for(&[-100.0, 40.0, -100.0, 40.0], &p, &mut out);
        assert_eq!(out, vec![0]);
        // southern hemisphere: no z1 row-0 tile (buffer zone excluded)
        w.slots_for(&[-100.0, -40.0, -100.0, -40.0], &p, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn push_capped_keeps_top_k_strictly_and_returns_the_loser() {
        let c = (0, 0);
        let mut h = Heap::new();
        assert!(push_capped(&mut h, (5, 0, 0, c), 2).is_none());
        assert!(push_capped(&mut h, (7, 0, 1, c), 2).is_none());
        assert_eq!(push_capped(&mut h, (5, 0, 2, c), 2), Some((5, 0, 2, c))); // equal to min: rejected
        assert_eq!(push_capped(&mut h, (6, 0, 3, c), 2), Some((5, 0, 0, c))); // evicts the 5
        let mut got: Vec<HeapEntry> = h.into_iter().map(|Reverse(e)| e).collect();
        got.sort();
        assert_eq!(got, vec![(6, 0, 3, c), (7, 0, 1, c)]);
    }

    #[test]
    fn demoted_large_features_become_dots() {
        let mut sel = TileSel::default();
        sel.offer((1, 1), false, (priority(50, 1), 0, 0), 1);
        sel.offer((2, 2), false, (priority(90, 1), 0, 1), 1); // evicts the first -> dot in (1,1)
        sel.offer((3, 3), false, (priority(10, 1), 0, 2), 1); // rejected -> dot in (3,3)
        assert_eq!(sel.heap.len(), 1);
        assert_eq!(sel.cells.len(), 2);
        assert!(sel.cells.contains_key(&(1, 1)) && sel.cells.contains_key(&(3, 3)));
    }
}
