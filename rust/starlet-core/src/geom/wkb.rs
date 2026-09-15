//! Minimal, allocation-light WKB parser producing a flat geometry
//! representation. Multi geometries are folded into their single-type
//! variant (a MultiPolygon is a `Polygon` with several parts).

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeomKind {
    Point = 1,
    Line = 2,
    Polygon = 3,
}

/// Geometry in an arbitrary planar coordinate system.
/// - Point: `parts` has one ring with all points.
/// - Line: each part is one linestring.
/// - Polygon: `parts` holds every ring of every polygon; `polys` marks the
///   index in `parts` where each polygon starts (first ring is the exterior).
#[derive(Clone, Debug, PartialEq)]
pub struct Geometry {
    pub kind: GeomKind,
    pub parts: Vec<Vec<[f64; 2]>>,
    pub polys: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BBox {
    pub minx: f64,
    pub miny: f64,
    pub maxx: f64,
    pub maxy: f64,
}

impl BBox {
    pub fn empty() -> Self {
        BBox { minx: f64::INFINITY, miny: f64::INFINITY, maxx: f64::NEG_INFINITY, maxy: f64::NEG_INFINITY }
    }
    pub fn from_arr(b: [f64; 4]) -> Self {
        BBox { minx: b[0], miny: b[1], maxx: b[2], maxy: b[3] }
    }
    pub fn arr(&self) -> [f64; 4] {
        [self.minx, self.miny, self.maxx, self.maxy]
    }
    #[inline]
    pub fn add(&mut self, p: [f64; 2]) {
        self.minx = self.minx.min(p[0]);
        self.miny = self.miny.min(p[1]);
        self.maxx = self.maxx.max(p[0]);
        self.maxy = self.maxy.max(p[1]);
    }
    #[inline]
    pub fn intersects(&self, o: &BBox) -> bool {
        !(self.maxx < o.minx || self.minx > o.maxx || self.maxy < o.miny || self.miny > o.maxy)
    }
    pub fn is_empty(&self) -> bool {
        self.minx > self.maxx
    }
    pub fn center(&self) -> [f64; 2] {
        [(self.minx + self.maxx) * 0.5, (self.miny + self.maxy) * 0.5]
    }
    pub fn width(&self) -> f64 {
        self.maxx - self.minx
    }
    pub fn height(&self) -> f64 {
        self.maxy - self.miny
    }
}

#[derive(Debug)]
pub struct WkbError(pub &'static str);
impl fmt::Display for WkbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "wkb: {}", self.0)
    }
}
impl std::error::Error for WkbError {}

struct Cur<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cur<'a> {
    #[inline]
    fn u8(&mut self) -> Result<u8, WkbError> {
        let v = *self.b.get(self.p).ok_or(WkbError("truncated"))?;
        self.p += 1;
        Ok(v)
    }
    #[inline]
    fn u32(&mut self, le: bool) -> Result<u32, WkbError> {
        let s = self.b.get(self.p..self.p + 4).ok_or(WkbError("truncated"))?;
        self.p += 4;
        let a = [s[0], s[1], s[2], s[3]];
        Ok(if le { u32::from_le_bytes(a) } else { u32::from_be_bytes(a) })
    }
    #[inline]
    fn f64(&mut self, le: bool) -> Result<f64, WkbError> {
        let s = self.b.get(self.p..self.p + 8).ok_or(WkbError("truncated"))?;
        self.p += 8;
        let a = [s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]];
        Ok(if le { f64::from_le_bytes(a) } else { f64::from_be_bytes(a) })
    }
    fn skip(&mut self, n: usize) -> Result<(), WkbError> {
        if self.p + n > self.b.len() {
            return Err(WkbError("truncated"));
        }
        self.p += n;
        Ok(())
    }
}

/// Decode the WKB type word into (base type 1..7, number of extra dims).
fn parse_type(t: u32) -> (u32, usize) {
    let has_z = (t & 0x8000_0000) != 0;
    let has_m = (t & 0x4000_0000) != 0;
    let t = t & 0x0FFF_FFFF;
    let (base, extra_iso) = match t {
        1..=7 => (t, 0),
        1001..=1007 => (t - 1000, 1),
        2001..=2007 => (t - 2000, 1),
        3001..=3007 => (t - 3000, 2),
        _ => (t, 0),
    };
    (base, extra_iso + has_z as usize + has_m as usize)
}

impl Geometry {
    pub fn empty(kind: GeomKind) -> Self {
        Geometry { kind, parts: Vec::new(), polys: Vec::new() }
    }

    pub fn from_wkb(b: &[u8]) -> Result<Geometry, WkbError> {
        let mut c = Cur { b, p: 0 };
        let mut out: Option<Geometry> = None;
        parse_into(&mut c, &mut out)?;
        out.ok_or(WkbError("empty"))
    }

    /// Bounding box of a WKB payload by scanning its coordinates in place
    /// (no allocation). Used for row filtering on datasets that lack bbox
    /// columns.
    pub fn wkb_bbox(b: &[u8]) -> Result<BBox, WkbError> {
        let mut c = Cur { b, p: 0 };
        let mut bb = BBox::empty();
        scan_bbox(&mut c, &mut bb)?;
        if bb.is_empty() {
            Err(WkbError("empty"))
        } else {
            Ok(bb)
        }
    }

    pub fn is_empty(&self) -> bool {
        self.parts.iter().all(|p| p.is_empty())
    }

    pub fn bbox(&self) -> BBox {
        let mut bb = BBox::empty();
        for part in &self.parts {
            for p in part {
                bb.add(*p);
            }
        }
        bb
    }

    pub fn vertex_count(&self) -> usize {
        self.parts.iter().map(|p| p.len()).sum()
    }

    /// Number of polygons (for `Polygon`) or parts otherwise.
    pub fn n_parts(&self) -> usize {
        match self.kind {
            GeomKind::Polygon => self.polys.len(),
            _ => self.parts.len(),
        }
    }

    /// Rings of polygon `i`.
    pub fn poly_rings(&self, i: usize) -> &[Vec<[f64; 2]>] {
        let start = self.polys[i];
        let end = self.polys.get(i + 1).copied().unwrap_or(self.parts.len());
        &self.parts[start..end]
    }

    /// Apply a point transform in place.
    pub fn map_coords(&mut self, f: impl Fn(f64, f64) -> (f64, f64)) {
        for part in &mut self.parts {
            for p in part.iter_mut() {
                let (x, y) = f(p[0], p[1]);
                p[0] = x;
                p[1] = y;
            }
        }
    }

    pub fn from_lonlat_to_merc(&mut self) {
        self.map_coords(super::mercator::lonlat_to_merc);
    }
}

/// Accumulate the bbox of a WKB geometry without building it.
fn scan_bbox(c: &mut Cur, bb: &mut BBox) -> Result<(), WkbError> {
    let le = c.u8()? == 1;
    let (t, extra) = parse_type(c.u32(le)?);
    let mut scan_pts = |c: &mut Cur, n: usize| -> Result<(), WkbError> {
        if n > (c.b.len() - c.p) / 16 + 1 {
            return Err(WkbError("bad ring length"));
        }
        for _ in 0..n {
            let x = c.f64(le)?;
            let y = c.f64(le)?;
            c.skip(extra * 8)?;
            if !x.is_nan() && !y.is_nan() {
                bb.add([x, y]);
            }
        }
        Ok(())
    };
    match t {
        1 => scan_pts(c, 1)?,
        2 => {
            let n = c.u32(le)? as usize;
            scan_pts(c, n)?;
        }
        3 => {
            let nr = c.u32(le)? as usize;
            for _ in 0..nr {
                let n = c.u32(le)? as usize;
                scan_pts(c, n)?;
            }
        }
        4..=7 => {
            let n = c.u32(le)?;
            for _ in 0..n {
                scan_bbox(c, bb)?;
            }
        }
        _ => return Err(WkbError("unsupported geometry type")),
    }
    Ok(())
}

fn read_ring(c: &mut Cur, le: bool, extra: usize) -> Result<Vec<[f64; 2]>, WkbError> {
    let n = c.u32(le)? as usize;
    if n > (c.b.len() - c.p) / 16 + 1 {
        return Err(WkbError("bad ring length"));
    }
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        let x = c.f64(le)?;
        let y = c.f64(le)?;
        c.skip(extra * 8)?;
        v.push([x, y]);
    }
    Ok(v)
}

fn push_kind(out: &mut Option<Geometry>, kind: GeomKind) -> Result<&mut Geometry, WkbError> {
    match out {
        None => {
            *out = Some(Geometry::empty(kind));
        }
        Some(g) if g.kind != kind => return Err(WkbError("mixed geometry collection")),
        _ => {}
    }
    Ok(out.as_mut().unwrap())
}

fn parse_into(c: &mut Cur, out: &mut Option<Geometry>) -> Result<(), WkbError> {
    let le = c.u8()? == 1;
    let (t, extra) = parse_type(c.u32(le)?);
    match t {
        1 => {
            let x = c.f64(le)?;
            let y = c.f64(le)?;
            c.skip(extra * 8)?;
            let g = push_kind(out, GeomKind::Point)?;
            if !x.is_nan() && !y.is_nan() {
                if g.parts.is_empty() {
                    g.parts.push(Vec::new());
                }
                g.parts[0].push([x, y]);
            }
        }
        2 => {
            let ring = read_ring(c, le, extra)?;
            let g = push_kind(out, GeomKind::Line)?;
            if ring.len() >= 2 {
                g.parts.push(ring);
            }
        }
        3 => {
            let nr = c.u32(le)? as usize;
            let g = push_kind(out, GeomKind::Polygon)?;
            let start = g.parts.len();
            for _ in 0..nr {
                let ring = read_ring(c, le, extra)?;
                if ring.len() >= 4 {
                    g.parts.push(ring);
                }
            }
            if g.parts.len() > start {
                g.polys.push(start);
            }
        }
        4..=7 => {
            let n = c.u32(le)?;
            for _ in 0..n {
                parse_into(c, out)?;
            }
        }
        _ => return Err(WkbError("unsupported geometry type")),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wkb_polygon() -> Vec<u8> {
        // POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))
        let mut b = vec![1u8];
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&5u32.to_le_bytes());
        for (x, y) in [(0., 0.), (4., 0.), (4., 4.), (0., 4.), (0., 0.)] {
            b.extend_from_slice(&f64::to_le_bytes(x));
            b.extend_from_slice(&f64::to_le_bytes(y));
        }
        b
    }

    #[test]
    fn parses_polygon() {
        let g = Geometry::from_wkb(&wkb_polygon()).unwrap();
        assert_eq!(g.kind, GeomKind::Polygon);
        assert_eq!(g.n_parts(), 1);
        assert_eq!(g.vertex_count(), 5);
        assert_eq!(g.bbox().arr(), [0., 0., 4., 4.]);
        assert_eq!(Geometry::wkb_bbox(&wkb_polygon()).unwrap().arr(), [0., 0., 4., 4.]);
    }

    #[test]
    fn rejects_truncated() {
        assert!(Geometry::from_wkb(&wkb_polygon()[..20]).is_err());
    }
}
