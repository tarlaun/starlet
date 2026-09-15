//! Rectangle clipping in tile coordinates (Sutherland-Hodgman for rings,
//! Liang-Barsky for lines).

use super::wkb::{GeomKind, Geometry};

#[derive(Clone, Copy)]
pub struct Rect {
    pub minx: f64,
    pub miny: f64,
    pub maxx: f64,
    pub maxy: f64,
}

impl Rect {
    pub fn new(minx: f64, miny: f64, maxx: f64, maxy: f64) -> Self {
        Rect { minx, miny, maxx, maxy }
    }
    #[inline]
    fn contains(&self, p: [f64; 2]) -> bool {
        p[0] >= self.minx && p[0] <= self.maxx && p[1] >= self.miny && p[1] <= self.maxy
    }
}

/// Clip a geometry to `r`. Returns `None` when nothing remains.
pub fn clip(g: &Geometry, r: &Rect) -> Option<Geometry> {
    match g.kind {
        GeomKind::Point => {
            let pts: Vec<[f64; 2]> = g.parts.iter().flatten().copied().filter(|p| r.contains(*p)).collect();
            if pts.is_empty() {
                None
            } else {
                Some(Geometry { kind: GeomKind::Point, parts: vec![pts], polys: vec![] })
            }
        }
        GeomKind::Line => {
            let mut parts = Vec::new();
            for part in &g.parts {
                clip_line(part, r, &mut parts);
            }
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
                let ext = clip_ring(&rings[0], r);
                if ext.len() < 4 {
                    continue;
                }
                let start = out.parts.len();
                out.parts.push(ext);
                for hole in &rings[1..] {
                    let h = clip_ring(hole, r);
                    if h.len() >= 4 {
                        out.parts.push(h);
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

/// Sutherland-Hodgman polygon clipping against an axis-aligned rectangle.
pub fn clip_ring(ring: &[[f64; 2]], r: &Rect) -> Vec<[f64; 2]> {
    let mut cur: Vec<[f64; 2]> = ring.to_vec();
    if let (Some(f), Some(l)) = (cur.first(), cur.last()) {
        if f == l {
            cur.pop();
        }
    }
    for edge in 0..4 {
        if cur.is_empty() {
            return Vec::new();
        }
        let inside = |p: &[f64; 2]| match edge {
            0 => p[0] >= r.minx,
            1 => p[0] <= r.maxx,
            2 => p[1] >= r.miny,
            _ => p[1] <= r.maxy,
        };
        let intersect = |a: &[f64; 2], b: &[f64; 2]| -> [f64; 2] {
            match edge {
                0 | 1 => {
                    let x = if edge == 0 { r.minx } else { r.maxx };
                    let t = (x - a[0]) / (b[0] - a[0]);
                    [x, a[1] + t * (b[1] - a[1])]
                }
                _ => {
                    let y = if edge == 2 { r.miny } else { r.maxy };
                    let t = (y - a[1]) / (b[1] - a[1]);
                    [a[0] + t * (b[0] - a[0]), y]
                }
            }
        };
        let mut next = Vec::with_capacity(cur.len() + 4);
        let n = cur.len();
        for i in 0..n {
            let a = cur[(i + n - 1) % n];
            let b = cur[i];
            let (ia, ib) = (inside(&a), inside(&b));
            if ib {
                if !ia {
                    next.push(intersect(&a, &b));
                }
                next.push(b);
            } else if ia {
                next.push(intersect(&a, &b));
            }
        }
        cur = next;
    }
    cur.dedup();
    if cur.len() >= 3 {
        let f = cur[0];
        cur.push(f);
        cur
    } else {
        Vec::new()
    }
}

/// Liang-Barsky clipping of a polyline; emits every visible sub-segment run.
pub fn clip_line(line: &[[f64; 2]], r: &Rect, out: &mut Vec<Vec<[f64; 2]>>) {
    let mut run: Vec<[f64; 2]> = Vec::new();
    for w in line.windows(2) {
        let (a, b) = (w[0], w[1]);
        if let Some((p, q, a_inside, b_inside)) = clip_segment(a, b, r) {
            if run.is_empty() || !a_inside {
                if run.len() >= 2 {
                    out.push(std::mem::take(&mut run));
                } else {
                    run.clear();
                }
                run.push(p);
            }
            run.push(q);
            if !b_inside {
                if run.len() >= 2 {
                    out.push(std::mem::take(&mut run));
                } else {
                    run.clear();
                }
            }
        } else if run.len() >= 2 {
            out.push(std::mem::take(&mut run));
        } else {
            run.clear();
        }
    }
    if run.len() >= 2 {
        out.push(run);
    }
}

fn clip_segment(a: [f64; 2], b: [f64; 2], r: &Rect) -> Option<([f64; 2], [f64; 2], bool, bool)> {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let mut t0 = 0.0f64;
    let mut t1 = 1.0f64;
    let checks = [(-dx, a[0] - r.minx), (dx, r.maxx - a[0]), (-dy, a[1] - r.miny), (dy, r.maxy - a[1])];
    for (p, q) in checks {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
        } else {
            let t = q / p;
            if p < 0.0 {
                if t > t1 {
                    return None;
                }
                if t > t0 {
                    t0 = t;
                }
            } else {
                if t < t0 {
                    return None;
                }
                if t < t1 {
                    t1 = t;
                }
            }
        }
    }
    let p = [a[0] + t0 * dx, a[1] + t0 * dy];
    let q = [a[0] + t1 * dx, a[1] + t1 * dy];
    Some((p, q, t0 == 0.0, t1 == 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clips_polygon_to_rect() {
        let ring = vec![[-2., -2.], [6., -2.], [6., 6.], [-2., 6.], [-2., -2.]];
        let out = clip_ring(&ring, &Rect::new(0., 0., 4., 4.));
        assert_eq!(out.len(), 5);
    }

    #[test]
    fn outside_polygon_is_dropped() {
        let ring = vec![[10., 10.], [12., 10.], [12., 12.], [10., 10.]];
        assert!(clip_ring(&ring, &Rect::new(0., 0., 4., 4.)).is_empty());
    }
}
