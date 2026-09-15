//! Douglas-Peucker simplification (starlet uses shapely.simplify with
//! `preserve_topology=False`, which is plain Douglas-Peucker).

use super::wkb::{GeomKind, Geometry};

pub fn douglas_peucker(pts: &[[f64; 2]], tol: f64) -> Vec<[f64; 2]> {
    if pts.len() <= 2 {
        return pts.to_vec();
    }
    let tol2 = tol * tol;
    let mut keep = vec![false; pts.len()];
    keep[0] = true;
    keep[pts.len() - 1] = true;
    let mut stack = vec![(0usize, pts.len() - 1)];
    while let Some((s, e)) = stack.pop() {
        if e <= s + 1 {
            continue;
        }
        let a = pts[s];
        let b = pts[e];
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        let len2 = dx * dx + dy * dy;
        let mut best = 0.0;
        let mut idx = s;
        for i in s + 1..e {
            let p = pts[i];
            let d2 = if len2 == 0.0 {
                (p[0] - a[0]).powi(2) + (p[1] - a[1]).powi(2)
            } else {
                let t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2;
                let t = t.clamp(0.0, 1.0);
                (p[0] - (a[0] + t * dx)).powi(2) + (p[1] - (a[1] + t * dy)).powi(2)
            };
            if d2 > best {
                best = d2;
                idx = i;
            }
        }
        if best > tol2 {
            keep[idx] = true;
            stack.push((s, idx));
            stack.push((idx, e));
        }
    }
    pts.iter().zip(keep).filter(|(_, k)| *k).map(|(p, _)| *p).collect()
}

/// Simplify every part of `g` with Douglas-Peucker at `tol` (tile units),
/// without snapping. Mirrors `shapely.simplify(geom, tol, preserve_topology=False)`
/// applied by starlet before clipping. Degenerate parts are dropped.
pub fn simplify_dp(g: &Geometry, tol: f64) -> Option<Geometry> {
    match g.kind {
        GeomKind::Point => Some(g.clone()),
        GeomKind::Line => {
            let mut parts = Vec::with_capacity(g.parts.len());
            for part in &g.parts {
                let s = if part.len() > 2 { douglas_peucker(part, tol) } else { part.clone() };
                if s.len() >= 2 {
                    parts.push(s);
                }
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
                let mut cleaned: Vec<Vec<[f64; 2]>> = Vec::new();
                for (ri, ring) in rings.iter().enumerate() {
                    let s = if ring.len() > 4 { douglas_peucker(ring, tol) } else { ring.clone() };
                    if s.len() >= 4 {
                        cleaned.push(s);
                    } else if ri == 0 {
                        break; // exterior collapsed: drop whole polygon
                    }
                }
                if !cleaned.is_empty() {
                    out.polys.push(out.parts.len());
                    out.parts.extend(cleaned);
                }
            }
            if out.polys.is_empty() {
                None
            } else {
                Some(out)
            }
        }
    }
}

/// Shoelace signed area (positive when counter-clockwise in a y-up frame).
pub fn signed_area(ring: &[[f64; 2]]) -> f64 {
    let n = ring.len();
    if n < 3 {
        return 0.0;
    }
    let mut a = 0.0;
    for i in 0..n {
        let p = ring[i];
        let q = ring[(i + 1) % n];
        a += p[0] * q[1] - q[0] * p[1];
    }
    a / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dp_removes_collinear() {
        let pts = vec![[0., 0.], [1., 0.01], [2., 0.], [3., 0.02], [4., 0.]];
        assert_eq!(douglas_peucker(&pts, 0.1), vec![[0., 0.], [4., 0.]]);
    }
}
