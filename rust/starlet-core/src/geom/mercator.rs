//! Web Mercator tile math (EPSG:3857), compatible with starlet's helpers
//! (`starlet._internal.mvt.helpers`).

pub const WORLD_MIN: f64 = -20037508.342789244;
pub const WORLD_MAX: f64 = 20037508.342789244;
pub const WORLD_W: f64 = WORLD_MAX - WORLD_MIN;
pub const DEFAULT_EXTENT: u32 = 4096;
pub const DEFAULT_BUFFER: u32 = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileId {
    pub z: u8,
    pub x: u32,
    pub y: u32,
}

impl TileId {
    pub fn new(z: u8, x: u32, y: u32) -> Self {
        TileId { z, x, y }
    }
    pub fn bounds(&self) -> [f64; 4] {
        tile_bounds(self.z, self.x, self.y)
    }
}

impl std::fmt::Display for TileId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}/{}", self.z, self.x, self.y)
    }
}

/// Longitude/latitude (degrees) to Web Mercator meters. Latitude is clamped
/// to the Mercator limits so poles do not produce infinities.
#[inline]
pub fn lonlat_to_merc(lon: f64, lat: f64) -> (f64, f64) {
    let x = lon.clamp(-180.0, 180.0) * WORLD_MAX / 180.0;
    let lat = lat.clamp(-85.051_128_78, 85.051_128_78);
    let y = ((90.0 + lat) * std::f64::consts::PI / 360.0).tan().ln() / std::f64::consts::PI * WORLD_MAX;
    (x, y)
}

#[inline]
pub fn merc_to_lonlat(x: f64, y: f64) -> (f64, f64) {
    let lon = x / WORLD_MAX * 180.0;
    let lat = (2.0 * (y / WORLD_MAX * std::f64::consts::PI).exp().atan() - std::f64::consts::FRAC_PI_2)
        * 180.0
        / std::f64::consts::PI;
    (lon, lat)
}

#[inline]
pub fn tile_width(z: u8) -> f64 {
    WORLD_W / (1u64 << z) as f64
}

/// `[minx, miny, maxx, maxy]` in Web Mercator for a tile.
#[inline]
pub fn tile_bounds(z: u8, x: u32, y: u32) -> [f64; 4] {
    let w = tile_width(z);
    let minx = WORLD_MIN + x as f64 * w;
    let maxy = WORLD_MAX - y as f64 * w;
    [minx, maxy - w, minx + w, maxy]
}

/// Affine transform from Mercator to tile coordinates (origin top-left,
/// `extent` units across the tile). Identical to starlet's affine params.
#[derive(Clone, Copy, Debug)]
pub struct TileTransform {
    pub minx: f64,
    pub maxy: f64,
    pub scale: f64,
    pub extent: u32,
    pub buffer: u32,
}

impl TileTransform {
    pub fn new(tile: TileId, extent: u32, buffer: u32) -> Self {
        let b = tile.bounds();
        TileTransform { minx: b[0], maxy: b[3], scale: extent as f64 / (b[2] - b[0]), extent, buffer }
    }
    #[inline]
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        ((x - self.minx) * self.scale, (self.maxy - y) * self.scale)
    }
    /// Mercator query bounds including the buffer (starlet's
    /// `_expand_tile_bounds_for_buffer`).
    pub fn buffered_bounds(&self) -> [f64; 4] {
        let pad = self.buffer as f64 / self.scale;
        let w = self.extent as f64 / self.scale;
        [self.minx - pad, self.maxy - w - pad, self.minx + w + pad, self.maxy + pad]
    }
}

/// Mercator bbox -> lon/lat bbox (for pruning against EPSG:4326 partitions).
pub fn merc_bbox_to_lonlat(b: &[f64; 4]) -> [f64; 4] {
    let (x0, y0) = merc_to_lonlat(b[0], b[1]);
    let (x1, y1) = merc_to_lonlat(b[2], b[3]);
    [x0.min(x1), y0.min(y1), x0.max(x1), y0.max(y1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lonlat_roundtrip() {
        let (x, y) = lonlat_to_merc(-117.4, 33.9);
        let (lon, lat) = merc_to_lonlat(x, y);
        assert!((lon + 117.4).abs() < 1e-9 && (lat - 33.9).abs() < 1e-9);
    }

    #[test]
    fn tile_bounds_z0_is_world() {
        assert_eq!(tile_bounds(0, 0, 0), [WORLD_MIN, WORLD_MIN, WORLD_MAX, WORLD_MAX]);
    }
}
