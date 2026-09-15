//! Planar geometry, Web Mercator tile math, clipping and simplification.
//! Adapted from TileAQP's `geom` module (same authors' Rust port of starlet).

pub mod clip;
pub mod mercator;
pub mod simplify;
pub mod wkb;

pub use mercator::*;
pub use wkb::{BBox, GeomKind, Geometry};
