//! Mapbox Vector Tile (spec 2.1) encoding. Hand-written protobuf, adapted
//! from TileAQP's `mvt` module.

pub mod encode;
pub mod proto;

pub use encode::{LayerBuilder, TileWriter, Value};
