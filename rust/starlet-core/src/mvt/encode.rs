//! MVT (Mapbox Vector Tile spec 2.1) encoder with key/value dictionaries.

use std::collections::HashMap;

use crate::geom::simplify::signed_area;
use crate::geom::{GeomKind, Geometry};

use super::proto::*;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Str(String),
    F64(f64),
    I64(i64),
    Bool(bool),
}

impl Value {
    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(12);
        match self {
            Value::Str(s) => put_bytes_field(&mut out, 1, s.as_bytes()),
            Value::F64(v) => {
                // double_value (field 3); starlet's encoder emits doubles too
                put_tag(&mut out, 3, 1);
                out.extend_from_slice(&v.to_le_bytes());
            }
            Value::I64(v) => {
                if *v >= 0 {
                    put_varint_field(&mut out, 5, *v as u64);
                } else {
                    put_varint_field(&mut out, 6, zigzag(*v));
                }
            }
            Value::Bool(b) => put_varint_field(&mut out, 7, *b as u64),
        }
        out
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum ValueKey {
    Str(String),
    F64(u64),
    I64(i64),
    Bool(bool),
}

impl ValueKey {
    fn of(v: &Value) -> Self {
        match v {
            Value::Str(s) => ValueKey::Str(s.clone()),
            Value::F64(f) => ValueKey::F64(f.to_bits()),
            Value::I64(i) => ValueKey::I64(*i),
            Value::Bool(b) => ValueKey::Bool(*b),
        }
    }
}

#[derive(Clone, Debug)]
pub struct FeatureRec {
    pub id: Option<u64>,
    pub geom_type: u8,
    pub geom: Vec<u8>,
    pub tags: Vec<(u32, u32)>,
}

#[derive(Clone)]
pub struct LayerBuilder {
    pub name: String,
    pub extent: u32,
    keys: Vec<String>,
    key_idx: HashMap<String, u32>,
    values: Vec<Value>,
    value_idx: HashMap<ValueKey, u32>,
    pub features: Vec<FeatureRec>,
    pub feature_count: usize,
}

impl LayerBuilder {
    pub fn new(name: &str, extent: u32) -> Self {
        LayerBuilder {
            name: name.to_string(),
            extent,
            keys: Vec::new(),
            key_idx: HashMap::new(),
            values: Vec::new(),
            value_idx: HashMap::new(),
            features: Vec::new(),
            feature_count: 0,
        }
    }

    pub fn key(&mut self, k: &str) -> u32 {
        if let Some(i) = self.key_idx.get(k) {
            return *i;
        }
        let i = self.keys.len() as u32;
        self.keys.push(k.to_string());
        self.key_idx.insert(k.to_string(), i);
        i
    }

    pub fn value(&mut self, v: &Value) -> u32 {
        let k = ValueKey::of(v);
        if let Some(i) = self.value_idx.get(&k) {
            return *i;
        }
        let i = self.values.len() as u32;
        self.values.push(v.clone());
        self.value_idx.insert(k, i);
        i
    }

    /// Add a feature whose geometry is in tile coordinates (rounded at
    /// encode time). `tags` are (key index, value index) pairs.
    pub fn add_feature(&mut self, id: Option<u64>, geom: &Geometry, tags: &[(u32, u32)]) -> usize {
        let g = encode_geometry(geom);
        let n = g.len() + tags.len() * 4 + 8;
        self.features.push(FeatureRec { id, geom_type: geom.kind as u8, geom: g, tags: tags.to_vec() });
        self.feature_count += 1;
        n
    }

    fn encode_feature(f: &FeatureRec, out: &mut Vec<u8>) {
        out.clear();
        if let Some(id) = f.id {
            put_varint_field(out, 1, id);
        }
        if !f.tags.is_empty() {
            let mut t = Vec::with_capacity(f.tags.len() * 4);
            for (k, v) in &f.tags {
                put_varint(&mut t, *k as u64);
                put_varint(&mut t, *v as u64);
            }
            put_bytes_field(out, 2, &t);
        }
        put_varint_field(out, 3, f.geom_type as u64);
        put_bytes_field(out, 4, &f.geom);
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut l = Vec::with_capacity(1024);
        put_varint_field(&mut l, 15, 2);
        put_bytes_field(&mut l, 1, self.name.as_bytes());
        let mut fb = Vec::with_capacity(256);
        for f in &self.features {
            Self::encode_feature(f, &mut fb);
            put_bytes_field(&mut l, 2, &fb);
        }
        for k in &self.keys {
            put_bytes_field(&mut l, 3, k.as_bytes());
        }
        for v in &self.values {
            put_bytes_field(&mut l, 4, &v.encode());
        }
        put_varint_field(&mut l, 5, self.extent as u64);
        l
    }
}

pub struct TileWriter {
    layers: Vec<Vec<u8>>,
}

impl TileWriter {
    pub fn new() -> Self {
        TileWriter { layers: Vec::new() }
    }
    pub fn add_layer(&mut self, l: &LayerBuilder) {
        self.layers.push(l.encode());
    }
    pub fn finish(self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.layers.iter().map(|l| l.len() + 6).sum());
        for l in &self.layers {
            put_bytes_field(&mut out, 3, l);
        }
        out
    }
}

impl Default for TileWriter {
    fn default() -> Self {
        Self::new()
    }
}

const MOVE_TO: u32 = 1;
const LINE_TO: u32 = 2;
const CLOSE_PATH: u32 = 7;

#[inline]
fn cmd(id: u32, count: u32) -> u64 {
    ((id & 0x7) | (count << 3)) as u64
}

/// Encode geometry commands; coordinates are rounded to integers here.
pub fn encode_geometry(geom: &Geometry) -> Vec<u8> {
    let mut out = Vec::with_capacity(geom.vertex_count() * 3 + 8);
    let mut cx = 0i64;
    let mut cy = 0i64;
    let mut emit = |out: &mut Vec<u8>, p: &[f64; 2]| {
        let x = p[0].round() as i64;
        let y = p[1].round() as i64;
        put_varint(out, zigzag(x - cx));
        put_varint(out, zigzag(y - cy));
        cx = x;
        cy = y;
    };
    match geom.kind {
        GeomKind::Point => {
            let pts: Vec<&[f64; 2]> = geom.parts.iter().flatten().collect();
            put_varint(&mut out, cmd(MOVE_TO, pts.len() as u32));
            for p in pts {
                emit(&mut out, p);
            }
        }
        GeomKind::Line => {
            for part in &geom.parts {
                if part.len() < 2 {
                    continue;
                }
                put_varint(&mut out, cmd(MOVE_TO, 1));
                emit(&mut out, &part[0]);
                put_varint(&mut out, cmd(LINE_TO, (part.len() - 1) as u32));
                for p in &part[1..] {
                    emit(&mut out, p);
                }
            }
        }
        GeomKind::Polygon => {
            for i in 0..geom.n_parts() {
                for (ri, ring) in geom.poly_rings(i).iter().enumerate() {
                    let open: &[[f64; 2]] =
                        if ring.len() > 1 && ring[0] == *ring.last().unwrap() { &ring[..ring.len() - 1] } else { ring };
                    if open.len() < 3 {
                        continue;
                    }
                    // In tile coords (y down) MVT exterior rings must have
                    // positive shoelace area (clockwise on screen).
                    let area = signed_area(open);
                    let want_positive = ri == 0;
                    let reversed: Vec<[f64; 2]>;
                    let pts: &[[f64; 2]] = if (area > 0.0) != want_positive {
                        reversed = open.iter().rev().copied().collect();
                        &reversed
                    } else {
                        open
                    };
                    put_varint(&mut out, cmd(MOVE_TO, 1));
                    emit(&mut out, &pts[0]);
                    put_varint(&mut out, cmd(LINE_TO, (pts.len() - 1) as u32));
                    for p in &pts[1..] {
                        emit(&mut out, p);
                    }
                    put_varint(&mut out, cmd(CLOSE_PATH, 1));
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_a_layer_with_tags() {
        let mut layer = LayerBuilder::new("layer0", 4096);
        let k = layer.key("name");
        let v = layer.value(&Value::Str("A".into()));
        let poly = Geometry {
            kind: GeomKind::Polygon,
            parts: vec![vec![[0., 0.], [10., 0.], [10., 10.], [0., 10.], [0., 0.]]],
            polys: vec![0],
        };
        layer.add_feature(None, &poly, &[(k, v)]);
        let mut w = TileWriter::new();
        w.add_layer(&layer);
        let bytes = w.finish();
        assert!(bytes.len() > 20);
        assert_eq!(layer.feature_count, 1);
    }
}
