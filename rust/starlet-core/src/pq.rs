//! Parquet/Arrow helpers for starlet's `parquet_tiles/` layout: cached
//! footers, projected row-group reads, and row-group pruning on starlet's
//! per-row bbox columns (`_bbox_xmin`, `_bbox_ymin`, `_bbox_xmax`, `_bbox_ymax`).

use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use arrow::array::{Array, ArrayRef, AsArray};
use arrow::datatypes::*;
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder};
use parquet::arrow::ProjectionMask;
use parquet::file::statistics::Statistics;

use crate::mvt::Value;

pub const BBOX_COLS: [&str; 4] = ["_bbox_xmin", "_bbox_ymin", "_bbox_xmax", "_bbox_ymax"];
pub const INTERNAL_COLS: [&str; 5] = ["_tile_id", "_bbox_xmin", "_bbox_ymin", "_bbox_xmax", "_bbox_ymax"];

/// Geometry column of a partition schema: `geometry` when present, else the
/// GeoParquet `primary_column`, else the first binary column that is not one
/// of starlet's internal columns, else the last non-internal column. The bbox
/// covering columns come last, so "the last column" alone would pick
/// `_bbox_ymax` for datasets whose geometry column has another name.
pub fn geometry_column(schema: &Schema) -> Option<String> {
    let fields = schema.fields();
    if fields.iter().any(|f| f.name() == "geometry") {
        return Some("geometry".to_string());
    }
    if let Some(geo) = schema.metadata().get("geo") {
        if let Some(primary) = json_string_field(geo, "primary_column") {
            if fields.iter().any(|f| f.name() == &primary) {
                return Some(primary);
            }
        }
    }
    let external = |f: &&FieldRef| !INTERNAL_COLS.contains(&f.name().as_str());
    if let Some(f) = fields.iter().filter(external).find(|f| {
        matches!(f.data_type(), DataType::Binary | DataType::LargeBinary | DataType::BinaryView)
    }) {
        return Some(f.name().clone());
    }
    fields.iter().filter(external).last().map(|f| f.name().clone())
}

/// The string value of top-level key `key` in a small JSON document
/// (enough for GeoParquet's `primary_column`; no serde dependency).
fn json_string_field(json: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let rest = &json[json.find(&needle)? + needle.len()..];
    let rest = rest.trim_start().strip_prefix(':')?.trim_start().strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Cached parquet footer + schema for one partition file.
#[derive(Clone)]
pub struct PqFile {
    pub path: PathBuf,
    pub meta: ArrowReaderMetadata,
}

impl PqFile {
    pub fn open(path: &Path) -> Result<PqFile> {
        let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let meta = ArrowReaderMetadata::load(&f, ArrowReaderOptions::new())?;
        Ok(PqFile { path: path.to_path_buf(), meta })
    }

    pub fn schema(&self) -> SchemaRef {
        self.meta.schema().clone()
    }

    pub fn num_row_groups(&self) -> usize {
        self.meta.metadata().num_row_groups()
    }

    pub fn has_column(&self, name: &str) -> bool {
        self.meta.schema().index_of(name).is_ok()
    }

    fn leaf_index(&self, name: &str) -> Option<usize> {
        self.meta.parquet_schema().columns().iter().position(|c| c.name() == name)
    }

    /// (min, max) statistics of a Float64 column in row group `rg`.
    pub fn f64_stats(&self, rg: usize, name: &str) -> Option<(f64, f64)> {
        let idx = self.leaf_index(name)?;
        let col = self.meta.metadata().row_group(rg).column(idx);
        match col.statistics()? {
            Statistics::Double(s) => Some((*s.min_opt()?, *s.max_opt()?)),
            Statistics::Float(s) => Some((*s.min_opt()? as f64, *s.max_opt()? as f64)),
            _ => None,
        }
    }

    /// Row groups whose bbox statistics overlap `q` (`[minx, miny, maxx, maxy]`,
    /// same CRS as the columns). Row groups without statistics are kept.
    pub fn prune_bbox(&self, q: &[f64; 4]) -> Vec<usize> {
        (0..self.num_row_groups())
            .filter(|&rg| {
                match (
                    self.f64_stats(rg, BBOX_COLS[0]),
                    self.f64_stats(rg, BBOX_COLS[1]),
                    self.f64_stats(rg, BBOX_COLS[2]),
                    self.f64_stats(rg, BBOX_COLS[3]),
                ) {
                    (Some(xmin), Some(ymin), Some(xmax), Some(ymax)) => {
                        // overlap iff row's xmax >= q.minx && row's xmin <= q.maxx (etc.)
                        !(xmax.1 < q[0] || xmin.0 > q[2] || ymax.1 < q[1] || ymin.0 > q[3])
                    }
                    _ => true,
                }
            })
            .collect()
    }

    /// Read selected row groups with a column projection by name.
    pub fn read(&self, row_groups: Vec<usize>, columns: &[&str], batch_size: usize) -> Result<Vec<RecordBatch>> {
        if row_groups.is_empty() {
            return Ok(Vec::new());
        }
        let f = File::open(&self.path)?;
        let schema = self.schema();
        let mut indices = Vec::with_capacity(columns.len());
        for c in columns {
            let i = schema.index_of(c).map_err(|_| anyhow!("column {c} not in {}", self.path.display()))?;
            indices.push(i);
        }
        let mask = ProjectionMask::roots(self.meta.parquet_schema(), indices);
        let reader = ParquetRecordBatchReaderBuilder::new_with_metadata(f, self.meta.clone())
            .with_projection(mask)
            .with_row_groups(row_groups)
            .with_batch_size(batch_size)
            .build()?;
        let mut out = Vec::new();
        for b in reader {
            out.push(b?);
        }
        Ok(out)
    }
}

/// Scalar MVT value from an Arrow cell, mirroring starlet's `_property_value`
/// (str/int/float/bool pass through; anything else becomes its string form).
#[inline]
pub fn arrow_value(arr: &ArrayRef, row: usize) -> Option<Value> {
    if arr.is_null(row) {
        return None;
    }
    Some(match arr.data_type() {
        DataType::Utf8 => Value::Str(arr.as_string::<i32>().value(row).to_string()),
        DataType::LargeUtf8 => Value::Str(arr.as_string::<i64>().value(row).to_string()),
        DataType::Utf8View => Value::Str(arr.as_string_view().value(row).to_string()),
        DataType::Int64 => Value::I64(arr.as_primitive::<Int64Type>().value(row)),
        DataType::Int32 => Value::I64(arr.as_primitive::<Int32Type>().value(row) as i64),
        DataType::Int16 => Value::I64(arr.as_primitive::<Int16Type>().value(row) as i64),
        DataType::Int8 => Value::I64(arr.as_primitive::<Int8Type>().value(row) as i64),
        DataType::UInt64 => Value::I64(arr.as_primitive::<UInt64Type>().value(row) as i64),
        DataType::UInt32 => Value::I64(arr.as_primitive::<UInt32Type>().value(row) as i64),
        DataType::UInt16 => Value::I64(arr.as_primitive::<UInt16Type>().value(row) as i64),
        DataType::UInt8 => Value::I64(arr.as_primitive::<UInt8Type>().value(row) as i64),
        DataType::Float64 => Value::F64(arr.as_primitive::<Float64Type>().value(row)),
        DataType::Float32 => Value::F64(arr.as_primitive::<Float32Type>().value(row) as f64),
        DataType::Boolean => Value::Bool(arr.as_boolean().value(row)),
        _ => Value::Str(arrow::util::display::array_value_to_string(arr, row).ok()?),
    })
}

/// Raw WKB bytes of a (Large)Binary cell.
#[inline]
pub fn wkb_at(b: &RecordBatch, col: usize, i: usize) -> Option<&[u8]> {
    let arr = b.column(col);
    if arr.is_null(i) {
        return None;
    }
    match arr.data_type() {
        DataType::Binary => Some(arr.as_binary::<i32>().value(i)),
        DataType::LargeBinary => Some(arr.as_binary::<i64>().value(i)),
        DataType::BinaryView => Some(arr.as_binary_view().value(i)),
        _ => None,
    }
}

/// Parse `tile_XXXXXX__minx_miny_maxx_maxy.parquet` where each coordinate is
/// an `int_decimal` pair (e.g. `-97_123` -> -97.123), as starlet writes them.
pub fn parse_filename_bbox(fname: &str) -> Option<[f64; 4]> {
    let stem = fname.strip_suffix(".parquet")?;
    let coord = stem.split("__").nth(1)?;
    let toks: Vec<&str> = coord.split('_').collect();
    if toks.len() != 8 {
        return None;
    }
    let mut nums = [0f64; 4];
    for (k, pair) in toks.chunks(2).enumerate() {
        nums[k] = format!("{}.{}", pair[0], pair[1]).parse().ok()?;
    }
    Some(nums)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_starlet_filename_bbox() {
        let b = parse_filename_bbox("tile_000012__-97_123_33_4_-96_5_34_0.parquet").unwrap();
        assert_eq!(b, [-97.123, 33.4, -96.5, 34.0]);
        assert!(parse_filename_bbox("nope.parquet").is_none());
    }
}

#[cfg(test)]
mod geometry_column_tests {
    use super::*;
    use std::collections::HashMap;

    fn schema(names: &[(&str, DataType)], geo: Option<&str>) -> Schema {
        let fields: Vec<Field> = names.iter().map(|(n, t)| Field::new(*n, t.clone(), true)).collect();
        let mut md = HashMap::new();
        if let Some(g) = geo {
            md.insert("geo".to_string(), g.to_string());
        }
        Schema::new(fields).with_metadata(md)
    }

    #[test]
    fn geometry_column_prefers_geometry_then_geo_metadata_then_binary() {
        let bbox = |v: &mut Vec<(&'static str, DataType)>| {
            for c in BBOX_COLS {
                v.push((c, DataType::Float64));
            }
        };
        let mut plain = vec![("id", DataType::Int64), ("geometry", DataType::Binary)];
        bbox(&mut plain);
        assert_eq!(geometry_column(&schema(&plain, None)).as_deref(), Some("geometry"));

        let mut named = vec![("id", DataType::Int64), ("tags", DataType::Utf8), ("wkb_geometry", DataType::Binary)];
        bbox(&mut named);
        let geo = r#"{"version": "1.1.0", "primary_column": "wkb_geometry", "columns": {}}"#;
        assert_eq!(geometry_column(&schema(&named, Some(geo))).as_deref(), Some("wkb_geometry"));
        // no metadata: the binary column, never the trailing bbox column
        assert_eq!(geometry_column(&schema(&named, None)).as_deref(), Some("wkb_geometry"));

        let mut none = vec![("id", DataType::Int64), ("x", DataType::Utf8)];
        bbox(&mut none);
        assert_eq!(geometry_column(&schema(&none, None)).as_deref(), Some("x"));
    }

    #[test]
    fn json_string_field_reads_primary_column() {
        assert_eq!(json_string_field(r#"{"a": 1, "primary_column" : "g"}"#, "primary_column").as_deref(), Some("g"));
        assert_eq!(json_string_field(r#"{"a": 1}"#, "primary_column"), None);
    }
}
