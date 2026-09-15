//! `starlet_core`: optional Rust acceleration for starlet.
//!
//! Exposes a `Dataset` that generates Mapbox Vector Tiles directly from a
//! starlet dataset directory (`parquet_tiles/`), single tiles for on-the-fly
//! serving and batches of tiles in parallel (rayon, no GIL) for pyramid
//! generation. Output semantics follow starlet's Python pipeline.

pub mod geom;
pub mod mvt;
pub mod pq;
pub mod pyramid;
pub mod tiler;

use std::path::Path;
use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;

use geom::TileId;
use tiler::Params;

fn to_py_err(e: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// A starlet dataset opened for tile generation.
#[pyclass(name = "Dataset", module = "starlet_core")]
pub struct PyDataset {
    inner: Arc<tiler::Dataset>,
}

#[pymethods]
impl PyDataset {
    /// Open `<path>/parquet_tiles`. `rg_cache` is the number of decoded row
    /// groups kept in memory across tile requests.
    #[new]
    #[pyo3(signature = (path, rg_cache = 256))]
    fn new(path: &str, rg_cache: usize) -> PyResult<Self> {
        let ds = tiler::Dataset::open(Path::new(path), rg_cache).map_err(to_py_err)?;
        Ok(PyDataset { inner: Arc::new(ds) })
    }

    /// Number of partition files.
    #[getter]
    fn num_partitions(&self) -> usize {
        self.inner.parts.len()
    }

    /// Whether every partition carries starlet's per-row bbox columns
    /// (the fast pruning path; without them rows are filtered by WKB bbox).
    #[getter]
    fn has_bbox_columns(&self) -> bool {
        !self.inner.parts.is_empty() && self.inner.parts.iter().all(|p| p.has_bbox_cols)
    }

    /// Generate one tile; returns the MVT bytes. Releases the GIL.
    #[pyo3(signature = (z, x, y, feature_capacity = 25000, extent = 4096, buffer = 256))]
    fn generate_tile<'py>(
        &self,
        py: Python<'py>,
        z: u8,
        x: u32,
        y: u32,
        feature_capacity: usize,
        extent: u32,
        buffer: u32,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let inner = self.inner.clone();
        let p = Params { feature_capacity, extent, buffer };
        let bytes = py
            .allow_threads(move || inner.generate(TileId::new(z, x, y), &p))
            .map_err(to_py_err)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Generate one tile and return `(bytes, stats_dict)` for diagnostics.
    #[pyo3(signature = (z, x, y, feature_capacity = 25000, extent = 4096, buffer = 256))]
    fn generate_tile_with_stats<'py>(
        &self,
        py: Python<'py>,
        z: u8,
        x: u32,
        y: u32,
        feature_capacity: usize,
        extent: u32,
        buffer: u32,
    ) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, pyo3::types::PyDict>)> {
        let inner = self.inner.clone();
        let p = Params { feature_capacity, extent, buffer };
        let (bytes, st) = py
            .allow_threads(move || inner.generate_with_stats(TileId::new(z, x, y), &p))
            .map_err(to_py_err)?;
        let d = pyo3::types::PyDict::new(py);
        d.set_item("partitions_total", st.partitions_total)?;
        d.set_item("partitions_read", st.partitions_read)?;
        d.set_item("row_groups_total", st.row_groups_total)?;
        d.set_item("row_groups_read", st.row_groups_read)?;
        d.set_item("candidates", st.candidates)?;
        d.set_item("retained", st.retained)?;
        Ok((PyBytes::new(py, &bytes), d))
    }

    /// Generate many tiles in parallel (rayon, GIL released). Returns a list
    /// aligned with `tiles`: MVT bytes, or `None` for tiles with no features.
    #[pyo3(signature = (tiles, feature_capacity = 25000, extent = 4096, buffer = 256))]
    fn generate_tiles<'py>(
        &self,
        py: Python<'py>,
        tiles: Vec<(u8, u32, u32)>,
        feature_capacity: usize,
        extent: u32,
        buffer: u32,
    ) -> PyResult<Vec<Option<Bound<'py, PyBytes>>>> {
        let inner = self.inner.clone();
        let p = Params { feature_capacity, extent, buffer };
        let results: Vec<anyhow::Result<(Vec<u8>, u64)>> = py.allow_threads(move || {
            tiles
                .par_iter()
                .map(|&(z, x, y)| inner.generate_with_stats(TileId::new(z, x, y), &p).map(|(b, s)| (b, s.retained)))
                .collect()
        });
        results
            .into_iter()
            .map(|r| {
                r.map(|(b, n)| if n == 0 { None } else { Some(PyBytes::new(py, &b)) })
                    .map_err(to_py_err)
            })
            .collect()
    }

    /// Generate many tiles in parallel and write each non-empty one to
    /// `<outdir>/<z>/<x>/<y>.mvt` (starlet's layout). Nothing crosses the GIL
    /// but the count of tiles written. Returns that count.
    #[pyo3(signature = (outdir, tiles, feature_capacity = 25000, extent = 4096, buffer = 256))]
    fn write_tiles(
        &self,
        py: Python<'_>,
        outdir: &str,
        tiles: Vec<(u8, u32, u32)>,
        feature_capacity: usize,
        extent: u32,
        buffer: u32,
    ) -> PyResult<usize> {
        let inner = self.inner.clone();
        let p = Params { feature_capacity, extent, buffer };
        let out = std::path::PathBuf::from(outdir);
        let written: anyhow::Result<usize> = py.allow_threads(move || {
            let counts: Vec<anyhow::Result<usize>> = tiles
                .par_iter()
                .map(|&(z, x, y)| {
                    let (bytes, st) = inner.generate_with_stats(TileId::new(z, x, y), &p)?;
                    if st.retained == 0 {
                        return Ok(0);
                    }
                    let dir = out.join(z.to_string()).join(x.to_string());
                    std::fs::create_dir_all(&dir)?;
                    std::fs::write(dir.join(format!("{y}.mvt")), &bytes)?;
                    Ok(1)
                })
                .collect();
            let mut n = 0;
            for c in counts {
                n += c?;
            }
            Ok(n)
        });
        written.map_err(to_py_err)
    }

    /// Generate a whole pyramid (any mix of zooms) with bounded memory —
    /// two streaming passes over the row groups instead of per-tile pulls —
    /// writing each non-empty tile to `<outdir>/<z>/<x>/<y>.mvt` as soon as
    /// it is complete. Returns a stats dict (`tiles_written`, `candidates`,
    /// `features`, `row_groups`, `tiles_requested`). Releases the GIL.
    #[pyo3(signature = (outdir, tiles, feature_capacity = 25000, extent = 4096, buffer = 256))]
    fn write_pyramid<'py>(
        &self,
        py: Python<'py>,
        outdir: &str,
        tiles: Vec<(u8, u32, u32)>,
        feature_capacity: usize,
        extent: u32,
        buffer: u32,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let inner = self.inner.clone();
        let p = Params { feature_capacity, extent, buffer };
        let out = std::path::PathBuf::from(outdir);
        let st = py
            .allow_threads(move || {
                let ids: Vec<TileId> = tiles.iter().map(|&(z, x, y)| TileId::new(z, x, y)).collect();
                inner.write_pyramid(&out, &ids, &p)
            })
            .map_err(to_py_err)?;
        let d = pyo3::types::PyDict::new(py);
        d.set_item("tiles_requested", st.tiles_requested)?;
        d.set_item("tiles_written", st.tiles_written)?;
        d.set_item("row_groups", st.row_groups)?;
        d.set_item("candidates", st.candidates)?;
        d.set_item("features", st.features)?;
        Ok(d)
    }
}

/// Number of worker threads rayon will use.
#[pyfunction]
fn num_threads() -> usize {
    rayon::current_num_threads()
}

#[pymodule]
fn starlet_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDataset>()?;
    m.add_function(wrap_pyfunction!(num_threads, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
