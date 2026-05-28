//! PyO3 bindings: expose `hammingbird-core` to Python with zero-copy access
//! to a contiguous uint8 numpy array.

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

/// Reject the degenerate `d_bytes == 0` input with a clear message.
/// numpy allows arrays of shape (n, 0); pigeon doesn't.
#[inline]
fn check_d_bytes(d_bytes: usize) -> PyResult<()> {
    if d_bytes == 0 {
        Err(pyo3::exceptions::PyValueError::new_err(
            "hammingbird: d_bytes (shape[1]) must be >= 1",
        ))
    } else {
        Ok(())
    }
}

#[pyfunction]
#[pyo3(signature = (a, k))]
fn find_pairs_self<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, u8>,
    k: u32,
) -> PyResult<Vec<(u32, u32, u32)>> {
    let arr = a.as_array();
    let shape = arr.shape();
    if shape.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("A must be 2D"));
    }
    let n = shape[0];
    let d_bytes = shape[1];
    check_d_bytes(d_bytes)?;
    if !a.is_c_contiguous() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "A must be C-contiguous; pass np.ascontiguousarray(A)",
        ));
    }
    let data: &[u8] = arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("array is not contiguous")
    })?;
    if !hammingbird_core::is_supported(k, d_bytes) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "hammingbird: k ({}) must be < d_bytes ({}). At k >= d_bytes the byte-aligned \
             chunk pigeonhole can miss pairs. For that regime use faiss.IndexBinaryFlat \
             or a bit-level chunk implementation (not yet shipped).",
            k, d_bytes
        )));
    }
    // SAFETY: numpy keeps the underlying buffer alive via the readonly borrow
    // held by `a` for the duration of this function. We release the GIL only
    // for the pure-Rust computation; we do not touch any Python object inside.
    Ok(py.allow_threads(|| hammingbird_core::find_pairs_self(data, n, d_bytes, k)))
}

/// Entropy-aware adaptive chunk planning (Task 4). Bit chunks are chosen
/// from the positive-entropy bits only, balanced across k+1 chunks.
#[pyfunction]
#[pyo3(signature = (a, k))]
fn find_pairs_self_adaptive<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, u8>,
    k: u32,
) -> PyResult<Vec<(u32, u32, u32)>> {
    let arr = a.as_array();
    let shape = arr.shape();
    if shape.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("A must be 2D"));
    }
    let n = shape[0];
    let d_bytes = shape[1];
    if !a.is_c_contiguous() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "A must be C-contiguous",
        ));
    }
    let data: &[u8] = arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("array is not contiguous")
    })?;
    check_d_bytes(d_bytes)?;
    Ok(py.allow_threads(|| hammingbird_core::find_pairs_self_adaptive(data, n, d_bytes, k)))
}

/// **Internal benchmark helper — DO NOT USE in application code.**
///
/// This is the pre-Task-3 implementation that uses a full popcount + compare
/// (no early exit). It exists only so the benchmark suite can quantify the
/// gain from `hamming_le_k` early-exit. Behavior is identical to
/// `find_pairs_self` on supported inputs, just slower. Will be removed
/// without notice in a future release.
#[pyfunction]
#[pyo3(signature = (a, k))]
fn _find_pairs_self_full_popcount<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, u8>,
    k: u32,
) -> PyResult<Vec<(u32, u32, u32)>> {
    let arr = a.as_array();
    let shape = arr.shape();
    if shape.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("A must be 2D"));
    }
    let n = shape[0];
    let d_bytes = shape[1];
    check_d_bytes(d_bytes)?;
    if !a.is_c_contiguous() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "A must be C-contiguous",
        ));
    }
    let data: &[u8] = arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("array is not contiguous")
    })?;
    if !hammingbird_core::is_supported(k, d_bytes) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "k ({}) must be < d_bytes ({}) for byte-level path", k, d_bytes
        )));
    }
    Ok(py.allow_threads(|| {
        hammingbird_core::find_pairs_self_no_dedup_par_prefetch_full(data, n, d_bytes, k)
    }))
}

#[pyfunction]
#[pyo3(signature = (a, b, k))]
fn find_pairs_cross<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, u8>,
    b: PyReadonlyArray2<'py, u8>,
    k: u32,
) -> PyResult<Vec<(u32, u32, u32)>> {
    let a_arr = a.as_array();
    let b_arr = b.as_array();
    let a_shape = a_arr.shape();
    let b_shape = b_arr.shape();
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("A and B must be 2D"));
    }
    let n_a = a_shape[0];
    let n_b = b_shape[0];
    let d_a = a_shape[1];
    let d_b = b_shape[1];
    if d_a != d_b {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "A and B must have the same d_bytes (shape[1])",
        ));
    }
    check_d_bytes(d_a)?;
    if !a.is_c_contiguous() || !b.is_c_contiguous() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "A and B must be C-contiguous; pass np.ascontiguousarray(...)",
        ));
    }
    let a_data: &[u8] = a_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("A array is not contiguous")
    })?;
    let b_data: &[u8] = b_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("B array is not contiguous")
    })?;
    if !hammingbird_core::is_supported(k, d_a) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "hammingbird: k ({}) must be < d_bytes ({}). At k >= d_bytes the byte-aligned \
             chunk pigeonhole can miss pairs. For that regime use faiss.IndexBinaryFlat.",
            k, d_a
        )));
    }
    Ok(py.allow_threads(|| {
        hammingbird_core::find_pairs_cross(a_data, n_a, b_data, n_b, d_a, k)
    }))
}

#[pyfunction]
#[pyo3(signature = (a, b, k))]
fn find_pairs_cross_bit<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, u8>,
    b: PyReadonlyArray2<'py, u8>,
    k: u32,
) -> PyResult<Vec<(u32, u32, u32)>> {
    let a_arr = a.as_array();
    let b_arr = b.as_array();
    let a_shape = a_arr.shape();
    let b_shape = b_arr.shape();
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("A and B must be 2D"));
    }
    let n_a = a_shape[0];
    let n_b = b_shape[0];
    let d_a = a_shape[1];
    let d_b = b_shape[1];
    if d_a != d_b {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "A and B must have the same d_bytes (shape[1])",
        ));
    }
    check_d_bytes(d_a)?;
    if !a.is_c_contiguous() || !b.is_c_contiguous() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "A and B must be C-contiguous; pass np.ascontiguousarray(...)",
        ));
    }
    let a_data: &[u8] = a_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("A array is not contiguous")
    })?;
    let b_data: &[u8] = b_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("B array is not contiguous")
    })?;
    if !hammingbird_core::is_supported_bit(k, d_a) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "hammingbird: k ({}) must be < 8*d_bytes ({}).", k, 8 * d_a
        )));
    }
    Ok(py.allow_threads(|| {
        hammingbird_core::find_pairs_cross_bit(a_data, n_a, b_data, n_b, d_a, k)
    }))
}

#[pyfunction]
#[pyo3(signature = (a, k))]
fn find_pairs_self_bit<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, u8>,
    k: u32,
) -> PyResult<Vec<(u32, u32, u32)>> {
    let arr = a.as_array();
    let shape = arr.shape();
    if shape.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("A must be 2D"));
    }
    let n = shape[0];
    let d_bytes = shape[1];
    check_d_bytes(d_bytes)?;
    if !a.is_c_contiguous() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "A must be C-contiguous; pass np.ascontiguousarray(A)",
        ));
    }
    let data: &[u8] = arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("array is not contiguous")
    })?;
    if !hammingbird_core::is_supported_bit(k, d_bytes) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "hammingbird: k ({}) must be < 8*d_bytes ({})",
            k, 8 * d_bytes
        )));
    }
    Ok(py.allow_threads(|| hammingbird_core::find_pairs_self_bit(data, n, d_bytes, k)))
}

/// Streaming near-duplicate index. Stateful, append-only. Bit-level chunks
/// so any `k < 8 * d_bytes` is supported.
///
/// # Thread safety
///
/// The Index is `Send`: it can be moved across threads and used from a
/// worker pool (Flask/FastAPI, Celery, asyncio threadpool). PyO3 enforces
/// Python-side `&self` / `&mut self` borrow rules:
///
/// - **Concurrent `query()` from multiple threads is safe** — `query` takes
///   `&self`, multiple readers run in parallel without contention.
/// - **A mutating call (`add`, `add_batch`) while another thread holds a
///   `query` reference raises `RuntimeError: Already mutably borrowed`** at
///   the Python level — a safe runtime check, NOT undefined behavior or a
///   crash. Production deployments that interleave add and query should
///   either:
///     1. funnel writes through a single dedicated writer thread, OR
///     2. wrap the index in a `threading.Lock` on the Python side, OR
///     3. accept the occasional `RuntimeError` and retry.
#[pyclass]
struct Index {
    inner: hammingbird_core::Index,
}

#[pymethods]
impl Index {
    #[new]
    #[pyo3(signature = (d_bytes, k))]
    fn new(d_bytes: usize, k: u32) -> PyResult<Self> {
        if d_bytes == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Index: d_bytes must be >= 1",
            ));
        }
        let d_bits = 8 * d_bytes;
        if (k as usize) >= d_bits {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Index: k ({}) must be < 8*d_bytes ({})",
                k, d_bits
            )));
        }
        Ok(Self { inner: hammingbird_core::Index::new(d_bytes, k) })
    }

    /// Add one row (1-D uint8 array of length d_bytes). Returns assigned id.
    fn add<'py>(&mut self, row: PyReadonlyArray1<'py, u8>) -> PyResult<u32> {
        if !row.is_c_contiguous() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "row must be C-contiguous; pass np.ascontiguousarray(row)",
            ));
        }
        let arr = row.as_array();
        if arr.shape()[0] != self.inner.d_bytes() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "row length {} != d_bytes {}",
                arr.shape()[0],
                self.inner.d_bytes()
            )));
        }
        let data: &[u8] = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("row not contiguous")
        })?;
        Ok(self.inner.add(data))
    }

    /// Add a batch (2-D uint8 array, shape (m, d_bytes)). Returns the id of
    /// the first row added, or `None` if the batch was empty.
    fn add_batch<'py>(
        &mut self,
        py: Python<'py>,
        batch: PyReadonlyArray2<'py, u8>,
    ) -> PyResult<Option<u32>> {
        if !batch.is_c_contiguous() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "batch must be C-contiguous; pass np.ascontiguousarray(batch)",
            ));
        }
        let arr = batch.as_array();
        let shape = arr.shape();
        if shape.len() != 2 || shape[1] != self.inner.d_bytes() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "batch must be 2-D with shape[1] == d_bytes ({}), got shape {:?}",
                self.inner.d_bytes(),
                shape
            )));
        }
        let data: &[u8] = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("batch not contiguous")
        })?;
        // Release the GIL for the bulk insert — useful for >10k row batches.
        Ok(py.allow_threads(|| self.inner.add_batch(data)))
    }

    /// Batched query (2-D uint8 array, shape (m, d_bytes)). Returns a list
    /// of result lists, one per input row. Cheaper than looping `query` in
    /// Python because the candidate-set buffer is reused across rows.
    fn query_batch<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<'py, u8>,
    ) -> PyResult<Vec<Vec<(u32, u32)>>> {
        if !queries.is_c_contiguous() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "queries must be C-contiguous; pass np.ascontiguousarray(...)",
            ));
        }
        let arr = queries.as_array();
        let shape = arr.shape();
        if shape.len() != 2 || shape[1] != self.inner.d_bytes() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "queries must be 2-D with shape[1] == d_bytes ({}), got shape {:?}",
                self.inner.d_bytes(),
                shape
            )));
        }
        let data: &[u8] = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("queries not contiguous")
        })?;
        Ok(py.allow_threads(|| self.inner.query_batch(data)))
    }

    /// Query one row (1-D uint8 array of length d_bytes). Returns
    /// `list[(id, hamming_dist)]` for all stored rows within distance ≤ k.
    ///
    /// Note: if the query row was previously added to the index, the
    /// corresponding `(stored_id, 0)` entry IS included in the result.
    /// Callers performing self-query (e.g. iterating over the corpus to
    /// find near-dups of each row) should filter `id != self_id` themselves.
    fn query<'py>(&self, row: PyReadonlyArray1<'py, u8>) -> PyResult<Vec<(u32, u32)>> {
        if !row.is_c_contiguous() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "row must be C-contiguous",
            ));
        }
        let arr = row.as_array();
        if arr.shape()[0] != self.inner.d_bytes() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "row length {} != d_bytes {}",
                arr.shape()[0],
                self.inner.d_bytes()
            )));
        }
        let data: &[u8] = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("row not contiguous")
        })?;
        // query is fast (sub-µs); GIL release overhead would dominate, so keep it held.
        Ok(self.inner.query(data))
    }

    fn __len__(&self) -> usize { self.inner.len() }

    #[getter]
    fn n(&self) -> usize { self.inner.len() }

    #[getter]
    fn d_bytes(&self) -> usize { self.inner.d_bytes() }

    #[getter]
    fn k(&self) -> u32 { self.inner.k() }
}

#[pymodule]
fn _hammingbird(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_pairs_self, m)?)?;
    m.add_function(wrap_pyfunction!(find_pairs_self_bit, m)?)?;
    m.add_function(wrap_pyfunction!(find_pairs_cross, m)?)?;
    m.add_function(wrap_pyfunction!(find_pairs_cross_bit, m)?)?;
    m.add_function(wrap_pyfunction!(_find_pairs_self_full_popcount, m)?)?;
    m.add_function(wrap_pyfunction!(find_pairs_self_adaptive, m)?)?;
    m.add_class::<Index>()?;
    m.add("__version__", "0.5.0")?;
    Ok(())
}
