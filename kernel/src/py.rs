//! PyO3 bindings. Build with `maturin develop --features python` from kernel/.
//!
//! Deliberately numpy-free: values cross as plain lists/Vec, which is fine for
//! validation and small matrices. Zero-copy numpy views are a Phase 2
//! optimization once the kernel replaces the Phase 1 eval forward.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::bitplane::{dequantize, pack, PackedMatrix};
use crate::gemv::{matvec, matvec_dequant};

#[pyclass(name = "PackedMatrix")]
struct PyPackedMatrix {
    inner: PackedMatrix,
}

#[pymethods]
impl PyPackedMatrix {
    #[getter]
    fn rows(&self) -> usize {
        self.inner.rows
    }

    #[getter]
    fn cols(&self) -> usize {
        self.inner.cols
    }

    #[getter]
    fn bits(&self) -> Vec<u8> {
        self.inner.bits.clone()
    }

    #[getter]
    fn scales(&self) -> Vec<f32> {
        self.inner.scales.clone()
    }

    fn plane_bytes(&self) -> usize {
        self.inner.plane_bytes()
    }

    fn mean_bits(&self) -> f64 {
        self.inner.mean_bits()
    }

    /// Reconstructed quantized weights, row-major.
    fn dequantize(&self) -> Vec<f32> {
        dequantize(&self.inner)
    }

    /// Bit-serial y = M x.
    fn matvec(&self, x: Vec<f32>) -> PyResult<Vec<f32>> {
        if x.len() != self.inner.cols {
            return Err(PyValueError::new_err(format!(
                "x length {} != cols {}",
                x.len(),
                self.inner.cols
            )));
        }
        Ok(matvec(&self.inner, &x))
    }

    /// Reference path (dequantize + f64 dot), for cross-checking.
    fn matvec_dequant(&self, x: Vec<f32>) -> PyResult<Vec<f32>> {
        if x.len() != self.inner.cols {
            return Err(PyValueError::new_err("x length mismatch"));
        }
        Ok(matvec_dequant(&self.inner, &x))
    }
}

/// Quantize + pack a row-major f32 matrix.
#[pyfunction]
fn pack_matrix(weights: Vec<f32>, rows: usize, cols: usize, bits: Vec<u8>) -> PyResult<PyPackedMatrix> {
    pack(&weights, rows, cols, &bits)
        .map(|inner| PyPackedMatrix { inner })
        .map_err(PyValueError::new_err)
}

#[pymodule]
fn myelin_kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPackedMatrix>()?;
    m.add_function(wrap_pyfunction!(pack_matrix, m)?)?;
    m.add("MAX_BITS", crate::bitplane::MAX_BITS)?;
    Ok(())
}
