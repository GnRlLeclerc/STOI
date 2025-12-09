use pyo3::prelude::*;

/// Python bindings for stoilib
#[pymodule]
mod stoi {
    use numpy::{PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::{exceptions::PyWarning, prelude::*};
    use rayon::iter::{ParallelBridge, ParallelIterator};

    #[pyfunction]
    fn stoi(
        x: PyReadonlyArray1<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        fs_sig: usize,
        extended: bool,
    ) -> PyResult<f64> {
        match stoilib::stoi(
            x.as_slice().expect("x is not contiguous"),
            y.as_slice().expect("y is not contiguous"),
            fs_sig,
            extended,
        ) {
            Ok(value) => Ok(value),
            Err(err) => Err(PyWarning::new_err(err.to_string())),
        }
    }

    #[pyfunction]
    fn par_stoi(
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray2<'_, f64>,
        fs_sig: usize,
        extended: bool,
    ) -> PyResult<Vec<f64>> {
        let x = x.as_array();
        let y = y.as_array();

        Ok(x.outer_iter()
            .zip(y.outer_iter())
            .par_bridge()
            .map(|(x, y)| {
                match stoilib::stoi(
                    x.as_slice().expect("x is not contiguous"),
                    y.as_slice().expect("y is not contiguous"),
                    fs_sig,
                    extended,
                ) {
                    Ok(value) => value,
                    Err(_) => 1e-5,
                }
            })
            .collect())
    }
}
