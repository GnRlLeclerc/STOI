use pyo3::prelude::*;

/// Python bindings for fast-stoi
#[pymodule]
mod fast_stoi {
    use numpy::ndarray::parallel::prelude::*;
    use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::{exceptions::PyWarning, prelude::*};

    #[pyfunction]
    fn stoi(
        x: PyReadonlyArray1<'_, f32>,
        y: PyReadonlyArray1<'_, f32>,
        fs_sig: usize,
        extended: bool,
    ) -> PyResult<f32> {
        match lib_fast_stoi::stoi(
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
    fn par_stoi<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
        y: PyReadonlyArray2<'_, f32>,
        fs_sig: usize,
        extended: bool,
    ) -> Bound<'py, PyArray1<f32>> {
        let x = x.as_array();
        let y = y.as_array();

        x.outer_iter()
            .into_par_iter()
            .zip(y.outer_iter().into_par_iter())
            .map(|(x, y)| {
                match lib_fast_stoi::stoi(
                    x.as_slice().expect("x is not contiguous"),
                    y.as_slice().expect("y is not contiguous"),
                    fs_sig,
                    extended,
                ) {
                    Ok(value) => value,
                    Err(_) => 1e-5,
                }
            })
            .collect::<Vec<_>>()
            .into_pyarray(py)
    }
}
