use numpy::PyArray2;
use pyo3::{
    prelude::*,
    types::{PyDict, PyTuple},
};

fn arg<T: ToPyObject>(py: Python<'_>, arg: T) -> Bound<'_, PyTuple> {
    PyTuple::new_bound(py, vec![arg])
}

#[pyclass(unsendable)]
pub struct CairoRenderer {
    camera: Py<PyAny>,
    file_writer: Py<PyAny>,
    animation_hashes: Vec<String>,
    time: f64,
    static_image: Option<PyArray2<f64>>,

    #[pyo3(get)]
    num_plays: u32,
    #[pyo3(set)]
    skip_animations: bool,
    _original_skip_animations: bool,
}

#[pymethods]
impl CairoRenderer {
    #[new]
    fn new(camera: Py<PyAny>, file_writer: Py<PyAny>, skip_animations: bool) -> Self {
        Self {
            camera,
            file_writer,
            skip_animations,
            _original_skip_animations: skip_animations,
            animation_hashes: vec![],
            num_plays: 0,
            time: 0.0,
            static_image: None,
        }
    }

    #[pyo3(signature=(scene, hash, *animations, **kwargs))]
    fn play(
        &mut self,
        py: Python<'_>,
        scene: Py<PyAny>,
        hash: String,
        animations: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        self.file_writer.call_method_bound(
            py,
            "add_partial_movie_file",
            arg(py, hash.clone()),
            None,
        )?;
        self.animation_hashes.push(hash);
        println!("List of the first few animation hashes of the scene: {:#?}", &self.animation_hashes[..5]);

        self.file_writer.call_method_bound(py, "begin_animation", arg(py, !self.skip_animations), None)?;
        scene.call_method_bound(py, "begin_animations", PyTuple::empty_bound(py), None)?;
        Ok(())
    }

    fn __str__(slf: &Bound<'_, Self>) -> PyResult<String> {
        slf.get_type().qualname()
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        slf.get_type().qualname()
    }
}
