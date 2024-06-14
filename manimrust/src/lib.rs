use pyo3::prelude::*;
mod cairo;
use crate::cairo::CairoRenderer;

fn register_child_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    // access this as import manimrust.cairo.CairoRenderer
    let py = parent_module.py();
    let cairo = PyModule::new_bound(py, "cairo")?;
    cairo.add_class::<CairoRenderer>()?;
    parent_module.add_submodule(&cairo)?;
    let sys = PyModule::import_bound(py, "sys")?;
    sys.getattr("modules")?.set_item("manimrust.cairo", cairo)?;
    Ok(())
}

/// Parts of manim that need to do heavy lifting are implemented in this
/// library using Rust. See the ``manimrust`` directory
#[pymodule]
fn manimrust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_child_module(m)?;
    Ok(())
}
