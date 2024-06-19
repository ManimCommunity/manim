use numpy::ndarray::{s, Array2, ArrayView1, ArrayView2, Axis};
use numpy::PyReadonlyArray2;
use pyo3::intern;
use pyo3::prelude::*;
use std::iter;

/// Check if two points are approximately equal
fn consider_points_equal_2d<T: num_traits::Float>(p1: ArrayView1<T>, p2: ArrayView1<T>) -> bool {
    let rtol = T::from(1e-5).expect("rtol is a float");
    let atol = T::from(1e-6).expect("atol is a float"); // TODO make this based off vmobject
    ((p1[0] - p2[0]).abs() <= atol + rtol * p1[0].abs())
        & ((p1[1] - p2[1]).abs() <= atol + rtol * p1[1].abs())
}

// fn consider_points_equal<T: num_traits::Float>(p1: ArrayView1<T>, p2: ArrayView1<T>) -> bool {
//     p1.iter().zip(p2.iter()).all(|(a, b)| a == b)
// }

fn gen_subpaths_from_points_2d<T: num_traits::Float + std::fmt::Debug>(
    points: ArrayView2<T>,
) -> Vec<Array2<T>> {
    let nppcc = 4;
    let filtered = (nppcc..points.len_of(Axis(0))).step_by(nppcc).filter(|&n| {
        !consider_points_equal_2d(
            points.index_axis(Axis(0), n - 1),
            points.index_axis(Axis(0), n),
        )
    });
    let split_indicies: Vec<usize> = iter::once(0)
        .chain(filtered)
        .chain(iter::once(points.len_of(Axis(0))))
        .collect();

    split_indicies
        .iter()
        .zip(split_indicies.iter().skip(1))
        .filter_map(|(&i1, &i2)| {
            if i2 - i1 >= nppcc {
                let path = points.slice(s![i1..i2, ..]).to_owned();
                return Some(path);
            }
            None
        })
        .collect()
}

fn gen_cubic_bezier_tuples_from_points<T>(points: ArrayView2<T>) -> Vec<Array2<T>>
where
    T: Clone,
{
    let nppcc = 4;
    let remainder = points.len() % nppcc;
    let points = points.slice(s![..points.len_of(Axis(0)) - remainder, ..]);
    (0..points.len_of(Axis(0)))
        .step_by(nppcc)
        .map(|i| points.slice(s![i..i + nppcc, ..]).to_owned())
        .collect()
}

/// The base class for Manim.Camera with --renderer=cairo
#[pyclass(subclass)]
pub struct CairoCamera;

#[pymethods]
impl CairoCamera {
    #[new]
    fn new() -> Self {
        Self
    }

    fn set_cairo_context_path<'py>(
        &self,
        py: Python<'py>,
        ctx: Py<PyAny>,
        _vmobject: Py<PyAny>,
        points: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<()> {
        let points = points.as_array();
        // We assume context is correct because serializing it into the Rust binding cairo::Context is too much work
        ctx.call_method0(py, intern!(py, "new_path"))?;
        let subpaths = gen_subpaths_from_points_2d(points);
        for subpath in subpaths {
            let quads = gen_cubic_bezier_tuples_from_points(subpath.view());
            ctx.call_method0(py, intern!(py, "new_sub_path"))?;
            let start = subpath.index_axis(Axis(0), 0);
            ctx.call_method1(py, intern!(py, "move_to"), (start[0], start[1]))?;
            for bezier_tuples in quads {
                let _p0 = bezier_tuples.index_axis(Axis(0), 0);
                let p1 = bezier_tuples.index_axis(Axis(0), 1);
                let p2 = bezier_tuples.index_axis(Axis(0), 2);
                let p3 = bezier_tuples.index_axis(Axis(0), 3);
                ctx.call_method1(
                    py,
                    intern!(py, "curve_to"),
                    (p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]),
                )?;
            }
            if consider_points_equal_2d(
                subpath.index_axis(Axis(0), 0),
                subpath.index_axis(Axis(0), subpath.len_of(Axis(0)) - 1),
            ) {
                ctx.call_method0(py, intern!(py, "close_path"))?;
            }

        }
        Ok(())
    }
}
