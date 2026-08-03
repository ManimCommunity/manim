import numpy as np
import pytest

from manim import ImageMobject


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_invert_image(dtype):
    rng = np.random.default_rng()
    array = (255 * rng.random((10, 10, 4))).astype(dtype)
    image = ImageMobject(array, pixel_array_dtype=dtype, invert=True)
    assert image.pixel_array.dtype == dtype

    array[:, :, :3] = np.iinfo(dtype).max - array[:, :, :3]
    assert np.allclose(array, image.pixel_array)
