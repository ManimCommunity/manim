from __future__ import annotations

import av
import numpy as np
import pytest

from manim import BLUE_E, VectorScene, config, tempconfig


@pytest.mark.parametrize(("pixel_width", "pixel_height"), [(160, 90), (144, 96)])
def test_lock_in_faded_grid_preserves_background_and_dimness(
    tmp_path, pixel_width, pixel_height
):
    with tempconfig(
        {
            "renderer": "cairo",
            "pixel_width": pixel_width,
            "pixel_height": pixel_height,
            "background_color": BLUE_E,
            "media_dir": str(tmp_path),
            "write_to_movie": False,
        }
    ):
        contrasts = []
        for dimness, axes_dimness in [(0, 0), (0.7, 0.5), (1, 1)]:
            scene = VectorScene()
            camera = scene.camera
            camera.frame_width = 12
            camera.frame_height = 8
            camera.frame_center = np.array([0.5, -0.25, 0])
            original_background = camera.background.copy()

            scene.lock_in_faded_grid(dimness=dimness, axes_dimness=axes_dimness)

            assert not scene.mobjects
            assert scene.camera.frame_width == 12
            assert scene.camera.frame_height == 8
            np.testing.assert_array_equal(scene.camera.frame_center, [0.5, -0.25, 0])
            frozen_grid = scene.camera.background.copy()
            assert frozen_grid.shape == (pixel_height, pixel_width, 4)

            scene.renderer.update_frame(scene)
            np.testing.assert_array_equal(scene.renderer.get_frame(), frozen_grid)
            contrasts.append(
                np.abs(
                    frozen_grid.astype(float) - original_background.astype(float)
                ).sum()
            )

        assert contrasts[0] > contrasts[1] > contrasts[2]
        assert contrasts[2] == 0


def test_lock_in_faded_grid_can_render_vector_animation(tmp_path):
    class LockedGridScene(VectorScene):
        def construct(self):
            self.lock_in_faded_grid()
            self.frozen_grid = self.camera.background.copy()
            self.add_vector([2, 1])
            self.wait(1 / config.frame_rate)
            self.frame_with_vector = self.renderer.get_frame().copy()
            self.clear()
            self.wait(1 / config.frame_rate)
            self.grid_only_frame = self.renderer.get_frame().copy()

    with tempconfig(
        {
            "renderer": "cairo",
            "pixel_width": 160,
            "pixel_height": 90,
            "frame_rate": 5,
            "media_dir": str(tmp_path),
            "format": "mp4",
            "write_to_movie": True,
            "save_last_frame": False,
            "disable_caching": True,
        }
    ):
        scene = LockedGridScene()
        scene.render()

        assert np.any(scene.frame_with_vector != scene.frozen_grid)
        np.testing.assert_array_equal(scene.grid_only_frame, scene.frozen_grid)
        with av.open(str(scene.renderer.file_writer.final_file_path)) as movie:
            frames = list(movie.decode(video=0))
        assert len(frames) >= 3
        assert all((frame.width, frame.height) == (160, 90) for frame in frames)
        assert any(
            not np.array_equal(frame.to_ndarray(), frames[0].to_ndarray())
            for frame in frames[1:]
        )
