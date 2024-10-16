"""Mobjects representing animated raster images."""

__all__ = ["VideoMobject"]

from ...animation.video import Video
from .image_mobject import ImageMobject


class VideoMobject(ImageMobject):
    """Displays a video as a series of images
    from a numpy array or a file.

    Parameters
    ----------
    filenames_or_arrays
        A sequence of numpy arrays or file names from which each image is loaded.


    Example
    -------
    .. manim:: RandomNoiseVideo

        class RandomNoiseVideo(Scene):
            def construct(self):
                size = (1000, 1000, 3)
                images = [(255 * np.random.rand(*size)).astype("uint8") for _ in range(100)]
                video = VideoMobject(images)

                self.play(GrowFromCenter(video))
                self.play(video.play(run_time=6.0))
                self.play(FadeOut(video))
    """

    def __init__(
        self,
        filenames_or_arrays,
        **kwargs,
    ):
        assert len(filenames_or_arrays) > 0, "Cannot create an empty video"
        self.filenames_or_arrays = filenames_or_arrays
        self._kwargs = kwargs
        super().__init__(self.filenames_or_arrays[0], **kwargs)

    def __len__(self):
        return len(self.filenames_or_arrays)

    def __getitem__(self, index):
        return ImageMobject(self.filenames_or_arrays[index], **self._kwargs)

    def play(self, **kwargs):
        """Create an animation that will play video frames."""
        return Video(self, **kwargs)
