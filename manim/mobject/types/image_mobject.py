"""Mobjects representing raster images."""

from __future__ import annotations

__all__ = ["AbstractImageMobject", "ImageMobject", "ImageMobjectFromCamera"]

import pathlib

import colour
import numpy as np
from PIL import Image
from PIL.Image import Resampling

from manim.mobject.geometry.shape_matchers import SurroundingRectangle

from ... import config
from ...constants import *
from ...mobject.mobject import Mobject
from ...utils.bezier import interpolate
from ...utils.color import WHITE, color_to_int_rgb
from ...utils.images import change_to_rgba_array, get_full_raster_image_path


class AbstractImageMobject(Mobject):
    """
    Automatically filters out black pixels

    Parameters
    ----------
    scale_to_resolution
        At this resolution the image is placed pixel by pixel onto the screen, so it
        will look the sharpest and best.
        This is a custom parameter of ImageMobject so that rendering a scene with
        e.g. the ``--quality low`` or ``--quality medium`` flag for faster rendering
        won't effect the position of the image on the screen.
    """

    def __init__(
        self,
        scale_to_resolution: int,
        pixel_array_dtype="uint8",
        resampling_algorithm=Resampling.BICUBIC,
        **kwargs,
    ):
        self.pixel_array_dtype = pixel_array_dtype
        self.scale_to_resolution = scale_to_resolution
        self.set_resampling_algorithm(resampling_algorithm)
        super().__init__(**kwargs)

    def get_pixel_array(self):
        raise NotImplementedError()

    def set_color(self, color, alpha=None, family=True):
        # Likely to be implemented in subclasses, but no obligation
        pass

    def set_resampling_algorithm(self, resampling_algorithm: int):
        """
        Sets the interpolation method for upscaling the image. By default the image is
        interpolated using bicubic algorithm. This method lets you change it.
        Interpolation is done internally using Pillow, and the function besides the
        string constants describing the algorithm accepts the Pillow integer constants.

        Parameters
        ----------
        resampling_algorithm
            An integer constant described in the Pillow library,
            or one from the RESAMPLING_ALGORITHMS global dictionary,
            under the following keys:

            * 'bicubic' or 'cubic'
            * 'nearest' or 'none'
            * 'box'
            * 'bilinear' or 'linear'
            * 'hamming'
            * 'lanczos' or 'antialias'
        """
        if isinstance(resampling_algorithm, int):
            self.resampling_algorithm = resampling_algorithm
        else:
            raise ValueError(
                "resampling_algorithm has to be an int, one of the values defined in "
                "RESAMPLING_ALGORITHMS or a Pillow resampling filter constant. "
                "Available algorithms: 'bicubic', 'nearest', 'box', 'bilinear', "
                "'hamming', 'lanczos'.",
            )

    def reset_points(self):
        # Corresponding corners of image are fixed to these 3 points
        self.points = np.array(
            [
                UP + LEFT,
                UP + RIGHT,
                DOWN + LEFT,
            ],
        )
        self.center()
        h, w = self.get_pixel_array().shape[:2]
        if self.scale_to_resolution:
            height = h / self.scale_to_resolution * config["frame_height"]
        else:
            height = 3  # this is the case for ImageMobjectFromCamera
        self.stretch_to_fit_height(height)
        self.stretch_to_fit_width(height * w / h)


class ImageMobject(AbstractImageMobject):
    """Displays an Image from a numpy array or a file.

    Parameters
    ----------
    scale_to_resolution
        At this resolution the image is placed pixel by pixel onto the screen, so it
        will look the sharpest and best.
        This is a custom parameter of ImageMobject so that rendering a scene with
        e.g. the ``--quality low`` or ``--quality medium`` flag for faster rendering
        won't effect the position of the image on the screen.


    Example
    -------
    .. manim:: ImageFromArray
        :save_last_frame:

        class ImageFromArray(Scene):
            def construct(self):
                image = ImageMobject(np.uint8([[0, 100, 30, 200],
                                               [255, 0, 5, 33]]))
                image.height = 7
                self.add(image)


    Changing interpolation style:

    .. manim:: ImageInterpolationEx
        :save_last_frame:

        class ImageInterpolationEx(Scene):
            def construct(self):
                img = ImageMobject(np.uint8([[63, 0, 0, 0],
                                                [0, 127, 0, 0],
                                                [0, 0, 191, 0],
                                                [0, 0, 0, 255]
                                                ]))

                img.height = 2
                img1 = img.copy()
                img2 = img.copy()
                img3 = img.copy()
                img4 = img.copy()
                img5 = img.copy()

                img1.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
                img2.set_resampling_algorithm(RESAMPLING_ALGORITHMS["lanczos"])
                img3.set_resampling_algorithm(RESAMPLING_ALGORITHMS["linear"])
                img4.set_resampling_algorithm(RESAMPLING_ALGORITHMS["cubic"])
                img5.set_resampling_algorithm(RESAMPLING_ALGORITHMS["box"])
                img1.add(Text("nearest").scale(0.5).next_to(img1,UP))
                img2.add(Text("lanczos").scale(0.5).next_to(img2,UP))
                img3.add(Text("linear").scale(0.5).next_to(img3,UP))
                img4.add(Text("cubic").scale(0.5).next_to(img4,UP))
                img5.add(Text("box").scale(0.5).next_to(img5,UP))

                x= Group(img1,img2,img3,img4,img5)
                x.arrange()
                self.add(x)
    """

    def __init__(
        self,
        filename_or_array,
        scale_to_resolution: int = QUALITIES[DEFAULT_QUALITY]["pixel_height"],
        invert=False,
        image_mode="RGBA",
        **kwargs,
    ):
        self.fill_opacity = 1
        self.stroke_opacity = 1
        self.invert = invert
        self.image_mode = image_mode
        if isinstance(filename_or_array, (str, pathlib.PurePath)):
            path = get_full_raster_image_path(filename_or_array)
            image = Image.open(path).convert(self.image_mode)
            self.pixel_array = np.array(image)
            self.path = path
        else:
            self.pixel_array = np.array(filename_or_array)
        self.pixel_array_dtype = kwargs.get("pixel_array_dtype", "uint8")
        self.pixel_array = change_to_rgba_array(
            self.pixel_array, self.pixel_array_dtype
        )
        if self.invert:
            self.pixel_array[:, :, :3] = 255 - self.pixel_array[:, :, :3]
        super().__init__(scale_to_resolution, **kwargs)

    def get_pixel_array(self):
        """A simple getter method."""
        return self.pixel_array

    def set_color(self, color, alpha=None, family=True):
        rgb = color_to_int_rgb(color)
        self.pixel_array[:, :, :3] = rgb
        if alpha is not None:
            self.pixel_array[:, :, 3] = int(255 * alpha)
        for submob in self.submobjects:
            submob.set_color(color, alpha, family)
        self.color = color
        return self

    def set_opacity(self, alpha: float):
        """Sets the image's opacity.

        Parameters
        ----------
        alpha
            The alpha value of the object, 1 being opaque and 0 being
            transparent.
        """
        self.pixel_array[:, :, 3] = int(255 * alpha)
        self.fill_opacity = alpha
        self.stroke_opacity = alpha
        return self

    def fade(self, darkness: float = 0.5, family: bool = True):
        """Sets the image's opacity using a 1 - alpha relationship.

        Parameters
        ----------
        darkness
            The alpha value of the object, 1 being transparent and 0 being
            opaque.
        family
            Whether the submobjects of the ImageMobject should be affected.
        """
        self.set_opacity(1 - darkness)
        super().fade(darkness, family)
        return self

    def interpolate_color(
        self, mobject1: ImageMobject, mobject2: ImageMobject, alpha: float
    ):
        """Interpolates the array of pixel color values from one ImageMobject
        into an array of equal size in the target ImageMobject.

        Parameters
        ----------
        mobject1
            The ImageMobject to transform from.

        mobject2
            The ImageMobject to transform into.

        alpha
            Used to track the lerp relationship. Not opacity related.
        """
        assert mobject1.pixel_array.shape == mobject2.pixel_array.shape, (
            f"Mobject pixel array shapes incompatible for interpolation.\n"
            f"Mobject 1 ({mobject1}) : {mobject1.pixel_array.shape}\n"
            f"Mobject 2 ({mobject2}) : {mobject2.pixel_array.shape}"
        )
        self.fill_opacity = interpolate(
            mobject1.fill_opacity,
            mobject2.fill_opacity,
            alpha,
        )
        self.stroke_opacity = interpolate(
            mobject1.stroke_opacity,
            mobject2.stroke_opacity,
            alpha,
        )
        self.pixel_array = interpolate(
            mobject1.pixel_array,
            mobject2.pixel_array,
            alpha,
        ).astype(self.pixel_array_dtype)

    def get_style(self):
        return {
            "fill_color": colour.rgb2hex(self.color.get_rgb()),
            "fill_opacity": self.fill_opacity,
        }


# TODO, add the ability to have the dimensions/orientation of this
# mobject more strongly tied to the frame of the camera it contains,
# in the case where that's a MovingCamera


class ImageMobjectFromCamera(AbstractImageMobject):
    def __init__(self, camera, default_display_frame_config=None, **kwargs):
        self.camera = camera
        if default_display_frame_config is None:
            default_display_frame_config = {
                "stroke_width": 3,
                "stroke_color": WHITE,
                "buff": 0,
            }
        self.default_display_frame_config = default_display_frame_config
        self.pixel_array = self.camera.pixel_array
        super().__init__(scale_to_resolution=False, **kwargs)

    # TODO: Get rid of this.
    def get_pixel_array(self):
        self.pixel_array = self.camera.pixel_array
        return self.pixel_array

    def add_display_frame(self, **kwargs):
        config = dict(self.default_display_frame_config)
        config.update(kwargs)
        self.display_frame = SurroundingRectangle(self, **config)
        self.add(self.display_frame)
        return self

    def interpolate_color(self, mobject1, mobject2, alpha):
        assert mobject1.pixel_array.shape == mobject2.pixel_array.shape, (
            f"Mobject pixel array shapes incompatible for interpolation.\n"
            f"Mobject 1 ({mobject1}) : {mobject1.pixel_array.shape}\n"
            f"Mobject 2 ({mobject2}) : {mobject2.pixel_array.shape}"
        )
        self.pixel_array = interpolate(
            mobject1.pixel_array,
            mobject2.pixel_array,
            alpha,
        ).astype(self.pixel_array_dtype)
