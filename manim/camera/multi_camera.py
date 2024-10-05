"""A camera supporting multiple perspectives."""

from __future__ import annotations

__all__ = ["MultiCamera"]


from manim.camera.moving_camera import MovingCamera
from manim.mobject.mobject import Mobject
from manim.mobject.types.image_mobject import ImageMobjectFromCamera
from manim.utils.iterables import list_difference_update


class MultiCamera(MovingCamera):
    """Subclass of :class:`~.MovingCamera` with the ability to access multiple
    other "subcameras", allowing for multiple perspectives for the same scene.

    In order to "add" a new :class:`~.Camera` to :class:`MultiCamera`, one has to
    create an :class:`~.ImageMobjectFromCamera` from it and add it to the
    :attr:`image_mobjects_from_cameras` list, via the constructor or the
    :meth:`add_image_mobject_from_camera` method. Then, :class:`MultiCamera` can
    access the new subcamera via the :attr:`~.ImageMobjectFromCamera.camera`
    attribute which references its corresponding source.

    .. warning::
        Currently :class:`MultiCamera` does not support 3D perspectives, as it does not
        inherit from :class:`~.ThreeDCamera` which contains the required attributes
        such as rotations and a focal distance.

    Attributes
    ----------
    image_mobjects_from_cameras : List[:class:`~.ImageMobjectFromCamera`]
        A list of instances of :class:`~.ImageMobjectFromCamera`,
        each one created from a different :class:`~.Camera`.

    allow_cameras_to_capture_their_own_display : bool
        When the subcameras capture Mobjects, it is possible that they capture the
        :class:`~.ImageMobjectFromCamera` display generated from themselves.
        If this attribute is ``True``, this display is included into the captured
        Mobjects. Otherwise, it's filtered out.

    Parameters
    ----------
    image_mobjects_from_cameras
        A list of instances of :class:`~.ImageMobjectFromCamera`,
        each one created from a different :class:`~.Camera`.

    allow_cameras_to_capture_their_own_display
        When the subcameras capture Mobjects, it is possible that they capture the
        :class:`~.ImageMobjectFromCamera` generated from themselves.
        If this parameter is ``True``, this display is included into the captured
        Mobjects. Otherwise, it's filtered out. Default is ``False``.

    kwargs
        Any valid keyword arguments for the parent class :class:`~.MovingCamera`.
    """

    def __init__(
        self,
        image_mobjects_from_cameras: ImageMobjectFromCamera
        | Iterable[ImageMobjectFromCamera]
        | None = None,
        allow_cameras_to_capture_their_own_display: bool = False,
        **kwargs,
    ) -> None:
        self.image_mobjects_from_cameras: list[ImageMobjectFromCamera] = []
        if image_mobjects_from_cameras is not None:
            for imfc in image_mobjects_from_cameras:
                self.add_image_mobject_from_camera(imfc)
        self.allow_cameras_to_capture_their_own_display: bool = (
            allow_cameras_to_capture_their_own_display
        )
        super().__init__(**kwargs)

    def add_image_mobject_from_camera(
        self,
        image_mobject_from_camera: ImgMobFromCam,
    ) -> None:
        """Takes an :class:`~.ImageMobjectFromCamera` created from a preexisting :class:`~.Camera`,
        and adds it into the :attr:`image_mobjects_from_cameras` list. In this way, the
        :class:`MultiCamera` can reference that camera through this image mobject, therefore
        considering it as a new "subcamera".

        Parameters
        ----------
        image_mobject_from_camera
            The :class:`~.ImageMobject` to add to :attr:`image_mobjects_from_cameras`.
        """
        # A silly method to have right now, but maybe there are things
        # we want to guarantee about any imfc's added later.
        imfc = image_mobject_from_camera
        assert isinstance(imfc.camera, MovingCamera)
        self.image_mobjects_from_cameras.append(imfc)

    def update_sub_cameras(self):
        """For each one of the subcameras referenced by :attr:`image_mobjects_from_cameras`,
        update its :attr:`frame_shape` and reset its pixel shape."""
        pixel_height, pixel_width = self.pixel_array.shape[:2]
        for imfc in self.image_mobjects_from_cameras:
            imfc.camera.frame_shape = (
                imfc.camera.frame.height,
                imfc.camera.frame.width,
            )
            imfc.camera.reset_pixel_shape(
                int(pixel_height * imfc.height / self.frame_height),
                int(pixel_width * imfc.width / self.frame_width),
            )

    def reset(self) -> Self:
        """Resets each of the subcameras referenced by :attr:`image_mobjects_from_cameras`,
        and then resets the :class:`MultiCamera` itself.

        Returns
        -------
        Self
            The :class:`MultiCamera` itself, after resetting itself and all of its subcameras.
        """
        for imfc in self.image_mobjects_from_cameras:
            imfc.camera.reset()
        super().reset()
        return self

    def capture_mobjects(
        self,
        mobjects: Iterable[Mobject],
        **kwargs,
    ) -> None:
        """Makes all the subcameras capture the :class:`~.Mobject` s passed.
        If any of the :class:`~.Mobject`s is already in the family of any
        :class:`~.ImageMobjectFromCamera` created from any of the subcameras,
        the :attr:`allow_cameras_to_capture_their_own_display` attribute decides
        whether to filter out the :class:`Mobject` for that specific subcamera
        (if ``False``), or allow that subcamera to capture it as well (if ``True``).

        Parameters
        ----------
        mobjects
            :class:`~.Mobject` s to capture by the subcameras.
        """
        self.update_sub_cameras()
        for imfc in self.image_mobjects_from_cameras:
            to_add = list(mobjects)
            if not self.allow_cameras_to_capture_their_own_display:
                to_add = list_difference_update(to_add, imfc.get_family())
            imfc.camera.capture_mobjects(to_add, **kwargs)
        super().capture_mobjects(mobjects, **kwargs)

    def get_mobjects_indicating_movement(self) -> list[Mobject]:
        """Returns all :class:`~.Mobject` s whose movement implies that
        the :class:`MultiCamera` should think of all the other :class:`~.Mobject` s
        on the screen as moving.

        Returns
        -------
        list[Mobject]
            List of :class:`~.Mobject`s indicating movement.
        """
        return [self.frame] + [
            imfc.camera.frame for imfc in self.image_mobjects_from_cameras
        ]
