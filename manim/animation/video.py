from .animation import Animation


class Video(Animation):
    """
    Animate a VideoMobject by playing its different frames, in order.

    The actual number of frames played will depend on the interpolation
    resolution, i.e., the frames per seconds and the animation runtime.

    If the interpolation resolution is too small, some frames may be skipped.
    """

    def __init__(self, video_mobject, **kwargs):
        self.video_mobject = video_mobject
        self.index = 0
        self.n_frames = len(self.video_mobject)
        self.dt = 1.0 / self.n_frames
        super().__init__(video_mobject, **kwargs)

    def interpolate_mobject(self, alpha):
        index = int(alpha / self.dt) % self.n_frames

        if index != self.index:
            self.video_mobject.pixel_array = self.video_mobject[index].pixel_array
            self.index = index

        return self
