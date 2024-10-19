from manim import *


class AnimatableNumberLine(NumberLine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_range_tracker = ValueTracker(self.x_range)

    def get_x_range(self):
        return self.x_range_tracker.get_value()

    def set_x_range(self, new_range):
        self.x_range_tracker.set_value(new_range)
        self.x_range = new_range
        self.x_min, self.x_max, self.x_step = new_range
        self.rebuild()

    def rebuild(self):
        self.clear()
        self.__init__(x_range=self.get_x_range(), length=self.length,
                      include_ticks=self.include_ticks, include_numbers=self.include_numbers,
                      **{k: v for k, v in self.__dict__.items() if
                         k not in ['x_range', 'length', 'include_ticks', 'include_numbers']})


class ChangeNumberLineRange(Animation):
    def __init__(self, number_line, new_range, **kwargs):
        self.new_range = new_range
        super().__init__(number_line, **kwargs)

    def interpolate_mobject(self, alpha):
        current_range = [
            interpolate(start, end, alpha)
            for start, end in zip(self.mobject.get_x_range(), self.new_range)
        ]
        self.mobject.set_x_range(current_range)


class AnimatableNumberPlane(NumberPlane):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_range_tracker = ValueTracker(self.x_range)
        self.y_range_tracker = ValueTracker(self.y_range)

    def get_x_range(self):
        return self.x_range_tracker.get_value()

    def get_y_range(self):
        return self.y_range_tracker.get_value()

    def set_x_range(self, new_range):
        self.x_range_tracker.set_value(new_range)
        self.x_range = new_range
        self.rebuild()

    def set_y_range(self, new_range):
        self.y_range_tracker.set_value(new_range)
        self.y_range = new_range
        self.rebuild()

    def rebuild(self):
        self.clear()
        self.__init__(x_range=self.get_x_range(), y_range=self.get_y_range(),
                      **{k: v for k, v in self.__dict__.items() if k not in ['x_range', 'y_range']})

    def get_lines(self):
        x_lines = VGroup()
        y_lines = VGroup()
        x_min, x_max = self.x_range[:2]
        y_min, y_max = self.y_range[:2]
        for x in self.get_x_axis().get_tick_range():
            if x_min <= x <= x_max:
                x_lines.add(self.get_vertical_line(self.c2p(x, 0)))
        for y in self.get_y_axis().get_tick_range():
            if y_min <= y <= y_max:
                y_lines.add(self.get_horizontal_line(self.c2p(0, y)))
        return VGroup(x_lines, y_lines)


class ChangeNumberPlaneRange(Animation):
    def __init__(self, number_plane, new_x_range=None, new_y_range=None, **kwargs):
        self.new_x_range = new_x_range
        self.new_y_range = new_y_range
        super().__init__(number_plane, **kwargs)

    def interpolate_mobject(self, alpha):
        if self.new_x_range:
            current_x_range = [
                interpolate(start, end, alpha)
                for start, end in zip(self.mobject.get_x_range(), self.new_x_range)
            ]
            self.mobject.set_x_range(current_x_range)
        if self.new_y_range:
            current_y_range = [
                interpolate(start, end, alpha)
                for start, end in zip(self.mobject.get_y_range(), self.new_y_range)
            ]
            self.mobject.set_y_range(current_y_range)