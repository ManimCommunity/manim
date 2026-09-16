from manim.mobject.abstract.positionable import Positionable


class PositionableWithFamily(Positionable):
    def __init__(self, submobjects: list[Positionable]) -> None:
        super().__init__()
        self.submobjects = submobjects

    def get_family(self) -> list[Positionable]:
        return [self, *(mob for sub in self.submobjects for mob in sub.get_family())]
