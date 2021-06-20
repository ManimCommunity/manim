from typing import Sequence, Union

import numpy as np

__all__ = [
    "Position",
]


class Position:
    MAX_DIMS = 3

    def __init__(
        self,
        *args: Union[float, int, Sequence[float], "Position"],
        default_val=0,
        dtype=np.float64
    ):
        if args is None or len(args) == 0:
            self.pos = np.array([default_val for _ in range(Position.MAX_DIMS)])
        elif isinstance(args[0], Position):
            self.pos = args[0]()
        else:
            self.pos: np.ndarray = np.array(args, dtype=dtype).flatten()[
                : Position.MAX_DIMS
            ]
        self.default_val = default_val
        self.add_padding()

    def add_padding(self, padding=3):
        zeros_to_pad = padding - len(self.pos)
        if zeros_to_pad <= 0:
            return
        self.pos = np.append(self.pos, [self.default_val] * zeros_to_pad)

    @property
    def x(self) -> float:
        return self.pos[0]

    @property
    def y(self) -> float:
        return self.pos[1]

    @property
    def z(self) -> float:
        return self.pos[2]

    def as_numpy(self):
        return self.pos

    def as_list(self) -> list:
        return [i.item() for i in self.pos]

    def as_tuple(self) -> tuple:
        return tuple(i.item() for i in self.pos)

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.as_numpy()

    def __add__(self, other):
        return Position(self.pos + other.pos)

    def __sub__(self, other):
        return Position(self.pos - other.pos)

    def __mul__(self, other):
        return Position(self.pos * other.pos)

    def __truediv__(self, other):
        return Position(self.pos / other.pos)

    def __floordiv__(self, other):
        return Position(self.pos // other.pos)

    def __mod__(self, other):
        return Position(self.pos % other.pos)
