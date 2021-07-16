import math
from typing import Dict, Iterable

import numpy as np

from ..mobject.numbers import DecimalNumber


class _ScaleBase:
    def __init__(self, custom_labels: bool = False):
        self.custom_labels = custom_labels

    def function(self, value: float) -> float:
        raise NotImplementedError

    def inverse_function(self, value: float) -> float:
        raise NotImplementedError

    def get_custom_labels(self, val_range: Iterable[float]) -> Dict:
        raise NotImplementedError


class LinearBase(_ScaleBase):
    def __init__(self, scale_factor: float = 1.0):
        super().__init__()
        self.scale_factor = scale_factor

    def function(self, value):
        return self.scale_factor * value

    def inverse_function(self, value):
        return value / self.scale_factor


class LogBase(_ScaleBase):
    def __init__(self, base: float, custom_labels: bool = True):
        super().__init__()
        self.base = base
        self.custom_labels = custom_labels

    def function(self, value: float) -> float:
        return self.base ** value

    def inverse_function(self, value: float) -> float:
        # if isinstance(value, np.ndarray):
        #     value[value== 0] = 1e-15
        # elif value == 0:
        #     value = 1e-15
        value = np.log(value) / np.log(self.base)
        return value

    def get_custom_labels(self, val_range: Iterable[float]) -> Dict:
        tex_labels = [
            DecimalNumber(
                self.base,
                num_decimal_places=0,
                unit="^{%s}" % int(round(self.inverse_function(i))),
            )
            for i in val_range
        ]
        labels = dict(zip(val_range, tex_labels))
        return labels
