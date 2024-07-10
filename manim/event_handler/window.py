from __future__ import annotations

from abc import ABC, abstractmethod


class WindowABC(ABC):
    is_closing: bool

    @abstractmethod
    def swap_buffers(self): ...

    @abstractmethod
    def close(self): ...

    @abstractmethod
    def clear(self): ...
