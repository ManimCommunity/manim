from __future__ import annotations

from abc import ABC, abstractmethod


class WindowABC(ABC):
    is_closing: bool

    @abstractmethod
    def swap_buffers(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def clear(self) -> None: ...
