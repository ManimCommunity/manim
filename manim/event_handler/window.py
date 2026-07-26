from __future__ import annotations

from typing import Protocol


class WindowProtocol(Protocol):
    @property
    def is_closing(self) -> bool: ...

    def swap_buffers(self) -> object: ...

    def close(self) -> object: ...

    def clear(self) -> object: ...
