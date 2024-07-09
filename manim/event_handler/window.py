from abc import ABC, abstractmethod


class WindowABC(ABC):
    is_closing: bool

    @abstractmethod
    def swap_buffers(self): ...

    @abstractmethod
    def close(self): ...
