"""
Abstract base class for several objects used by manim.  In particular, both
:class:`~.Scene` and :class:`~.Mobject` inherit from Container.
"""


__all__ = ["Container"]


from abc import ABC, abstractmethod


class Container(ABC):
    """Abstract base class for several objects used by manim.  In particular, both
    :class:`~.Scene` and :class:`~.Mobject` inherit from Container.

    Parameters
    ----------
    kwargs : Any

    """

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def add(self, *items):
        """Abstract method to add items to Container.

        Parameters
        ----------
        items : Any
            Objects to be added.
        """

    @abstractmethod
    def remove(self, *items):
        """Abstract method to remove items from Container.

        Parameters
        ----------
        items : Any
            Objects to be added.
        """
