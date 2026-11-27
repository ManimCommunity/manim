"""When a renderer can reuse the drawing of the previous frame.

A frame draws a list of render snapshots in order. The nodes at the start of the
list that have the same keys (:attr:`.RNode.key`) as in the previous frame, from the
same view, give the same pixels: a renderer keeps their drawing and draws only the
rest. That holds whatever changed them, so animations, updaters and direct changes
between frames need no announcement. A node without a key (``None``) never equals
another: it ends the reusable prefix.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias

from manim.renderer.protocol import CameraState

if TYPE_CHECKING:
    from manim.renderer.snapshot import RNode

#: What a node is compared by between frames (see :func:`node_key`).
NodeKey: TypeAlias = Hashable
#: What a frame's drawing depends on besides its nodes (see :func:`view_key`).
ViewKey: TypeAlias = tuple[Hashable, ...]

__all__: list[str] = []


def node_key(node: RNode) -> NodeKey:
    """The key a node is compared by: its :attr:`.RNode.key`, or an object equal to
    nothing if it has none.
    """
    key = node.key
    return _Unequal() if key is None else key


class _Unequal:
    """A key that equals nothing, not even itself."""

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        return False

    def __hash__(self) -> int:
        return id(self)


def view_key(
    state: CameraState,
    size: tuple[int, int],
    reference_size: tuple[int, int],
    background: Hashable,
) -> ViewKey:
    """What a frame's drawing depends on besides its nodes: every field of the
    camera state, the pixel size, the reference pixel size (which scales widths
    given in pixels) and the background the target starts from.
    """
    return (
        size,
        reference_size,
        tuple(_value(getattr(state, field.name)) for field in _STATE_FIELDS),
        background,
    )


_STATE_FIELDS = dataclasses.fields(CameraState)


def _value(value: Any) -> Hashable:
    """A comparable and hashable form of a camera state field."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    try:
        return tuple(float(x) for x in value.ravel())
    except AttributeError:
        return tuple(float(x) for x in value)


def common_prefix(a: Sequence[Any], b: Sequence[Any]) -> int:
    """The number of equal items at the start of ``a`` and ``b``."""
    count = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        count += 1
    return count


class PrefixMemo:
    """The keys of the last frame drawn into one target, and of its view.

    :meth:`count_reusable` tells how many leading nodes of a new frame can be copied
    from a saved drawing, and :meth:`count_worth_saving` how many to save after this
    frame: the
    nodes that did not change since the previous frame, which are likely to stay
    unchanged in the next ones.
    """

    def __init__(self) -> None:
        self._previous_keys: list[NodeKey] = []
        self._previous_view: ViewKey | None = None
        #: Keys of the nodes whose drawing is saved, and their view.
        self.saved_keys: list[NodeKey] = []
        self.saved_view: ViewKey | None = None

    def count_reusable(self, keys: list[NodeKey], view: ViewKey) -> int:
        """How many leading nodes of ``keys`` the saved drawing holds (0 or all)."""
        count = len(self.saved_keys)
        if (
            count
            and view == self.saved_view
            and common_prefix(keys, self.saved_keys) == count
        ):
            return count
        return 0

    def count_worth_saving(self, keys: list[NodeKey], view: ViewKey) -> int:
        """How many leading nodes to save after this frame, if the saved drawing
        cannot be reused: those unchanged since the previous frame.
        """
        if view != self._previous_view:
            return 0
        return common_prefix(keys, self._previous_keys)

    def mark_saved(self, keys: list[NodeKey], view: ViewKey) -> None:
        """Remember that the drawing of the nodes with ``keys`` was saved."""
        self.saved_keys, self.saved_view = keys, view

    def forget_saved(self) -> None:
        """Forget the saved drawing; nothing is reused until one is saved again."""
        self.saved_keys, self.saved_view = [], None

    def record_drawn(self, keys: list[NodeKey], view: ViewKey) -> None:
        """Remember the frame just drawn, for :meth:`count_worth_saving`."""
        self._previous_keys, self._previous_view = keys, view

    def clear(self) -> None:
        self.forget_saved()
        self._previous_keys, self._previous_view = [], None
