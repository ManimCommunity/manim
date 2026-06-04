from __future__ import annotations

__all__ = [
    "EndSceneEarlyException",
    "RerunSceneException",
    "MultiAnimationOverrideException",
]


class EndSceneEarlyException(Exception):
    pass


class RerunSceneException(Exception):
    pass


class MultiAnimationOverrideException(Exception):
    pass
