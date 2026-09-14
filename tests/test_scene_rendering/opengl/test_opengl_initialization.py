"""Exercise initialization rollback against native standalone OpenGL resources."""

from unittest.mock import Mock

import moderngl
import pytest

from manim import Scene


@pytest.mark.parametrize("failure_type", [ValueError, KeyboardInterrupt, SystemExit])
def test_native_standalone_resources_are_released(
    using_temp_opengl_config, monkeypatch, failure_type
):
    create_context = moderngl.create_context
    acquired = []
    failure = failure_type("blend setup failed")

    def create(*args, **kwargs):
        context = create_context(*args, **kwargs)
        acquired.append(context)
        for name in ("texture", "depth_renderbuffer", "framebuffer"):
            method = getattr(context, name)

            def track(*args, _method=method, **kwargs):
                resource = _method(*args, **kwargs)
                acquired.append(resource)
                return resource

            monkeypatch.setattr(context, name, track)
        monkeypatch.setattr(context, "enable", Mock(side_effect=failure))
        return context

    monkeypatch.setattr(moderngl, "create_context", create)
    try:
        with pytest.raises(failure_type) as caught:
            Scene().renderer.open()
        assert caught.value is failure
        assert len(acquired) == 4
        # These are real driver-backed objects, not just release-call spies.
        assert all(
            isinstance(resource.mglo, moderngl.InvalidObject) for resource in acquired
        )
    finally:
        for resource in reversed(acquired):
            if not isinstance(resource.mglo, moderngl.InvalidObject):
                resource.release()
