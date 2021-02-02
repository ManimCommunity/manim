from types import MappingProxyType

import pytest
from manim.animation.animation import Animation
from manim.debug.scene_debugger import SceneDebugger
from manim.mobject.geometry import Square

s = Square()
a = Animation(s)
dummy_scene_vars = MappingProxyType({"animations": [a], "mobjects": s, "dummy_attr": 1})
dummy_renderer_vars = MappingProxyType(
    {"time": 1, "number_frame": 2, "animations_hashes": ["999_999"], "num_plays": 0}
)


def test_scene_debugger_scene_infos():
    s = SceneDebugger()
    s.set_scene_vars(dummy_scene_vars)
    s.set_renderer_vars(dummy_renderer_vars)
    assert str(s._get_scene_dict_info()["animations"]) == "[Animation(Square)]"
    assert s._get_scene_dict_info()["number_frame"] == 2


def test_scene_debugger_animation_info():
    s = SceneDebugger()
    s.set_scene_vars(dummy_scene_vars)
    s.set_renderer_vars(dummy_renderer_vars)
    print(s._get_current_animations_dict_info())
    print(s._get_current_animations_dict_info())
    assert s._get_current_animations_dict_info()["Animation(Square)"]["run_time"] == 1.0


def test_add_attributes_to_watch():
    s = SceneDebugger()
    scene_vars = MappingProxyType({"dummy_attr": 1})
    s.set_scene_vars(scene_vars)
    s.debug_scene_attributes.add("dummy_attr")
    assert s._get_scene_dict_info()["dummy_attr"] == 1


def test_changing_object():
    class DummyObject:
        def __init__(self) -> None:
            self.element_to_look = 4

    o = DummyObject()
    s = SceneDebugger()
    s.set_scene_vars(MappingProxyType(vars(o)))
    s.debug_scene_attributes.add("element_to_look")
    assert s._get_scene_dict_info()["element_to_look"] == 4
    o.element_to_look = 5
    assert s._get_scene_dict_info()["element_to_look"] == 5


def test_spy_function_with_force_call():
    element_to_look = 1

    def func():
        return element_to_look

    s = SceneDebugger()
    s.set_renderer_vars(dummy_renderer_vars)
    s.spy_function(func, force_call=True)
    s.update()
    assert s._record_spied_functions[func.__name__] == "1 (fra. 2)"
    element_to_look = 2
    s.update()
    assert s._record_spied_functions[func.__name__] == "2 (fra. 2)"


def f1():
    return 3


def test_spy_function():
    s = SceneDebugger()
    s.set_renderer_vars(dummy_renderer_vars)
    s.spy_function(f1)
    print(s._record_spied_functions)
    assert s._record_spied_functions[f1.__name__] == "Not called"
    f1()
    assert s._record_spied_functions[f1.__name__] == "3 (fra. 2)"


def test_spy_inner_function():
    def inner_func():
        return 3

    s = SceneDebugger()
    with pytest.raises(
        ValueError, match="Inner functions are not yet supported by scene-debugger."
    ):
        s.spy_function(inner_func)
