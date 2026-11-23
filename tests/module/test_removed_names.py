"""The names removed by paints, colour fields, lights and materials stay removed
(the table of docs/source/guides/upgrading.rst).
"""

from __future__ import annotations

import importlib

import pytest

import manim
from manim import (
    StreamLines,
    ThreeDAxes,
    ThreeDCamera,
    ThreeDScene,
    VMobject,
    tempconfig,
)
from manim.mobject.three_d.mesh import MeshMobject

REMOVED_ATTRIBUTES = [
    (VMobject, name)
    for name in (
        "fill_rgbas",
        "stroke_rgbas",
        "background_stroke_rgbas",
        "get_fill_rgbas",
        "get_stroke_rgbas",
        "generate_rgbas_array",
        "update_rgbas_array",
        "align_rgbas",
        "set_sheen",
        "set_sheen_direction",
        "rotate_sheen_direction",
        "get_sheen_direction",
        "get_sheen_factor",
        "get_gradient_start_and_end_points",
        "shade_in_3d",
        "set_shade_in_3d",
        "color_using_background_image",
        "match_background_image",
        "get_background_image",
        "z_index_group",
    )
] + [
    (MeshMobject, "color_using_background_image"),
    (MeshMobject, "set_sheen"),
    (StreamLines, "get_colored_background_image"),
    (ThreeDCamera, "light_source"),
    (ThreeDCamera, "should_apply_shading"),
    (ThreeDCamera, "shading_factor"),
    (ThreeDAxes, "light_source"),
]


@pytest.mark.parametrize(
    ("owner", "name"),
    REMOVED_ATTRIBUTES,
    ids=[f"{owner.__name__}.{name}" for owner, name in REMOVED_ATTRIBUTES],
)
def test_removed_attributes_are_gone(owner, name):
    assert not hasattr(owner, name)


def test_removed_attributes_are_gone_from_instances():
    vmobject = VMobject()
    for name in ("shade_in_3d", "sheen_factor", "background_image", "fill_rgbas"):
        assert not hasattr(vmobject, name)
    with tempconfig({"dry_run": True}):
        camera = ThreeDScene().camera
    for name in ("light_source", "light_source_start_point", "shading_factor"):
        assert not hasattr(camera, name)


@pytest.mark.parametrize(
    ("owner", "argument"),
    [
        (VMobject, "shade_in_3d"),
        (VMobject, "sheen_factor"),
        (VMobject, "sheen_direction"),
        (VMobject, "background_image"),
        (ThreeDAxes, "light_source"),
    ],
)
def test_removed_arguments_are_refused(owner, argument):
    with pytest.raises(TypeError):
        owner(**{argument: 1})


@pytest.mark.parametrize("name", ["get_shaded_rgb", "get_3d_vmob_start_corner"])
def test_removed_functions_are_not_exported(name):
    assert not hasattr(manim, name)


def test_removed_modules_are_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("manim.mobject.three_d.three_d_utils")
