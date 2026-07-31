from __future__ import annotations

from manim import Dot, ZoomedScene, tempconfig


def test_activate_deactivate_zooming():
    with tempconfig({"dry_run": True, "quality": "low_quality"}):
        scene = ZoomedScene()
        scene.setup()
        dot = Dot()
        scene.add(dot)

        scene.activate_zooming(animate=False)
        assert scene.zoom_activated
        assert scene.zoomed_camera.frame in scene.foreground_mobjects
        assert scene.zoomed_display in scene.foreground_mobjects
        assert scene.zoomed_display in scene.camera.image_mobjects_from_cameras

        scene.deactivate_zooming()
        assert not scene.zoom_activated
        assert scene.zoomed_camera.frame not in scene.mobjects
        assert scene.zoomed_display not in scene.mobjects
        assert scene.zoomed_camera.frame not in scene.foreground_mobjects
        assert scene.zoomed_display not in scene.foreground_mobjects
        assert scene.zoomed_display not in scene.camera.image_mobjects_from_cameras
        assert dot in scene.mobjects


def test_deactivate_zooming_when_inactive():
    with tempconfig({"dry_run": True, "quality": "low_quality"}):
        scene = ZoomedScene()
        scene.setup()

        # Deactivating before activating must not raise.
        scene.deactivate_zooming()
        assert not scene.zoom_activated


def test_reactivate_zooming_after_deactivation():
    with tempconfig({"dry_run": True, "quality": "low_quality"}):
        scene = ZoomedScene()
        scene.setup()

        scene.activate_zooming(animate=False)
        scene.deactivate_zooming()
        scene.activate_zooming(animate=False)
        assert scene.zoom_activated
        assert scene.zoomed_display in scene.camera.image_mobjects_from_cameras


def test_deactivate_zooming_animated_restores_geometry():
    with tempconfig({"dry_run": True, "quality": "low_quality"}):
        scene = ZoomedScene()
        scene.setup()

        frame_before = scene.zoomed_camera.frame.copy()
        display_before = scene.zoomed_display.copy()

        scene.activate_zooming(animate=True)
        scene.deactivate_zooming(animate=True)

        assert not scene.zoom_activated
        assert scene.zoomed_camera.frame.width == frame_before.width
        assert scene.zoomed_camera.frame.height == frame_before.height
        assert (
            scene.zoomed_camera.frame.get_center() == frame_before.get_center()
        ).all()
        assert scene.zoomed_display.width == display_before.width
        assert scene.zoomed_display.height == display_before.height
        assert (scene.zoomed_display.get_center() == display_before.get_center()).all()
