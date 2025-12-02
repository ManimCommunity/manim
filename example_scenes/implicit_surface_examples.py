"""Example scenes demonstrating ImplicitSurface usage."""

from __future__ import annotations

import numpy as np

from manim import *


class ImplicitSphereExample(ThreeDScene):
    """Demonstrate a unit sphere defined implicitly by x^2 + y^2 + z^2 = 1."""

    def construct(self):
        axes = ThreeDAxes(
            x_range=(-1.5, 1.5, 1),
            y_range=(-1.5, 1.5, 1),
            z_range=(-1.5, 1.5, 1),
        )

        def sphere_func(x, y, z):
            return x**2 + y**2 + z**2 - 1.0

        surface = ImplicitSurface(
            sphere_func,
            x_range=(-1.3, 1.3),
            y_range=(-1.3, 1.3),
            z_range=(-1.3, 1.3),
            resolution=40,
            fill_color=BLUE,
            fill_opacity=0.8,
        )

        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
        self.add(axes, surface)
        self.wait()


class ImplicitTorusExample(ThreeDScene):
    """Demonstrate a torus defined implicitly."""

    def construct(self):
        R, r = 2.0, 0.5  # Major and minor radii

        def torus_func(x, y, z):
            return (np.sqrt(x**2 + y**2) - R) ** 2 + z**2 - r**2

        surface = ImplicitSurface(
            torus_func,
            x_range=(-3, 3),
            y_range=(-3, 3),
            z_range=(-1, 1),
            resolution=50,
            fill_color=GREEN,
            fill_opacity=0.9,
        )

        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        self.add(surface)
        self.wait()


class ImplicitGyroidExample(ThreeDScene):
    """Demonstrate a gyroid minimal surface with gradient coloring."""

    def construct(self):
        def gyroid_func(x, y, z):
            return (
                np.sin(x) * np.cos(y)
                + np.sin(y) * np.cos(z)
                + np.sin(z) * np.cos(x)
            )

        # Define salmon to pink gradient colors
        salmon = ManimColor("#FA8072")
        hot_pink = ManimColor("#FF69B4")

        surface = ImplicitSurface(
            gyroid_func,
            x_range=(-PI, PI),
            y_range=(-PI, PI),
            z_range=(-PI, PI),
            resolution=50,
            fill_opacity=0.9,
        )

        # Apply gradient coloring based on z-position
        for tri in surface.submobjects:
            # Get the center z-coordinate of each triangle
            z_val = tri.get_center()[2]
            # Normalize z to [0, 1] range
            t = (z_val + PI) / (2 * PI)
            t = max(0, min(1, t))  # Clamp to [0, 1]
            # Interpolate from salmon to pink
            tri.set_fill(
                color=interpolate_color(salmon, hot_pink, t),
                opacity=0.9,
            )

        self.set_camera_orientation(phi=70 * DEGREES, theta=-30 * DEGREES)
        self.add(surface)
        self.wait()


class ImplicitBlobbySurfaceExample(ThreeDScene):
    """Demonstrate a 'blobby' surface made from sum of Gaussians."""

    def construct(self):
        centers = [
            (-0.5, 0.0, 0.0),
            (0.5, 0.0, 0.0),
            (0.0, 0.8, 0.0),
        ]

        def blobby_func(x, y, z):
            # Sum of Gaussians centered at different points
            field = 0.0
            for cx, cy, cz in centers:
                r2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
                field += np.exp(-4 * r2)
            return field - 0.4  # 0-level of this isosurface

        surface = ImplicitSurface(
            blobby_func,
            x_range=(-2, 2),
            y_range=(-2, 2),
            z_range=(-2, 2),
            resolution=40,
            fill_color=ORANGE,
            fill_opacity=0.8,
        )

        axes = ThreeDAxes(
            x_range=(-2, 2, 1),
            y_range=(-2, 2, 1),
            z_range=(-2, 2, 1),
        )

        self.set_camera_orientation(phi=70 * DEGREES, theta=-60 * DEGREES)
        self.add(axes, surface)
        self.wait()


class ImplicitCylinderExample(ThreeDScene):
    """Demonstrate a cylinder defined implicitly by x^2 + y^2 = 1."""

    def construct(self):
        def cylinder_func(x, y, z):
            return x**2 + y**2 - 1.0

        surface = ImplicitSurface(
            cylinder_func,
            x_range=(-1.5, 1.5),
            y_range=(-1.5, 1.5),
            z_range=(-1.5, 1.5),
            resolution=40,
            fill_color=YELLOW,
            fill_opacity=0.7,
        )

        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
        self.add(surface)
        self.wait()


class ImplicitHeartExample(ThreeDScene):
    """Demonstrate a 3D heart shape defined implicitly."""

    def construct(self):
        def heart_func(x, y, z):
            # Heart surface equation
            return (
                (x**2 + (9 / 4) * y**2 + z**2 - 1) ** 3
                - x**2 * z**3
                - (9 / 80) * y**2 * z**3
            )

        surface = ImplicitSurface(
            heart_func,
            x_range=(-1.5, 1.5),
            y_range=(-1.5, 1.5),
            z_range=(-1.5, 1.5),
            resolution=50,
            fill_color=RED,
            fill_opacity=0.9,
        )

        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
        self.add(surface)
        self.wait()
