"""Unit tests for ImplicitSurface."""

from __future__ import annotations

import numpy as np

from manim import GREEN, ImplicitSurface


class TestImplicitSurface:
    """Tests for the ImplicitSurface class."""

    def test_sphere_creation(self):
        """Test that a sphere can be created."""

        def sphere_func(x, y, z):
            return x**2 + y**2 + z**2 - 1.0

        surface = ImplicitSurface(
            sphere_func,
            x_range=(-1.5, 1.5),
            y_range=(-1.5, 1.5),
            z_range=(-1.5, 1.5),
            resolution=32,
        )

        # Should have triangular faces
        assert len(surface.submobjects) > 0

    def test_sphere_vertices_on_surface(self):
        """Test that sphere vertices approximately satisfy the implicit equation."""

        def sphere_func(x, y, z):
            return x**2 + y**2 + z**2 - 1.0

        surface = ImplicitSurface(
            sphere_func,
            x_range=(-1.5, 1.5),
            y_range=(-1.5, 1.5),
            z_range=(-1.5, 1.5),
            resolution=40,
        )

        errors = surface.verify_surface()
        # Mean error should be small
        assert errors["mean_error"] < 0.1
        # Max error should be reasonable
        assert errors["max_error"] < 0.2

    def test_no_surface_in_domain(self):
        """Test that no triangles are created when surface is outside domain."""

        def f(x, y, z):
            return x**2 + y**2 + z**2 - 100.0  # Sphere of radius 10

        surface = ImplicitSurface(
            f,
            x_range=(-1, 1),
            y_range=(-1, 1),
            z_range=(-1, 1),
            resolution=20,
        )

        # No triangles should be created
        assert len(surface.submobjects) == 0

    def test_level_parameter(self):
        """Test that the level parameter works correctly."""

        def f(x, y, z):
            return x**2 + y**2 + z**2

        # Create sphere of radius 2 using level=4
        surface = ImplicitSurface(
            f,
            x_range=(-2.5, 2.5),
            y_range=(-2.5, 2.5),
            z_range=(-2.5, 2.5),
            level=4.0,  # radius squared = 4, so radius = 2
            resolution=32,
        )

        assert len(surface.submobjects) > 0

        # Check vertices have approximately radius 2
        vertices = surface.get_vertices()
        if len(vertices) > 0:
            radii = np.sqrt(np.sum(vertices**2, axis=1))
            mean_radius = np.mean(radii)
            assert abs(mean_radius - 2.0) < 0.2

    def test_color_parameters(self):
        """Test that color parameters are applied."""

        def f(x, y, z):
            return x**2 + y**2 + z**2 - 1.0

        surface = ImplicitSurface(
            f,
            resolution=20,
            fill_color=GREEN,
            fill_opacity=0.5,
            stroke_width=1.0,
        )

        if len(surface.submobjects) > 0:
            # Check that faces have the expected properties
            face = surface.submobjects[0]
            assert face.fill_opacity == 0.5
            assert face.stroke_width == 1.0

    def test_cylinder_surface(self):
        """Test creating a cylinder using implicit function."""

        def cylinder_func(x, y, z):
            return x**2 + y**2 - 1.0

        surface = ImplicitSurface(
            cylinder_func,
            x_range=(-1.5, 1.5),
            y_range=(-1.5, 1.5),
            z_range=(-1, 1),
            resolution=32,
        )

        assert len(surface.submobjects) > 0

    def test_torus_surface(self):
        """Test creating a torus using implicit function."""
        R, r = 2.0, 0.5

        def torus_func(x, y, z):
            return (np.sqrt(x**2 + y**2) - R) ** 2 + z**2 - r**2

        surface = ImplicitSurface(
            torus_func,
            x_range=(-3, 3),
            y_range=(-3, 3),
            z_range=(-1, 1),
            resolution=40,
        )

        assert len(surface.submobjects) > 0

    def test_get_vertices(self):
        """Test getting vertices from the surface."""

        def f(x, y, z):
            return x**2 + y**2 + z**2 - 1.0

        surface = ImplicitSurface(f, resolution=32)
        vertices = surface.get_vertices()

        # Should have vertices
        assert len(vertices) > 0
        # Each vertex should be 3D
        assert vertices.shape[1] == 3

    def test_empty_surface_get_vertices(self):
        """Test getting vertices from an empty surface."""

        def f(x, y, z):
            return x**2 + y**2 + z**2 - 100.0

        surface = ImplicitSurface(
            f,
            x_range=(-1, 1),
            y_range=(-1, 1),
            z_range=(-1, 1),
            resolution=20,
        )

        vertices = surface.get_vertices()
        assert len(vertices) == 0

    def test_plane_surface(self):
        """Test creating a plane using implicit function."""

        def plane_func(x, y, z):
            return z - 0.5  # Plane at z = 0.5

        surface = ImplicitSurface(
            plane_func,
            x_range=(-1, 1),
            y_range=(-1, 1),
            z_range=(0, 1),
            resolution=32,
        )

        assert len(surface.submobjects) > 0

        # Vertices should have z close to 0.5
        vertices = surface.get_vertices()
        if len(vertices) > 0:
            z_values = vertices[:, 2]
            assert np.allclose(z_values, 0.5, atol=0.1)
