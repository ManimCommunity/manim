from manim.scene.vector_space_scene import VectorScene
from tests.test_graphical_units.testing.frames_comparison import frames_comparison

__module_test__ = "vector_scene"


@frames_comparison(base_scene=VectorScene, last_frame=False)
def test_vector_to_coords(scene):
    scene.add_plane().add_coordinates()
    vector = scene.add_vector([-3, -2])
    basis = scene.get_basis_vectors()
    scene.add(basis)
    scene.vector_to_coords(vector=vector)
