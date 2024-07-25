from manim import RIGHT, UP, LinearTransformationScene, Vector, VGroup

__module_test__ = "vector_space_scene"


def test_ghost_vectors_len_and_types():
    scene = LinearTransformationScene()
    scene.leave_ghost_vectors = True

    # prepare vectors (they require a vmobject as their target)
    v1, v2 = Vector(RIGHT), Vector(RIGHT)
    v1.target, v2.target = Vector(UP), Vector(UP)

    # ghost_vector addition is in this method
    scene.get_piece_movement((v1, v2))

    ghosts = scene.get_ghost_vectors()
    assert len(ghosts) == 1
    # check if there are two vectors in the ghost vector VGroup
    assert len(ghosts[0]) == 2

    # check types of ghost vectors
    assert isinstance(ghosts, VGroup)
    assert isinstance(ghosts[0], VGroup)
    assert all(isinstance(x, Vector) for x in ghosts[0])
