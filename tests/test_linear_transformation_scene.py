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


def test_moving_mobjects_survive_plays(dry_run):
    # Scene used to replace moving_mobjects at the start of every play.
    class Moving(LinearTransformationScene):
        def construct(self):
            vector = Vector([1, 0])
            self.add_moving_mobject(vector, target_mobject=Vector([0, 1]))
            self.wait(0.1)
            self.moving = list(self.moving_mobjects)

    scene = Moving()
    scene.render()
    assert [type(mobject) for mobject in scene.moving] == [Vector]
