from manim import *

"""
These examples showcase the new `scaling_type` argument for the tree layout.
They use unnecessecarily large vertex size given vertices are just empty circles,
but it becomes useful when there are some labels and/or graphics attached to
vertices
"""


class TreeAnimationAbsolute(MovingCameraScene):
    """
    This scene uses the value `"absolute"` for `scaling_type`.

    Notice how the horizontal component of the graph (`0.5`) is exactly
    twice the radius of a vertex (`0.25`), and how vertices that are closest
    to each other just barely touch each other horizontally.
    The vertical component being larger controls the vertical spacing of the nodes.

    The scale parameter controls the vertex spacing in the same units you use to
    control vertex size. This means that larger trees can easily grow outside of
    a fixed camera view. We use the `auto_zoom` method of a `MovingCameraScene`'s
    camera to zoom out as needed.
    """

    DEPTH = 4
    CHILDREN_PER_VERTEX = 3
    SCALE = (0.5, 1)
    LAYOUT_CONFIG = {"scaling_type": "absolute"}
    VERTEX_TYPE = Circle
    VERTEX_CONF = {"radius": 0.25, "color": BLUE_B, "fill_opacity": 1}
    T_STEP = 0.5
    AUTOZOOM = True

    def construct(self):
        self.g = Graph(
            ["ROOT"], [], vertex_type=self.VERTEX_TYPE, vertex_config=self.VERTEX_CONF
        )
        self.expand_vertex("ROOT", 1)

    def expand_vertex(self, vertex_id: str, depth: int):
        new_vertices = [f"{vertex_id}/{i}" for i in range(self.CHILDREN_PER_VERTEX)]
        new_edges = [(vertex_id, child_id) for child_id in new_vertices]
        anim = AnimationGroup(
            self.g.animate.add_vertices(
                *new_vertices,
                vertex_type=self.VERTEX_TYPE,
                vertex_config=self.VERTEX_CONF,
                positions={
                    k: self.g.vertices[vertex_id].get_center() + 0.1 * DOWN
                    for k in new_vertices
                },
            ),
            self.g.animate.add_edges(*new_edges),
            self.g.animate.change_layout(
                "tree",
                root_vertex="ROOT",
                layout_scale=self.SCALE,
                layout_config=self.LAYOUT_CONFIG,
            ),
        )
        self.play(anim, run_time=self.T_STEP)
        if self.AUTOZOOM:
            camera_anim = self.camera.auto_zoom(self.g, margin=1)
            self.play(camera_anim, run_time=self.T_STEP)

        if depth < self.DEPTH:
            for child_id in new_vertices:
                self.expand_vertex(child_id, depth + 1)


class TreeAnimationRelative(TreeAnimationAbsolute):
    """
    This scene builds the same tree as the `TreeAnimationAbsolute` scene, but it uses
    the value `"relative"` for the `scaling_type` parameter. This matches the old behaviour,
    and is used as a default value for the argument.

    The scale parameter now defines size of the whole tree in the same units used to control
    vertex size. Notice how vertices start overlapping when the tree grows larger.
    """

    LAYOUT_CONFIG = {"scaling_type": "relative"}
    SCALE = (6, 3)
    AUTOZOOM = False
