from manim import *

class C2PBehavior(Scene):
    def construct(self):
        # Create an Axes object
        ax = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 5, 1],
            axis_config={"include_tip": False},
        )
        self.add(ax)

        # --- Single points ---
        self.add(Dot(ax.c2p(1, 2), color=BLUE))                # correct single point
        self.add(Dot(ax.c2p(1, 2), color=GREEN))               # replaced [1,2] with separate args
        self.add(Dot(ax.c2p(1, 2, 1), color=RED))             # optional 3D point

        # --- Multiple points ---
        points = [[1, 2], [3, 4], [2, 1]]                     # list of (x,y) points
        for p in points:
            self.add(Dot(ax.c2p(*p), color=YELLOW))           # use *p to unpack x,y

        
