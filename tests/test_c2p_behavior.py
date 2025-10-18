from manim import *

class C2PBehavior(Scene):
    def construct(self):
        axes = Axes()
        
        # Single coordinate input
        print("Single coordinate:", axes.c2p(1, 2))
        
        # List input
        print("List input:", axes.c2p([1, 2]))
        
        # List of lists input
        print("List of lists input:", axes.c2p([[1, 2], [3, 4]]))
        
        # Numpy array input
        import numpy as np
        print("Numpy array input:", axes.c2p(np.array([1, 2])))
