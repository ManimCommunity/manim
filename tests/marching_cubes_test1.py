import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def implicitiFuncition():
    def sphere_f(x, y, z, r=1.0):
        return x**2 + y**2 + z**2 - r**2

    number_polygon = int(input("The number of polygon: "))
    x = float(input("Value of X: "))
    y = float(input("Value of Y: "))
    z = float(input("Value of Z: "))

    n = number_polygon
    x = np.linspace(-x, x, n)
    y = np.linspace(-y, y, n)
    z = np.linspace(-z, z, n)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    F = sphere_f(X, Y, Z, r=1.0)

    verts, faces, normals, values = measure.marching_cubes(F, level=0.0, spacing=(x[1]-x[0], y[1]-y[0], z[1]-z[0]))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces], alpha=0.7)
    mesh.set_facecolor("cyan")
    mesh.set_edgecolor("k")
    ax.add_collection3d(mesh)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_box_aspect([1,1,1]) 

    plt.tight_layout()
    plt.show()
