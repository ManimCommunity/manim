import numpy as np
from skimage import measure
from manim import *

class ImplicitSurface(ThreeDVMobject):
    """Renderiza uma isosuperfície implícita usando Marching Cubes.

    Parameters
    ----------
    func : Callable[[float, float, float], float]
        Função implícita f(x,y,z). A superfície é definida onde f(x,y,z) = isolevel.
    resolution : int, default=25
        Número de divisões por eixo.
    isolevel : float, default=0.0
        Nível da isosuperfície.
    x_range, y_range, z_range : list[float], default=[-2, 2]
        Intervalos de amostragem.
    color : Color, default=BLUE
        Cor da superfície.
    **kwargs
        Argumentos adicionais para ThreeDVMobject.
    """

    def __init__(
        self,
        func,
        resolution=25,
        isolevel=0.0,
        x_range=[-2, 2],
        y_range=[-2, 2],
        z_range=[-2, 2],
        color=BLUE,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Gera a grade 3D:
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        z = np.linspace(z_range[0], z_range[1], resolution)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        values = func(X, Y, Z)

        # Extrai a malha:
        verts, faces, _, _ = measure.marching_cubes(values, level=isolevel)

        # Normaliza para o domínio real:
        scale_x = (x_range[1] - x_range[0]) / resolution
        scale_y = (y_range[1] - y_range[0]) / resolution
        scale_z = (z_range[1] - z_range[0]) / resolution
        verts = np.array([
            [x_range[0] + v[0]*scale_x, y_range[0] + v[1]*scale_y, z_range[0] + v[2]*scale_z]
            for v in verts
        ])

        # Constrói os polígonos:
        for face in faces:
            tri = [verts[i] for i in face]
            self.add(Polygon(*tri, color=color, fill_opacity=1))
