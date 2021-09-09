__all__ = ["OBJMobject"]
from pathlib import Path
from typing import *

import numpy as np
from colour import Color

from .. import config, logger
from ..constants import *
from ..mobject.mobject import *

# from ..mobject.opengl_mobject import OpenGLMobject
from ..mobject.types.vectorized_mobject import VGroup
from ..utils.color import *
from .three_dimensions import ThreeDVMobject

# from manim.mobject.opengl_compatibility import ConvertToOpenGL


class OBJMobject(VGroup):
    # TODO: Better Colours; textures; Free-form geometry; smoothing groups

    def get_faces_by_group(self, *groups):
        return VGroup(
            *(
                self.submobjects[i]
                for i, faceinfo in enumerate(self.faces_info)
                if all(group in faceinfo["groups"] for group in groups)
            )
        )

    def _parse_obj_file(self):
        isint = lambda s: s.isdigit() or (s.startswith("-") and s[1:].isdigit())

        groups = None  # Not all obj files specify groups. Set default group to None
        material = None  # Default material is white in colour. Gets handled in self._build_faces
        with open(self.fp) as f:
            for line in f.readlines():
                if line.startswith("#") or len(line.strip()) == 0:
                    continue
                linedata = line.split()
                if linedata[0] == "mtllib":
                    mtlfilepaths = list(map(Path, linedata[1:]))
                    for mtlfilepath in mtlfilepaths:
                        if not mtlfilepath.is_absolute():
                            mtlfilepath = self.fp.parent / mtlfilepath
                        self._parse_mtl_file(mtlfilepath)
                if linedata[0] == "v":
                    self.vertices.append(list(map(float, linedata[1:4])))
                elif linedata[0] == "vn":
                    self.vertex_normals.append(list(map(float, linedata[1:4])))
                elif linedata[0] == "vt":
                    self.texture_vertices.append(list(map(float, linedata[1:4])))
                elif linedata[0] == "g":
                    # If no group names are mentioned use a list with None.
                    # None is put in a list so iterating doesn't break.
                    groups = linedata[1:] if len(linedata) > 1 else [None]
                    for group in groups:
                        if group not in self.group_names:
                            self.group_names.append(group)

                elif linedata[0] == "usemtl":
                    material = linedata[1]
                elif linedata[0] == "f":
                    face_info = {"groups": groups, "material": material}
                    face_info["vertices"] = [
                        {
                            ("vertex", "texture_vertex", "vertex_normal")[i]: (
                                self.vertices,
                                self.texture_vertices,
                                self.vertex_normals,
                            )[i][
                                int(index) - 1 if int(index) >= 1 else int(index)
                            ]  # Indexing starts from 1. Reverse indexing also possible.
                            if isint(index)
                            else None
                            for i, index in enumerate(text_vertex_data.split("/"))
                        }
                        for text_vertex_data in linedata[1:]
                    ]
                    self.faces_info.append(face_info)

    def _parse_mtl_file(self, mtlfilepath):
        with open(mtlfilepath) as f:
            for line in f.readlines():
                if line.startswith("#") or len(line.strip()) == 0:
                    continue
                linedata = line.split()
                if linedata[0] == "newmtl":
                    material_name = linedata[1]
                    self.materials_dict[material_name] = {}
                elif linedata[0] == "Ka":
                    if linedata[1] == "spectral" or len(linedata[1:]) > 4:
                        logger.warning(
                            "Spectral and CIEXYZ colour is not supported yet. "
                            "Your model may be coloured strangely.",
                        )
                        continue
                    self.materials_dict[material_name]["ambient_reflectivity"] = list(
                        map(float, linedata[1:]),
                    )
                elif linedata[0] == "Kd":
                    if linedata[1] == "spectral" or len(linedata[1:]) > 4:
                        logger.warning(
                            "Spectral and CIEXYZ colour is not supported yet. "
                            "Your model may be coloured strangely.",
                        )
                        continue
                    self.materials_dict[material_name]["diffuse_reflectivity"] = list(
                        map(float, linedata[1:]),
                    )
                elif linedata[0] == "Ks":
                    if linedata[1] == "spectral" or len(linedata[1:]) > 4:
                        logger.warning(
                            "Spectral and CIEXYZ colour is not supported yet. "
                            "Your model may be coloured strangely.",
                        )
                        continue
                    self.materials_dict[material_name]["specular_reflectivity"] = list(
                        map(float, linedata[1:]),
                    )

    def _build_faces(self):
        for face_info in self.faces_info:
            vertex_list = [
                face_vertex["vertex"] for face_vertex in face_info["vertices"]
            ]
            face = (
                ThreeDVMobject()
                .set_points_as_corners(vertex_list + [vertex_list[0]])
                .set_fill(
                    WHITE
                    if face_info["material"] is None
                    or "diffuse_reflectivity"
                    not in self.materials_dict[face_info["material"]]
                    else rgb_to_color(
                        # Use diffuse reflectivity as it is closest to actual color.
                        self.materials_dict[face_info["material"]][
                            "diffuse_reflectivity"
                        ],
                    ),
                )
            )
            self.add(face)

    def __init__(self, fp, **kwargs):

        VGroup.__init__(self, **kwargs)

        self.fp = Path(fp)
        self.vertices = []
        self.vertex_normals = []
        self.texture_vertices = []
        self.materials_dict = {}
        self.faces_info = []
        self.group_names = []

        self._parse_obj_file()
        self._build_faces()

        self.set_fill(opacity=1)
        self.set_stroke(width=0)
