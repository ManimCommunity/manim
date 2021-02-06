import itertools as it
import typing
from functools import wraps
from inspect import getmodule, isfunction, ismethod
from types import MappingProxyType

import numpy as np
from manim.mobject.mobject import Mobject
from PIL import Image, ImageDraw, ImageFont

from ... import logger
from ...constants import DOWN, LEFT, RIGHT, UP

__all__ = ["SceneDebugger"]


class SceneDebugger:
    def __init__(self) -> None:
        self._scene_info = None
        self._renderer_info = None

        # Default values. they can be modified through self.debug_<>_attribues.add(element)/remove(element) etc
        self.debug_animation_attributes = {"run_time", "lag_ratio"}

        self.debug_mobjects_attributes = {"color"}

        self.debug_scene_attributes = {
            "num_plays",
            "time",
            "animations",
            "number_frame",
            "time",
            "current_animation_hash",
        }

        self._record_spied_functions = {}
        self._force_called_spied_functions = set()

        self.VISUAL_OFFSET = 10
        self.VECTOR_OFFSET = np.full(2, self.VISUAL_OFFSET)

    def set_scene_vars(self, scene_vars: MappingProxyType) -> None:
        """Internally used to allow the debugger to access scene debug values. MappingProxy is used to prevent any
        monkeypatch from happening (sneaky modifications of scene_vars, that could affect the original scene object.)

        Parameters
        ----------
        scene_vars : MappingProxyType
            Variables of the scene.
        """
        self._scene_info = scene_vars

    def set_renderer_vars(self, renderer_vars: MappingProxyType) -> None:
        """Internally used to allow the debugger to access rendered debug values. ``MappingProxy`` is used to prevent any
        monkeypatch from happening (sneaky modifications of renderer_vars, that could affect the original renderer object.)

        Parameters
        ----------
        renderer_vars : MappingProxyType
            Bariables of the renderer.
        """
        self._renderer_info = renderer_vars

    def _get_scene_dict_info(self) -> dict:
        # NOTE : Renderer vars ar mixed up with scene vars here.
        # This is done on purpose, as there are attributes that the user could need (current hash, etc) but that are in renderer_vars.
        debug_dict_info = {}
        if self._scene_info is None:
            self._scene_info = {}
        if self._renderer_info is None:
            self._renderer_info = {}
        for key, value in it.chain(
            self._scene_info.items(), self._renderer_info.items()
        ):
            if key in self.debug_scene_attributes:
                debug_dict_info[key] = value
        return debug_dict_info

    def _get_mobjects_dict_info(self) -> dict:
        mobjects = set(self._scene_info.get("mobjects"))
        if mobjects is None or len(mobjects) == 0:
            return None
        debug_dict_mobjects = {}
        for mobject in mobjects:
            temp_mobject_info = {}
            # If the mobject can't be vars. (this shouldn't in theory happen)
            if not hasattr(mobject, "__dict__"):
                debug_dict_mobjects[str(mobject)] = "Not displayable"
                continue
            for k, v in vars(mobject).items():
                if k in self.debug_mobjects_attributes:
                    temp_mobject_info[k] = v

            debug_dict_mobjects[str(mobject)] = temp_mobject_info
        return debug_dict_mobjects

    def _get_current_animations_dict_info(self) -> dict:
        debug_dict_animations = {}
        for animation in self._scene_info["animations"]:
            if not hasattr(animation, "__dict__"):
                debug_dict_animations[str(animation)] = "Not displayable"
                continue
            debug_dict_animations[str(animation)] = {}
            for key, value in vars(animation).items():
                if key in self.debug_animation_attributes:
                    debug_dict_animations[str(animation)][key] = value
        return debug_dict_animations if len(debug_dict_animations) > 0 else None

    def get_layout(
        self,
        shape_original_frame: tuple,
    ) -> np.array:
        """Retrieves the debug-layout, as a numpy array for convenience.

        Parameters
        ----------
        shape_original_frame : tuple
            The shape of the frame where this layout will be added

        Returns
        -------
        np.array
            The debug-layout.
        """
        # TODO Change the color to the reverse of the background
        out = Image.new("RGBA", shape_original_frame, (0, 0, 0, 0))
        draw_layer = ImageDraw.Draw(out)
        starting_position = np.zeros(2)
        offset = self._draw_debug_box(
            draw_layer, "SCENE ATTR.", self._get_scene_dict_info(), starting_position
        )
        # WARNING : Up and Down are inversed in the PIL referential ..
        position = starting_position + np.multiply(
            offset + self.VECTOR_OFFSET, -DOWN[:2]
        )
        self._draw_debug_box(
            draw_layer,
            "CURRENT ANIMATIONS ATTR.",
            self._get_current_animations_dict_info(),
            position,
        )

        position += np.multiply(offset + self.VECTOR_OFFSET, -DOWN[:2])
        self._draw_debug_box(
            draw_layer, "SPIED FUNCTIONS", self._record_spied_functions, position
        )

        last_position = starting_position
        direction = RIGHT
        mobjects_info = self._get_mobjects_dict_info()
        if  mobjects_info is not None and len(mobjects_info) > 0: 
            for mob, mobject_info in mobjects_info.items():
                # Element-wise product between the direction vector (tronqued so it is in 2 dim) to get
                # an offset vector, that we add to he previous pos to get the new one.
                last_position += np.multiply(offset + self.VECTOR_OFFSET, direction[:2])
                offset = self._draw_debug_box(
                    draw_layer, str(mob), mobject_info, last_position
                )
        return np.asarray(out)

    def _draw_debug_box(
        self,
        draw_layer: ImageDraw,
        title: str,
        debug_content: str,
        start_position: np.array,
    ) -> np.array:
        text = f"{title} : \n"
        for key, value in debug_content.items():
            text += f"{key} : {value}\n"
        draw_layer.multiline_text(start_position, text)
        return np.asarray(draw_layer.multiline_textsize(text))

    def _place_spy(self, spied_func):
        @wraps(spied_func)
        def wrapper(*args, **kwargs):
            # TODO : keep a track of the current frame from where the functions has been called?
            res = spied_func(*args, **kwargs)
            if hasattr(res, "__str__"):
                self._record_spied_functions[
                    spied_func.__name__
                ] = f"{res} (fra. {self._renderer_info['number_frame']})"
            else:
                self._record_spied_functions[
                    spied_func.__name__
                ] = f"Not conv. to str. {self._renderer_info['number_frame']}"
            return res

        return wrapper

    def spy_function(
        self, func: typing.Callable, force_call=False, args=[], kwargs={}
    ) -> None:
        """Enable listening for a given function. Its return value(s) will then be displayed in the debug layout.

        If the function is not called by manim, you can force the debugger to call it and to display its return value(s) by setting
        ``force_call`` to true. Put as well eventual args and kwargs if needed.

        WARNING: When ``force_call`` is enabled, the debugger will call the function and keep a track of return value(s). Beware of side effects of the force-called function!

        Parameters
        ----------
        func : typing.Callable
            The func to spy.
        force_call : bool, optional
            Whether the function should be called by the debugger at each frame., by default False
        args : list, optional
            Args, if needed during a force call, by default []
        kwargs : dict, optional
            Kwargs, if needed during a force call, by default {}

        Raises
        ------
        ValueError
            If ``func`` is not callable, or if func is an inner function.
        """
        if "locals" in func.__qualname__ and not force_call:
            # Using force call with a nested funciton will still work, because there is no need to monkey-patch the function because it's the debugger that forces call it.
            raise ValueError(
                "Inner functions are not yet supported by scene-debugger. Nevertheless, you can force call inner functions."
            )
        # Redefine the old function by a decorated one.
        new_func = self._place_spy(func)
        if ismethod(func):
            setattr(func.__self__, func.__name__, new_func)
        elif isfunction(func):
            setattr(getmodule(func), func.__name__, new_func)
        else:
            raise ValueError("Only functions can be spied.")
        self._record_spied_functions[func.__name__] = "Not called"
        if force_call:
            self._force_called_spied_functions.add(lambda: new_func(*args, **kwargs))

    def update(self):
        """Used internally to update some debug values."""
        for func in self._force_called_spied_functions:
            func()
