"""Utilities for scene caching."""

import collections
import copy
import inspect
import json
import typing
import zlib
from time import perf_counter
from types import FunctionType, MappingProxyType, MethodType, ModuleType
from typing import Any

import numpy as np

from manim.utils.exceptions import EndSceneEarlyException

from .. import logger

class _Memoizer:

    _already_processed = set()

    # Can be changed to whatever string to help debugging the JSon generation.
    _ALREADY_PROCESSED_PLACEHOLDER = None

    @classmethod
    def reset_already_processed(cls):
        cls._already_processed.clear()
    
    @classmethod
    def check_already_processed_decorator(cls:"_Memoizer", is_method = False):
        def layer(func):
            # NOTE : There is probably a better to separate both case when func is a method or a function.
            if is_method:
                return lambda self, obj: cls._handle_already_processed(
                    obj, default_function=lambda obj: func(self, obj)
                )
            return lambda obj: cls._handle_already_processed(obj, default_function=func)

        return layer

    @classmethod
    def check_already_processed(cls, obj):
        # When the object is not memorized, we return the object itself.
        return cls._handle_already_processed(obj, lambda x: x)
        
    @classmethod
    def _handle_already_processed(cls, obj, default_function: typing.Callable[[Any], Any]):
        if isinstance(
            obj,
            (
                int,
                float,
                str,
                complex,
            ),
        ) and obj not in [None, cls._ALREADY_PROCESSED_PLACEHOLDER]:
            # It makes no sense (and it'd slower) to memoize objects of these primitive types.
            # Hence, we simply return the object.
            return obj
        if isinstance(obj, collections.Hashable):
            return cls._return_with_memoizing(obj, hash, default_function)
        else:
            return cls._return_with_memoizing(obj, id, default_function)

    @classmethod
    def _return_with_memoizing(
        cls, obj: typing.Any, obj_to_membership_sign: typing.Callable[[Any], int], default_func
    ) -> typing.Union[str, Any]:
        
        obj_membership_sign = obj_to_membership_sign(obj)
        if obj_membership_sign in cls._already_processed:
            return cls._ALREADY_PROCESSED_PLACEHOLDER
        cls._already_processed.add(obj_membership_sign)
        return default_func(obj)

class _CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        """
        This method is used to serialize objects to JSON format.

        If obj is a function, then it will return a dict with two keys : 'code', for the code source, and 'nonlocals' for all nonlocalsvalues. (including nonlocals functions, that will be serialized as this is recursive.)
        if obj is a np.darray, it converts it into a list.
        if obj is an object with __dict__ attribute, it returns its __dict__.
        Else, will let the JSONEncoder do the stuff, and throw an error if the type is not suitable for JSONEncoder.

        Parameters
        ----------
        obj : Any
            Arbitrary object to convert

        Returns
        -------
        Any
            Python object that JSON encoder will recognize

        """
        if not (isinstance(obj, ModuleType)) and isinstance(
            obj, (MethodType, FunctionType)
        ):
            cvars = inspect.getclosurevars(obj)
            cvardict = {**copy.copy(cvars.globals), **copy.copy(cvars.nonlocals)}
            for i in list(cvardict):
                # NOTE : All module types objects are removed, because otherwise it throws ValueError: Circular reference detected if not. TODO
                if isinstance(cvardict[i], ModuleType):
                    del cvardict[i]
            try:
                code = inspect.getsource(obj)
            except OSError:
                # This happens when rendering videos included in the documentation
                # within doctests and should be replaced by a solution avoiding
                # hash collision (due to the same, empty, code strings) at some point.
                # See https://github.com/ManimCommunity/manim/pull/402.
                code = ""
            return self._cleaned_iterable({"code": code, "nonlocals": cvardict})
        elif isinstance(obj, np.ndarray):
            if obj.size > 1000:
                obj = np.resize(obj, (100, 100))
                return f"TRUNCATED ARRAY: {repr(obj)}"
            # We return the repr and not a list to avoid the JsonEncoder to iterate over it.
            return repr(obj)
        elif hasattr(obj, "__dict__"):
            temp = getattr(obj, "__dict__")
            # MappingProxy is scene-caching nightmare. It contains all of the object methods and attributes. We skip it as the mechanism will at some point process the object, but instantiated.
            # Indeed, there is certainly no case where scene-caching will receive only a non instancied object, as this is never used in the library or encouraged to be used user-side.
            if isinstance(temp, MappingProxyType):
                return "MappingProxy"
            return self._cleaned_iterable(temp)
        elif isinstance(obj, np.uint8):
            return int(obj)
        return f"Unsupported type for serializing -> {str(type(obj))}"

    def _cleaned_iterable(self, iterable):
        """Check for circular reference at each iterable that will go through the JSONEncoder, as well as key of the wrong format.

        If a key with a bad format is found (i.e not a int, string, or float), it gets replaced byt its hash using the same process implemented here.
        If a circular reference is found within the iterable, it will be replaced by the string "already processed".

        Parameters
        ----------
        iterable : Iterable[Any]
            The iterable to check.
        """
        def _key_to_hash(key):
            return zlib.crc32(json.dumps(key, cls=_CustomEncoder).encode())

        @_Memoizer.check_already_processed_decorator(is_method=False)
        def _iter_check_list(lst):
            processed_list = [None] * len(lst)
            for i, el in enumerate(lst):
                if isinstance(el, (list, tuple)):
                    processed_list[i] = _iter_check_list(el)
                elif isinstance(el, dict):
                    processed_list[i] = _iter_check_dict(el)
                else:
                    processed_list[i] = _Memoizer.check_already_processed(el)
            return processed_list

        @_Memoizer.check_already_processed_decorator(is_method=False)
        def _iter_check_dict(dct):
            processed_dict = {}
            for k, v in dct.items():
                # We check if the k is of the right format (supporter by Json)
                if not isinstance(k, (str, int, float, bool)) and k is not None:
                    k_new = _key_to_hash(k)
                else:
                    k_new = k
                if isinstance(v, dict):
                    processed_dict[k_new] = _iter_check_dict(v)
                elif isinstance(v, (list, tuple)):
                    processed_dict[k_new] = _iter_check_list(v)
                else:
                    processed_dict[k_new] = _Memoizer.check_already_processed(v)
            return processed_dict

        if isinstance(iterable, (list, tuple)):
            return _iter_check_list(iterable)
        elif isinstance(iterable, dict):
            return _iter_check_dict(iterable)

    def encode(self, obj):
        """Overriding of :meth:`JSONEncoder.encode`, to make our own process.

        Parameters
        ----------
        obj: Any
            The object to encode in JSON.

        Returns
        -------
        :class:`str`
           The object encoder with the standard json process.
        """
        # We need to mark as already processed the first object to go in the process,
        # As after, only objects that come from iterables will be marked as such.
        if isinstance(obj, (dict, list, tuple)):
            return super().encode(self._cleaned_iterable(obj))
        return super().encode(obj)


def get_json(obj):
    """Recursively serialize `object` to JSON using the :class:`CustomEncoder` class.

    Parameters
    ----------
    dict_config : :class:`dict`
        The dict to flatten

    Returns
    -------
    :class:`str`
        The flattened object
    """
    return json.dumps(obj, cls=_CustomEncoder)


def get_camera_dict_for_hashing(camera_object):
    """Remove some keys from `camera_object.__dict__` that are very heavy and useless for the caching functionality.

    Parameters
    ----------
    camera_object : :class:`~.Camera`
        The camera object used in the scene

    Returns
    -------
    :class:`dict`
        `Camera.__dict__` but cleaned.
    """
    camera_object_dict = copy.copy(camera_object.__dict__)
    # We have to clean a little bit of camera_dict, as pixel_array and background are two very big numpy arrays. They
    # are not essential to caching process. We also have to remove pixel_array_to_cairo_context as it contains used
    # memory address (set randomly). See l.516 get_cached_cairo_context in camera.py
    for to_clean in ["background", "pixel_array", "pixel_array_to_cairo_context"]:
        camera_object_dict.pop(to_clean, None)
    return camera_object_dict


def get_hash_from_play_call(
    scene_object, camera_object, animations_list, current_mobjects_list
) -> str:
    """Take the list of animations and a list of mobjects and output their hashes. This is meant to be used for `scene.play` function.

    Parameters
    -----------
    scene_object : :class:`~.Scene`
        The scene object.

    camera_object : :class:`~.Camera`
        The camera object used in the scene.

    animations_list : Iterable[:class:`~.Animation`]
        The list of animations.

    current_mobjects_list : Iterable[:class:`~.Mobject`]
        The list of mobjects.

    Returns
    -------
    :class:`str`
        A string concatenation of the respective hashes of `camera_object`, `animations_list` and `current_mobjects_list`, separated by `_`.
    """
    logger.debug("Hashing ...")
    t_start = perf_counter()
    _Memoizer.check_already_processed(scene_object)
    camera_json = get_json(get_camera_dict_for_hashing(camera_object))
    animations_list_json = [get_json(x) for x in sorted(animations_list, key=str)]
    current_mobjects_list_json = [get_json(x) for x in current_mobjects_list]
    hash_camera, hash_animations, hash_current_mobjects = [
        zlib.crc32(repr(json_val).encode())
        for json_val in [camera_json, animations_list_json, current_mobjects_list_json]
    ]
    hash_complete = f"{hash_camera}_{hash_animations}_{hash_current_mobjects}"
    t_end = perf_counter()
    logger.debug("Hashing done in %(time)s s.", {"time": str(t_end - t_start)[:8]})
    # End of the hashing for the animation, reset all the memoize.
    _Memoizer.reset_already_processed()
    logger.debug("Hash generated :  %(h)s", {"h": hash_complete})
    return hash_complete
