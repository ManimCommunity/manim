"""Utilities for scene caching."""

from __future__ import annotations

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

from manim.animation.animation import Animation
from manim.camera.camera import Camera
from manim.mobject.mobject import Mobject

from .. import config, logger

if typing.TYPE_CHECKING:
    from manim.scene.scene import Scene

__all__ = ["KEYS_TO_FILTER_OUT", "get_hash_from_play_call", "get_json"]

# Sometimes there are elements that are not suitable for hashing (too long or
# run-dependent).  This is used to filter them out.
KEYS_TO_FILTER_OUT = {
    "original_id",
    "background",
    "pixel_array",
    "pixel_array_to_cairo_context",
}


class _Memoizer:
    """Implements the memoization logic to optimize the hashing procedure and prevent
    the circular references within iterable processed.

    Keeps a record of all the processed objects, and handle the logic to return a place
    holder instead of the original object if the object has already been processed
    by the hashing logic (i.e, recursively checked, converted to JSON, etc..).

    This class uses two signatures functions to keep a track of processed objects :
    hash or id. Whenever possible, hash is used to ensure a broader object
    content-equality detection.
    """

    _already_processed = set()

    # Can be changed to whatever string to help debugging the JSon generation.
    ALREADY_PROCESSED_PLACEHOLDER = "AP"
    THRESHOLD_WARNING = 170_000

    @classmethod
    def reset_already_processed(cls):
        cls._already_processed.clear()

    @classmethod
    def check_already_processed_decorator(cls: _Memoizer, is_method: bool = False):
        """Decorator to handle the arguments that goes through the decorated function.
        Returns _ALREADY_PROCESSED_PLACEHOLDER if the obj has been processed, or lets
        the decorated function call go ahead.

        Parameters
        ----------
        is_method
            Whether the function passed is a method, by default False.
        """

        def layer(func):
            # NOTE : There is probably a better way to separate both case when func is
            # a method or a function.
            if is_method:
                return lambda self, obj: cls._handle_already_processed(
                    obj,
                    default_function=lambda obj: func(self, obj),
                )
            return lambda obj: cls._handle_already_processed(obj, default_function=func)

        return layer

    @classmethod
    def check_already_processed(cls, obj: Any) -> Any:
        """Checks if obj has been already processed. Returns itself if it has not been,
        or the value of _ALREADY_PROCESSED_PLACEHOLDER if it has.
        Marks the object as processed in the second case.

        Parameters
        ----------
        obj
            The object to check.

        Returns
        -------
        Any
            Either the object itself or the placeholder.
        """
        # When the object is not memoized, we return the object itself.
        return cls._handle_already_processed(obj, lambda x: x)

    @classmethod
    def mark_as_processed(cls, obj: Any) -> None:
        """Marks an object as processed.

        Parameters
        ----------
        obj
            The object to mark as processed.
        """
        cls._handle_already_processed(obj, lambda x: x)
        return cls._return(obj, id, lambda x: x, memoizing=False)

    @classmethod
    def _handle_already_processed(
        cls,
        obj,
        default_function: typing.Callable[[Any], Any],
    ):
        if isinstance(
            obj,
            (
                int,
                float,
                str,
                complex,
            ),
        ) and obj not in [None, cls.ALREADY_PROCESSED_PLACEHOLDER]:
            # It makes no sense (and it'd slower) to memoize objects of these primitive
            # types.  Hence, we simply return the object.
            return obj
        if isinstance(obj, collections.abc.Hashable):
            try:
                return cls._return(obj, hash, default_function)
            except TypeError:
                # In case of an error with the hash (eg an object is marked as hashable
                # but contains a non hashable within it)
                # Fallback to use the built-in function id instead.
                pass
        return cls._return(obj, id, default_function)

    @classmethod
    def _return(
        cls,
        obj: typing.Any,
        obj_to_membership_sign: typing.Callable[[Any], int],
        default_func,
        memoizing=True,
    ) -> str | Any:
        obj_membership_sign = obj_to_membership_sign(obj)
        if obj_membership_sign in cls._already_processed:
            return cls.ALREADY_PROCESSED_PLACEHOLDER
        if memoizing:
            if (
                not config.disable_caching_warning
                and len(cls._already_processed) == cls.THRESHOLD_WARNING
            ):
                logger.warning(
                    "It looks like the scene contains a lot of sub-mobjects. Caching "
                    "is sometimes not suited to handle such large scenes, you might "
                    "consider disabling caching with --disable_caching to potentially "
                    "speed up the rendering process.",
                )
                logger.warning(
                    "You can disable this warning by setting disable_caching_warning "
                    "to True in your config file.",
                )

            cls._already_processed.add(obj_membership_sign)
        return default_func(obj)


class _CustomEncoder(json.JSONEncoder):
    def default(self, obj: Any):
        """
        This method is used to serialize objects to JSON format.

        If obj is a function, then it will return a dict with two keys : 'code', for
        the code source, and 'nonlocals' for all nonlocalsvalues. (including nonlocals
        functions, that will be serialized as this is recursive.)
        if obj is a np.darray, it converts it into a list.
        if obj is an object with __dict__ attribute, it returns its __dict__.
        Else, will let the JSONEncoder do the stuff, and throw an error if the type is
        not suitable for JSONEncoder.

        Parameters
        ----------
        obj
            Arbitrary object to convert

        Returns
        -------
        Any
            Python object that JSON encoder will recognize

        """
        if not (isinstance(obj, ModuleType)) and isinstance(
            obj,
            (MethodType, FunctionType),
        ):
            cvars = inspect.getclosurevars(obj)
            cvardict = {**copy.copy(cvars.globals), **copy.copy(cvars.nonlocals)}
            for i in list(cvardict):
                # NOTE : All module types objects are removed, because otherwise it
                # throws ValueError: Circular reference detected if not. TODO
                if isinstance(cvardict[i], ModuleType):
                    del cvardict[i]
            try:
                code = inspect.getsource(obj)
            except (OSError, TypeError):
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
        # Serialize it with only the type of the object. You can change this to whatever string when debugging the serialization process.
        return str(type(obj))

    def _cleaned_iterable(self, iterable: typing.Iterable[Any]):
        """Check for circular reference at each iterable that will go through the JSONEncoder, as well as key of the wrong format.

        If a key with a bad format is found (i.e not a int, string, or float), it gets replaced byt its hash using the same process implemented here.
        If a circular reference is found within the iterable, it will be replaced by the string "already processed".

        Parameters
        ----------
        iterable
            The iterable to check.
        """

        def _key_to_hash(key):
            return zlib.crc32(json.dumps(key, cls=_CustomEncoder).encode())

        def _iter_check_list(lst):
            processed_list = [None] * len(lst)
            for i, el in enumerate(lst):
                el = _Memoizer.check_already_processed(el)
                if isinstance(el, (list, tuple)):
                    new_value = _iter_check_list(el)
                elif isinstance(el, dict):
                    new_value = _iter_check_dict(el)
                else:
                    new_value = el
                processed_list[i] = new_value
            return processed_list

        def _iter_check_dict(dct):
            processed_dict = {}
            for k, v in dct.items():
                v = _Memoizer.check_already_processed(v)
                if k in KEYS_TO_FILTER_OUT:
                    continue
                # We check if the k is of the right format (supporter by Json)
                if not isinstance(k, (str, int, float, bool)) and k is not None:
                    k_new = _key_to_hash(k)
                else:
                    k_new = k
                if isinstance(v, dict):
                    new_value = _iter_check_dict(v)
                elif isinstance(v, (list, tuple)):
                    new_value = _iter_check_list(v)
                else:
                    new_value = v
                processed_dict[k_new] = new_value
            return processed_dict

        if isinstance(iterable, (list, tuple)):
            return _iter_check_list(iterable)
        elif isinstance(iterable, dict):
            return _iter_check_dict(iterable)

    def encode(self, obj: Any):
        """Overriding of :meth:`JSONEncoder.encode`, to make our own process.

        Parameters
        ----------
        obj
            The object to encode in JSON.

        Returns
        -------
        :class:`str`
           The object encoder with the standard json process.
        """
        _Memoizer.mark_as_processed(obj)
        if isinstance(obj, (dict, list, tuple)):
            return super().encode(self._cleaned_iterable(obj))
        return super().encode(obj)


def get_json(obj: dict):
    """Recursively serialize `object` to JSON using the :class:`CustomEncoder` class.

    Parameters
    ----------
    obj
        The dict to flatten

    Returns
    -------
    :class:`str`
        The flattened object
    """
    return json.dumps(obj, cls=_CustomEncoder)


def get_hash_from_play_call(
    scene_object: Scene,
    camera_object: Camera,
    animations_list: typing.Iterable[Animation],
    current_mobjects_list: typing.Iterable[Mobject],
) -> str:
    """Take the list of animations and a list of mobjects and output their hashes. This is meant to be used for `scene.play` function.

    Parameters
    -----------
    scene_object
        The scene object.

    camera_object
        The camera object used in the scene.

    animations_list
        The list of animations.

    current_mobjects_list
        The list of mobjects.

    Returns
    -------
    :class:`str`
        A string concatenation of the respective hashes of `camera_object`, `animations_list` and `current_mobjects_list`, separated by `_`.
    """
    logger.debug("Hashing ...")
    t_start = perf_counter()
    _Memoizer.mark_as_processed(scene_object)
    camera_json = get_json(camera_object)
    animations_list_json = [get_json(x) for x in sorted(animations_list, key=str)]
    current_mobjects_list_json = [get_json(x) for x in current_mobjects_list]
    hash_camera, hash_animations, hash_current_mobjects = (
        zlib.crc32(repr(json_val).encode())
        for json_val in [camera_json, animations_list_json, current_mobjects_list_json]
    )
    hash_complete = f"{hash_camera}_{hash_animations}_{hash_current_mobjects}"
    t_end = perf_counter()
    logger.debug("Hashing done in %(time)s s.", {"time": str(t_end - t_start)[:8]})
    # End of the hashing for the animation, reset all the memoize.
    _Memoizer.reset_already_processed()
    logger.debug("Hash generated :  %(h)s", {"h": hash_complete})
    return hash_complete
