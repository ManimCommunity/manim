import json
import zlib
import inspect
import copy
import dis
import numpy as np
from types import ModuleType

from ..logger import logger


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        """
        This method is used to serialize objects to JSON format.
        If obj is a function, then it will return a dict with two keys : 'code', for the code source, and 'nonlocals' for all nonlocalsvalues. (including nonlocals functions, that will be serialized as this is recursive.)
        if obj is a np.darray, it converts it into a list.
        if obj is an object with __dict__ attribute, it returns its __dict__.
        Else, will let the JSONEncoder do the stuff, and throw an error if the type is not suitable for JSONEncoder. 

        Parameters
        ----------
        obj : `any`
            Arbitrary object to convert

        Returns
        -------
        `any`
            Python object that JSON encoder will recognize

        """
        if inspect.isfunction(obj) and not isinstance(obj, ModuleType):
            r = inspect.getclosurevars(obj)
            x = {**copy.copy(r.globals), **copy.copy(r.nonlocals)}
            for i in list(x):
                # NOTE : All module types objects are removed, because otherwise it throws ValueError: Circular reference detected if not. TODO
                if isinstance(x[i], ModuleType):
                    del x[i]
            return {'code': inspect.getsource(obj),
                    'nonlocals': x}
        elif isinstance(obj, np.ndarray):
            return list(obj)
        elif hasattr(obj, "__dict__"):
            return getattr(obj, '__dict__')
        elif isinstance(obj, np.uint8):
            return int(obj)
        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError:
            # This is used when the user enters an unknown type in CONFIG. Rather than throwing an error, we transform
            # it into a string "Unsupported type for hashing" so it won't affect the hash.
            return "Unsupported type for hashing"


def get_json(object):
    """Recursively serialize object to JSON. Use CustomEncoder class above.

    Paramaters
    ----------
    dict_config : :class:`dict`
        The dict to flatten

    Returns
    -------
    `str` 
        the object flattened
    """
    return json.dumps(object, cls=CustomEncoder)


def get_camera_dict_for_hashing(camera_object):
    """Remove some keys from cameraobject.__dict__ that are useless for the caching functionnality and very heavy. 

    Parameters
    ----------
    object_camera : :class:``~.Camera`
        The camera object used in the scene

    Returns
    -------
    `dict`
        Camera.__dict__ but cleaned.
    """
    camera_object_dict = copy.copy(camera_object.__dict__)
    # We have to clean a little bit camera_dict, as pixel_array and background are two very big numpy array.
    # They are not essential to caching process.
    # We also have to remove pixel_array_to_cairo_context as it conntains uses memory adress (set randomly). See l.516 get_cached_cairo_context in camera.py
    for to_clean in ['background', 'pixel_array', 'pixel_array_to_cairo_context']:
        camera_object_dict.pop(to_clean, None)
    return camera_object_dict


def get_hash_from_play_call(camera_object, animations_list, current_mobjects_list):
    """Take the list of animations and a list of mobjects and output their hash. Is meant to be used for `scene.play` function.

    Parameters
    -----------
    object_camera : :class:``~.Camera`
        The camera object used in the scene

    animations_list : :class:`list`
        The list of animations 

    current_mobjects_list : :class:`list`
        The list of mobjects.

    Returns
    -------
    `str` 
        concatenation of the hash of object_camera, animations_list and current_mobjects_list separated by '_'.
    """
    camera_json = get_json(get_camera_dict_for_hashing(camera_object))
    animations_list_json = [get_json(x) for x in sorted(
        animations_list, key=lambda obj: str(obj))]
    current_mobjects_list_json = [get_json(x) for x in sorted(
        current_mobjects_list, key=lambda obj: str(obj))]
    hash_camera = zlib.crc32(repr(camera_json).encode())
    hash_animations = zlib.crc32(repr(animations_list_json).encode())
    hash_current_mobjects = zlib.crc32(
        repr(current_mobjects_list_json).encode())
    return "{}_{}_{}".format(hash_camera, hash_animations, hash_current_mobjects)


def get_hash_from_wait_call(camera_object, wait_time, stop_condition_function, current_mobjects_list):
    """Take a wait time, a boolean function as stop_condition and a list of mobjects output their hash. Is meant to be used for `scene.wait` function.

    Parameters
    -----------
    wait_time : :class:`float`
        The time to wait

    stop_condition_function : :class:`func`
        Boolean function used as a stop_condition in `wait`.

    Returns
    -------
    `str` 
        concatenation of the hash of animations_list and current_mobjects_list separated by '_'.
    """
    camera_json = get_json(get_camera_dict_for_hashing(camera_object))
    current_mobjects_list_json = [get_json(x) for x in sorted(
        current_mobjects_list, key=lambda obj: str(obj))]
    hash_current_mobjects = zlib.crc32(
        repr(current_mobjects_list_json).encode())
    hash_camera = zlib.crc32(repr(camera_json).encode())
    if stop_condition_function != None:
        hash_function = zlib.crc32(get_json(stop_condition_function).encode())
        return "{}_{}{}_{}".format(hash_camera, str(wait_time).replace('.', '-'), hash_function, hash_current_mobjects)
    else:
        return "{}_{}_{}".format(hash_camera, str(wait_time).replace('.', '-'), hash_current_mobjects)
