import json
import zlib
import inspect
import numpy as np


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        """
        This method is used to encode objects contained in some dict in an object that JSON encoder will recognize.
        If obj is a function, then it will return the code of this function.
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
        if callable(obj):
            return inspect.getsource(obj)  # Improvement to do ?
        elif isinstance(obj, np.ndarray):
            return list(obj)
        elif hasattr(obj, "__dict__"):
            return getattr(obj, '__dict__')
        return json.JSONEncoder.default(self, obj)


def get_json(dict_config):
    """Flatten a dictionnary of objects by tranforming these objects into their __dict__ or another type defined in the CustomEncoder class

    Paramaters
    ----------
    dict_config : :class:`dict`
        The dict to flatten

    Returns
    -------
    `str` 
        the dict flattened
    """
    return json.dumps(dict_config, cls=CustomEncoder)


def get_hash_from_play_call(animations_list, current_mobjects_list):
    """Take the list of animations and a list of mobjects and output their hash. Is meant to be used for `scene.play` function.

    Parameters
    -----------
    animations_list : :class:`list`
        The list of animations 

    current_mobjects_list : :class:`list`
        The list of mobjects.

    Returns
    -------
    `str` 
        concatenation of the hash of animations_list and current_mobjects_list separated by '_'.
    """
    animations_list_json = [get_json(x).encode() for x in sorted(
        animations_list, key=lambda obj: str(obj))]
    current_mobjects_list_json = [get_json(x).encode() for x in sorted(
        current_mobjects_list, key=lambda obj: str(obj))]
    hash_animations = zlib.crc32(repr(animations_list_json).encode())
    hash_current_mobjects = zlib.crc32(
        repr(current_mobjects_list_json).encode())
    return "{}_{}".format(hash_animations, hash_current_mobjects)


def get_hash_from_wait_call(wait_time, stop_condition_function, current_mobjects_list):
    """Take a wait time, a boolean function as stop_condition and a list of mobjects output their hash. Is meant to be used for `scene.wait` function.

    Parameters
    -----------
    wait_time : :class:`int`
        The time to wait

    stop_condition_function : :class:`func`
        Boolean function used as a stop_condition in `wait`.

    Returns
    -------
    `str` 
        concatenation of the hash of animations_list and current_mobjects_list separated by '_'.
    """
    current_mobjects_list_json = [get_json(x).encode() for x in sorted(
        current_mobjects_list, key=lambda obj: str(obj))]
    hash_current_mobjects = zlib.crc32(
        repr(current_mobjects_list_json).encode())
    if stop_condition_function != None:
        hash_function = zlib.crc32(inspect.getsource(
            stop_condition_function).encode())
        return "{}{}_{}".format(wait_time, hash_function, hash_current_mobjects)
    else:
        return "{}_{}".format(wait_time, hash_current_mobjects)
