import json
import zlib
import inspect
import numpy as np


class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        '''
        Convert objects unrecognized by the default encoder

        Parameters
        ----------
        obj : any
                Arbitrary object to convert

        Returns
        -------
        any
                Python object that JSON encoder will recognize

        '''
        if callable(o):
            return inspect.getsource(o)  # Improvement to do ?
        elif isinstance(o, np.ndarray):
            return list(o)
        elif hasattr(o, "__dict__"):
            return getattr(o, '__dict__')
        return json.JSONEncoder.default(self, o)


def get_json(obj):
    return json.dumps(obj, cls=CustomEncoder,)


def get_hash_from_play_call(animations_list, current_mobjects_list):
    animations_list_json = [get_json(x).encode() for x in sorted(
        animations_list, key=lambda obj: str(obj))]
    current_mobjects_list_json = [get_json(x).encode() for x in sorted(
        current_mobjects_list, key=lambda obj: str(obj))]
    hash_animations = zlib.crc32(repr(animations_list_json).encode())
    hash_current_mobjects = zlib.crc32(
        repr(current_mobjects_list_json).encode())
    return "{}_{}".format(hash_animations, hash_current_mobjects)
