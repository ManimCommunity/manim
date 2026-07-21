from __future__ import annotations

import json
from zlib import crc32

import numpy as np
import pytest

import manim.utils.hashing as hashing
from manim import ImageMobject, Square

ALREADY_PROCESSED_PLACEHOLDER = hashing._Memoizer.ALREADY_PROCESSED_PLACEHOLDER


@pytest.fixture(autouse=True)
def reset_already_processed():
    hashing._Memoizer.reset_already_processed()


def test_JSON_basic():
    o = {"test": 1, 2: 4, 3: 2.0}
    o_serialized = hashing.get_json(o)
    assert isinstance(o_serialized, str)
    assert o_serialized == str({"test": 1, "2": 4, "3": 2.0}).replace("'", '"')


def test_JSON_with_object():
    class Obj:
        def __init__(self, a):
            self.a = a
            self.b = 3.0
            self.c = [1, 2, "test", ["nested list"]]
            self.d = {2: 3, "2": "salut"}

    o = Obj(2)
    o_serialized = hashing.get_json(o)
    assert (
        str(o_serialized)
        == '{"a": 2, "b": 3.0, "c": [1, 2, "test", ["nested list"]], "d": {"2": 3, "2": "salut"}}'
    )


def test_JSON_with_function():
    def test(uhu):
        uhu += 2
        return uhu

    o_serialized = hashing.get_json(test)
    dict_o = json.loads(o_serialized)
    assert "code" in dict_o
    assert "nonlocals" in dict_o
    assert (
        str(o_serialized)
        == r'{"code": "    def test(uhu):\n        uhu += 2\n        return uhu\n", "nonlocals": {}}'
    )


def test_JSON_with_function_and_external_val():
    external = 2

    def test(uhu):
        uhu += external
        return uhu

    o_ser = hashing.get_json(test)
    external = 3
    o_ser2 = hashing.get_json(test)
    assert json.loads(o_ser2)["nonlocals"] == {"external": 3}
    assert o_ser != o_ser2


def test_JSON_with_method():
    class A:
        def __init__(self):
            self.a = self.method
            self.b = 3

        def method(self, b):
            b += 3
            return b

    o_ser = hashing.get_json(A())
    dict_o = json.loads(o_ser)
    assert dict_o["a"]["nonlocals"] == {}


def test_JSON_with_wrong_keys():
    def test():
        return 3

    class Test:
        def __init__(self):
            self.a = 2

    a = {(1, 2): 3}
    b = {Test(): 3}
    c = {test: 3}
    for el in [a, b, c]:
        o_ser = hashing.get_json(el)
        dict_o = json.loads(o_ser)
        # check if this is an int (it meant that the lkey has been hashed)
        assert int(list(dict_o.keys())[0])


def test_JSON_with_circular_references():
    B = {1: 2}

    class A:
        def __init__(self):
            self.b = B

    B["circular_ref"] = A()
    o_ser = hashing.get_json(B)
    dict_o = json.loads(o_ser)
    assert dict_o["circular_ref"]["b"] == ALREADY_PROCESSED_PLACEHOLDER


def test_JSON_with_big_np_array():
    a = np.zeros((1000, 1000))
    serialized = hashing.get_json(a)
    hashing._Memoizer.reset_already_processed()
    assert hashing.get_json(a.copy()) == serialized

    a[500, 500] = 1
    hashing._Memoizer.reset_already_processed()
    assert hashing.get_json(a) != serialized


def test_JSON_with_equivalent_np_array_layouts():
    logical = np.arange(24, dtype=np.float64).reshape(4, 6) / 3
    serialized = hashing.get_json(np.array(logical, order="C"))

    for equivalent in [
        np.array(logical, order="F"),
        logical.astype(logical.dtype.newbyteorder(">")),
    ]:
        hashing._Memoizer.reset_already_processed()
        assert hashing.get_json(equivalent) == serialized

    backing = np.empty((4, 12), dtype=np.float64)
    backing[:, ::2] = logical
    backing[:, 1::2] = -1
    hashing._Memoizer.reset_already_processed()
    assert hashing.get_json(backing[:, ::2]) == serialized


def test_JSON_with_out_of_order_and_overlapping_structured_arrays():
    for dtype in [
        np.dtype(
            {
                "names": ["late", "early"],
                "formats": ["<i4", "<i2"],
                "offsets": [8, 0],
                "itemsize": 12,
            }
        ),
        np.dtype(
            {
                "names": ["wide", "narrow"],
                "formats": ["<u4", "<u2"],
                "offsets": [0, 2],
                "itemsize": 4,
            }
        ),
    ]:
        array = np.zeros(8, dtype=dtype)
        first_field = dtype.names[0]
        array[first_field] = np.arange(8) + 100
        hashing._Memoizer.reset_already_processed()
        serialized = hashing.get_json(array)

        changed = array.copy()
        changed[first_field][4] += 1
        hashing._Memoizer.reset_already_processed()
        assert hashing.get_json(changed) != serialized


def test_JSON_preserves_structured_dtype_identity_without_hashing_padding():
    compact_dtype = np.dtype([("value", "<i4")])
    padded_dtype = np.dtype(
        {
            "names": ["value"],
            "formats": ["<i4"],
            "offsets": [0],
            "itemsize": 12,
        }
    )
    compact = np.zeros(4, dtype=compact_dtype)
    padded = np.zeros(4, dtype=padded_dtype)
    compact["value"] = padded["value"] = np.arange(4)

    compact_json = hashing.get_json(compact)
    hashing._Memoizer.reset_already_processed()
    assert hashing.get_json(padded) != compact_json

    shifted_dtype = np.dtype(
        {
            "names": ["value"],
            "formats": ["<i4"],
            "offsets": [4],
            "itemsize": 12,
        }
    )
    shifted = np.zeros(4, dtype=shifted_dtype)
    shifted["value"] = np.arange(4)
    hashing._Memoizer.reset_already_processed()
    shifted_json = hashing.get_json(shifted)
    hashing._Memoizer.reset_already_processed()
    assert shifted_json != hashing.get_json(padded)

    alternate_padding = padded.copy()
    alternate_padding.view(np.uint8).reshape(4, 12)[:, 4:] = 0xFF
    hashing._Memoizer.reset_already_processed()
    padded_json = hashing.get_json(padded)
    hashing._Memoizer.reset_already_processed()
    assert hashing.get_json(alternate_padding) == padded_json

    titled = np.zeros(4, dtype=np.dtype([(("title", "value"), "<i4")]))
    titled["value"] = np.arange(4)
    hashing._Memoizer.reset_already_processed()
    assert hashing.get_json(titled) != compact_json

    aligned_dtype = np.dtype([("flag", "u1"), ("value", "<u4")], align=True)
    unaligned_dtype = np.dtype(
        {
            "names": ["flag", "value"],
            "formats": ["u1", "<u4"],
            "offsets": [0, 4],
            "itemsize": 8,
        },
        align=False,
    )
    aligned = np.zeros(4, dtype=aligned_dtype)
    unaligned = np.zeros(4, dtype=unaligned_dtype)
    aligned["value"] = unaligned["value"] = np.arange(4)
    hashing._Memoizer.reset_already_processed()
    aligned_json = hashing.get_json(aligned)
    hashing._Memoizer.reset_already_processed()
    assert aligned_json != hashing.get_json(unaligned)

    hashing._Memoizer.reset_already_processed()
    empty_struct_json = hashing.get_json(np.empty(4, dtype=np.dtype([])))
    hashing._Memoizer.reset_already_processed()
    assert hashing.get_json(np.empty(4, dtype=np.dtype("V0"))) != empty_struct_json

    empty_padded_dtype = np.dtype({"names": [], "formats": [], "itemsize": 8})
    empty_padded = np.zeros(4, dtype=empty_padded_dtype)
    alternate_empty_padding = empty_padded.copy()
    alternate_empty_padding.view(np.uint8)[:] = 0xFF
    hashing._Memoizer.reset_already_processed()
    empty_padded_json = hashing.get_json(empty_padded)
    hashing._Memoizer.reset_already_processed()
    assert hashing.get_json(alternate_empty_padding) == empty_padded_json
    hashing._Memoizer.reset_already_processed()
    assert hashing.get_json(np.zeros(4, dtype="V8")) != empty_padded_json

    bytes_title = np.zeros(4, dtype=np.dtype([((b"title", "value"), "<i4")]))
    scalar_title = np.zeros(
        4,
        dtype=np.dtype([((np.int64(7), "value"), "<i4")]),
    )
    bytes_title["value"] = scalar_title["value"] = np.arange(4)
    hashing._Memoizer.reset_already_processed()
    bytes_title_json = hashing.get_json(bytes_title)
    hashing._Memoizer.reset_already_processed()
    assert bytes_title_json != hashing.get_json(scalar_title)


def test_play_hash_includes_mobject_pixels_but_not_camera_pixels():
    class HashableObject:
        def __init__(self, name: str, pixels: np.ndarray) -> None:
            self.name = name
            self.pixel_array = pixels

        def __str__(self) -> str:
            return self.name

    scene = HashableObject("scene", np.arange(8, dtype=np.uint8))
    camera = HashableObject("camera", np.arange(8, dtype=np.uint8))
    mobject = ImageMobject(np.zeros((8, 8, 4), dtype=np.uint8))

    original = hashing.get_hash_from_play_call(scene, camera, [], [mobject])
    mobject.pixel_array[4, 4, 0] ^= 1
    assert hashing.get_hash_from_play_call(scene, camera, [], [mobject]) != original

    mobject.pixel_array[4, 4, 0] ^= 1
    camera.pixel_array[0] ^= 1
    assert hashing.get_hash_from_play_call(scene, camera, [], [mobject]) == original


def test_play_hash_keeps_distinct_mobjects_with_equal_python_hashes():
    class CollidingObject:
        def __init__(self, name: str) -> None:
            self.name = name

        def __str__(self) -> str:
            return self.name

        def __hash__(self) -> int:
            return 1

    scene = CollidingObject("scene")
    camera = CollidingObject("camera")
    mobjects = [CollidingObject("first"), CollidingObject("second")]

    original = hashing.get_hash_from_play_call(scene, camera, [], mobjects)
    mobjects[1].name = "changed"
    assert hashing.get_hash_from_play_call(scene, camera, [], mobjects) != original


def test_JSON_with_tuple():
    o = [(1, [1])]
    o_ser = hashing.get_json(o)
    assert o_ser == "[[1, [1]]]"


def test_JSON_with_object_that_is_itself_circular_reference():
    class T:
        def __init__(self) -> None:
            self.a = None

    o = T()
    o.a = o
    hashing.get_json(o)


def test_hash_consistency():
    def assert_two_objects_produce_same_hash(obj1, obj2, debug=False):
        """
        When debug is True, if the hashes differ an assertion comparing (element-wise) the two objects will be raised,
        and pytest will display a nice difference summary making it easier to debug.
        """
        json1 = hashing.get_json(obj1)
        hashing._Memoizer.reset_already_processed()
        json2 = hashing.get_json(obj2)
        hashing._Memoizer.reset_already_processed()
        hash1 = crc32(repr(json1).encode())
        hash2 = crc32(repr(json2).encode())
        if hash1 != hash2 and debug:
            dict1 = json.loads(json1)
            dict2 = json.loads(json2)
            assert dict1 == dict2
        assert hash1 == hash2, f"{obj1} and {obj2} have different hashes."

    assert_two_objects_produce_same_hash(Square(), Square())
    s = Square()
    assert_two_objects_produce_same_hash(s, s.copy())
