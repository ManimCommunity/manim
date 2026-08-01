from __future__ import annotations

import gc
import json
import weakref
from zlib import crc32

import manim.utils.hashing as hashing
from manim import Square

ALREADY_PROCESSED_PLACEHOLDER = hashing._Memoizer.ALREADY_PROCESSED_PLACEHOLDER


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
    import numpy as np

    a = np.zeros((1000, 1000))
    o_ser = hashing.get_json(a)
    assert "TRUNCATED ARRAY" in o_ser


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
        json2 = hashing.get_json(obj2)
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


def test_memoizer_does_not_confuse_hash_and_id_signatures():
    """
    hash() values and id() addresses are recorded in the same set, so a numeric coincidence between the two kinds
    must not mark a never-processed object as already processed. Whether such a coincidence occurs varies from
    process to process (str hashes are randomized), so an untagged set makes play hashes unstable across runs.
    """
    memoizer = hashing._Memoizer()
    id_signed = {"a": 1}  # dicts are unhashable, so this is recorded under its id.
    memoizer.check_already_processed(id_signed)

    class HashCollidingWithRecordedId:
        def __hash__(self):
            return id(id_signed)

    collider = HashCollidingWithRecordedId()
    assert memoizer.check_already_processed(collider) is collider

    # The other direction: an id() address colliding with a recorded hash() value.
    unhashable = {"b": 2}

    class HashCollidingWithLiveId:
        def __hash__(self):
            return id(unhashable)

    memoizer.check_already_processed(HashCollidingWithLiveId())
    assert memoizer.check_already_processed(unhashable) is unhashable


def test_memoizer_does_not_confuse_fresh_objects_with_dead_ones():
    """
    A fresh object must never be reported as already processed just because it was allocated at the address of a
    dead object recorded earlier in the same pass (membership signs are id-derived, and ids are only unique among
    simultaneously live objects).
    """
    memoizer = hashing._Memoizer()
    transient = {"value": 1}
    memoizer.check_already_processed(transient)
    del transient
    for i in range(10):
        fresh = {"value": i + 2}
        assert memoizer.check_already_processed(fresh) is fresh
        del fresh


def test_memoizer_retains_each_distinct_tracked_object_once():
    memoizer = hashing._Memoizer()
    hash_tracked = object()
    memoizer.check_already_processed(hash_tracked)
    memoizer.check_already_processed(hash_tracked)

    id_tracked = {}
    memoizer.check_already_processed(id_tracked)
    memoizer.check_already_processed(id_tracked)

    assert memoizer._keep_alive == {
        id(hash_tracked): hash_tracked,
        id(id_tracked): id_tracked,
    }


def test_get_json_does_not_retain_tracked_objects_after_returning():
    class HashTracked:
        pass

    class IdTracked:
        __hash__ = None

    references = []
    for tracked_type in (HashTracked, IdTracked):
        obj = tracked_type()
        references.append(weakref.ref(obj))
        hashing.get_json(obj)
        del obj

    gc.collect()
    assert all(reference() is None for reference in references)


def test_get_json_calls_have_independent_memoizers():
    shared = {}
    assert hashing.get_json([shared]) == "[{}]"
    assert hashing.get_json([shared]) == "[{}]"


def test_shared_memoizer_spans_component_serializations():
    shared = {}
    memoizer = hashing._Memoizer()
    assert hashing._get_json([shared], memoizer) == "[{}]"
    assert hashing._get_json([shared], memoizer) == '["AP"]'


def test_sibling_closures_are_not_collapsed_to_placeholder():
    """
    Serializing many closures in one pass builds a transient closure-vars dict per closure (via inspect.getclosurevars).
    These share nothing, so none of them may collapse to the already-processed placeholder — yet without the
    keep-alive fix, dead dicts' reused addresses collapse most of them.
    """

    def make_closure(k):
        def closure(x):
            return x + k

        return closure

    closures = [make_closure(k) for k in range(20)]
    entries = json.loads(hashing.get_json(closures))
    assert len(entries) == 20
    for k, entry in enumerate(entries):
        assert entry != ALREADY_PROCESSED_PLACEHOLDER
        assert entry["nonlocals"] == {"k": k}, (
            f"closure {k} collapsed: {entry['nonlocals']!r}"
        )
