import pytest

from manim.utils.iterables import list_difference_update, list_update

ORDERED_ITERABLE_TYPES = [
    # Common non-list iterables
    pytest.param(tuple, id="tuple"),
    pytest.param(lambda values: dict.fromkeys(values).keys(), id="dictkeys"),
    # One-shot iterables
    pytest.param(iter, id="iter"),
    pytest.param(
        lambda values: (value for value in values),
        id="generator",
    ),
]

STANDARD_LIST_DIFFERENCE_CASES = [
    ([1, 2, 3, 4], [4, 2], [1, 3]),
    ([], [1, 2], []),
    ([1, 2, 3], [], [1, 2, 3]),
    # Disjoint input
    ([1, 2, 3], [4, 5], [1, 2, 3]),
    # Input with same elements
    ([1, 2, 3], [3, 2, 1], []),
    # Non-int input with equality semantics
    (["a", "b", "A", "C", "D"], ["A", "d"], ["a", "b", "C", "D"]),
    # Input without equality semantics
    ([ob1 := object(), ob2 := object(), ob3 := object()], [ob1, ob3], [ob2]),
]

# Reuse cases, since list_update is equivalent to list_difference_update(l1, l2) + l2
STANDARD_LIST_UPDATE_CASES = [
    (l1, l2, expected + l2) for l1, l2, expected in STANDARD_LIST_DIFFERENCE_CASES
]


@pytest.mark.parametrize(("l1", "l2", "expected"), STANDARD_LIST_DIFFERENCE_CASES)
def test_list_difference_update_removes_matching_items(l1, l2, expected):
    assert list_difference_update(l1, l2) == expected


@pytest.mark.parametrize("l1_type", ORDERED_ITERABLE_TYPES)
@pytest.mark.parametrize("l2_type", ORDERED_ITERABLE_TYPES)
def test_list_difference_update_accepts_iterables(l1_type, l2_type):
    l1, l2, expected = STANDARD_LIST_DIFFERENCE_CASES[0]
    assert list_difference_update(l1_type(l1), l2_type(l2)) == expected


def test_list_difference_update_preserves_l1_duplicates_and_order():
    assert list_difference_update([3, 1, 3, 2, 1], [1]) == [3, 3, 2]


def test_list_difference_update_accepts_duplicates_in_l2():
    assert list_difference_update([1, 2, 3], [2, 4, 4]) == [1, 3]


def test_list_difference_update_preserves_l1_order_and_duplicates():
    assert list_difference_update([3, 1, 3, 2, 1], [1]) == [3, 3, 2]


def test_list_difference_update_with_unordered_l1_removes_matching_items():
    """Test that matching items are correctly removed, even if there is no order to
    preserve in l1.
    """
    l1 = {(0, 4), (1, 3), (3, "A")}
    result = list_difference_update(l1, [(5, "B"), (1, 3)])
    assert sorted(result) == [(0, 4), (3, "A")]


@pytest.mark.parametrize(("l1", "l2", "expected"), STANDARD_LIST_DIFFERENCE_CASES)
def test_list_difference_update_with_key_none_uses_default_equality(l1, l2, expected):
    assert list_difference_update(l1, l2, key=None) == expected


@pytest.mark.parametrize(("l1", "l2", "expected"), STANDARD_LIST_DIFFERENCE_CASES)
def test_list_difference_update_with_no_key_function_uses_default_equality(
    l1, l2, expected
):
    assert list_difference_update(l1, l2) == expected


@pytest.mark.parametrize(("l1", "l2", "expected"), STANDARD_LIST_DIFFERENCE_CASES)
def test_list_difference_update_with_identity_function_uses_default_equality(
    l1, l2, expected
):
    """Test that it is possible to pass through the element itself and use default
    equality semantics.
    """
    assert list_difference_update(l1, l2, key=lambda x: x) == expected


@pytest.mark.parametrize(
    ("l1", "l2", "key", "expected"),
    [
        (["a", "b", "C", "A", "C", "a"], ["A", "D"], str.lower, ["b", "C", "C"]),
        ([1, 2, 11, 111, 12, 2, 20, 1], [2, 4], lambda x: x % 10, [1, 11, 111, 20, 1]),
        # Unhashable input
        ([1, [1], 2, 3, [3], [1]], [[1], 3], str, [1, 2, [3]]),
    ],
)
def test_list_difference_update_with_key_function_matches_unequal_items(
    l1, l2, key, expected
):
    """Test that items in l1 are removed if they match items in l2 according to the key
    function, even if they are not equal.
    """
    assert list_difference_update(l1, l2, key=key) == expected


@pytest.mark.parametrize("l1_type", ORDERED_ITERABLE_TYPES)
@pytest.mark.parametrize("l2_type", ORDERED_ITERABLE_TYPES)
def test_list_difference_update_with_key_function_accepts_iterables(l1_type, l2_type):
    l1, l2, expected = (["a", "b", "C", "A", "a"], ["A", "D"], ["b", "C"])
    assert list_difference_update(l1_type(l1), l2_type(l2), key=str.lower) == expected


@pytest.mark.parametrize(
    ("l1", "l2", "key", "expected"),
    [
        ([1, "1", 1.0, 2, "2.0", "A"], [1, "2"], str, [1.0, "2.0", "A"]),
        ([1, "1", 1.0, "1.0", 1.5, 2, "3"], [1, "2", "3.0"], float, [1.5]),
    ],
)
def test_list_difference_update_with_key_function_matches_different_types(
    l1, l2, key, expected
):
    """Test that items in l1 are removed if they match items in l2 according to the key
    function, even if they are not equal.
    """
    assert list_difference_update(l1, l2, key=key) == expected


@pytest.mark.parametrize(("l1", "l2", "expected"), STANDARD_LIST_UPDATE_CASES)
def test_list_update_removes_overlap_and_appends_l2(l1, l2, expected):
    assert list_update(l1, l2) == expected


@pytest.mark.parametrize("l1_type", ORDERED_ITERABLE_TYPES)
@pytest.mark.parametrize("l2_type", ORDERED_ITERABLE_TYPES)
def test_list_update_accepts_iterables(l1_type, l2_type):
    l1, l2, expected = STANDARD_LIST_UPDATE_CASES[0]
    assert list_update(l1_type(l1), l2_type(l2)) == expected


def test_list_update_preserves_l1_duplicates_and_order():
    assert list_update([3, 1, 3, 2, 1], [1]) == [3, 3, 2, 1]


def test_list_update_preserves_duplicates_in_l2():
    assert list_update([1, 2, 3], [2, 4, 4]) == [1, 3, 2, 4, 4]


def test_list_update_with_unordered_l1_removes_matching_items():
    """Test that matching items are correctly removed, even if there is no order to
    preserve in l1.
    """
    l1 = {(0, 4), (1, 3), (3, "A")}
    result = list_update(l1, [(5, "B"), (1, 3)])
    assert sorted(result) == [(0, 4), (1, 3), (3, "A"), (5, "B")]
    assert result[-2:] == [(5, "B"), (1, 3)]  # l2 is appended to the end of the result


@pytest.mark.parametrize(("l1", "l2", "expected"), STANDARD_LIST_UPDATE_CASES)
def test_list_update_with_key_none_uses_default_equality(l1, l2, expected):
    assert list_update(l1, l2, key=None) == expected


@pytest.mark.parametrize(("l1", "l2", "expected"), STANDARD_LIST_UPDATE_CASES)
def test_list_update_with_identity_function_uses_default_equality(l1, l2, expected):
    """Test that it is possible to pass through the element itself and use default
    equality semantics.
    """
    assert list_update(l1, l2, key=lambda x: x) == expected


@pytest.mark.parametrize(
    ("l1", "l2", "key", "expected"),
    [
        (["a", "b", "A", "C"], ["A", "B", "D"], str.lower, ["C", "A", "B", "D"]),
        ([1, 2, 11, 111, 12, 2, 1], [2, 4], lambda x: x % 10, [1, 11, 111, 1, 2, 4]),
        # Unhashable input
        ([1, [1], 2, 3, [3]], [[1], 3], str, [1, 2, [3], [1], 3]),
    ],
)
def test_list_update_with_key_function_matches_unequal_items(l1, l2, key, expected):
    """Test that items in l1 are removed if they match items in l2 according to the key
    function, even if they are not equal.
    """
    assert list_update(l1, l2, key=key) == expected


@pytest.mark.parametrize("l1_type", ORDERED_ITERABLE_TYPES)
@pytest.mark.parametrize("l2_type", ORDERED_ITERABLE_TYPES)
def test_list_update_with_key_function_accepts_iterables(l1_type, l2_type):
    """Test that items in l1 are removed if they match items in l2 according to the key
    function, even if they are not equal.
    """
    l1, l2, expected = (["a", "b", "C", "A", "a"], ["A", "D"], ["b", "C", "A", "D"])
    assert list_update(l1_type(l1), l2_type(l2), key=str.lower) == expected


@pytest.mark.parametrize(
    ("l1", "l2", "key", "expected"),
    [
        ([1, "1", 1.0, 2, "2.0", "A"], [1, "2"], str, [1.0, "2.0", "A", 1, "2"]),
        (["1", 1.0, "1.0", 1.5, 2, "3"], [1, "2", "3.0"], float, [1.5, 1, "2", "3.0"]),
    ],
)
def test_list_update_with_key_function_matches_different_types(l1, l2, key, expected):
    """Test that items in l1 are removed if they match items in l2 according to the key
    function, even if they are not equal.
    """
    assert list_update(l1, l2, key=key) == expected
