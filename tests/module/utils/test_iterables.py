import pytest

from manim.utils.iterables import list_difference_update, list_update


def test_list_difference_update_removes_matching_items():
    assert list_difference_update([1, 2, 3, 4], [2, 4]) == [1, 3]


def test_list_difference_update_preserves_duplicates_and_order():
    assert list_difference_update([3, 1, 3, 2, 1], [1]) == [3, 3, 2]


@pytest.mark.parametrize(
    ("l1", "l2", "expected"),
    [([1, 2, 3, 4], [2, 4], [1, 3]), ([], [1, 2], [])],
)
def test_list_difference_update_removes_elements(l1, l2, expected):
    assert list_difference_update(l1, l2) == expected


def test_list_difference_update_preserves_l1_order_and_duplicates():
    assert list_difference_update([3, 1, 3, 2, 1], [1]) == [3, 3, 2]


def test_list_difference_update_with_key_none_uses_default_equality():
    assert list_difference_update([1, 2, 3], [2], key=None) == [1, 3]


def test_list_difference_update_uses_key_function():
    result = list_difference_update(["a", "b", "A", "C"], ["A", "D"], key=str.lower)
    assert result == ["b", "C"]


@pytest.mark.parametrize(
    ("l1", "l2", "expected"),
    [([1, 2, 3], [2, 4], [1, 3, 2, 4]), ([], [1, 2], [1, 2])],
)
def test_list_update_removes_overlap_and_appends_l2(l1, l2, expected):
    assert list_update(l1, l2) == expected


def test_list_update_preserves_duplicates_in_l2():
    assert list_update([1, 2, 3], [2, 4, 4]) == [1, 3, 2, 4, 4]


def test_list_update_with_key_none_uses_default_equality():
    assert list_update([1, 2, 3], [2, 4], key=None) == [1, 3, 2, 4]


def test_list_update_uses_key_function():
    result = list_update(["a", "b", "c", "A", "B", "C"], ["A", "b", "D"], key=str.lower)
    assert result == ["c", "C", "A", "b", "D"]
