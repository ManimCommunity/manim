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


@pytest.mark.parametrize(
    ("l1", "l2", "expected"),
    [([1, 2, 3], [2, 4], [1, 3, 2, 4]), ([], [1, 2], [1, 2])],
)
def test_list_update_removes_overlap_and_appends_l2(l1, l2, expected):
    assert list_update(l1, l2) == expected


def test_list_update_preserves_duplicates_in_l2():
    assert list_update([1, 2, 3], [2, 4, 4]) == [1, 3, 2, 4, 4]
