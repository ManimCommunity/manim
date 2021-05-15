from manim.mobject.geometry import Dot
from manim.mobject.value_tracker import ComplexValueTracker, ValueTracker


def test_value_tracker_set_value():
    """Test ValueTracker.set_value()"""
    tracker = ValueTracker()
    tracker.set_value(10.0)
    assert tracker.get_value() == 10.0


def test_value_tracker_get_value():
    """Test ValueTracker.get_value()"""
    tracker = ValueTracker(10.0)
    assert tracker.get_value() == 10.0


def test_value_tracker_interpolate():
    """Test ValueTracker.interpolate()"""
    tracker1 = ValueTracker(1.0)
    tracker2 = ValueTracker(2.5)
    tracker3 = ValueTracker().interpolate(tracker1, tracker2, 0.7)
    assert tracker3.get_value() == 2.05


def test_value_tracker_increment_value():
    """Test ValueTracker.increment_value()"""
    tracker = ValueTracker(0.0)
    tracker.increment_value(10.0)
    assert tracker.get_value() == 10.0


def test_value_tracker_bool():
    """Test ValueTracker.__bool__()"""
    tracker = ValueTracker(0.0)
    assert not tracker
    tracker.increment_value(1.0)
    assert tracker


def test_value_tracker_iadd():
    """Test ValueTracker.__iadd__()"""
    tracker = ValueTracker(0.0)
    tracker += 10.0
    assert tracker.get_value() == 10.0


def test_value_tracker_ifloordiv():
    """Test ValueTracker.__ifloordiv__()"""
    tracker = ValueTracker(5.0)
    tracker //= 2.0
    assert tracker.get_value() == 2.0


def test_value_tracker_imod():
    """Test ValueTracker.__imod__()"""
    tracker = ValueTracker(20.0)
    tracker %= 3.0
    assert tracker.get_value() == 2.0


def test_value_tracker_imul():
    """Test ValueTracker.__imul__()"""
    tracker = ValueTracker(3.0)
    tracker *= 4.0
    assert tracker.get_value() == 12.0


def test_value_tracker_ipow():
    """Test ValueTracker.__ipow__()"""
    tracker = ValueTracker(3.0)
    tracker **= 3.0
    assert tracker.get_value() == 27.0


def test_value_tracker_isub():
    """Test ValueTracker.__isub__()"""
    tracker = ValueTracker(20.0)
    tracker -= 10.0
    assert tracker.get_value() == 10.0


def test_value_tracker_itruediv():
    """Test ValueTracker.__itruediv__()"""
    tracker = ValueTracker(5.0)
    tracker /= 2.0
    assert tracker.get_value() == 2.5


def test_complex_value_tracker_set_value():
    """Test ComplexValueTracker.set_value()"""
    tracker = ComplexValueTracker()
    tracker.set_value(1 + 2j)
    assert tracker.get_value() == 1 + 2j


def test_complex_value_tracker_get_value():
    """Test ComplexValueTracker.get_value()"""
    tracker = ComplexValueTracker(2.0 - 3.0j)
    assert tracker.get_value() == 2.0 - 3.0j
