import math
import pytest
from utils import add, subtract, multiply, divide

@pytest.mark.parametrize(
    "a,b,expected",
    [
        (1.2, 3.4, 4.6),
        (-5.5, 4.2, -1.3),
        (0.0, 0.0, 0.0),
    ],
)
def test_add(a, b, expected):
    assert math.isclose(add(a, b), expected, rel_tol=1e-12)

@pytest.mark.parametrize(
    "a,b,expected",
    [
        (5.5, 4.2, 1.3),
        (-2.0, -3.0, 1.0),
        (0.0, 5.0, -5.0),
    ],
)
def test_subtract(a, b, expected):
    assert math.isclose(subtract(a, b), expected, rel_tol=1e-12)

@pytest.mark.parametrize(
    "a,b,expected",
    [
        (2.0, 3.5, 7.0),
        (-1.2, 4.0, -4.8),
        (0.0, 5.0, 0.0),
    ],
)
def test_multiply(a, b, expected):
    assert math.isclose(multiply(a, b), expected, rel_tol=1e-12)

@pytest.mark.parametrize(
    "a,b,expected",
    [
        (7.0, 2.0, 3.5),
        (-3.5, 2.1, -1.6666666666666665),
        (5.0, -2.0, -2.5),
    ],
)
def test_divide(a, b, expected):
    result = divide(a, b)
    assert result is not None
    # ``divide`` uses ``math.nextafter`` to move the value toward zero,
    # therefore we compare with ``expected`` using a tolerance that tolerates the
    # tiny adjustment.
    assert math.isclose(result, expected, rel_tol=1e-15)

def test_divide_by_zero():
    assert divide(1.0, 0.0) is None
