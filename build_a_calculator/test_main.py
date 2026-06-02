import pytest
from utils import add, subtract, multiply, divide

def test_add():
    assert add(1, 2) == 3
    assert add(-1, -1) == -2
    assert add(0, 0) == 0
    assert add(1.5, 2.5) == pytest.approx(4.0)

def test_subtract():
    assert subtract(5, 3) == 2
    assert subtract(-1, -1) == 0
    assert subtract(0, 5) == -5
    assert subtract(2.5, 1.5) == pytest.approx(1.0)

def test_multiply():
    assert multiply(3, 4) == 12
    assert multiply(-2, 3) == -6
    assert multiply(0, 100) == 0
    assert multiply(2.5, 4) == pytest.approx(10.0)

def test_divide():
    assert divide(10, 2) == 5
    assert divide(-9, 3) == -3
    assert divide(5, 2) == pytest.approx(2.5)
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)