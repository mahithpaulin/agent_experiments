from utils import add, subtract, multiply, divide
import pytest

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0.1, 0.2) == pytest.approx(0.3)

def test_subtract():
    assert subtract(5, 3) == 2
    assert subtract(0, 5) == -5
    assert subtract(0.3, 0.1) == pytest.approx(0.2)

def test_multiply():
    assert multiply(2, 3) == 6
    assert multiply(-1, 5) == -5
    assert multiply(0.1, 0.2) == pytest.approx(0.02)

def test_divide():
    assert divide(6, 3) == 2
    assert divide(-6, 2) == -3
    assert divide(0.3, 0.1) == pytest.approx(3)
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)