import pytest
from utils import add, subtract, multiply, divide

def test_add_integers():
    assert add(2, 3) == 5
    assert add(-1, -4) == -5
    assert add(0, 0) == 0

def test_subtract_integers():
    assert subtract(10, 4) == 6
    assert subtract(-5, -2) == -3
    assert subtract(0, 5) == -5

def test_multiply_mixed():
    assert multiply(7, 6) == 42
    assert multiply(-3, 5) == -15
    assert multiply(0, 100) == 0

def test_divide_floats():
    assert divide(10, 4) == pytest.approx(2.5)
    assert divide(-9, 3) == pytest.approx(-3.0)
    assert divide(5, 2) == pytest.approx(2.5)

def test_divide_by_zero():
    with pytest.raises(ValueError) as excinfo:
        divide(10, 0)
    assert "zero" in str(excinfo.value).lower()

def test_commutative_addition():
    a, b = 12.7, 3.3
    assert add(a, b) == pytest.approx(add(b, a))

def test_associative_multiplication():
    a, b, c = 2, 3, 4
    left = multiply(multiply(a, b), c)
    right = multiply(a, multiply(b, c))
    assert left == right

def test_divide_precision():
    result = divide(1, 3)
    assert result == pytest.approx(0.3333333333333333, rel=1e-12)