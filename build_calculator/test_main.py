import pytest
from utils import add, subtract, multiply, divide
from main import calculate

def test_add():
    assert add(5, 3) == 8
    assert add(-2, 7) == 5
    assert add(0, 0) == 0
    assert add(2.5, 3.7) == 6.2


def test_subtract():
    assert subtract(5, 3) == 2
    assert subtract(-2, 7) == -9
    assert subtract(0, 0) == 0
    assert subtract(3.7, 2.5) == pytest.approx(1.2, rel=1e-4)


def test_multiply():
    assert multiply(5, 3) == 15
    assert multiply(-2, 7) == -14
    assert multiply(0, 5) == 0
    assert multiply(2.5, 4) == 10.0


def test_divide():
    assert divide(6, 3) == 2
    assert divide(-10, 2) == -5
    assert divide(0, 5) == 0
    assert divide(9, 4) == 2.25
    assert divide(5, 2) == 2.5
    assert divide(10, 3) == pytest.approx(3.333333, rel=1e-4)


def test_divide_by_zero():
    assert divide(5, 0) == "Error: Division by zero"
    assert divide(-3, 0) == "Error: Division by zero"
    assert divide(0, 0) == "Error: Division by zero"


def test_calculate():
    # Test with positive numbers
    result = calculate(6, 3)
    assert result['add'] == 9
    assert result['subtract'] == 3
    assert result['multiply'] == 18
    assert result['divide'] == 2

    # Test with negative numbers
    result = calculate(-4, 2)
    assert result['add'] == -2
    assert result['subtract'] == -6
    assert result['multiply'] == -8
    assert result['divide'] == -2

    # Test with zero
    result = calculate(5, 0)
    assert result['add'] == 5
    assert result['subtract'] == 5
    assert result['multiply'] == 0
    assert result['divide'] == "Error: Division by zero"

    # Test with floating point numbers
    result = calculate(2.5, 1.5)
    assert result['add'] == 4.0
    assert result['subtract'] == 1.0
    assert result['multiply'] == 3.75
    assert result['divide'] == pytest.approx(1.666667, rel=1e-4)