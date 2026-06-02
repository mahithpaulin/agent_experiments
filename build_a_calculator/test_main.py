import pytest
from calculator import Calculator

def test_addition():
    calc = Calculator()
    assert calc.add(3, 5) == 8
    assert calc.add(-3, 5) == 2
    assert calc.add(0, 0) == 0

def test_subtraction():
    calc = Calculator()
    assert calc.subtract(10, 5) == 5
    assert calc.subtract(0, 5) == -5
    assert calc.subtract(-3, -7) == 4

def test_multiplication():
    calc = Calculator()
    assert calc.multiply(3, 5) == 15
    assert calc.multiply(-3, 5) == -15
    assert calc.multiply(0, 5) == 0

def test_division():
    calc = Calculator()
    assert calc.divide(10, 2) == pytest.approx(5)
    assert calc.divide(7, 3) == pytest.approx(2.333333, rel=1e-5)
    with pytest.raises(ValueError):
        calc.divide(10, 0)
