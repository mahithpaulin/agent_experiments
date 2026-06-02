# test_main.py
import pytest
from main import Calculator
from utils import validate_numbers, safe_divide, safe_multiply, safe_add, safe_subtract

@pytest.fixture
def calculator():
    return Calculator()

def test_add(calculator):
    assert calculator.add(3, 2) == 5
    assert calculator.add(-3, 2) == -1
    assert calculator.add(0, 0) == 0
    assert calculator.add(1.5, 2.5) == pytest.approx(4.0)

def test_subtract(calculator):
    assert calculator.subtract(5, 3) == 2
    assert calculator.subtract(-5, -3) == -2
    assert calculator.subtract(0, 0) == 0
    assert calculator.subtract(2.5, 1.5) == pytest.approx(1.0)

def test_multiply(calculator):
    assert calculator.multiply(4, 3) == 12
    assert calculator.multiply(-4, 3) == -12
    assert calculator.multiply(0, 5) == 0
    assert calculator.multiply(1.5, 2) == pytest.approx(3.0)

def test_divide(calculator):
    assert calculator.divide(6, 3) == pytest.approx(2.0)
    assert calculator.divide(-6, 3) == pytest.approx(-2.0)
    assert calculator.divide(1.5, 0.5) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        calculator.divide(5, 0)

def test_validate_numbers():
    assert validate_numbers(3, 2) is True
    assert validate_numbers(1.5, -2.5) is True
    with pytest.raises(TypeError):
        validate_numbers(3, "a")
    with pytest.raises(TypeError):
        validate_numbers("b", 2)

def test_safe_add():
    assert safe_add(3, 2) == 5
    assert safe_add(-3, 2) == -1
    assert safe_add(1.5, 2.5) == pytest.approx(4.0)
    with pytest.raises(TypeError):
        safe_add(3, "a")

def test_safe_subtract():
    assert safe_subtract(5, 3) == 2
    assert safe_subtract(-5, -3) == -2
    assert safe_subtract(2.5, 1.5) == pytest.approx(1.0)
    with pytest.raises(TypeError):
        safe_subtract("a", 3)

def test_safe_multiply():
    assert safe_multiply(4, 3) == 12
    assert safe_multiply(-4, 3) == -12
    assert safe_multiply(1.5, 2) == pytest.approx(3.0)
    with pytest.raises(TypeError):
        safe_multiply(4, "b")

def test_safe_divide():
    assert safe_divide(6, 3) == pytest.approx(2.0)
    assert safe_divide(-6, 3) == pytest.approx(-2.0)
    assert safe_divide(1.5, 0.5) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        safe_divide(5, 0)
    with pytest.raises(TypeError):
        safe_divide("a", 3)