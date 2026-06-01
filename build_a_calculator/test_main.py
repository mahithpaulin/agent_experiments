import pytest
from main import calculator
from utils import add, subtract, multiply, divide

def test_add():
    assert add(10, 2) == 12
    assert add(-5, 3) == -2
    assert add(0, 0) == 0

def test_subtract():
    assert subtract(10, 2) == 8
    assert subtract(-5, 3) == -8
    assert subtract(0, 0) == 0

def test_multiply():
    assert multiply(10, 2) == 20
    assert multiply(-5, 3) == -15
    assert multiply(0, 0) == 0

def test_divide():
    assert divide(10, 2) == pytest.approx(5.0)
    assert divide(-5, 3) == pytest.approx(-1.6666666666666667)
    with pytest.raises(ZeroDivisionError):
        divide(10, 0)

def test_calculator(capsys):
    calculator()
    captured = capsys.readouterr()
    assert "Addition: 10 + 2 = 12" in captured.out
    assert "Subtraction: 10 - 2 = 8" in captured.out
    assert "Multiplication: 10 * 2 = 20" in captured.out
    assert "Division: 10 / 2 = 5.0" in captured.out