import pytest
from utils import calculate

def test_add_positive():
    assert calculate('add', 1, 2) == 3

def test_sub_positive():
    assert calculate('sub', 5, 3) == 2

def test_mul_positive():
    assert calculate('mul', 4, 5) == 20

def test_div_positive():
    assert calculate('div', 10, 2) == 5.0

def test_div_by_zero():
    with pytest.raises(ValueError, match="Cannot divide by zero."):
        calculate('div', 10, 0)

def test_unsupported_op():
    with pytest.raises(ValueError, match="Unsupported operation: mod"):
        calculate('mod', 1, 2)

def test_negative_addition():
    assert calculate('add', -5, 3) == -2

def test_negative_multiplication():
    assert calculate('mul', 5, -2) == -10