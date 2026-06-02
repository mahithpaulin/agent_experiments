import pytest
from utils import add, subtract, multiply

def test_add():
    """Test addition functionality with positive, negative and zero values."""
    assert add(10, 5) == 15
    assert add(-1, 1) == 0
    assert add(-5, -5) == -10
    assert add(0, 0) == 0
    assert add(1.5, 2.5) == 4.0

def test_subtract():
    """Test subtraction functionality with positive, negative and zero values."""
    assert subtract(10, 5) == 5
    assert subtract(5, 10) == -5
    assert subtract(-1, -1) == 0
    assert subtract(0, 5) == -5
    assert subtract(2.5, 1.5) == 1.0

def test_multiply():
    """Test multiplication functionality with positive, negative and zero values."""
    assert multiply(10, 5) == 50
    assert multiply(-2, 3) == -6
    assert multiply(-2, -4) == 8
    assert multiply(100, 0) == 0
    assert multiply(0.5, 2) == 1.0

def test_float_precision():
    """Test that float operations are handled with precision using pytest.approx."""
    assert add(0.1, 0.2) == pytest.approx(0.3)
    assert subtract(0.3, 0.1) == pytest.approx(0.2)
    assert multiply(0.1, 0.1) == pytest.approx(0.01)

def test_large_numbers():
    """Test calculations with large numbers."""
    assert add(1e10, 1e10) == 2e10
    assert subtract(1e10, 1e10) == 0
    assert multiply(1e5, 1e5) == 1e10