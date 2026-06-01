import pytest
from main import calculate

@pytest.mark.parametrize("a,b,expected", [
    (10, 5, {"add": 15, "subtract": 5, "multiply": 50, "divide": 2}),
    (3, 0, {"add": 3, "subtract": 3, "multiply": 0, "divide": None}),
    (-2, 8, {"add": 6, "subtract": -10, "multiply": -16, "divide": -0.25}),
    (0, 0, {"add": 0, "subtract": 0, "multiply": 0, "divide": None}),
    (5, -5, {"add": 0, "subtract": 10, "multiply": -25, "divide": -1}),
])
def test_calculate(a, b, expected):
    result = calculate(a, b)
    assert result["add"] == expected["add"]
    assert result["subtract"] == expected["subtract"]
    assert result["multiply"] == expected["multiply"]
    if expected["divide"] is None:
        assert result["divide"] is None
    else:
        assert result["divide"] == pytest.approx(expected["divide"])