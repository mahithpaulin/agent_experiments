import pytest
import math
from f import Const, Var, Add, Sub, Mul, Div, Pow, Log

def test_const_simplify_and_evaluate():
    c = Const(5)
    assert c.simplify() == c
    assert c.evaluate({}) == 5
    assert str(c) == "5"

def test_var_simplify_and_evaluate():
    x = Var("x")
    assert x.simplify() == x
    assert x.evaluate({"x": 3}) == 3
    with pytest.raises(ValueError):
        x.evaluate({})
    assert str(x) == "x"

def test_add_simplify_and_derivative():
    x = Var("x")
    zero = Const(0)
    one = Const(1)
    expr = Add(x, zero)
    simplified = expr.simplify()
    assert simplified == x
    expr2 = Add(Const(2), Const(3))
    assert expr2.simplify() == Const(5)
    d = expr.derivative("x")
    assert d == Add(Const(1), Const(0)).simplify()
    env = {"x": 2}
    assert math.isclose(expr.evaluate(env), 2)
    assert str(expr) == f"({x} + {zero})"

def test_sub_simplify_and_derivative():
    x = Var("x")
    zero = Const(0)
    expr = Sub(x, zero)
    assert expr.simplify() == x
    expr2 = Sub(Const(5), Const(3))
    assert expr2.simplify() == Const(2)
    expr3 = Sub(x, x)
    assert expr3.simplify() == Const(0)
    d = expr.derivative("x")
    assert d == Sub(Const(1), Const(0)).simplify()
    env = {"x": 4}
    assert math.isclose(expr.evaluate(env), 4)
    assert str(expr) == f"({x} - {zero})"

def test_mul_simplify_and_derivative():
    x = Var("x")
    zero = Const(0)
    one = Const(1)
    expr = Mul(x, one)
    assert expr.simplify() == x
    expr2 = Mul(Const(0), x)
    assert expr2.simplify() == zero
    expr3 = Mul(Const(2), Const(3))
    assert expr3.simplify() == Const(6)
    d = expr.derivative("x")
    # derivative of x*1 = 1*1 + x*0 = 1
    assert d == Add(Mul(Const(1), one), Mul(x, Const(0))).simplify()
    env = {"x": 5}
    assert math.isclose(expr.evaluate(env), 5)
    assert str(expr) == f"({x} * {one})"

def test_div_simplify_and_derivative():
    x = Var("x")
    one = Const(1)
    zero = Const(0)
    expr = Div(x, one)
    assert expr.simplify() == x
    expr2 = Div(zero, x)
    assert expr2.simplify() == zero
    expr3 = Div(Const(6), Const(3))
    assert expr3.simplify() == Const(2)
    expr4 = Div(x, x)
    assert expr4.simplify() == one
    d = expr.derivative("x")
    # derivative of x/1 = (1*1 - x*0)/1^2 = 1
    assert d == Div(Sub(Mul(Const(1), one), Mul(x, Const(0))), Pow(one, Const(2))).simplify()
    env = {"x": 4}
    assert math.isclose(expr.evaluate(env), 4)
    assert str(expr) == f"({x} / {one})"

def test_pow_simplify_and_derivative():
    x = Var("x")
    zero = Const(0)
    one = Const(1)
    two = Const(2)
    expr = Pow(x, zero)
    assert expr.simplify() == one
    expr2 = Pow(x, one)
    assert expr2.simplify() == x
    expr3 = Pow(Const(0), two)
    assert expr3.simplify() == zero
    expr4 = Pow(Const(2), Const(3))
    assert expr4.simplify() == Const(8)
    d = Pow(x, two).derivative("x")
    # derivative of x^2 = 2*x^(2-1)*1 = 2*x
    expected = Mul(Mul(two, Pow(x, Const(1))), Const(1)).simplify()
    assert d == expected
    env = {"x": 3}
    assert math.isclose(expr4.evaluate(env), 8)
    assert str(expr4) == f"({Const(2)} ^ {Const(3)})"

def test_log_simplify_and_derivative():
    x = Var("x")
    one = Const(1)
    e = Const(math.e)
    expr = Log(x, x)
    assert expr.simplify() == Const(1)
    expr2 = Log(x, one)
    assert expr2.simplify() == Const(0)
    expr3 = Log(Const(2), Const(8))
    assert math.isclose(expr3.simplify().value, math.log(8, 2))
    d = Log(e, x).derivative("x")
    # derivative of ln(x) = 1/x
    expected = Div(Const(1), x).simplify()
    assert d == expected
    env = {"x": math.e}
    assert math.isclose(expr3.evaluate(env), math.log(8, 2))
    assert str(expr3).startswith("log_")

def test_expression_str_and_eq():
    x = Var("x")
    y = Var("y")
    expr1 = Add(x, Const(2))
    expr2 = Add(Var("x"), Const(2))
    expr3 = Add(x, Const(3))
    assert expr1 == expr2
    assert expr1 != expr3
    assert str(expr1) == f"({x} + {Const(2)})"
    expr4 = Mul(expr1, y)
    assert str(expr4) == f"({expr1} * {y})"

def test_derivative_chain_rule():
    x = Var("x")
    expr = Pow(x, Const(3))  # x^3
    d = expr.derivative("x")
    # derivative should be 3*x^2
    expected = Mul(Const(3), Pow(x, Const(2))).simplify()
    assert d == expected
    env = {"x": 2}
    val = expr.evaluate(env)
    dval = d.evaluate(env)
    assert math.isclose(val, 8)
    assert math.isclose(dval, 12)

def test_simplify_nested():
    x = Var("x")
    expr = Add(Const(0), Mul(Const(1), x))
    simplified = expr.simplify()
    assert simplified == x
    expr2 = Mul(Add(Const(0), x), Const(1))
    assert expr2.simplify() == x
    expr3 = Sub(x, x)
    assert expr3.simplify() == Const(0)
    expr4 = Div(x, x)
    assert expr4.simplify() == Const(1)