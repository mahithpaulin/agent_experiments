import pytest
from main import compile_source, Lexer, Parser, Compiler

def run_compiler_and_exec(source):
    pycode = compile_source(source)
    local_env = {}
    exec(pycode, {}, local_env)
    return local_env

def test_variable_assignment_and_arithmetic():
    source = """
    a = 5;
    b = 10;
    c = a + b * 2;
    """
    env = run_compiler_and_exec(source)
    assert env['a'] == 5
    assert env['b'] == 10
    assert env['c'] == 5 + 10 * 2

def test_if_else_statement():
    source = """
    a = 10;
    if (a > 5) {
        b = 1;
    } else {
        b = 2;
    }
    """
    env = run_compiler_and_exec(source)
    assert env['b'] == 1

    source2 = """
    a = 3;
    if (a > 5) {
        b = 1;
    } else {
        b = 2;
    }
    """
    env2 = run_compiler_and_exec(source2)
    assert env2['b'] == 2

def test_while_loop():
    source = """
    i = 0;
    s = 0;
    while (i < 5) {
        s = s + i;
        i = i + 1;
    }
    """
    env = run_compiler_and_exec(source)
    assert env['s'] == sum(range(5))
    assert env['i'] == 5

def test_function_definition_and_call():
    source = """
    def add(x, y) {
        return x + y;
    }
    a = add(3, 4);
    """
    env = run_compiler_and_exec(source)
    assert env['a'] == 7
    assert callable(env['add'])

def test_function_without_return():
    source = """
    def foo() {
        a = 5;
    }
    b = foo();
    """
    env = run_compiler_and_exec(source)
    assert env['b'] is None

def test_nested_function_calls_and_conditionals():
    source = """
    def max(a, b) {
        if (a > b) {
            return a;
        } else {
            return b;
        }
    }
    x = max(10, 20);
    y = max(5, 3);
    """
    env = run_compiler_and_exec(source)
    assert env['x'] == 20
    assert env['y'] == 5

def test_boolean_values_and_comparisons():
    source = """
    a = true;
    b = false;
    c = (5 == 5);
    d = (5 != 3);
    e = (4 < 3);
    f = (4 <= 4);
    g = (5 > 2);
    h = (5 >= 5);
    """
    env = run_compiler_and_exec(source)
    assert env['a'] is True
    assert env['b'] is False
    assert env['c'] is True
    assert env['d'] is True
    assert env['e'] is False
    assert env['f'] is True
    assert env['g'] is True
    assert env['h'] is True

def test_unary_minus_and_plus():
    source = """
    a = -5;
    b = +10;
    c = -(-3);
    d = +(+4);
    """
    env = run_compiler_and_exec(source)
    assert env['a'] == -5
    assert env['b'] == 10
    assert env['c'] == 3
    assert env['d'] == 4

def test_function_call_as_expression():
    source = """
    def square(x) {
        return x * x;
    }
    a = square(3) + square(4);
    """
    env = run_compiler_and_exec(source)
    assert env['a'] == 9 + 16

def test_parser_errors():
    # Missing semicolon
    source = "a = 5"
    with pytest.raises(Exception):
        lexer = Lexer(source)
        parser = Parser(lexer)
        ast = parser.program()

    # Invalid token
    source2 = "a = 5 $;"
    with pytest.raises(Exception):
        lexer = Lexer(source2)
        parser = Parser(lexer)
        ast = parser.program()

def test_empty_program():
    source = ""
    env = run_compiler_and_exec(source)
    assert isinstance(env, dict)

def test_return_without_expression():
    source = """
    def foo() {
        return 5;
    }
    a = foo();
    """
    env = run_compiler_and_exec(source)
    assert env['a'] == 5

def test_multiple_statements():
    source = """
    a = 1;
    b = 2;
    c = 3;
    d = a + b + c;
    """
    env = run_compiler_and_exec(source)
    assert env['d'] == 6
