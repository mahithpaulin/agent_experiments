import pytest
from compiler import compile_source, LexerError, ParserError, CompilerError

def test_var_decl_and_assign():
    source = """
    var x = 10;
    x = x + 5;
    """
    pycode = compile_source(source)
    # Execute generated code and check variable x
    local_env = {}
    exec(pycode, {}, local_env)
    assert local_env["x"] == 15

def test_if_else():
    source = """
    var x = 10;
    if (x > 5) {
        x = 1;
    } else {
        x = 2;
    }
    """
    pycode = compile_source(source)
    local_env = {}
    exec(pycode, {}, local_env)
    assert local_env["x"] == 1

def test_while_loop():
    source = """
    var x = 0;
    var i = 0;
    while (i < 5) {
        x = x + i;
        i = i + 1;
    }
    """
    pycode = compile_source(source)
    local_env = {}
    exec(pycode, {}, local_env)
    assert local_env["x"] == 10  # 0+1+2+3+4=10

def test_function_def_and_call():
    source = """
    function add(a, b) {
        return a + b;
    }
    var result = add(3, 4);
    """
    pycode = compile_source(source)
    local_env = {}
    exec(pycode, {}, local_env)
    assert local_env["result"] == 7

def test_nested_function_and_conditionals():
    source = """
    function factorial(n) {
        if (n == 0) {
            return 1;
        } else {
            return n * factorial(n - 1);
        }
    }
    var f5 = factorial(5);
    """
    pycode = compile_source(source)
    local_env = {}
    exec(pycode, {}, local_env)
    assert local_env["f5"] == 120

def test_unary_operations():
    source = """
    var x = -5;
    var y = +10;
    var z = -x + y;
    """
    pycode = compile_source(source)
    local_env = {}
    exec(pycode, {}, local_env)
    assert local_env["x"] == -5
    assert local_env["y"] == 10
    assert local_env["z"] == 15

def test_syntax_error():
    source = """
    var x = 10
    """
    with pytest.raises(ParserError):
        compile_source(source)

def test_lexer_error():
    source = """
    var x = 10;
    var y = @;
    """
    with pytest.raises(LexerError):
        compile_source(source)

def test_unknown_operator_error():
    source = """
    var x = 10;
    x = x ^ 2;
    """
    with pytest.raises(CompilerError):
        compile_source(source)