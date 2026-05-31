import sys
from utils import add, sub, mul, div


def calculate_pair(a, b):
    """
    Compute addition, subtraction, multiplication, and division for a pair of numbers.

    Returns a dictionary with keys: 'add', 'sub', 'mul', 'div'.
    If division by zero occurs, the 'div' value is set to None.
    """
    results = {
        "add": add(a, b),
        "sub": sub(a, b),
        "mul": mul(a, b),
    }
    try:
        results["div"] = div(a, b)
    except ZeroDivisionError:
        results["div"] = None
    return results


def _format_number(val):
    """Format numbers: drop trailing .0 for whole floats."""
    if isinstance(val, float) and val.is_integer():
        return int(val)
    return val


def _format_result(a, b, results):
    a_fmt = _format_number(a)
    b_fmt = _format_number(b)
    add_fmt = _format_number(results["add"])
    sub_fmt = _format_number(results["sub"])
    mul_fmt = _format_number(results["mul"])
    div_val = results["div"]
    if div_val is None:
        div_str = "Error(div/0)"
    else:
        div_str = str(_format_number(div_val))
    return f"{a_fmt} + {b_fmt} = {add_fmt}; {a_fmt} - {b_fmt} = {sub_fmt}; {a_fmt} * {b_fmt} = {mul_fmt}; {a_fmt} / {b_fmt} = {div_str}"


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            print(f"Ignoring invalid line: {line}", file=sys.stderr)
            continue
        try:
            a = float(parts[0]) if "." in parts[0] else int(parts[0])
            b = float(parts[1]) if "." in parts[1] else int(parts[1])
        except ValueError:
            print(f"Ignoring non-numeric line: {line}", file=sys.stderr)
            continue
        results = calculate_pair(a, b)
        print(_format_result(a, b, results))


if __name__ == "__main__":
    main()