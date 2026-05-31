import argparse
from utils import add, subtract, multiply, divide

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple command‑line calculator")
    parser.add_argument("a", type=float, help="First operand")
    parser.add_argument("b", type=float, help="Second operand")
    parser.add_argument(
        "op",
        choices=["add", "sub", "mul", "div"],
        help="Operation to perform",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    a, b = args.a, args.b
    if args.op == "add":
        result = add(a, b)
    elif args.op == "sub":
        result = subtract(a, b)
    elif args.op == "mul":
        result = multiply(a, b)
    elif args.op == "div":
        result = divide(a, b)
        if result is None:
            print("Error: division by zero")
            return
    else:
        raise AssertionError("Unsupported operation")

    print(result)

if __name__ == "__main__":
    main()
