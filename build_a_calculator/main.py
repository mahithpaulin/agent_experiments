import sys
from utils import add, subtract, multiply, divide

def main() -> None:
    """Simple demonstration of calculator operations."""
    a, b = 12, 4
    try:
        print(f"{a} + {b} = {add(a, b)}")
        print(f"{a} - {b} = {subtract(a, b)}")
        print(f"{a} * {b} = {multiply(a, b)}")
        print(f"{a} / {b} = {divide(a, b)}")
        # Edge case demonstration
        print(f"{a} / 0 = ", end="")
        print(divide(a, 0))
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()