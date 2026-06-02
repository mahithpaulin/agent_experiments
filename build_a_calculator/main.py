import sys
from utils import add, subtract, multiply

def main():
    if len(sys.argv) != 4:
        print("Usage: python main.py <number1> <operator> <number2>")
        return
    try:
        num1 = float(sys.argv[1])
        num2 = float(sys.argv[3])
    except ValueError:
        print("Both operands must be numbers")
        return

    operator = sys.argv[2]
    if operator == '+':
        result = add(num1, num2)
    elif operator == '-':
        result = subtract(num1, num2)
    elif operator == '*':
        result = multiply(num1, num2)
    else:
        print(f"Unsupported operator '{operator}'")
        return

    print(result)

if __name__ == "__main__":
    main()