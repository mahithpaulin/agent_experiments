from utils import add, subtract, multiply, divide

def calculator():
    print("Simple Calculator")
    print("Operations: add, subtract, multiply, divide")
    while True:
        op = input("Enter operation (or 'exit' to quit): ").strip().lower()
        if op == 'exit':
            print("Exiting calculator.")
            break
        if op not in ('add', 'subtract', 'multiply', 'divide'):
            print("Invalid operation. Try again.")
            continue
        try:
            x = float(input("Enter first number: "))
            y = float(input("Enter second number: "))
        except ValueError:
            print("Invalid number input. Try again.")
            continue
        try:
            if op == 'add':
                result = add(x, y)
            elif op == 'subtract':
                result = subtract(x, y)
            elif op == 'multiply':
                result = multiply(x, y)
            elif op == 'divide':
                result = divide(x, y)
            print(f"Result: {result}")
        except ZeroDivisionError:
            print("Error: Division by zero.")

if __name__ == "__main__":
    calculator()