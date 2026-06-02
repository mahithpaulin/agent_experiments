from utils import add, subtract, multiply, divide

def calculator():
    operation = input("Choose operation (add, subtract, multiply, divide): ").strip().lower()
    try:
        num1 = float(input("Enter first number: "))
        num2 = float(input("Enter second number: "))
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return

    if operation == "add":
        print(f"Result: {add(num1, num2)}")
    elif operation == "subtract":
        print(f"Result: {subtract(num1, num2)}")
    elif operation == "multiply":
        print(f"Result: {multiply(num1, num2)}")
    elif operation == "divide":
        try:
            print(f"Result: {divide(num1, num2)}")
        except ZeroDivisionError:
            print("Error: Division by zero is not allowed.")
    else:
        print("Invalid operation. Please choose add, subtract, multiply, or divide.")

if __name__ == "__main__":
    calculator()