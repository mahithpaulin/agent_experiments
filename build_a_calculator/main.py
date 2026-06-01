import matplotlib.pyplot as plt
from utils import add, subtract, multiply, divide

def calculator():
    num1 = 10
    num2 = 2

    add_result = add(num1, num2)
    subtract_result = subtract(num1, num2)
    multiply_result = multiply(num1, num2)
    divide_result = divide(num1, num2)

    print(f"Addition: {num1} + {num2} = {add_result}")
    print(f"Subtraction: {num1} - {num2} = {subtract_result}")
    print(f"Multiplication: {num1} * {num2} = {multiply_result}")
    print(f"Division: {num1} / {num2} = {divide_result}")

    plt.bar(['Addition', 'Subtraction', 'Multiplication', 'Division'], [add_result, subtract_result, multiply_result, divide_result])
    plt.savefig('calculator_results.png')
    plt.show()

if __name__ == "__main__":
    calculator()