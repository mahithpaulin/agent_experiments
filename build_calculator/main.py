from utils import add, subtract, multiply, divide

def calculate(a, b):
    return {
        'add': add(a, b),
        'subtract': subtract(a, b),
        'multiply': multiply(a, b),
        'divide': divide(a, b),
    }

if __name__ == '__main__':
    results = calculate(5, 2)
    for op, val in results.items():
        print(f"{op}: {val}")