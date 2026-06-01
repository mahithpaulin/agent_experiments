from utils import add, subtract, multiply, divide

def calculate(a, b):
    results = {
        "add": add(a, b),
        "subtract": subtract(a, b),
        "multiply": multiply(a, b),
        "divide": divide(a, b)
    }
    return results

def main():
    pairs = [(10, 5), (3, 0), (-2, 8)]
    for a, b in pairs:
        res = calculate(a, b)
        print(f"Calculations for {a} and {b}:")
        print(f"Add: {res['add']}")
        print(f"Subtract: {res['subtract']}")
        print(f"Multiply: {res['multiply']}")
        print(f"Divide: {res['divide']}")
        print()

if __name__ == "__main__":
    main()