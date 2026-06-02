class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

if __name__ == "__main__":
    calc = Calculator()
    result_add = calc.add(5, 3)
    result_subtract = calc.subtract(5, 3)
    result_multiply = calc.multiply(5, 3)
    result_divide = calc.divide(5, 3)
    print(f"Addition: {result_add}, Subtraction: {result_subtract}, Multiplication: {result_multiply}, Division: {result_divide}")