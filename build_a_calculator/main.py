import operator

class NaturalLanguageCalculator:
    def __init__(self):
        self.operations = {
            "add": operator.add,
            "subtract": operator.sub,
            "multiply": operator.mul,
            "divide": operator.truediv
        }

    def calculate(self, operation, a, b):
        if operation in self.operations:
            return self.operations[operation](a, b)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
