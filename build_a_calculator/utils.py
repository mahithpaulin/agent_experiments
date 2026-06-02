def validate_numbers(a, b):
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Both arguments must be numbers")
    return True

def safe_divide(a, b):
    validate_numbers(a, b)
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

def safe_multiply(a, b):
    validate_numbers(a, b)
    return a * b

def safe_add(a, b):
    validate_numbers(a, b)
    return a + b

def safe_subtract(a, b):
    validate_numbers(a, b)
    return a - b