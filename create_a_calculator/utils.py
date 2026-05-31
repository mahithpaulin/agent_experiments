def calculate(op, num1, num2):
    if op == 'add':
        return num1 + num2
    elif op == 'sub':
        return num1 - num2
    elif op == 'mul':
        return num1 * num2
    elif op == 'div':
        if num2 == 0:
            raise ValueError("Cannot divide by zero.")
        return num1 / num2
    else:
        raise ValueError(f"Unsupported operation: {op}")