from abc import ABC, abstractmethod
import math

class Expr(ABC):
    @abstractmethod
    def simplify(self):
        pass

    @abstractmethod
    def derivative(self, var):
        pass

    @abstractmethod
    def evaluate(self, env):
        pass

    @abstractmethod
    def __str__(self):
        pass

    def __add__(self, other):
        return Add(self, other if isinstance(other, Expr) else Const(other)).simplify()

    def __radd__(self, other):
        return Add(Const(other) if not isinstance(other, Expr) else other, self).simplify()

    def __sub__(self, other):
        return Sub(self, other if isinstance(other, Expr) else Const(other)).simplify()

    def __rsub__(self, other):
        return Sub(Const(other) if not isinstance(other, Expr) else other, self).simplify()

    def __mul__(self, other):
        return Mul(self, other if isinstance(other, Expr) else Const(other)).simplify()

    def __rmul__(self, other):
        return Mul(Const(other) if not isinstance(other, Expr) else other, self).simplify()

    def __truediv__(self, other):
        return Div(self, other if isinstance(other, Expr) else Const(other)).simplify()

    def __rtruediv__(self, other):
        return Div(Const(other) if not isinstance(other, Expr) else other, self).simplify()

    def __pow__(self, other):
        return Pow(self, other if isinstance(other, Expr) else Const(other)).simplify()

    def __rpow__(self, other):
        return Pow(Const(other) if not isinstance(other, Expr) else other, self).simplify()

class Const(Expr):
    def __init__(self, value):
        self.value = float(value)

    def simplify(self):
        return self

    def derivative(self, var):
        return Const(0)

    def evaluate(self, env):
        return self.value

    def __str__(self):
        if self.value.is_integer():
            return str(int(self.value))
        return str(self.value)

    def __eq__(self, other):
        if isinstance(other, Const):
            return math.isclose(self.value, other.value, rel_tol=1e-9)
        return False

class Var(Expr):
    def __init__(self, name):
        self.name = name

    def simplify(self):
        return self

    def derivative(self, var):
        if self.name == var:
            return Const(1)
        else:
            return Const(0)

    def evaluate(self, env):
        if self.name in env:
            return env[self.name]
        raise ValueError(f"Variable '{self.name}' not found in environment")

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, Var):
            return self.name == other.name
        return False

class Add(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def simplify(self):
        left = self.left.simplify()
        right = self.right.simplify()
        # 0 + x = x
        if isinstance(left, Const) and left.value == 0:
            return right
        if isinstance(right, Const) and right.value == 0:
            return left
        # both constants
        if isinstance(left, Const) and isinstance(right, Const):
            return Const(left.value + right.value)
        return Add(left, right)

    def derivative(self, var):
        return Add(self.left.derivative(var), self.right.derivative(var)).simplify()

    def evaluate(self, env):
        return self.left.evaluate(env) + self.right.evaluate(env)

    def __str__(self):
        return f"({self.left} + {self.right})"

    def __eq__(self, other):
        if isinstance(other, Add):
            return self.left == other.left and self.right == other.right
        return False

class Sub(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def simplify(self):
        left = self.left.simplify()
        right = self.right.simplify()
        # x - 0 = x
        if isinstance(right, Const) and right.value == 0:
            return left
        # both constants
        if isinstance(left, Const) and isinstance(right, Const):
            return Const(left.value - right.value)
        # x - x = 0
        if left == right:
            return Const(0)
        return Sub(left, right)

    def derivative(self, var):
        return Sub(self.left.derivative(var), self.right.derivative(var)).simplify()

    def evaluate(self, env):
        return self.left.evaluate(env) - self.right.evaluate(env)

    def __str__(self):
        return f"({self.left} - {self.right})"

    def __eq__(self, other):
        if isinstance(other, Sub):
            return self.left == other.left and self.right == other.right
        return False

class Mul(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def simplify(self):
        left = self.left.simplify()
        right = self.right.simplify()
        # 0 * x = 0
        if (isinstance(left, Const) and left.value == 0) or (isinstance(right, Const) and right.value == 0):
            return Const(0)
        # 1 * x = x
        if isinstance(left, Const) and left.value == 1:
            return right
        if isinstance(right, Const) and right.value == 1:
            return left
        # both constants
        if isinstance(left, Const) and isinstance(right, Const):
            return Const(left.value * right.value)
        return Mul(left, right)

    def derivative(self, var):
        # product rule: f'g + fg'
        return Add(Mul(self.left.derivative(var), self.right), Mul(self.left, self.right.derivative(var))).simplify()

    def evaluate(self, env):
        return self.left.evaluate(env) * self.right.evaluate(env)

    def __str__(self):
        return f"({self.left} * {self.right})"

    def __eq__(self, other):
        if isinstance(other, Mul):
            return self.left == other.left and self.right == other.right
        return False

class Div(Expr):
    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator

    def simplify(self):
        numerator = self.numerator.simplify()
        denominator = self.denominator.simplify()
        # 0 / x = 0 (x != 0)
        if isinstance(numerator, Const) and numerator.value == 0:
            return Const(0)
        # x / 1 = x
        if isinstance(denominator, Const) and denominator.value == 1:
            return numerator
        # both constants
        if isinstance(numerator, Const) and isinstance(denominator, Const):
            return Const(numerator.value / denominator.value)
        # x / x = 1 (if equal)
        if numerator == denominator:
            return Const(1)
        return Div(numerator, denominator)

    def derivative(self, var):
        # quotient rule: (f'g - fg')/g^2
        f = self.numerator
        g = self.denominator
        f_prime = f.derivative(var)
        g_prime = g.derivative(var)
        numerator = Sub(Mul(f_prime, g), Mul(f, g_prime))
        denominator = Pow(g, Const(2))
        return Div(numerator, denominator).simplify()

    def evaluate(self, env):
        return self.numerator.evaluate(env) / self.denominator.evaluate(env)

    def __str__(self):
        return f"({self.numerator} / {self.denominator})"

    def __eq__(self, other):
        if isinstance(other, Div):
            return self.numerator == other.numerator and self.denominator == other.denominator
        return False

class Pow(Expr):
    def __init__(self, base, exponent):
        self.base = base
        self.exponent = exponent

    def simplify(self):
        base = self.base.simplify()
        exponent = self.exponent.simplify()
        # x^0 = 1 (x != 0)
        if isinstance(exponent, Const) and exponent.value == 0:
            return Const(1)
        # x^1 = x
        if isinstance(exponent, Const) and exponent.value == 1:
            return base
        # 0^x = 0 if x != 0
        if isinstance(base, Const) and base.value == 0:
            if isinstance(exponent, Const) and exponent.value != 0:
                return Const(0)
        # both constants
        if isinstance(base, Const) and isinstance(exponent, Const):
            return Const(base.value ** exponent.value)
        return Pow(base, exponent)

    def derivative(self, var):
        base = self.base
        exponent = self.exponent
        # Cases:
        # 1) exponent is constant: d/dx (f^c) = c * f^(c-1) * f'
        if isinstance(exponent, Const):
            new_exp = Const(exponent.value - 1)
            return Mul(Mul(exponent, Pow(base, new_exp)), base.derivative(var)).simplify()
        # 2) base is constant: d/dx (c^g) = c^g * ln(c) * g'
        if isinstance(base, Const):
            return Mul(Mul(Pow(base, exponent), Log(Const(math.e), base)), exponent.derivative(var)).simplify()
        # 3) general case: d/dx (f^g) = f^g * (g' * ln(f) + g * f'/f)
        return Mul(
            Pow(base, exponent),
            Add(
                Mul(exponent.derivative(var), Log(Const(math.e), base)),
                Mul(exponent, Div(base.derivative(var), base))
            )
        ).simplify()

    def evaluate(self, env):
        return self.base.evaluate(env) ** self.exponent.evaluate(env)

    def __str__(self):
        return f"({self.base} ^ {self.exponent})"

    def __eq__(self, other):
        if isinstance(other, Pow):
            return self.base == other.base and self.exponent == other.exponent
        return False

class Log(Expr):
    # Log base argument: Log(base, argument)
    def __init__(self, base, argument):
        self.base = base
        self.argument = argument

    def simplify(self):
        base = self.base.simplify()
        argument = self.argument.simplify()
        # log_b(b) = 1
        if base == argument:
            return Const(1)
        # log_b(1) = 0
        if isinstance(argument, Const) and argument.value == 1:
            return Const(0)
        # both constants
        if isinstance(base, Const) and isinstance(argument, Const):
            return Const(math.log(argument.value, base.value))
        return Log(base, argument)

    def derivative(self, var):
        # d/dx log_b(f) = f' / (f * ln(b))
        f = self.argument
        b = self.base
        f_prime = f.derivative(var)
        ln_b = Log(Const(math.e), b)
        return Div(f_prime, Mul(f, ln_b)).simplify()

    def evaluate(self, env):
        base_val = self.base.evaluate(env)
        arg_val = self.argument.evaluate(env)
        return math.log(arg_val, base_val)

    def __str__(self):
        return f"log_{self.base}({self.argument})"

    def __eq__(self, other):
        if isinstance(other, Log):
            return self.base == other.base and self.argument == other.argument
        return False
