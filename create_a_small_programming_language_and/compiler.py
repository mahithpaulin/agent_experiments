import re

class Token:
    def __init__(self, type_, value, line, col):
        self.type = type_
        self.value = value
        self.line = line
        self.col = col
    def __repr__(self):
        return f"Token({self.type}, {self.value}, {self.line}, {self.col})"

class LexerError(Exception):
    pass

class ParserError(Exception):
    pass

class CompilerError(Exception):
    pass

KEYWORDS = {
    "if", "else", "while", "function", "return", "var"
}

TOKEN_SPECIFICATION = [
    ("NUMBER",   r"\d+(\.\d*)?"),  # Integer or decimal number
    ("ID",       r"[A-Za-z_][A-Za-z0-9_]*"),
    ("OP",       r"==|!=|<=|>=|[+\-*/<>=%]"),
    ("LPAREN",   r"\("),
    ("RPAREN",   r"\)"),
    ("LBRACE",   r"\{"),
    ("RBRACE",   r"\}"),
    ("SEMICOLON",r";"),
    ("COMMA",    r","),
    ("NEWLINE",  r"\n"),
    ("SKIP",     r"[ \t]+"),
    ("MISMATCH", r".")
]

class Lexer:
    def __init__(self, code):
        self.code = code
        self.tokens = []
        self.pos = 0
        self.line = 1
        self.col = 1
        self.regex = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in TOKEN_SPECIFICATION))
    def tokenize(self):
        for mo in self.regex.finditer(self.code):
            kind = mo.lastgroup
            value = mo.group()
            if kind == "NUMBER":
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
                self.tokens.append(Token("NUMBER", value, self.line, self.col))
                self.col += len(mo.group())
            elif kind == "ID":
                if value in KEYWORDS:
                    self.tokens.append(Token(value.upper(), value, self.line, self.col))
                else:
                    self.tokens.append(Token("ID", value, self.line, self.col))
                self.col += len(mo.group())
            elif kind == "OP":
                self.tokens.append(Token("OP", value, self.line, self.col))
                self.col += len(mo.group())
            elif kind in ("LPAREN","RPAREN","LBRACE","RBRACE","SEMICOLON","COMMA"):
                self.tokens.append(Token(kind, value, self.line, self.col))
                self.col += len(mo.group())
            elif kind == "NEWLINE":
                self.line += 1
                self.col = 1
            elif kind == "SKIP":
                self.col += len(mo.group())
            elif kind == "MISMATCH":
                raise LexerError(f"Unexpected character {value!r} at line {self.line} col {self.col}")
        self.tokens.append(Token("EOF", None, self.line, self.col))
        return self.tokens

# AST Nodes
class Node:
    pass

class Program(Node):
    def __init__(self, statements):
        self.statements = statements

class VarDecl(Node):
    def __init__(self, name, expr):
        self.name = name
        self.expr = expr

class Assign(Node):
    def __init__(self, name, expr):
        self.name = name
        self.expr = expr

class If(Node):
    def __init__(self, cond, then_block, else_block):
        self.cond = cond
        self.then_block = then_block
        self.else_block = else_block

class While(Node):
    def __init__(self, cond, body):
        self.cond = cond
        self.body = body

class FunctionDef(Node):
    def __init__(self, name, params, body):
        self.name = name
        self.params = params
        self.body = body

class Return(Node):
    def __init__(self, expr):
        self.expr = expr

class ExprStmt(Node):
    def __init__(self, expr):
        self.expr = expr

class BinaryOp(Node):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

class UnaryOp(Node):
    def __init__(self, op, expr):
        self.op = op
        self.expr = expr

class Number(Node):
    def __init__(self, value):
        self.value = value

class Var(Node):
    def __init__(self, name):
        self.name = name

class FuncCall(Node):
    def __init__(self, name, args):
        self.name = name
        self.args = args

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[self.pos]

    def error(self, msg=None):
        t = self.current_token
        raise ParserError(msg or f"Unexpected token {t.type}({t.value}) at line {t.line} col {t.col}")

    def eat(self, token_type=None, token_value=None):
        t = self.current_token
        if token_type and t.type != token_type:
            self.error(f"Expected token type {token_type} but got {t.type}")
        if token_value and t.value != token_value:
            self.error(f"Expected token value {token_value} but got {t.value}")
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = Token("EOF", None, t.line, t.col)
        return t

    def parse(self):
        stmts = []
        while self.current_token.type != "EOF":
            stmts.append(self.statement())
        return Program(stmts)

    def statement(self):
        t = self.current_token
        if t.type == "VAR":
            return self.var_decl()
        elif t.type == "ID":
            # Could be assign or func call
            next_t = self.tokens[self.pos+1]
            if next_t.type == "OP" and next_t.value == "=":
                return self.assign()
            else:
                expr = self.expr()
                self.eat("SEMICOLON")
                return ExprStmt(expr)
        elif t.type == "IF":
            return self.if_stmt()
        elif t.type == "WHILE":
            return self.while_stmt()
        elif t.type == "FUNCTION":
            return self.function_def()
        elif t.type == "RETURN":
            return self.return_stmt()
        else:
            expr = self.expr()
            self.eat("SEMICOLON")
            return ExprStmt(expr)

    def var_decl(self):
        self.eat("VAR")
        name = self.eat("ID").value
        self.eat("OP","=")
        expr = self.expr()
        self.eat("SEMICOLON")
        return VarDecl(name, expr)

    def assign(self):
        name = self.eat("ID").value
        self.eat("OP","=")
        expr = self.expr()
        self.eat("SEMICOLON")
        return Assign(name, expr)

    def if_stmt(self):
        self.eat("IF")
        self.eat("LPAREN")
        cond = self.expr()
        self.eat("RPAREN")
        then_block = self.block()
        else_block = None
        if self.current_token.type == "ELSE":
            self.eat("ELSE")
            else_block = self.block()
        return If(cond, then_block, else_block)

    def while_stmt(self):
        self.eat("WHILE")
        self.eat("LPAREN")
        cond = self.expr()
        self.eat("RPAREN")
        body = self.block()
        return While(cond, body)

    def function_def(self):
        self.eat("FUNCTION")
        name = self.eat("ID").value
        self.eat("LPAREN")
        params = []
        if self.current_token.type != "RPAREN":
            params.append(self.eat("ID").value)
            while self.current_token.type == "COMMA":
                self.eat("COMMA")
                params.append(self.eat("ID").value)
        self.eat("RPAREN")
        body = self.block()
        return FunctionDef(name, params, body)

    def return_stmt(self):
        self.eat("RETURN")
        expr = self.expr()
        self.eat("SEMICOLON")
        return Return(expr)

    def block(self):
        self.eat("LBRACE")
        stmts = []
        while self.current_token.type != "RBRACE":
            stmts.append(self.statement())
        self.eat("RBRACE")
        return stmts

    # Expression parsing with precedence climbing
    # Operators precedence (lowest to highest):
    # = (assignment handled separately)
    # == != < > <= >=
    # + -
    # * /
    # unary + -
    # function calls and parentheses

    def expr(self):
        return self._expr_relational()

    def _expr_relational(self):
        node = self._expr_additive()
        while self.current_token.type == "OP" and self.current_token.value in ("==","!=","<",">","<=",">="):
            op = self.eat("OP").value
            right = self._expr_additive()
            node = BinaryOp(node, op, right)
        return node

    def _expr_additive(self):
        node = self._expr_multiplicative()
        while self.current_token.type == "OP" and self.current_token.value in ("+","-"):
            op = self.eat("OP").value
            right = self._expr_multiplicative()
            node = BinaryOp(node, op, right)
        return node

    def _expr_multiplicative(self):
        node = self._expr_unary()
        while self.current_token.type == "OP" and self.current_token.value in ("*","/"):
            op = self.eat("OP").value
            right = self._expr_unary()
            node = BinaryOp(node, op, right)
        return node

    def _expr_unary(self):
        if self.current_token.type == "OP" and self.current_token.value in ("+","-"):
            op = self.eat("OP").value
            expr = self._expr_unary()
            return UnaryOp(op, expr)
        else:
            return self._expr_primary()

    def _expr_primary(self):
        t = self.current_token
        if t.type == "NUMBER":
            self.eat("NUMBER")
            return Number(t.value)
        elif t.type == "ID":
            # Could be var or func call
            name = self.eat("ID").value
            if self.current_token.type == "LPAREN":
                self.eat("LPAREN")
                args = []
                if self.current_token.type != "RPAREN":
                    args.append(self.expr())
                    while self.current_token.type == "COMMA":
                        self.eat("COMMA")
                        args.append(self.expr())
                self.eat("RPAREN")
                return FuncCall(name, args)
            else:
                return Var(name)
        elif t.type == "LPAREN":
            self.eat("LPAREN")
            node = self.expr()
            self.eat("RPAREN")
            return node
        else:
            self.error("Expected number, identifier or '('")

class CodeGenerator:
    def __init__(self):
        self.indent_level = 0
        self.lines = []

    def indent(self):
        return "    " * self.indent_level

    def emit(self, line):
        self.lines.append(self.indent() + line)

    def generate(self, node):
        if isinstance(node, Program):
            for stmt in node.statements:
                self.generate(stmt)
        elif isinstance(node, VarDecl):
            # var x = expr; -> x = expr
            code = f"{node.name} = {self.gen_expr(node.expr)}"
            self.emit(code)
        elif isinstance(node, Assign):
            code = f"{node.name} = {self.gen_expr(node.expr)}"
            self.emit(code)
        elif isinstance(node, If):
            cond = self.gen_expr(node.cond)
            self.emit(f"if {cond}:")
            self.indent_level += 1
            for stmt in node.then_block:
                self.generate(stmt)
            self.indent_level -= 1
            if node.else_block is not None:
                self.emit("else:")
                self.indent_level += 1
                for stmt in node.else_block:
                    self.generate(stmt)
                self.indent_level -= 1
        elif isinstance(node, While):
            cond = self.gen_expr(node.cond)
            self.emit(f"while {cond}:")
            self.indent_level += 1
            for stmt in node.body:
                self.generate(stmt)
            self.indent_level -= 1
        elif isinstance(node, FunctionDef):
            params = ", ".join(node.params)
            self.emit(f"def {node.name}({params}):")
            self.indent_level += 1
            if not node.body:
                self.emit("pass")
            else:
                for stmt in node.body:
                    self.generate(stmt)
            self.indent_level -= 1
        elif isinstance(node, Return):
            expr = self.gen_expr(node.expr)
            self.emit(f"return {expr}")
        elif isinstance(node, ExprStmt):
            expr = self.gen_expr(node.expr)
            self.emit(expr)
        else:
            raise CompilerError(f"Unknown node type: {type(node)}")

    def gen_expr(self, node):
        if isinstance(node, Number):
            return repr(node.value)
        elif isinstance(node, Var):
            return node.name
        elif isinstance(node, BinaryOp):
            left = self.gen_expr(node.left)
            right = self.gen_expr(node.right)
            op = node.op
            if op == "==":
                op = "=="
            elif op == "!=":
                op = "!="
            elif op == "<":
                op = "<"
            elif op == ">":
                op = ">"
            elif op == "<=":
                op = "<="
            elif op == ">=":
                op = ">="
            elif op == "+":
                op = "+"
            elif op == "-":
                op = "-"
            elif op == "*":
                op = "*"
            elif op == "/":
                op = "/"
            else:
                raise CompilerError(f"Unknown binary operator {op}")
            return f"({left} {op} {right})"
        elif isinstance(node, UnaryOp):
            op = node.op
            expr = self.gen_expr(node.expr)
            return f"({op}{expr})"
        elif isinstance(node, FuncCall):
            args = ", ".join(self.gen_expr(arg) for arg in node.args)
            return f"{node.name}({args})"
        else:
            raise CompilerError(f"Unknown expr node type: {type(node)}")

def compile_source(source_code):
    lexer = Lexer(source_code)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    codegen = CodeGenerator()
    codegen.generate(ast)
    return "\n".join(codegen.lines)
