import re

# Token types
(
    NUMBER, IDENT, KEYWORD, OP, LPAREN, RPAREN, LBRACE, RBRACE, SEMI, COMMA, EOF
) = (
    'NUMBER', 'IDENT', 'KEYWORD', 'OP', 'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE', 'SEMI', 'COMMA', 'EOF'
)

KEYWORDS = {'if', 'else', 'while', 'def', 'return', 'true', 'false'}

class Token:
    def __init__(self, type_, value, pos):
        self.type = type_
        self.value = value
        self.pos = pos

    def __repr__(self):
        return f'Token({self.type}, {self.value})'

class Lexer:
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.current_char = self.text[self.pos] if self.text else None

    def error(self):
        raise Exception(f'Invalid character {self.current_char} at pos {self.pos}')

    def advance(self):
        self.pos += 1
        if self.pos >= len(self.text):
            self.current_char = None
        else:
            self.current_char = self.text[self.pos]

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def number(self):
        result = ''
        while self.current_char is not None and (self.current_char.isdigit() or self.current_char == '.'):
            result += self.current_char
            self.advance()
        if result.count('.') > 1:
            self.error()
        if '.' in result:
            return Token(NUMBER, float(result), self.pos)
        else:
            return Token(NUMBER, int(result), self.pos)

    def ident_or_keyword(self):
        result = ''
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
            result += self.current_char
            self.advance()
        if result in KEYWORDS:
            return Token(KEYWORD, result, self.pos)
        else:
            return Token(IDENT, result, self.pos)

    def get_next_token(self):
        while self.current_char is not None:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char.isdigit():
                return self.number()

            if self.current_char.isalpha() or self.current_char == '_':
                return self.ident_or_keyword()

            if self.current_char == '=':
                self.advance()
                if self.current_char == '=':
                    self.advance()
                    return Token(OP, '==', self.pos)
                else:
                    return Token(OP, '=', self.pos)

            if self.current_char == '!':
                self.advance()
                if self.current_char == '=':
                    self.advance()
                    return Token(OP, '!=', self.pos)
                else:
                    self.error()

            if self.current_char == '<':
                self.advance()
                if self.current_char == '=':
                    self.advance()
                    return Token(OP, '<=', self.pos)
                else:
                    return Token(OP, '<', self.pos)

            if self.current_char == '>':
                self.advance()
                if self.current_char == '=':
                    self.advance()
                    return Token(OP, '>=', self.pos)
                else:
                    return Token(OP, '>', self.pos)

            if self.current_char in '+-*/%':
                ch = self.current_char
                self.advance()
                return Token(OP, ch, self.pos)

            if self.current_char == '(':  # LPAREN
                self.advance()
                return Token(LPAREN, '(', self.pos)

            if self.current_char == ')':  # RPAREN
                self.advance()
                return Token(RPAREN, ')', self.pos)

            if self.current_char == '{':  # LBRACE
                self.advance()
                return Token(LBRACE, '{', self.pos)

            if self.current_char == '}':  # RBRACE
                self.advance()
                return Token(RBRACE, '}', self.pos)

            if self.current_char == ';':  # SEMI
                self.advance()
                return Token(SEMI, ';', self.pos)

            if self.current_char == ',':  # COMMA
                self.advance()
                return Token(COMMA, ',', self.pos)

            self.error()

        return Token(EOF, None, self.pos)

# AST nodes
class AST:
    pass

class Program(AST):
    def __init__(self, statements):
        self.statements = statements

class Block(AST):
    def __init__(self, statements):
        self.statements = statements

class VarAssign(AST):
    def __init__(self, name, expr):
        self.name = name
        self.expr = expr

class Var(AST):
    def __init__(self, name):
        self.name = name

class Number(AST):
    def __init__(self, value):
        self.value = value

class Boolean(AST):
    def __init__(self, value):
        self.value = value

class BinOp(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

class UnaryOp(AST):
    def __init__(self, op, expr):
        self.op = op
        self.expr = expr

class If(AST):
    def __init__(self, cond, then_block, else_block=None):
        self.cond = cond
        self.then_block = then_block
        self.else_block = else_block

class While(AST):
    def __init__(self, cond, block):
        self.cond = cond
        self.block = block

class FuncDef(AST):
    def __init__(self, name, params, block):
        self.name = name
        self.params = params
        self.block = block

class FuncCall(AST):
    def __init__(self, name, args):
        self.name = name
        self.args = args

class Return(AST):
    def __init__(self, expr=None):
        self.expr = expr

class Parser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()

    def error(self, msg=None):
        if msg is None:
            msg = f'Unexpected token {self.current_token}'
        raise Exception(msg)

    def eat(self, token_type, value=None):
        if self.current_token.type == token_type and (value is None or self.current_token.value == value):
            self.current_token = self.lexer.get_next_token()
        else:
            self.error(f'Expected token type {token_type} with value {value}, got {self.current_token}')

    def program(self):
        statements = []
        while self.current_token.type != EOF:
            if self.current_token.type == SEMI:
                self.eat(SEMI)
                continue
            stmt = self.statement()
            statements.append(stmt)
        return Program(statements)

    def statement(self):
        if self.current_token.type == KEYWORD:
            if self.current_token.value == 'if':
                return self.if_statement()
            elif self.current_token.value == 'while':
                return self.while_statement()
            elif self.current_token.value == 'def':
                return self.func_def()
            elif self.current_token.value == 'return':
                return self.return_statement()
            else:
                self.error(f'Unknown keyword {self.current_token.value}')
        elif self.current_token.type == IDENT:
            # Could be assignment or function call
            next_token = self.lexer.get_next_token()
            self.lexer.pos -= len(next_token.value) if next_token.value else 0
            self.lexer.current_char = self.lexer.text[self.lexer.pos] if self.lexer.pos < len(self.lexer.text) else None
            if next_token.type == OP and next_token.value == '=':
                return self.assignment()
            else:
                expr = self.expr()
                self.eat(SEMI)
                return expr
        else:
            expr = self.expr()
            self.eat(SEMI)
            return expr

    def assignment(self):
        var_name = self.current_token.value
        self.eat(IDENT)
        self.eat(OP, '=')
        expr = self.expr()
        self.eat(SEMI)
        return VarAssign(var_name, expr)

    def if_statement(self):
        self.eat(KEYWORD, 'if')
        self.eat(LPAREN)
        cond = self.expr()
        self.eat(RPAREN)
        then_block = self.block()
        else_block = None
        if self.current_token.type == KEYWORD and self.current_token.value == 'else':
            self.eat(KEYWORD, 'else')
            else_block = self.block()
        return If(cond, then_block, else_block)

    def while_statement(self):
        self.eat(KEYWORD, 'while')
        self.eat(LPAREN)
        cond = self.expr()
        self.eat(RPAREN)
        block = self.block()
        return While(cond, block)

    def func_def(self):
        self.eat(KEYWORD, 'def')
        name = self.current_token.value
        self.eat(IDENT)
        self.eat(LPAREN)
        params = []
        if self.current_token.type == IDENT:
            params.append(self.current_token.value)
            self.eat(IDENT)
            while self.current_token.type == COMMA:
                self.eat(COMMA)
                params.append(self.current_token.value)
                self.eat(IDENT)
        self.eat(RPAREN)
        block = self.block()
        return FuncDef(name, params, block)

    def return_statement(self):
        self.eat(KEYWORD, 'return')
        if self.current_token.type == SEMI:
            self.eat(SEMI)
            return Return()
        expr = self.expr()
        self.eat(SEMI)
        return Return(expr)

    def block(self):
        self.eat(LBRACE)
        statements = []
        while self.current_token.type != RBRACE:
            if self.current_token.type == SEMI:
                self.eat(SEMI)
                continue
            stmt = self.statement()
            statements.append(stmt)
        self.eat(RBRACE)
        return Block(statements)

    def expr(self):
        return self.expr_logic()

    def expr_logic(self):
        node = self.expr_arith()
        while self.current_token.type == OP and self.current_token.value in ('==', '!=', '<', '<=', '>', '>='):
            op = self.current_token.value
            self.eat(OP)
            right = self.expr_arith()
            node = BinOp(node, op, right)
        return node

    def expr_arith(self):
        node = self.term()
        while self.current_token.type == OP and self.current_token.value in ('+', '-'):
            op = self.current_token.value
            self.eat(OP)
            right = self.term()
            node = BinOp(node, op, right)
        return node

    def term(self):
        node = self.factor()
        while self.current_token.type == OP and self.current_token.value in ('*', '/', '%'):
            op = self.current_token.value
            self.eat(OP)
            right = self.factor()
            node = BinOp(node, op, right)
        return node

    def factor(self):
        token = self.current_token
        if token.type == OP and token.value in ('+', '-'):
            self.eat(OP)
            node = UnaryOp(token.value, self.factor())
            return node
        elif token.type == NUMBER:
            self.eat(NUMBER)
            return Number(token.value)
        elif token.type == KEYWORD and token.value in ('true', 'false'):
            self.eat(KEYWORD)
            return Boolean(token.value == 'true')
        elif token.type == IDENT:
            # Could be function call or variable
            name = token.value
            self.eat(IDENT)
            if self.current_token.type == LPAREN:
                self.eat(LPAREN)
                args = []
                if self.current_token.type != RPAREN:
                    args.append(self.expr())
                    while self.current_token.type == COMMA:
                        self.eat(COMMA)
                        args.append(self.expr())
                self.eat(RPAREN)
                return FuncCall(name, args)
            else:
                return Var(name)
        elif token.type == LPAREN:
            self.eat(LPAREN)
            node = self.expr()
            self.eat(RPAREN)
            return node
        else:
            self.error()

class Compiler:
    def __init__(self):
        self.indent_level = 0
        self.lines = []

    def indent(self):
        return '    ' * self.indent_level

    def compile(self, node):
        method_name = 'compile_' + type(node).__name__
        method = getattr(self, method_name, self.generic_compile)
        return method(node)

    def generic_compile(self, node):
        raise Exception(f'No compile_{type(node).__name__} method')

    def compile_Program(self, node):
        for stmt in node.statements:
            self.compile(stmt)
        return '\n'.join(self.lines)

    def compile_Block(self, node):
        self.indent_level += 1
        for stmt in node.statements:
            self.compile(stmt)
        self.indent_level -= 1

    def compile_VarAssign(self, node):
        expr_code = self.compile(node.expr)
        self.lines.append(f'{self.indent()}{node.name} = {expr_code}')

    def compile_Var(self, node):
        return node.name

    def compile_Number(self, node):
        return repr(node.value)

    def compile_Boolean(self, node):
        return 'True' if node.value else 'False'

    def compile_BinOp(self, node):
        left = self.compile(node.left)
        right = self.compile(node.right)
        op = node.op
        return f'({left} {op} {right})'

    def compile_UnaryOp(self, node):
        expr = self.compile(node.expr)
        return f'({node.op}{expr})'

    def compile_If(self, node):
        cond = self.compile(node.cond)
        self.lines.append(f'{self.indent()}if {cond}:')
        self.compile(node.then_block)
        if node.else_block:
            self.lines.append(f'{self.indent()}else:')
            self.compile(node.else_block)

    def compile_While(self, node):
        cond = self.compile(node.cond)
        self.lines.append(f'{self.indent()}while {cond}:')
        self.compile(node.block)

    def compile_FuncDef(self, node):
        params = ', '.join(node.params)
        self.lines.append(f'{self.indent()}def {node.name}({params}):')
        self.indent_level += 1
        if not node.block.statements:
            self.lines.append(f'{self.indent()}pass')
        else:
            self.compile(node.block)
        self.indent_level -= 1

    def compile_FuncCall(self, node):
        args = ', '.join(self.compile(arg) for arg in node.args)
        return f'{node.name}({args})'

    def compile_Return(self, node):
        if node.expr is None:
            self.lines.append(f'{self.indent()}return')
        else:
            expr = self.compile(node.expr)
            self.lines.append(f'{self.indent()}return {expr}')


def compile_source(source):
    lexer = Lexer(source)
    parser = Parser(lexer)
    ast = parser.program()
    compiler = Compiler()
    pycode = compiler.compile(ast)
    return pycode
