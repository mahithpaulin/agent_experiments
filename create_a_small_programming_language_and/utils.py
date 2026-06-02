def indent_lines(code, level=1):
    prefix = '    ' * level
    return '\n'.join(prefix + line if line.strip() else line for line in code.split('\n'))

def is_identifier(s):
    import re
    return re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', s) is not None

def escape_string(s):
    return s.replace('\\', '\\\\').replace('"', '\\"')

def flatten(lst):
    return [item for sublist in lst for item in sublist] if lst and isinstance(lst[0], list) else lst

def join_lines(lines):
    return '\n'.join(lines)