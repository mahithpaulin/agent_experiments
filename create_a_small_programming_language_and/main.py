import sys
from compiler import compile_source

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <source_file>")
        sys.exit(1)
    source_file = sys.argv[1]
    with open(source_file, "r") as f:
        source_code = f.read()
    python_code = compile_source(source_code)
    print(python_code)

if __name__ == "__main__":
    main()