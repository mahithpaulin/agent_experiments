import argparse
from utils import calculate

def main():
    parser = argparse.ArgumentParser(description='Simple calculator')
    parser.add_argument('op', choices=['add','sub','mul','div'], help='operation')
    parser.add_argument('numbers', nargs=2, type=float, help='two numbers')
    args = parser.parse_args()
    result = calculate(args.op, args.numbers[0], args.numbers[1])
    print(result)

if __name__ == '__main__':
    main()