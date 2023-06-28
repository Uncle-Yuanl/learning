import sys
import argparse


def func(a):
    if a < 10:
        return a
    else:
        raise AttributeError


def navie(num):
    print(func(num))


def main(num):
    try:
        print(func(num))
        sys.exit(0)
    except Exception as e:
        sys.exit(-1)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('-i', '--integer', type=int)
    args = parse.parse_args()

    num = args.integer
    # navie(num)
    main(num)
    print('python continue')