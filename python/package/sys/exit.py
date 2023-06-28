import sys

def raw_exit():
    print('start')
    sys.exit(0)
    print('end')


def try_exit():
    try:
        print('start')
        sys.exit(0)
        print('middle')
        raise AttributeError('try exit error')
    except Exception as e:
        print(e)
        sys.exit(0)
    finally:
        print('end')


def except_exit():
    try:
        print('start')
        a = 1 / 0
    except Exception as e:
        print(e)
        sys.exit(0)
    finally:
        print('end')


if __name__ == "__main__":
    # raw_exit()
    # try_exit()
    except_exit()
    print('out end')