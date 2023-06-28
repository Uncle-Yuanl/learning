import time
from contextlib import contextmanager


@contextmanager
def timer(name, logger):
    start = time.time()
    yield
    cost = round(time.time() - start, 2)
    logger.info(f"[{name}] 耗时 {cost} s")


if __name__ == "__main__":
    pass