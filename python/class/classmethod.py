import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
from functools import wraps


def retry(times):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            trytime = 0
            while trytime < times:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    trytime += 1
        return wrapper
    return decorator


class DataProcess:

    logger = logging.getLogger("【DataPropressor】")

    def __init__(self, logger):
        self.logger = logger
        
    @classmethod
    def info(cls):
        cls.logger.info('test')

    def infoi(self):
        self.logger.info('test')

    @classmethod
    @retry(5)
    def func_need_retry(cls):
        cls.logger.info('retry')

    @retry(5)
    @classmethod
    def func_need_retry_rev(cls):
        """这种方式不行，貌似都没执行
        """
        cls.logger.info('retry rev')
    

if __name__ == "__main__":
    DataProcess.info()
    logger = logging.getLogger("【outer】")
    d = DataProcess(logger)
    d.info()
    d.infoi()
    DataProcess.func_need_retry()
    DataProcess.func_need_retry_rev()