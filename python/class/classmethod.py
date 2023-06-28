import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class DataProcess:

    logger = logging.getLogger("【DataPropressor】")

    def __init__(self, logger):
        self.logger = logger
        
    @classmethod
    def info(cls):
        cls.logger.info('test')

    def infoi(self):
        self.logger.info('test')


if __name__ == "__main__":
    DataProcess.info()
    logger = logging.getLogger("【outer】")
    d = DataProcess(logger)
    d.info()
    d.infoi()

