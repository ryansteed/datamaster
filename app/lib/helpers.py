import time

from app.config import logger


class Timer:
    def __init__(self, name="Timer"):
        self.start = time.time()
        self.name = name

    def elapsed(self):
        return time.time() - self.start

    def log(self):
        logger.info("{} done ({}s)".format(self.name, self.elapsed()))
        return self

    def reset(self, name=None):
        self.start = time.time()
        if name is not None:
            self.name = name
        return self
