import time
from app.config import logger


class Timer:
    """A helpful timer for tracking processes with the logger."""
    def __init__(self, name="Timer", verbose=True):
        self.init = None
        self.name = name
        self.verbose = verbose
        self.start()

    def start(self):
        self.init = time.time()
        if self.verbose:
            logger.info("{} started".format(self.name))

    def elapsed(self):
        return time.time() - self.init

    def log(self):
        if self.verbose:
            logger.info("{} done ({}s)".format(self.name, self.elapsed()))
        return self

    def reset(self, name=None):
        if name is not None:
            self.name = name
        self.start()
        return self
