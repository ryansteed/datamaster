import sys

import app.lib.metrics
from app.config import logger


## endpoints
def metrics_test():
    app.lib.metrics.main()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Missing arg.\n USAGE: python main.py [test_type]")

    test = sys.argv[1]
    logger.info("## Testing {} ##".format(test))

    if test == "metrics":
        metrics_test()
