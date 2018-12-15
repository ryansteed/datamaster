import sys

import app.lib.metrics
from app.config import logger


# endpoints
def metrics_test():
    app.lib.metrics.main()


def query_test():
    app.lib.munge.test_query()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Missing arg.\n USAGE: python main.py [test_type]")

    test = sys.argv[1]
    logger.info("## Testing {} ##".format(test))

    if test == "metrics":
        metrics_test()

    if test == "query":
        query_test()
