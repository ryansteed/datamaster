import sys

from app.config import logger
from app.tests import *

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Missing arg.\n USAGE: python main.py [test_type]")

    test = sys.argv[1]
    logger.info("## Testing {} ##".format(test))

    if test == "metrics":
        metrics_test()

    if test == "query":
        query_test()

    if test == "root":
        root_test()
