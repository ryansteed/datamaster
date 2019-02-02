from app.config import logger
from app.tests import *

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Missing arg.\n USAGE: python main.py [test_type]")

    test = sys.argv[1]
    logger.info("## Testing {} ##".format(test))

    """
    The query endpoint collects patents for a query, constructs a citation network, 
    and conducts metric calculations.
    """
    if test == "query":
        query_test()

    """
    The root endpoint constructs a descendant citation tree for one or more patents and calculates metrics for the root.
    """
    if test == "root":
        root_test_single()
    if test == "root_all":
        root_test_multiple()

    """
    The feature endpoint constructs descendant trees for a series of roots from a single query, but does not conduct
    time series analysis. It also collects additional observable features for use as controls in multiple regression.
    """
    if test == "features":
        root_test_multiple(bin_size=None)
