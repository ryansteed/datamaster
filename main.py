import sys

import app.lib.metrics
from app.lib.munge import Munger
from app.config import Config
from app.config import logger
from app.lib.metrics import CitationNetwork
import json


# endpoints
def metrics_test():
    app.lib.metrics.main()


def query_test():
    if len(sys.argv) < 3:
        raise ValueError("No query passed.\n USAGE: python main.py query [json file]")

    with open(sys.argv[2], 'r') as f:
        query = json.load(f)
    munger = Munger(limit=Config.DOC_LIMIT)
    munger.load_data_from_query(query)
    G = munger.get_network(metadata=True)
    cn = CitationNetwork(G)
    cn.eval_all()
    cn.summary()
    cn.file_custom_metrics(munger.make_query_filename(query))

    # Possible queries:
    # test from https://ropensci.github.io/patentsview/articles/citation-networks.html
    # {"cpc_subgroup_id": "Y10S707/933"}

    # test from https://link.springer.com/article/10.1007/s11192-017-2252-y
    # {"uspc_mainclass_id": "372"}

    # artificial intelligence
    # {"uspc_mainclass_id": "706"}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Missing arg.\n USAGE: python main.py [test_type]")

    test = sys.argv[1]
    logger.info("## Testing {} ##".format(test))

    if test == "metrics":
        metrics_test()

    if test == "query":
        query_test()
