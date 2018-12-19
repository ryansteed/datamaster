from app.lib.metrics import CitationNetwork
import app.lib.metrics
from app.lib.munge import QueryMunger, RootMunger
from app.config import Config

import json
import sys


# endpoints
def metrics_test():
    app.lib.metrics.main()


def root_test():
    check_args(3, "root [patent_number]")
    patent = int(sys.argv[2])
    munger = RootMunger(patent, depth=1, limit=Config.DOC_LIMIT)
    eval_and_sum(munger)

    # sample patents
    # 3961197


def query_test():
    check_args(3, "query [json file]")
    with open(sys.argv[2], 'r') as f:
        query = json.load(f)
    munger = QueryMunger(query, limit=Config.DOC_LIMIT)
    eval_and_sum(munger)
    # Possible queries:
    # test from https://ropensci.github.io/patentsview/articles/citation-networks.html
    # {"cpc_subgroup_id": "Y10S707/933"}

    # test from https://link.springer.com/article/10.1007/s11192-017-2252-y
    # {"uspc_mainclass_id": "372"}

    # artificial intelligence
    # {"uspc_mainclass_id": "706"}


def eval_and_sum(munger):
    G = munger.get_network()
    cn = CitationNetwork(G, custom_centrality=False)
    # cn.draw()
    cn.eval_all()
    cn.summary()
    cn.file_custom_metrics((sys.argv[2].strip("json")))


def check_args(num, usage):
    if len(sys.argv) < num:
        raise ValueError("No query passed.\n USAGE: python main.py {}".format(usage))
