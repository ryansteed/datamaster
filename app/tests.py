from app.lib.metrics import CitationNetwork, TreeCitationNetwork
import app.lib.metrics
from app.lib.munge import QueryMunger, RootMunger
from app.config import Config

import json
import sys


# endpoints
def metrics_test():
    app.lib.metrics.main()


def root_test(patent, depth):
    munger = RootMunger(patent, depth=depth, limit=Config.DOC_LIMIT)
    # eval_and_sum(munger)
    cn = TreeCitationNetwork(munger.get_network(), patent)
    cn.write_graphml("{}_{}".format(patent, depth))
    cn.eval_binned(20)
    # cn.summary()

    # sample patents
    # 3961197


def root_test_single():
    check_args(4, "root [patent_number] [depth]")
    patent = sys.argv[2]
    depth = int(sys.argv[3])
    root_test(patent, depth)


def root_test_multiple():
    check_args(4, "root_all [query json file] [limit]")
    G = get_query_munger(sys.argv[2]).get_network()
    patents = [i for i in G.nodes][:int(sys.argv[3])]
    for patent in patents:
        root_test(str(patent), 2)


def query_test():
    check_args(3, "query [json file]")
    munger = get_query_munger(sys.argv[2])
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


def get_query_munger(query_file):
    with open(query_file, 'r') as f:
        query = json.load(f)
    return QueryMunger(query, limit=Config.DOC_LIMIT)
