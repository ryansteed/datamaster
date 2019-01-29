from app.lib.metrics import CitationNetwork, TreeCitationNetwork
import app.lib.metrics
from app.lib.munge import QueryMunger, RootMunger
from app.config import Config

import json
import sys


# endpoints
def metrics_test():
    app.lib.metrics.main()


def root_test_single():
    check_args(4, "root [patent_number] [depth]")
    patent = sys.argv[2]
    depth = int(sys.argv[3])
    munger = RootMunger(patent, depth=depth, limit=Config.DOC_LIMIT)
    cn = TreeCitationNetwork(munger.get_network(), patent)
    cn.eval_binned(20, plot=True)
    cn.write_graphml("{}_{}".format(patent, depth))


def root_test_multiple(bin_size=20):
    # TODO: build this as a function of the full network, handling empty root networks automatically - then build a full dataframe and save to file
    check_args(4, "root_all [query json file] [limit]")
    munger = get_query_munger(sys.argv[2], limit=int(sys.argv[3]))
    G = munger.get_network()
    cn = CitationNetwork(G, custom_centrality=False)
    cn.root_analysis(
        3,
        munger.make_filename(prefix="TIME-DATA_{}".format(sys.argv[3])),
        limit=int(sys.argv[3]),
        bin_size=bin_size
    )


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


def get_query_munger(query_file, limit=Config.DOC_LIMIT):
    with open(query_file, 'r') as f:
        query = json.load(f)
    return QueryMunger(query, limit=limit)
