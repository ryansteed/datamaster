import json
import sys

from app.metrics import CitationNetwork, TreeCitationNetwork
from app.munge import QueryMunger, RootMunger
from app.config import Config


def root_test_single(bin_size=20):
    """
    Analyzes knowledge impact over time for a single patent
    :param bin_size: the bin size in weeks
    """
    check_args(4, "root [patent_number] [depth]")
    patent = sys.argv[2]
    depth = int(sys.argv[3])
    munger = RootMunger(patent, depth=depth, limit=Config.DOC_LIMIT)
    cn = TreeCitationNetwork(munger.get_network(), patent)
    cn.eval_binned(bin_size, plot=True)
    cn.write_graphml("{}_{}".format(patent, depth))


def root_test_multiple(bin_size=20):
    """
    Analyzes knowledge impact over time for a query of patents
    :param bin_size: bin size in weeks
    """
    # TODO: build this as a function of the full network, handling empty root networks automatically
    #  - then build a full dataframe and save to file
    check_args(4, "root_all [query json file] [limit]")
    munger = get_query_munger(sys.argv[2], limit=int(sys.argv[3]))
    G = munger.get_network()
    cn = CitationNetwork(G, custom_centrality=False)
    cn.root_analysis(
        3,
        munger.make_filename(prefix="TIME-DATA_{}".format(sys.argv[3])),
        bin_size=bin_size
    )


def query_test():
    """
    Evaluates all metrics for a query, breadth-wise.
    """
    check_args(3, "query [json file]")
    munger = get_query_munger(sys.argv[2])
    eval_and_sum(munger)


def eval_and_sum(munger):
    """
    Evaluates all metrics and summarize using the graph output from a munger.
    :param munger: the munger to analyze
    """
    G = munger.get_network()
    cn = CitationNetwork(G, custom_centrality=False)
    # cn.draw()
    cn.eval_all()
    cn.summary()
    cn.file_custom_metrics((sys.argv[2].strip("json")))


def check_args(num, usage):
    """
    Verifies the proper number of arguments have been submitted. Print usage if not.
    :param num: The proper number of arguments.
    :param usage: A string describing the usage for this endpoint.
    :return:
    """
    if len(sys.argv) < num:
        raise ValueError("No query passed.\n USAGE: python main.py {}".format(usage))


def get_query_munger(query_file, limit=Config.DOC_LIMIT):
    """
    Construct a query munger for a given query, stored in a JSON file.
    :param query_file: the path to the query file
    :param limit: the maximum number of docs to query
    :return: a QueryMunger with this configuration
    """
    with open(query_file, 'r') as f:
        query = json.load(f)
    return QueryMunger(query, limit=limit)
