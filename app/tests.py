import json
import sys
import os

from app.metrics import CitationNetwork, TreeCitationNetwork
from app.munge import QueryMunger, RootMunger
from app.config import Config

"""
Official API endpoint script. All non-developer calls should run through here for convenience.
"""


def root_test_single(patent, depth, bin_size=20):
    """
    The root endpoint constructs a descendant citation tree for one or more patents and calculates metrics for the root.

    :param patent: the patent number
    :type patent: str
    :param depth: the graph search depth
    :type depth: int
    :param bin_size: the bin size in weeks
    :type bin_size: int
    """
    munger = RootMunger(patent, depth=depth, limit=Config.DOC_LIMIT)
    cn = TreeCitationNetwork(munger.get_network(), patent)
    cn.eval_binned(bin_size, plot=True)
    cn.write_graphml("{}_{}".format(patent, depth))


def root_test_multiple(query_json_file, limit, bin_size=20):
    """
    The root endpoint constructs a descendant citation tree for one or more patents and calculates metrics for the root.

    :param query_json_file: path to a JSON file containing the query to be queried
    :type query_json_file: str
    :param limit: the maximum number of docs to munge
    :type limit: int
    :param bin_size: bin size in weeks
    :type bin_size: int
    """
    # TODO: build this as a function of the full network, handling empty root networks automatically
    #  - then build a full dataframe and save to file
    munger = get_query_munger(query_json_file, limit=limit)
    G = munger.get_network(limit=limit)
    cn = CitationNetwork(G, custom_centrality=False)
    cn.root_analysis(
        3,
        munger.make_filename(prefix="TIME-DATA_{}".format(limit)),
        limit=limit,
        bin_size=bin_size
    )


def query_test(query_json_file):
    """
    The query endpoint collects patents for a query, constructs a citation network,
    and conducts metric calculations breadth-wise.

    :param query_json_file: path to a JSON file containing the query to be queried
    :type query_json_file: str
    """
    munger = get_query_munger(query_json_file)
    eval_and_sum(munger)


def feature_test(query_json_file, limit):
    """
    The feature endpoint constructs descendant trees for a series of roots from a single query, but does not conduct
    time series analysis. It also collects additional observable features for use as controls in multiple regression.

    :param query_json_file: path to a JSON file containing the query to be queried
    :type query_json_file: str
    :param limit: the maximum number of docs to munge
    :type limit: int
    """
    root_test_multiple(query_json_file, limit, bin_size=None)


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
    cn.file_custom_metrics(munger.make_filename())


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
