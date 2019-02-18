import json
import sys
import os

from app.metrics import CitationNetwork, TreeCitationNetwork
from app.munge import QueryMunger, RootMunger
from app.config import Config, logger

"""
Official API endpoint script. All non-developer calls should run through here for convenience.
"""


def root_test_single(patent, depth, weighting_keys, bin_size=20):
    """
    The root endpoint constructs a descendant citation tree for one or more patents and calculates metrics for the root.

    :param patent: the patent number
    :type patent: str
    :param depth: the graph search depth
    :type depth: int
    :param bin_size: the bin size in weeks
    :type bin_size: int
    :param weighting_keys: the weighting key to use for knowledge calculation
    :type weighting_keys: list
    """
    munger = RootMunger(patent, depth=depth, limit=Config.DOC_LIMIT)
    cn = TreeCitationNetwork(munger.get_network(), patent, weighting_methods=weighting_keys)
    cn.eval_binned(bin_size, plot=True)
    cn.write_graphml("{}_{}".format(patent, depth))


def root_test_multiple(query_json_file, limit, weighting_keys, k_depth, bin_size=20, prefix="TIME-DATA"):
    """
    The root endpoint constructs a descendant citation tree for one or more patents and calculates metrics for the root.

    :param query_json_file: path to a JSON file containing the query to be queried
    :type query_json_file: str
    :param limit: the maximum number of docs to munge
    :type limit: int
    :param k_depth: the maximum depth to evaluate k
    :type k_depth: int
    :param bin_size: bin size in weeks
    :type bin_size: int
    :param weighting_keys: the weighting key to use for knowledge calculation
    :type weighting_keys: str
    :param prefix: prefix for final storage file name
    :type prefix: str
    """
    # TODO: build this as a function of the full network, handling empty root networks automatically
    #  - then build a full dataframe and save to file
    munger = get_query_munger(query_json_file, limit=limit)
    G = munger.get_network(limit=limit)
    cn = CitationNetwork(G, custom_centrality=False, weighting_methods=weighting_keys, k_depth=k_depth)
    cn.root_analysis(
        3,
        munger.make_filename(prefix="{}_{}".format(prefix, limit)),
        limit=limit,
        bin_size=bin_size
    )


def query_test(query_json_file, limit, weighting_keys, k_depth, write_graph=False):
    """
    The query endpoint collects patents for a query, constructs a citation network,
    and conducts metric calculations breadth-wise.

    :param query_json_file: path to a JSON file containing the query to be queried
    :type query_json_file: str
    :param limit: the maximum number of docs to munge
    :type limit: int
    :param k_depth: the maximum depth to evaluate k
    :type k_depth: int
    :param write_graph: whether or not to write the network to a graph ml file
    :type write_graph: bool
    :param weighting_keys: the weighting key to use for knowledge calculation
    :type weighting_keys: list
    """
    logger.debug(weighting_keys)
    munger = get_query_munger(query_json_file, limit=limit)
    eval_and_sum(munger, k_depth=k_depth, weighting_keys=weighting_keys, write_graph=write_graph)


def feature_test(query_json_file, limit, weighting_keys, k_depth):
    """
    The feature endpoint constructs descendant trees for a series of roots from a single query, but does not conduct
    time series analysis. It also collects additional observable features for use as controls in multiple regression.

    :param query_json_file: path to a JSON file containing the query to be queried
    :type query_json_file: str
    :param limit: the maximum number of docs to munge
    :type limit: int
    :param k_depth: the maximum depth to evaluate k
    :type k_depth: int
    :param weighting_keys: the weighting key to use for knowledge calculation
    :type weighting_keys: list
    """
    root_test_multiple(query_json_file, limit, k_depth, bin_size=None, weighting_keys=weighting_keys, prefix="FEATURE")


def eval_and_sum(munger,  weighting_keys, k_depth, write_graph=False):
    """
    Evaluates all metrics and summarize using the graph output from a munger.
    :param munger: the munger to analyze
    :param write_graph: whether or not to write the network to a graph ml file
    :param weighting_keys: the weighting key to use for knowledge calculation
    :param k_depth: the maximum depth to evaluate k
    :type k_depth: int
    """
    G = munger.get_network()
    cn = CitationNetwork(G, k_depth=k_depth, custom_centrality=False, knowledge=(not write_graph), weighting_methods=weighting_keys)
    filename = munger.make_filename(prefix="METRICS_{}".format(str(weighting_keys).strip(" ")))
    # cn.draw()
    cn.eval_all(file_early=filename)
    cn.summary()
    cn.file_custom_metrics(filename)
    if write_graph:
        cn.write_graphml(munger.make_filename(dirname="graph"))


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
