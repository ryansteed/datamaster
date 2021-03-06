import networkx as nx
import numpy as np
from scipy.stats import sem
import csv
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
from overrides import overrides
import enlighten
from collections import defaultdict

from app.config import Config, logger
from app.helpers import Timer
from app.munge import RootMunger


class CitationNetwork:
    """
    Tracks the graph and meta-attributes for any citation network
    """
    def __init__(
            self, G, weighting_methods=("forward_cites", "h_index"),
            quality=True, h_index=True, custom_centrality=False, knowledge=True,
            k_depth=Config.K_DEPTH, discount=Config.DISCOUNT
            ):
        """
        Initializes the CitationNetwork

        :param G: the citation graph
        :param weighting_methods: the primary importance weighting methods for the knowledge impact metric
        :param quality: whether to use the quality metric
        :param h_index: whether to use the h_index
        :param custom_centrality: whether to use custom_centrality
        :param knowledge: whether to calculate knowledge impact
        :param k_depth: maximum depth for knowledge evaluation
        """
        self.G = G
        self.weighting_methods = weighting_methods
        self.quality = quality
        self.h_index = h_index
        self.custom_centrality = custom_centrality
        self.knowledge = knowledge
        self.k_depth = k_depth
        self.discount = discount
        self.attributes = []
        self.search_tracker = GraphSearchTracker()
        self.k_search_tracker = GraphSearchTracker()
        if quality:
            self.attributes += ['forward_cites', 'backward_cites', 'family_size', 'num_claims']
        if h_index:
            self.attributes.append("h_index")
        if custom_centrality:
            self.attributes.append("custom_centrality")
        if knowledge:
            self.attributes += [self.make_knowledge_name(key) for key in weighting_methods]

    # Metric Calculation #
    def eval_all(self, weighting_keys=None, verbose=True, file_early=None, knowledge=True):
        """
        Calculates all custom metrics, if requested during instantiation

        :param weighting_keys: the preferred weighting key to use for knowledge impact
        :param verbose: whether or not to log progress
        """
        if verbose:
            logger.info("Calculating metrics")
            t = Timer("Metric calculation")
        if self.quality:
            if verbose: logger.info("Calculating quality")
            self.eval_quality()
            if verbose: t.log()
        if self.h_index:
            if verbose: logger.info("Calculating H-index")
            self.eval_h()
            if verbose: t.log()
        if self.custom_centrality:
            if verbose: logger.info("Calculating centralities")
            self.eval_centrality()
            if verbose: t.log()
        if file_early is not None:
            self.file_custom_metrics(file_early)
        if self.knowledge and knowledge:
            if verbose: logger.info("Calculating knowledge")
            self.eval_k(self.weighting_methods if weighting_keys is None else weighting_keys)
            if verbose: t.log()

    def eval_h(self):
        """
        calculates the h_index and set it as a node attribute
        """
        h_indices = {}
        for node in self.G:
            forward_cites = [self.G.nodes[child]['forward_cites'] for child in self.G.successors(node)]
            h_indices[node] = CitationNetwork.h_index(forward_cites)
        nx.set_node_attributes(self.G, h_indices, 'h_index')

    def eval_centrality(self):
        """
        calculates the centrality and set it as a node attribute
        """

        # For each centrality, need:
        # - Maximum
        # - Minimum
        # - Variance ratio
        c = {
            '+': [],
            '-': [],
            'var': [],
            'c': []
        }
        custom_centralities = {}

        # Centralities
        centralities = [
            nx.degree_centrality(self.G),
            nx.betweenness_centrality(self.G),
            nx.closeness_centrality(self.G)
        ]
        try:
            centralities.append(nx.eigenvector_centrality_numpy(self.G))
            # Local vals
            for centrality in centralities:
                c['+'].append(centrality[max(centrality, key=centrality.get)])
                c['-'].append(centrality[min(centrality, key=centrality.get)])
                c['var'].append(np.var([val for key, val in centrality.items()]))

            # Centrality metric
            var_t = sum(c['var'])
            s_optimums = []
            for node in self.G:
                for key in ['-', '+']:
                    s_optimums.append(np.sqrt(
                        np.sum(
                            [
                                c['var'][i] * (centralities[i][node] - c[key][i] / var_t) ** 2
                                for i in range(len(centralities))
                            ]
                        )
                    ))
                custom_centralities[node] = s_optimums[0] / (s_optimums[0] + s_optimums[1])
        except Exception as e:
            logger.warn("Failed to calculate eigenvector centralities with error {}".format(e))

        nx.set_node_attributes(self.G, custom_centralities, 'custom_centrality')

    def eval_quality(self):
        """
        determines patent quality with established metrics; saves metrics as node attributes
        """
        nx.set_node_attributes(
            self.G,
            {node: out_degree for node, out_degree in self.G.out_degree()},
            'forward_cites'
        )
        nx.set_node_attributes(
            self.G,
            {node: in_degree for node, in_degree in self.G.in_degree()},
            'backward_cites'
        )
        nx.set_node_attributes(
            self.G,
            {node: int(np.random.normal(3, 2)) for node in self.G},
            'family_size'
        )
        nx.set_node_attributes(
            self.G,
            {node: int(np.random.normal(2.5, 0.5)) for node in self.G},
            'num_claims'
        )

    # Total Knowledge Contribution #
    def eval_k(self, weighting_keys, verbose=False):
        """
        Evaluates the knowledge impact metric for every node and saves as node attribute
        :param weighting_keys: the quality metric to use as a weight
        :param verbose: whether or not to print out progress, as opposed to using the ticker
        """
        node_attrs = {}

        manager = enlighten.get_manager()
        ticker = manager.counter(total=len(self.G.nodes), desc='Ticks', unit='ticks')
        div = Config.PROGRESS_DIV
        if verbose:
            t = Timer("{}%".format(round(1/div*100)))

        for i, node in enumerate(self.G.nodes):
            # keep track of visited nodes to avoid graph cycles
            self.k_search_tracker.reset([node])
            # save the k value for this node
            node_attrs[node] = self.k(node, node, weighting_keys, 0)
            if verbose and i % int(len(self.G.nodes) / div) == 0:
                t.log()
                if i / int(len(self.G.nodes) / div) < div:
                    t.reset("{}%".format(round((i / int(len(self.G.nodes))+1/div)*100)))
            ticker.update()

        ticker.close()

        # add the k values as attributes in the graph
        for key in weighting_keys:
            nx.set_node_attributes(
                self.G,
                {k: v[key] for k, v in node_attrs.items()},
                self.make_knowledge_name(key)
            )

    def k(self, root, node, weighting_keys, depth, verbose=False):
        """
        Recursively calculates the knowledge impact for a single node

        :math:`K_i = W_i + \sum_{j=1}^{n_i} \lambda P_jK_{j}`

        :param root: the root node
        :param node: the current node
        :param weighting_keys: the quality weighting key to use for knowledge impact
        :param depth: the current search depth
        :param verbose: whether or not to print progress to stdout
        :return: a dictionary containing the total knowledge impact score keyed by the weighting metric used
        """
        # base case - exceeded depth allowed
        if self.k_depth is not None and depth > self.k_depth:
            return {key: 0 for key in weighting_keys}

        # keep track of the sum of child scores in a dictionary keyed by weighting method used
        sum_children = defaultdict(int)
        # generate the list of this node's children, excluding visited nodes
        children = [x for x in self.G.successors(node) if x is not None and not self.k_search_tracker.is_visited(x)]
        for child in children:
            # recursively evaluate k for this child (returns dict ordered by weighting key)
            # add the k score to the sum total by weighting key
            for key, val in self.k(root, child, weighting_keys, depth+1).items():
                sum_children[key] += val

        # keep track of the total in a dictionary keyed by weighting method
        total_k = defaultdict(int)
        # calculate the persistence index for this node
        p = self.p(root, node)
        # calculate total knowledge contribution by weighting method
        for key in weighting_keys:
            # Note that p is applied to ALL calculations - p for the root is simply 1,
            # and since p is distributive over the children it can be applied to each child separately.
            # Makes it easier to evaluate persistence (would otherwise have to remember each child).
            total_k[key] = (self.G.nodes[node][key] + self.discount * sum_children[key]) * p

        if verbose:
            logger.info('node', node)
            logger.info('> w: ', self.G.nodes[node][weighting_keys])
            logger.info('> p: ', self.p(root, node))
            logger.info('> k: ', total_k)

        return total_k

    def p(self, root, node):
        return 1 if node == root else 1 / int(self.G.in_degree(node))

    # Time Series Analysis #
    def root_analysis(self, depth, filename, allow_external=Config.ALLOW_EXTERNAL, limit=Config.DOC_LIMIT, bin_size=20):
        """
        Instead of evaluating knowledge impact within the network (breadth-first), conducts a depth-first calculation
        for every node in the network up to some limit.
        Then, calculate knowledge impact in bins of a given size by time.
        Write time-series results to a file.

        :param depth: the depth of the search (how many generations of children to examine)
        :param filename: the filename to save results
        :param limit: the maximum number of nodes to evaluate
        :param bin_size: the size of the time bins in weeks
        :param allow_external: whether or not to allow external patents in the analysis
        :param query: a query to help speed up feature querying, if no external patents allowed
        """
        df = None
        patents = [i for i in self.G.nodes]
        if limit is not None:
            patents = patents[:limit+1]

        manager = enlighten.get_manager()
        ticker = manager.counter(total=len(patents), desc='Patent Trees Analyzed', unit='patents')

        for i, patent in enumerate(patents):
            if allow_external:
                munger = RootMunger(patent, depth=depth, limit=Config.DOC_LIMIT)
                network = munger.get_network()
                features = munger.features
            else:
                self.search_tracker.reset([patent])
                network = nx.DiGraph(self.G.subgraph([patent] + self.get_successors_recursively(patent, depth, 0)))
                try:
                    features = network.nodes[patent]['features']
                except KeyError:
                    logger.warn("Bad query input file. For root analysis, need to first collect features. "
                                "Try again with cache=False.")
                    features = RootMunger.query_features_single(patent)

            cn = TreeCitationNetwork(
                network,
                patent,
                weighting_methods=self.weighting_methods,
                k_depth=self.k_depth,
            )
            if not cn.is_empty():
                data = cn.eval_binned(bin_size, weighting_keys=self.weighting_methods, plot=False)
                if df is None:
                    df = self.make_df(features, data)
                else:
                    df = df.append(self.make_df(features, data), ignore_index=True)
                with open(filename, "w+") as file:
                    df.to_csv(file, index=False, header=True)
            ticker.update()

        ticker.close()

    def make_df(self, features, data):
        return pd.DataFrame(
            data=[
                [x[key] for key in self.weighting_methods] +
                [str(i)] +
                [str(val).replace(",", " ") for val in features.values()] for i, x in enumerate(data)
            ],
            columns=[self.make_knowledge_name(key) for key in self.weighting_methods] + ["t"] + list(features.keys())
        )

    def get_successors_recursively(self, patent, max_depth, depth):
        successors = []
        children = [patent for patent in self.G.successors(patent) if not self.search_tracker.is_visited(patent)]
        for child in children:
            successors.append(child)
            if depth < max_depth:
                successors += self.get_successors_recursively(child, depth+1, max_depth)
        return successors

    # Helpers #
    @staticmethod
    def h_index(m):
        """
        Calculates the h index for a list of values
        :param m: list of values
        :return: the h-index
        """
        s = [0] * (len(m) + 1)
        for i in range(len(m)):
            s[min([len(m), m[i]])] += 1
        x = 0
        for i in reversed(range(len(s))):
            x += s[i]
            if x >= i:
                return i
        return 0

    @staticmethod
    def make_knowledge_name(weighting_key):
        return "knowledge_{}".format(weighting_key)

    @staticmethod
    def str_to_datetime(date):
        return datetime.strptime(date, '%Y-%M-%d')

    # Descriptions #
    def summary(self):
        """
        Prints summary statistics for the graph metrics
        """
        custom = ""
        custom += "Connected components: {}\n".format(nx.number_connected_components(self.G.to_undirected()))
        # average metrics
        for key, values in {attribute: list(nx.get_node_attributes(self.G, attribute).values()) for attribute in
                            self.attributes}.items():
            custom += "{}: {} ({})\n".format(key, round(np.average(values), 3), round(sem(values), 3))
        logger.info("\n== CN Summary ==\n{}\n{}====".format(nx.info(self.G), custom))

    def print_custom_metrics(self):
        """
        Summarize the calculated metrics
        """
        logger.info("== Calculated Metrics ==")
        for node in self.G.nodes:
            logger.info(self.G.nodes[node])
        logger.info("====")

    def draw(self):
        """
        Draws the network
        """
        nx.draw_networkx(self.G, pos=nx.kamada_kawai_layout(self.G))
        plt.show()

    def write_graphml(self, filepath):
        """
        Writes the graph to a graphml file

        :param filepath: the filename to store, as generated by a QueryMunger instance
        :type filepath: str
        """
        filepath = "{}.graphml".format(filepath.strip(".csv"))
        t = Timer("Writing to graphml {}".format(filepath))
        nx.write_graphml(self.G, filepath)
        t.log()

    def file_custom_metrics(self, filename):
        """
        Files the calculated metrics in a CSV
        :param filename: the filename
        """
        logger.info("Filing calculated metrics in {}".format(filename))
        t = Timer("Filing metrics")
        with open(filename, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(['node'] + self.attributes)
            for node in self.G.nodes:
                row = [node] + list(self.G.nodes[node].values())
                writer.writerow(row)
        t.log()


class TreeCitationNetwork(CitationNetwork):
    """
    A special variation on the CitationNetwork built specifically to track the descendants of a single patent.
    """
    def __init__(
            self, G, root, weighting_methods=("h_index", "forward_cites"),
            k_depth=Config.K_DEPTH, quality=True, h_index=True, custom_centrality=False, knowledge=True
    ):
        super().__init__(
            G,
            weighting_methods=weighting_methods,
            quality=quality,
            h_index=h_index,
            custom_centrality=custom_centrality,
            knowledge=knowledge,
            k_depth=k_depth
        )
        self.root = root

    def is_empty(self):
        return self.G.size() == 0

    def eval_binned(self, bin_size_weeks, plot=False, weighting_keys=["h_index", "forward_cites"]):
        """
        Evaluates knowledge impact in time-based bins.

        :param bin_size_weeks: the bin size in weeks
        :param plot: whether or not to display a plot of knowledge over time
        :param weighting_keys: the weighting key for knowledge impact calculation
        :return: a list of knowledge impact metrics, one for each bin
        """
        bin_size = timedelta(weeks=bin_size_weeks) if bin_size_weeks is not None else None
        dates = [self.str_to_datetime(self.G.edges[edge]['date']) for edge in nx.get_edge_attributes(self.G, "date")]

        k = []
        bins = int((max(dates) - min(dates)) / bin_size) if bin_size is not None else 1
        # TODO: this is inefficient; create a hashtable and store edges that way,
        #  then generate the network from the hash tables
        for i in range(bins):
            date_min = min(dates)
            date_max = min(dates) + (i + 1) * bin_size if bin_size is not None else max(dates)
            remove = []
            for edge in self.G.edges:
                date = self.str_to_datetime(self.G.edges[edge]['date'])
                if date < date_min or date > date_max:
                    remove.append(edge)
            G_copy = self.G.copy()
            G_copy.remove_edges_from(remove)
            tn = TreeCitationNetwork(
                G_copy,
                self.root,
                weighting_methods=self.weighting_methods,
                k_depth=self.k_depth,
                quality=self.quality,
                h_index=self.h_index,
                custom_centrality=self.custom_centrality,
                knowledge=self.knowledge
            )
            tn.eval_all(weighting_keys=weighting_keys, verbose=False, knowledge=False)
            k.append(tn.k(self.root, self.root, weighting_keys, 0))

        if plot:
            plt.plot(k)
            plt.show()

        return k

    @overrides
    def summary(self):
        custom = ""
        custom += "Connected components: {}\n".format(nx.number_connected_components(self.G.to_undirected()))
        # average metrics
        for key, values in {attribute: self.G.nodes[self.root][attribute] for attribute in
                            self.attributes}.items():
            custom += "{}: {} ({})\n".format(key, round(np.average(values), 3), round(sem(values), 3))
        logger.info("\n== CN Summary ==\n{}\n{}====".format(nx.info(self.G), custom))


class GraphSearchTracker:
    def __init__(self):
        self.visited = []

    def is_visited(self, val):
        if val in self.visited:
            return True
        self.visited.append(val)
        return False

    def reset(self, already_visited=[]):
        self.visited = already_visited
