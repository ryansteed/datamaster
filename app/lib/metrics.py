import networkx as nx
import numpy as np
from scipy.stats import sem
import app.lib.munge as munge
import csv
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import pandas as pd
from overrides import overrides

from app.config import Config, logger
from app.lib.helpers import Timer
from app.lib.munge import RootMunger


class CitationNetwork:

    def __init__(
            self, G, weighting_method="forward_cites",
            quality=True, h_index=True, custom_centrality=True, knowledge=True
            ):
        self.G = G
        self.weighting_method = weighting_method
        self.quality = quality
        self.h_index = h_index
        self.custom_centrality = custom_centrality
        self.knowledge = knowledge
        self.attributes = []
        if quality:
            self.attributes += ['forward_cites', 'backward_cites', 'family_size', 'num_claims']
        if h_index:
            self.attributes.append("h_index")
        if custom_centrality:
            self.attributes.append("custom_centrality")
        if knowledge:
            self.attributes.append("knowledge")

    def summary(self):
        custom = ""
        custom += "Connected components: {}\n".format(nx.number_connected_components(self.G.to_undirected()))
        # average metrics
        for key, values in {attribute: list(nx.get_node_attributes(self.G, attribute).values()) for attribute in self.attributes}.items():
            custom += "{}: {} ({})\n".format(key, round(np.average(values), 3), round(sem(values), 3))
        logger.info("\n== CN Summary ==\n{}\n{}====".format(nx.info(self.G), custom))

    def draw(self):
        nx.draw_networkx(self.G, pos=nx.kamada_kawai_layout(self.G))
        plt.show()

    def write_graphml(self, name):
        nx.write_graphml(self.G, "{}.graphml".format(os.path.abspath(os.path.join("./data/graph", name))))

    def print_custom_metrics(self):
        logger.info("== Calculated Metrics ==")
        for node in self.G.nodes:
            logger.info(self.G.nodes[node])
        logger.info("====")

    def file_custom_metrics(self, prefix):
        logger.info("Filing calculated metrics in {}.csv".format(prefix))
        t = Timer("Filing metrics")
        with open(Config.DATA_PATH+'/'+prefix+'.csv', 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(['node']+self.attributes)
            for node in self.G.nodes:
                row = [node]+list(self.G.nodes[node].values())
                writer.writerow(row)
        t.log()

    # Analytics #
    def eval_all(self, weighting_key=None, verbose=True):
        if verbose:
            logger.info("Calculating metrics")
            t = Timer("Metric calculation")
        if self.quality:
            if verbose:
                logger.info("Calculating quality")
            self.eval_quality()
            if verbose:
                t.log()
        if self.h_index:
            if verbose:
                logger.info("Calculating H-index")
            self.eval_h()
            if verbose:
                t.log()
        if self.custom_centrality:
            if verbose:
                logger.info("Calculating centralities")
            self.eval_centrality()
            if verbose:
                t.log()
        if self.knowledge:
            if verbose:
                logger.info("Calculating knowledge")
            self.eval_k(self.weighting_method if weighting_key is None else weighting_key)
            if verbose:
                t.log()

    @staticmethod
    def str_to_datetime(date):
        return datetime.strptime(date, '%Y-%M-%d')

    def eval_h(self):
        h_indices = {}
        for node in self.G:
            forward_cites = [self.G.nodes[child]['forward_cites'] for child in self.G.successors(node)]
            h_indices[node] = h_index(forward_cites)
        nx.set_node_attributes(self.G, h_indices, 'h_index')

    def eval_centrality(self):
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
        nx.set_node_attributes(
            self.G,
            {node: int(out_degree * (len(self.G) - 1)) for node, out_degree in self.G.out_degree()},
            'forward_cites'
        )
        nx.set_node_attributes(
            self.G,
            {node: int(in_degree * (len(self.G) - 1)) for node, in_degree in self.G.in_degree()},
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

    def eval_k(self, weighting_key):
        nx.set_node_attributes(
            self.G,
            {node: self.k(node, node, weighting_key) for node in self.G.nodes},
            'knowledge'
        )

    def k(self, root, node, weighting_key, verbose=False):
        sum_children = 0
        for child in [x for x in self.G.successors(node) if x is not None]:
            sum_children += self.k(root, child, weighting_key)
        total_k = (self.G.nodes[node][weighting_key] + sum_children) * self.p(root, node)
        if verbose:
            logger.info('node', node)
            logger.info('> w: ', self.G.nodes[node][weighting_key])
            logger.info('> p: ', self.p(root, node))
            logger.info('> k: ', total_k)
        return total_k

    def p(self, root, node):
        return 1 if node == root else 1 / int(self.G.in_degree(node))

    def root_analysis(self, depth, filename, limit=Config.DOC_LIMIT):
        df = None
        patents = [i for i in self.G.nodes][:limit+1]
        for patent in patents:
            munger = RootMunger(patent, depth=depth, limit=Config.DOC_LIMIT)
            # eval_and_sum(munger)
            cn = TreeCitationNetwork(munger.get_network(), patent)
            if not cn.is_empty():
                data = cn.eval_binned(20, plot=False)
                if df is None:
                    df = pd.DataFrame(
                        data=[[x, i]+list(munger.features.values()) for i,x in enumerate(data)],
                        columns=["t", "k"]+munger.features.keys()
                    )
                else:
                    df_new = pd.DataFrame(
                        data=list(enumerate(data) + munger.features.values()),
                        columns=["k", "t"]+list(munger.features.keys())
                    )
                    df = df.append(df_new, ignore_index=True)
                t = Timer("Writing data to file {}".format(filename))
                with open(filename, "w+") as file:
                    df.to_csv(file, index=False, sep='\t')
                t.log()


class TreeCitationNetwork(CitationNetwork):
    def __init__(
            self, G, root, weighting_method="forward_cites",
            quality=True, h_index=True, custom_centrality=True, knowledge=True
    ):
        super().__init__(
            G,
            weighting_method=weighting_method,
            quality=quality,
            h_index=h_index,
            custom_centrality=custom_centrality,
            knowledge=knowledge
        )
        self.root = root

    def is_empty(self):
        return self.G.size() == 0

    def eval_binned(self, bin_size_weeks, plot=False, weighting_key=None):
        bin_size = timedelta(weeks=bin_size_weeks)
        full_G = self.G.copy()
        logger.debug(self.str_to_datetime('2015-01-01'))
        dates = [self.str_to_datetime(self.G.edges[edge]['date']) for edge in nx.get_edge_attributes(full_G, "date")]
        logger.debug("Range: {}".format(max(dates) - min(dates)))

        k = []
        bins = int((max(dates) - min(dates)) / bin_size)
        # TODO: this is inefficient; create a hashtable and store edges that way, then generate the network from the hash tables
        for i in range(bins):
            date_min = min(dates)
            date_max = min(dates) + (i + 1) * bin_size
            # logger.debug("{} to {}".format(date_min, date_max))
            remove = []
            for edge in self.G.edges:
                date = self.str_to_datetime(self.G.edges[edge]['date'])
                if date < date_min or date > date_max:
                    remove.append(edge)
            self.G.remove_edges_from(remove)
            # TODO: only evaluate root node, rather than all nodes
            self.eval_all(weighting_key=weighting_key, verbose=False)
            k.append(self.G.nodes[self.root]["knowledge"])
            logger.debug("Bin {}/{}".format(i+1, bins))
            self.G = full_G.copy()

        logger.debug(k)
        if plot:
            plt.plot(k)
            plt.show()

        return k

    @overrides
    def eval_k(self, weighting_key):
        nx.set_node_attributes(
            self.G,
            {self.root: self.k(self.root, self.root, weighting_key)},
            'knowledge'
        )

    @overrides
    def summary(self):
        custom = ""
        custom += "Connected components: {}\n".format(nx.number_connected_components(self.G.to_undirected()))
        # average metrics
        for key, values in {attribute: self.G.nodes[self.root][attribute] for attribute in
                            self.attributes}.items():
            custom += "{}: {} ({})\n".format(key, round(np.average(values), 3), round(sem(values), 3))
        logger.info("\n== CN Summary ==\n{}\n{}====".format(nx.info(self.G), custom))


# h-index calculation
def h_index(m):
    s = [0]*(len(m)+1)
    for i in range(len(m)):
        s[min([len(m), m[i]])] += 1
    x = 0
    for i in reversed(range(len(s))):
        x += s[i]
        if x >= i:
            return i
    return 0


def test(G, name="test"):
    cn = CitationNetwork(G)
    cn.eval_all()
    cn.summary()
    cn.file_custom_metrics(name)


def main():
    # Test network
    G = nx.DiGraph()
    G.add_nodes_from(['a', 'b', 'c', 'd', 'e', 'f', 'i', 'g', 'h'])
    G.add_edges_from([('a', 'f'), ('f', 'i'), ('f', 'h'), ('b', 'g'), ('c', 'g'), ('d', 'g'), ('e', 'h'), ('g', 'h')])
    # nx.draw_networkx(G, pos=nx.spring_layout(G))
    # test(G, "manual")
    G = munge.test(limit=Config.DOC_LIMIT).get_network()
    test(G, "sample")
