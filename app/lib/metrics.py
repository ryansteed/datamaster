import networkx as nx
import numpy as np
from scipy.stats import sem
import app.lib.munge as munge
import csv
from app.config import Config, logger

class CitationNetwork:
    attributes = ['forward_cites', 'backward_cites', 'family_size', 'num_claims', 'h_index', 'custom_centrality',
                       'knowledge']

    def __init__(self, G, weighting_method="custom_centrality"):
        self.G = G
        self.weighting_method = weighting_method
        self.evaluate()

    def summary(self, attributes=tuple(attributes)):
        logger.info(nx.info(self.G))
        # average metrics
        metrics = {attribute: list(nx.get_node_attributes(self.G, attribute).values()) for attribute in attributes}
        for key, values in metrics.items():
            print(key+":", round(np.average(values), 3), "(",round(sem(values), 3), ")")

    def print_custom_metrics(self):
        for node in self.G.nodes:
            logger.info(self.G.nodes[node])

    def file_custom_metrics(self, prefix):
        with open(Config.DATA_PATH+'/'+prefix+'.csv', 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(['node']+self.attributes)
            for node in self.G.nodes:
                row = [node]+list(self.G.nodes[node].values())
                writer.writerow(row)

    # Analytics #

    def evaluate(self, weighting_key=None):
        self.eval_weights()
        self.eval_k(self.weighting_method if weighting_key is None else weighting_key)

    def eval_weights(self):
        self.eval_quality()
        self.eval_h()
        self.eval_centrality()

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
            nx.eigenvector_centrality_numpy(self.G),
            nx.closeness_centrality(self.G)
        ]
        ## Local vals
        for centrality in centralities:
            c['+'].append(centrality[max(centrality, key=centrality.get)])
            c['-'].append(centrality[min(centrality, key=centrality.get)])
            c['var'].append(np.var([val for key, val in centrality.items()]))

        ## Centrality metric
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

        nx.set_node_attributes(self.G, custom_centralities, 'custom_centrality')

    def eval_quality(self):
        nx.set_node_attributes(
            self.G,
            {node: int(out_degree * (len(self.G) - 1)) for node, out_degree in nx.out_degree_centrality(self.G).items()},
            'forward_cites'
        )
        nx.set_node_attributes(
            self.G,
            {node: int(in_degree * (len(self.G) - 1)) for node, in_degree in nx.in_degree_centrality(self.G).items()},
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
        for child in self.G.successors(node):
            sum_children += self.k(root, child, weighting_key)
        total_k = (self.G.nodes[node][weighting_key] + sum_children) * self.p(root, node)
        if verbose:
            logger.info('node', node)
            logger.info('> w: ', self.G.nodes[node][weighting_key])
            logger.info('> p: ', self.p(root, node))
            logger.info('> k: ', total_k)
        return total_k

    def p(self, root, node):
        return 1 if node == root else 1 / self.G.in_degree(node)


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
    # cn.print_custom_metrics()
    cn.summary()
    cn.file_custom_metrics(name)

def main():
    # Test network
    G = nx.DiGraph()
    G.add_nodes_from(['a', 'b', 'c', 'd', 'e', 'f', 'i', 'g', 'h'])
    G.add_edges_from([('a', 'f'), ('f', 'i'), ('f', 'h'), ('b', 'g'), ('c', 'g'), ('d', 'g'), ('e', 'h'), ('g', 'h')])
    nx.draw_networkx(G, pos=nx.spring_layout(G))
    test(G, "manual")
    G = munge.test(limit=Config.DOC_LIMIT).get_network()
    test(G, "sample")