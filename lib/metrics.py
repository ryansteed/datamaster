import networkx as nx
import numpy as np

import munge

class CitationNetwork:
    G = None
    weighting_method = None

    def __init__(self, G, weighting_method="custom_centrality"):
        self.G = G
        self.weighting_method = weighting_method
        self.evaluate()

    def print_custom_metrics(self):
        for node in self.G.nodes:
            print(self.G.nodes[node])

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
            {node: int(in_degree * (len(self.G) - 1)) for node, in_degree in nx.out_degree_centrality(self.G).items()},
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
            print('node', node)
            print('> w: ', self.G.nodes[node][weighting_key])
            print('> p: ', self.p(root, node))
            print('> k: ', total_k)
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

def test_manual():
    G = nx.DiGraph()
    G.add_nodes_from(['a', 'b', 'c', 'd', 'e', 'f', 'i', 'g', 'h'])
    G.add_edges_from([('a', 'f'), ('f', 'i'), ('f', 'h'), ('b', 'g'), ('c', 'g'), ('d', 'g'), ('e', 'h'), ('g', 'h')])
    nx.draw_networkx(G, pos=nx.spring_layout(G))

    cn = CitationNetwork(G)
    cn.print_custom_metrics()

def test_sample():
    munger = munge.test(limit=10000)
    cn = CitationNetwork(munger.get_network())
    cn.print_custom_metrics()

if __name__ == "__main__":
    # Test network
    test_manual()
    test_sample()