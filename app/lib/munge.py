import pandas as pd
import numpy as np
import networkx as nx
import requests
from app.config import Config, logger


class Munger:

    def __init__(self, datafile, limit=None):
        self.datafile = datafile
        self.limit = limit
        self.load_data(datafile)

    def load_data(self, datafile):
        if self.limit is not None:
            self.df = pd.read_csv(datafile, delimiter='\t', nrows=self.limit)
            return
        self.df = pd.read_csv(datafile, delimiter='\t')

    def get_network(self, metadata=False, limit=None):
        df_edges = self.get_edges()
        if limit is not None:
            df_edges = df_edges.head(limit)
        for key in self.get_citation_keys():
            df_edges[key] = df_edges[key].str.strip()

        G = nx.from_pandas_edgelist(df_edges, source='patent_id', target='citation_id', create_using=nx.DiGraph())

        if metadata:
            self.ensure_meta()
            for entry in self.df_meta.to_dict(orient='records'):
                try:
                    G.nodes[entry['patent_number']].update(
                        {key: val for key, val in entry.items() if key != 'patent_number'})
                except KeyError:
                    logger.error("Couldn't find network entry for {}".format(entry['patent_number']))

        return G

    def get_edges(self):
        return self.df[self.get_citation_keys()]

    @staticmethod
    def get_citation_keys():
        return ['patent_id', 'citation_id']

    def load_metadata(self):
        self.df_meta = None

        nodes = np.unique(pd.concat([self.df[key] for key in self.get_citation_keys()]))

        for i, chunk in enumerate(chunks(nodes, 25)):
            r = requests.post(
                'http://www.patentsview.org/api/patents/query',
                json={
                    "q": {"patent_number": [str(num) for num in chunk]},
                    "f": ["patent_number", "patent_title", "assignee_id", "cpc_category", "nber_category_title"]
                }
            )
            info = r.json()
            if i == 0:
                self.df_meta = pd.DataFrame(info['patents'])
                continue
            self.df_meta = self.df_meta.append(pd.DataFrame(info['patents']), ignore_index=True, verify_integrity=True)

    def summary(self):
        logger.info(self.df.info())

    def summary_meta(self):
        self.ensure_meta()
        logger.info(self.df_meta.info())

    def ensure_meta(self):
        if self.df_meta is None:
            self.load_metadata()


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def test(limit=Config.DOC_LIMIT):
    return Munger(Config.DATA_PATH+'/uspatentcitation.tsv', limit=limit)


def main():
    # Test data
    munger = test()
    G = munger.get_network(metadata=True)
    logger.debug(G.nodes.data())
    munger.summary()
    munger.summary_meta()


