import pandas as pd
import numpy as np
import networkx as nx
import requests

from app.config import Config, logger
from app.lib.helpers import Timer


class Munger:

    def __init__(self, limit=Config.DOC_LIMIT):
        self.limit = limit
        self.df = None

    def load_data_from_query(self, query):
        logger.info("Munging data from query {}".format(query))
        t = Timer("Querying")
        r = requests.post(
            'http://www.patentsview.org/api/patents/query',
            json={
                "q": query,
                "f": ["patent_number", "cited_patent_number", "citedby_patent_number"]
            }
        )
        info = r.json()
        t.log()

        t.reset(name="Parsing to dataframe")
        data = set()
        for patent in info['patents']:
            logger.debug(patent)
            for bcite in patent.get('cited_patents'):
                edge = (patent['patent_number'], bcite['cited_patent_number'])
                if None not in edge: data.add(edge)
            for fcite in patent.get('citedby_patents'):
                edge = (fcite['citedby_patent_number'], patent['patent_number'])
                if None not in edge: data.add(edge)
        df = pd.DataFrame(list(data), columns=self.get_citation_keys())

        self.df = df
        t.log()

        logger.info("Collected {} documents with query {}".format(df.size, query))
        return self

    def load_data_from_file(self, datafile):
        logger.info("Munging data from {}".format(datafile))
        if self.limit is not None:
            self.df = pd.read_csv(datafile, delimiter='\t', nrows=self.limit)
            return
        self.df = pd.read_csv(datafile, delimiter='\t')
        logger.info("Loaded {} documents from dataframe {}".format(self.df.size, datafile))
        return self

    def get_network(self, metadata=False, limit=None):
        logger.info("Generating network from data (metadata={}, limit={}".format(metadata, limit))
        df_edges = self.get_edges()
        if limit is not None:
            df_edges = df_edges.head(limit)

        # for key in self.get_citation_keys():
        #     df_edges[key] = df_edges[key].str.strip()

        G = nx.from_pandas_edgelist(df_edges, source='patent_id', target='citation_id', create_using=nx.DiGraph())

        if metadata:
            self.ensure_meta()
            for entry in self.df_meta.to_dict(orient='records'):
                try:
                    G.nodes[entry['patent_number']].update(
                        {key: val for key, val in entry.items() if key != 'patent_number'})
                except KeyError:
                    logger.error("Couldn't find network entry for {}".format(entry['patent_number']))

        logger.info("Generated network")
        return G

    def get_edges(self):
        self.ensure_data()
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

    def ensure_data(self):
        if self.df is None:
            raise ValueError("Please load data first.")


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def test(limit=Config.DOC_LIMIT):
    munger = Munger(limit=limit)

    # from uspto file
    # munger.load_data_from_file(Config.DATA_PATH+'/uspatentcitation.tsv')

    # test from https://ropensci.github.io/patentsview/articles/citation-networks.html
    # munger.load_data_from_query({"cpc_subgroup_id": "Y10S707/933"})

    # test from https://link.springer.com/article/10.1007/s11192-017-2252-y
    munger.load_data_from_query({"uspc_mainclass_id": "372"})

    # artificial intelligence
    # munger.load_data_from_query({"uspc_mainclass_id": "706"})

    return munger


def test_query():
    munger = Munger()
    print(munger.load_data_from_query({"cpc_subgroup_id": "Y10S707/933"}))


def main():
    # Test data
    munger = test()
    G = munger.get_network(metadata=True)
    munger.summary()
    munger.summary_meta()


