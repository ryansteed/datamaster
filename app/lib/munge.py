import pandas as pd
import numpy as np
import networkx as nx
import requests
import os
import math
import json
import enlighten
import time

from app.config import Config, logger
from app.lib.helpers import Timer


class Munger:

    query_fields = ["patent_number", "cited_patent_number", "citedby_patent_number"]

    def __init__(self, limit=Config.DOC_LIMIT):
        self.limit = limit
        self.df = None

    def load_data_from_query(self, query, cache=Config.USE_CACHED_QUERIES, per_page=100):
        if cache:
            try:
                self.load_data_from_file(self.make_query_filename(query))
                return self
            except (FileNotFoundError, DataFormatError) as e:
                if isinstance(e, FileNotFoundError):
                    logger.info("Missing query data file, querying USPTO")
                if isinstance(e, DataFormatError):
                    logger.info("Problem loading data file, querying USPTO")

        t = Timer("Querying USPTO: {}".format(query))
        count_patents = self.query_sounding(query)
        count_to_collect = self.limit if self.limit is not None and self.limit < count_patents else count_patents
        pages = math.ceil(count_to_collect / per_page)
        logger.info("Collecting {}/{} docs in {} page{}".format(
            count_to_collect,
            count_patents,
            pages,
            "s" if pages > 0 else ""
        ))

        self.df = pd.DataFrame(columns=self.get_citation_keys())

        manager = enlighten.get_manager()
        ticker = manager.counter(total=pages, desc='Ticks', unit='ticks')
        for i in range(pages):
            if Config.ENV_NAME != "local":
                logger.info("{}/{}".format(i, pages))
            self.df.append(self.query_paginated(query, per_page))
            ticker.update()
        t.log()

        logger.info("Collected {} edges".format(self.df.shape[0]))

        self.ensure_data()

        if Config.USE_CACHED_QUERIES:
            self.write_data_to_file(self.make_query_filename(query))

        return self

    def query_paginated(self, query, per_page):
        r = requests.post(
            'http://www.patentsview.org/api/patents/query',
            json={
                "q": query,
                "f": self.query_fields,
                "o": {
                    "per_page": str(per_page)
                }
            }
        )
        info = r.json()

        data = set()
        for patent in info['patents']:
            for bcite in patent.get('cited_patents'):
                edge = (patent['patent_number'], bcite['cited_patent_number'])
                if None not in edge: data.add(edge)
            for fcite in patent.get('citedby_patents'):
                edge = (fcite['citedby_patent_number'], patent['patent_number'])
                if None not in edge: data.add(edge)
        df = pd.DataFrame(list(data), columns=self.get_citation_keys())

        return df

    def query_sounding(self, query):
        t = Timer("Sounding")
        r = requests.post(
            'http://www.patentsview.org/api/patents/query',
            json={
                "q": query,
                "f": [self.query_fields[0]],
                "o": {
                    "per_page": "1",
                    "include_subentity_total_counts": "true"
                }
            }
        )
        info = r.json()
        t.log()
        return info['total_patent_count'] #, info['total_citedby_patent_count'], info['total_cited_patent_count']

    def write_data_to_file(self, filename):
        t = Timer("Writing data to file {}".format(filename))
        with open(filename, "w+") as file:
            self.df.to_csv(file, index=False, sep='\t')
        t.log()

    @staticmethod
    def make_query_filename(query):
        file_string = json.dumps(query)
        logger.debug(os.getcwd())
        for c in '"{} /':
            file_string = file_string.replace(c, '')
        return "{}.csv".format(os.path.abspath(os.path.join("./query_data", file_string)))

    def load_data_from_file(self, datafile):
        logger.info("Munging data from {}".format(datafile))
        self.df = pd.read_csv(datafile, delimiter='\t', nrows=self.limit) if self.limit is not None else pd.read_csv(datafile, delimiter='\t')
        logger.info("Loaded {} documents from file {}".format(self.df.size, datafile))
        self.ensure_data()
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
        if not {'patent_id', 'citation_id'}.issubset(self.df.columns):
            raise DataFormatError("Missing patent and citation columns in dataset")


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


class DataFormatError(Exception):
    pass


def test_query():
    munger = Munger()
    print(munger.load_data_from_query({"cpc_subgroup_id": "Y10S707/933"}))


def main():
    # Test data
    munger = test()
    G = munger.get_network(metadata=True)
    munger.summary()
    munger.summary_meta()


