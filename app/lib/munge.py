import pandas as pd
import numpy as np
import networkx as nx
import requests
import os
import math
import json
import enlighten
from overrides import overrides

from app.config import Config, logger
from app.lib.helpers import Timer


class Munger:

    query_fields = ["patent_number", "cited_patent_number", "cited_patent_date", "citedby_patent_number", "citedby_patent_date"]

    def __init__(self, limit, cache):
        self.limit = limit
        self.cache = cache
        self.df = None
        self.df_meta = None
        self.load_data()

    def load_data(self):
        if self.cache:
            try:
                self.load_data_from_file(self.make_filename())
                return self
            except (FileNotFoundError, DataFormatError) as e:
                if isinstance(e, FileNotFoundError):
                    logger.info("Missing data file, querying USPTO")
                if isinstance(e, DataFormatError):
                    logger.info("Problem loading data file, querying USPTO")
        self.query_data()
        self.ensure_data()
        if Config.USE_CACHED_QUERIES:
            self.write_data_to_file(self.make_filename())
        return self

    def query_data(self):
        raise NotImplementedError

    def make_filename(self):
        raise NotImplementedError

    @staticmethod
    def query(json_query):
        r = requests.post(
            'http://www.patentsview.org/api/patents/query',
            json=json_query
        )
        return r.json()

    def write_data_to_file(self, filename):
        t = Timer("Writing data to file {}".format(filename))
        with open(filename, "w+") as file:
            self.df.to_csv(file, index=False, sep='\t')
        t.log()

    def query_to_dataframe(self, info, bcites=True):
        data = set()
        for patent in info['patents']:
            if bcites:
                for bcite in patent.get('cited_patents'):
                    edge = (patent['patent_number'], bcite['cited_patent_number'], bcite['cited_patent_date'])
                    if None not in edge: data.add(edge)
            for fcite in patent.get('citedby_patents'):
                edge = (fcite['citedby_patent_number'], patent['patent_number'], fcite['citedby_patent_date'])
                if None not in edge: data.add(edge)
        return pd.DataFrame(list(data), columns=self.get_citation_keys()+['date'])

    def load_data_from_file(self, datafile):
        logger.info("Munging data from {}".format(datafile))
        self.df = pd.read_csv(datafile, delimiter='\t', nrows=self.limit) if self.limit is not None \
            else pd.read_csv(datafile, delimiter='\t')
        logger.info("Loaded {} documents from file {}".format(self.df.shape[0], datafile))
        self.ensure_data()
        return self

    def get_network(self, metadata=False, limit=None):
        logger.info("Generating network from data (metadata={}, limit={})".format(metadata, limit))
        df_edges = self.get_edges()
        if limit is not None:
            df_edges = df_edges.head(limit)

        # for key in self.get_citation_keys():
        #     df_edges[key] = df_edges[key].str.strip()
        G = nx.from_pandas_edgelist(df_edges, source='patent_id', target='citation_id', edge_attr="date", create_using=nx.DiGraph())

        logger.debug(np.unique(df_edges['patent_id']).size)
        logger.debug(np.unique(df_edges['citation_id']).size)
        if metadata:
            self.ensure_meta()
            for entry in self.df_meta.to_dict(orient='records'):
                try:
                    G.nodes[entry['patent_number']].update(
                        {key: val for key, val in entry.items() if key != 'patent_number'})
                except KeyError:
                    logger.error("Couldn't find network entry for {}".format(entry['patent_number']))

        logger.info("Generated network with {} nodes and {} edges".format(len(G.nodes), len(G.edges)))
        return G

    def get_edges(self):
        self.ensure_data()
        return self.df[self.get_citation_keys()+['date']]

    @staticmethod
    def get_citation_keys():
        return ['patent_id', 'citation_id']

    def load_metadata(self):
        self.df_meta = None

        nodes = np.unique(pd.concat([self.df[key] for key in self.get_citation_keys()]))

        for i, chunk in enumerate(chunks(nodes, 25)):
            info = self.query({
                "q": {"patent_number": [str(num) for num in chunk]},
                "f": ["patent_number", "patent_title", "assignee_id", "cpc_category", "nber_category_title"]
            })
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

    @staticmethod
    def get_filename_from_stem(file_string):
        return "{}.csv".format(os.path.abspath(os.path.join("./data/query", file_string)))


class QueryMunger(Munger):
    def __init__(self, query_json, limit=Config.DOC_LIMIT, cache=Config.USE_CACHED_QUERIES, per_page=100):
        self.query_json = query_json
        self.per_page = per_page
        super().__init__(limit, cache)

    @overrides
    def query_data(self):
        t = Timer("Querying USPTO: {}".format(self.query_json))
        count_patents = self.query_sounding(self.query_json)
        count_to_collect = self.limit if self.limit is not None and self.limit < count_patents else count_patents
        pages = math.ceil(count_to_collect / self.per_page)
        logger.info("Collecting {}/{} docs in {} page{}".format(
            count_to_collect,
            count_patents,
            pages,
            "s" if pages > 0 else ""
        ))

        manager = enlighten.get_manager()
        ticker = manager.counter(total=pages, desc='Ticks', unit='ticks')
        for i in range(pages):
            if Config.ENV_NAME != "local":
                logger.info("{}/{}".format(i, pages))
            page_df = self.query_paginated(self.query, i + 1, self.per_page)
            if self.df is None:
                self.df = page_df
            else:
                self.df = self.df.append(page_df, ignore_index=True)
            ticker.update()
        t.log()

        logger.info("Collected {} edges".format(self.df.shape[0]))

    def query_paginated(self, query, page, per_page):
        info = query({
            "q": query,
            "f": self.query_fields,
            "o": {
                "page": page,
                "per_page": str(per_page)
            }
        })
        return self.query_to_dataframe(info)

    def query_sounding(self, query_json):
        t = Timer("Sounding")
        info = self.query({
            "q": query_json,
            "f": [self.query_fields[0]],
            "o": {
                "per_page": "1",
                "include_subentity_total_counts": "true"
            }
        })
        t.log()
        return info['total_patent_count'] #, info['total_citedby_patent_count'], info['total_cited_patent_count']

    @overrides
    def make_filename(self):
        file_string = json.dumps(self.query)
        for c in '"{} /':
            file_string = file_string.replace(c, '')
        return self.get_filename_from_stem("QUERY_{}".format(file_string))


class RootMunger(Munger):

    def __init__(self, patent_number, depth, limit=Config.DOC_LIMIT, cache=Config.USE_CACHED_QUERIES):
        self.patent_number = patent_number
        self.depth = depth
        self.completed_branches = 0
        super().__init__(limit, cache)

    @overrides
    def make_filename(self):
        filename = self.get_filename_from_stem("PATENT_{}_{}".format(self.patent_number, self.depth))
        return filename

    @overrides
    def query_data(self):
        logger.debug(self.patent_number)
        t = Timer("Fetching children recursively")
        self.get_children(self.patent_number, 0)
        logger.debug("Examined {} branches".format(self.completed_branches))
        t.log()

    def get_children(self, curr_num, curr_depth):
        # logger.debug("At depth {}/{}".format(curr_depth, self.depth))
        if curr_depth == 1:
            self.completed_branches += 1
            logger.info("Branch {}".format(self.completed_branches))
        info = self.query({
            "q": {"patent_number": curr_num},
            "f": self.query_fields
        })
        if curr_depth == 0:
            logger.debug("Exploring {} branches".format(len(info['patents'][0]['citedby_patents'])))
        if info.get('patents') is not None:
            # TODO: include bcites, and recurse once more to get bcites for the leaves but not fcites
            df = self.query_to_dataframe(info, bcites=False)
            if self.df is None:
                self.df = df
            else:
                self.df = self.df.append(df, ignore_index=True)
            # iterate through all children, recursively
            if curr_depth+1 < self.depth:
                for patent in info['patents']:
                    for fcite in patent.get('citedby_patents'):
                        self.get_children(fcite['citedby_patent_number'], curr_depth+1)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


class DataFormatError(Exception):
    pass


# TESTS

def test(limit=Config.DOC_LIMIT):
    munger = QueryMunger(limit=limit)

    # from uspto file
    # munger.load_data_from_file(Config.DATA_PATH+'/uspatentcitation.tsv')

    # test from https://ropensci.github.io/patentsview/articles/citation-networks.html
    # munger.load_data_from_query({"cpc_subgroup_id": "Y10S707/933"})

    # test from https://link.springer.com/article/10.1007/s11192-017-2252-y
    munger.load_data({"uspc_mainclass_id": "372"})

    # artificial intelligence
    # munger.load_data_from_query({"uspc_mainclass_id": "706"})

    return munger

def test_query():
    munger = QueryMunger()
    print(munger.load_data({"cpc_subgroup_id": "Y10S707/933"}))


def main():
    # Test data
    munger = test()
    G = munger.get_network(metadata=True)
    munger.summary()
    munger.summary_meta()


