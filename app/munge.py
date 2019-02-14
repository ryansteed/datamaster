import pandas as pd
import numpy as np
import networkx as nx
import requests
import os
import math
import time
import json
import enlighten
from collections import defaultdict
from overrides import overrides

from app.config import Config, logger
from app.helpers import Timer


class Munger:
    """
    Collects patent data into a graph by querying the USPTO.
    """
    query_fields = [
        "patent_number",
        "cited_patent_number",
        "cited_patent_date",
        "citedby_patent_number",
        "citedby_patent_date"
    ]

    def __init__(self, limit, cache):
        """
        Initializes a munger.
        :param limit: the maximum number of patents to process and query
        :param cache: whether or not to use data cached in a csv file or make a fresh query
        """
        self.limit = limit
        self.cache = cache
        self.df = None
        self.df_meta = None
        self.load_data()

    def load_data(self):
        """
        Loads data from query or file to a dataframe.
        :return: the instance
        """
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
        """
        Queries data from the USPTO to dataframe.
        """
        raise NotImplementedError

    def make_filename(self):
        """
        Creates a filename to under which to store the munged data.
        :return: the filename
        """
        raise NotImplementedError

    def query(self, json_query):
        """
        Makes a query to the USPTO using a JSON attributes object.
        :param json_query: the json query according to the PatentsView API.
        :return: the return query in JSON format
        """
        error = None
        for i in range(10):
            try:
                info = self.post_request(json_query)
            except json.JSONDecodeError as e:
                error = e
                time.sleep(10)
                continue
            return info
        raise QueryError("Tried receiving response several times, repeated JSONDecodeError:\n{}".format(error))

    @staticmethod
    def post_request(json_query):
        r = requests.post(
            'http://www.patentsview.org/api/patents/query',
            json=json_query
        )
        if not r.ok:
            try:
                r.raise_for_status()
            except Exception as e:
                raise QueryError("Bad response", e)
        return r.json()

    def write_data_to_file(self, filename):
        """
        Write the data collected to a file
        :param filename: the name of the file, typically the query name
        """
        t = Timer("Writing data to file {}".format(filename))
        with open(filename, "w+") as file:
            self.df.to_csv(file, index=False, sep='\t')
        t.log()

    def query_to_dataframe(self, info, bcites=Config.COLLECT_BCITES):
        """
        Converts the JSON query results from PatentsView to an edge list dataframe.
        :param info: the query json output
        :param bcites: whether or not to include backward citations
        :return: the dataframe containing an edge list wtih the query results
        """
        data = set()
        for patent in info['patents']:
            if bcites:
                for bcite in patent.get('cited_patents'):
                    edge = (patent['patent_number'], bcite['cited_patent_number'], bcite['cited_patent_date'])
                    if None not in edge:
                        data.add(edge)
            for fcite in patent.get('citedby_patents'):
                edge = (fcite['citedby_patent_number'], patent['patent_number'], fcite['citedby_patent_date'])
                if None not in edge:
                    data.add(edge)
        return pd.DataFrame(list(data), columns=self.get_citation_keys()+['date'])

    def load_data_from_file(self, datafile):
        """
        Load data from file for this query (using the unique make_filename function)
        :param datafile: the file to search for
        :return: this instance
        """
        logger.info("Munging data from {}".format(datafile))
        self.df = pd.read_csv(datafile, delimiter='\t', nrows=self.limit) if self.limit is not None \
            else pd.read_csv(datafile, delimiter='\t')
        logger.info("Loaded {} documents from file {}".format(self.df.shape[0], datafile))
        self.ensure_data()
        return self

    def get_network(self, metadata=False, limit=None):
        """
        Constructs a citation network from the edge list.
        :param metadata: whether or not to include metadata
        :param limit: a limit to the number of documents to return
        :return: the NetworkX graph
        """
        logger.info("Generating network from data (metadata={}, limit={})".format(metadata, limit))
        df_edges = self.get_edges()
        if limit is not None:
            df_edges = df_edges.head(limit)

        # for key in self.get_citation_keys():
        #     df_edges[key] = df_edges[key].str.strip()
        G = nx.from_pandas_edgelist(
            df_edges,
            source='citation_id',
            target='patent_id',
            edge_attr="date",
            create_using=nx.DiGraph()
        )

        # logger.debug(np.unique(df_edges['patent_id']).size)
        # logger.debug(np.unique(df_edges['citation_id']).size)
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
        """
        Return the edges from this query, if it has been made; else, load data
        :return: the edge list in a dataframe, including date
        """
        self.ensure_data()
        return self.df[self.get_citation_keys()+['date']].astype(str)

    @staticmethod
    def get_citation_keys():
        return ['patent_id', 'citation_id']

    def load_metadata(self):
        """
        Query for metadata about each patent and add to dataframe
        """
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
        """
        Summarize the edge list
        """
        logger.info(self.df.info())

    def summary_meta(self):
        """
        Summarize the metadata
        """
        self.ensure_meta()
        logger.info(self.df_meta.info())

    def ensure_meta(self):
        """
        Check that metadata has been loaded
        """
        if self.df_meta is None:
            self.load_metadata()

    def ensure_data(self):
        """
        Check that edge list has been minimally loaded
        """
        if self.df is None:
            raise ValueError("Please load data first.")
        if not {'patent_id', 'citation_id'}.issubset(self.df.columns):
            raise DataFormatError("Missing patent and citation columns in dataset")

    @staticmethod
    def get_filename_from_stem(file_string, dir_name):
        return "{}.csv".format(os.path.abspath(os.path.join("./data/{}".format(dir_name), file_string)))


class QueryMunger(Munger):
    """
    A special munger designed to make a specific query to the PatentsView API
    """
    def __init__(self, query_json, limit=Config.DOC_LIMIT, cache=Config.USE_CACHED_QUERIES, per_page=1000):
        """
        Initializes the query munger
        :param query_json: the JSON for the query
        :param limit: the maximum number of documents to munge
        :param cache: whether or not to use cached query data
        :param per_page: the number of patents to request in each individual query
        """
        self.query_json = query_json
        self.per_page = per_page
        super().__init__(limit, cache)

    @overrides
    def query_data(self):
        t = Timer("Querying USPTO: {}".format(self.query_json))
        count_patents = self.query_sounding()
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
            page_df = self.query_paginated(i + 1, self.per_page)
            if self.df is None:
                self.df = page_df
            else:
                self.df = self.df.append(page_df, ignore_index=True)
            ticker.update()
        t.log()

        logger.info("Collected {} edges".format(self.df.shape[0]))

    def query_paginated(self, page, per_page):
        """
        Iteratively queries the PatentsView API (so as not to receive a timeout, and to gather data to file over time)
        :param page: the current page number to query
        :param per_page: the number of patents per query
        :return: a dataframe containing the query page results
        """
        info = self.query({
            "q": self.query_json,
            "f": self.query_fields,
            "o": {
                "page": page,
                "per_page": str(per_page)
            }
        })
        return self.query_to_dataframe(info)

    def query_sounding(self):
        """
        Sends a sounding query to establish the number of documents to scrape
        :return: the number of patents to scrape
        """
        t = Timer("Sounding")
        info = self.query({
            "q": self.query_json,
            "f": [self.query_fields[0]],
            "o": {
                "per_page": "1",
                "include_subentity_total_counts": "true"
            }
        })
        t.log()
        return info['total_patent_count']  # , info['total_citedby_patent_count'], info['total_cited_patent_count']

    @overrides
    def make_filename(self, prefix="QUERY", dirname="query"):
        file_string = json.dumps(self.query_json)
        for c in '"{} /':
            file_string = file_string.replace(c, '')
        return self.get_filename_from_stem("{}_{}_{}".format(prefix, self.limit, file_string), dirname)


class RootMunger(Munger):
    """
    A special munger to fetch the descendants of a given patent number
    """
    def __init__(self, patent_number, depth, limit=Config.DOC_LIMIT, cache=Config.USE_CACHED_QUERIES):
        """
        Initializes the root munger
        :param patent_number: the root patent number
        :param depth: the depth of the search (the number of generations of children)
        :param limit: a document limit
        :param cache: whether to use a cached query in the filesystem
        """
        self.patent_number = patent_number
        self.depth = depth
        self.completed_branches = 0
        t = Timer("Fetching root features")
        features = self.query({
            "q": {"patent_number": self.patent_number},
            "f": [
                "cpc_category",
                "cpc_group_id",
                "assignee_type",
                "assignee_total_num_patents",
                "assignee_id",
                "inventor_id",
                "inventor_total_num_patents",
                "ipc_class",
                "ipc_main_group",
                "nber_category_id",
                "nber_subcategory_id",
                # TODO handle the abstract
                # "patent_abstract",
                "patent_date",
                "patent_num_claims",
                "patent_num_cited_by_us_patents",
                "patent_processing_time",
                "uspc_mainclass_id",
                "uspc_subclass_id",
                "wipo_field_id"
            ]
        }).get('patents')[0]
        # TODO - unnest the return to self.features - should be flat dict
        features_categorical = ["inventors", "assignees", "cpcs", "nbers", "uspcs", "IPCs", "wipos"]
        self.features = {key: val for key, val in features.items() if key not in features_categorical}
        for category in features_categorical:
            unpacked = defaultdict(list)
            for item in features[category]:
                for key, val in item.items():
                    unpacked[key].append(val)
            self.features.update(unpacked)
        t.log()
        super().__init__(limit, cache)

    @overrides
    def make_filename(self, dirname="query"):
        filename = self.get_filename_from_stem("PATENT_{}_{}".format(self.patent_number, self.depth), dirname)
        return filename

    @overrides
    def query_data(self):
        # logger.debug(self.patent_number)
        t = Timer("Fetching children recursively")
        # TODO - also query patent features and include as attributes in network
        self.get_children(self.patent_number, 0)
        logger.debug("Examined {} branches".format(self.completed_branches))
        t.log()

    def get_children(self, curr_num, curr_depth):
        """
        Recursively fetches the children of the current patent up to the maximum depth and add to the edge list
        :param curr_num: the current patent being munged
        :param curr_depth: the current depth away from the root patent number
        """
        # logger.debug("At depth {}/{}".format(curr_depth, self.depth))
        if curr_depth == 1:
            self.completed_branches += 1
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
    """Yields successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


class DataFormatError(Exception):
    pass

class QueryError(Exception):
    pass
