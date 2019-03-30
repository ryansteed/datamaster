from app.config import logger
import app.tests
from app.config import Config

import argparse


# --- Factory class --- #
class JobFactory:
    """
    https://realpython.com/factory-method-python/
    """
    def __init__(self):
        self._subclasses = {}

    def register(self, key, subclass):
        self._subclasses[key] = subclass

    def create(self, key):
        subclass = self._subclasses.get(key)
        if not subclass:
            raise ValueError(key)
        return subclass(key)


# --- Job handling protocol --- #
class JobHandler:
    def __init__(self, description):
        self.description = description
        self.parser = argparse.ArgumentParser(description="{} endpoint".format(description))
        self.parser.add_argument(
            'endpoint',
            type=str,
            help="the name of the endpoint to be called"
        )
        self.test_runnable = None
        self.add_args()
        self.set_test()

    def set_test(self):
        self.test_runnable = self.get_test_fxn()

    def get_args(self):
        args = vars(self.parser.parse_args())
        args.pop('endpoint')
        return args

    def execute(self):
        logger.info("==== NEW TEST: {} ====".format(self.description))
        self.test_runnable(**self.get_args())

    def add_args(self):
        raise NotImplementedError

    def get_test_fxn(self):
        raise NotImplementedError


# --- Base handlers --- #
class KnowledgeHandler(JobHandler):
    def add_args(self):
        self.parser.add_argument(
            '-w',
            '--weighting_keys',
            nargs='+',
            default=("forward_cites", "h_index"),
            help="the weighting keys for knowledge calculation (e.g. 'forward_cites', 'h_index')"
        )

    def get_test_fxn(self):
        raise NotImplementedError


class RecursiveKnowledgeHandler(KnowledgeHandler):
    def add_args(self):
        super().add_args()
        self.parser.add_argument(
            'query_json_file',
            type=str,
            help="path to a JSON file containing the query to be queried"
        )
        self.parser.add_argument(
            '-l',
            '--limit',
            type=int,
            default=None,
            help="the maximum number of docs to munge"
        )
        self.parser.add_argument(
            '-k',
            '--k_depth',
            type=int,
            default=Config.K_DEPTH,
            help="the maximum number of docs to munge"
        )
        self.parser.add_argument(
            '-d',
            '--discount',
            type=float,
            default=Config.DISCOUNT,
            help="the generational discount rate"
        )

    def get_test_fxn(self):
        raise NotImplementedError


class RootHandler(KnowledgeHandler):
    def add_args(self):
        super().add_args()
        self.parser.add_argument(
            '-b',
            '--bin_size',
            type=int,
            default=20,
            help="the bin size in weeks"
        )

    def get_test_fxn(self):
        raise NotImplementedError


# --- Endpoint handlers --- #
class QueryHandler(RecursiveKnowledgeHandler):
    def add_args(self):
        super().add_args()
        self.parser.add_argument(
            '-g',
            '--write_graph',
            action='store_true',
            help="whether or not to write the network to a graph ml file"
        )

    def get_test_fxn(self):
        return app.tests.query_test


class FeaturesHandler(RecursiveKnowledgeHandler):
    def get_test_fxn(self):
        return app.tests.feature_test


class RootSingleHandler(RootHandler):
    def add_args(self):
        super().add_args()
        self.parser.add_argument(
            'patent',
            metavar="patent_number",
            type=str,
            help="number for the root patent"
        )
        self.parser.add_argument(
            'depth',
            type=int,
            help="the graph search depth"
        )

    def get_test_fxn(self):
        return app.tests.root_test_single


class RootAllHandler(RecursiveKnowledgeHandler, RootHandler):
    def get_test_fxn(self):
        return app.tests.root_test_multiple


class RegressionHandler(JobHandler):
    def add_args(self):
        pass

    def get_test_fxn(self):
        return app.tests.regression


class ForecastingHandler(JobHandler):
    def add_args(self):
        self.parser.add_argument(
            'forecast_type',
            type=str,
            help="which forecast type to use - either 'arima' for ARIMA or 'pooled' for Pooled OLS"
        )
        self.parser.add_argument(
            '-r',
            '--relative_series',
            action='store_true',
            help="whether or not to use a time-relative series"
        )

    def get_test_fxn(self):
        return app.tests.forecasting


# --- Main driver --- #
if __name__ == "__main__":
    # Factory for endpoint handling
    factory = JobFactory()

    # Endpoint registration
    factory.register("query", QueryHandler)
    factory.register("features", FeaturesHandler)
    factory.register("root", RootSingleHandler)
    factory.register("root_all", RootAllHandler)
    factory.register("regression", RegressionHandler)
    factory.register("forecasting", ForecastingHandler)

    # Handling command line input
    parser = argparse.ArgumentParser("DataMaster job handler")
    parser.add_argument(
        'endpoint',
        type=str,
        help="the name of the endpoint to be called"
    )
    parser.add_argument(
        'args',
        nargs=argparse.REMAINDER,
        help="endpoint-specific arguments"
    )
    test = vars(parser.parse_args()).get('endpoint')
    # Job creation
    job = factory.create(test)
    # Job execution
    job.execute()

