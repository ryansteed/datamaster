from app.config import logger
import app.tests

import argparse
import sys

from app.config import Config


class JobHandler:
    def __init__(self, description):
        self.parser = argparse.ArgumentParser(description="{} endpoint".format(description))
        self.parser.add_argument(
            'endpoint',
            type=str,
            help="the name of the endpoint to be called"
        )
        self.test_runnable = None

    def set_test(self, test_fxn):
        self.test_runnable = test_fxn

    def get_args(self):
        args = vars(self.parser.parse_args())
        args.pop('endpoint')
        return args

    def execute(self):
        logger.debug(self.get_args())
        logger.info("==== NEW SESSION ====")
        self.test_runnable(**self.get_args())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Missing endpoint arg.\n USAGE: python main.py [endpoint_string]")

    parser = argparse.ArgumentParser("Datamaster job handler")
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
    job = JobHandler(test)
    logger.info("## Testing {} ##".format(test))

    # set runnables
    if test == "query":
        job.set_test(app.tests.query_test)
        job.parser.add_argument(
            '-g',
            '--write_graph',
            action='store_true',
            help="whether or not to write the network to a graph ml file"
        )
    if test == "root":
        job.set_test(app.tests.root_test_single)
        job.parser.add_argument(
            'patent',
            metavar="patent_number",
            type=str,
            help="number for the root patent"
        )
        job.parser.add_argument(
            'depth',
            type=int,
            help="the graph search depth"
        )
    if test == "root_all":
        job.set_test(app.tests.root_test_multiple)
    if test == "features":
        job.set_test(app.tests.feature_test)

    # add common args
    if test in ("root_all", "features", "query"):
        job.parser.add_argument(
            'query_json_file',
            type=str,
            help="path to a JSON file containing the query to be queried"
        )
        job.parser.add_argument(
            '-l',
            '--limit',
            type=int,
            default=None,
            help="the maximum number of docs to munge"
        )
        job.parser.add_argument(
            '-k',
            '--k_depth',
            type=int,
            default=Config.K_DEPTH,
            help="the maximum number of docs to munge"
        )
        job.parser.add_argument(
            '-d',
            '--discount',
            type=float,
            default=Config.DISCOUNT,
            help="the generational discount rate"
        )
    if "root" in test:
        job.parser.add_argument(
            '-b',
            '--bin_size',
            type=int,
            default=20,
            help="the bin size in weeks"
        )

    job.parser.add_argument(
        '-w',
        '--weighting_keys',
        nargs='+',
        default=("forward_cites", "h_index"),
        help="the weighting keys for knowledge calculation (e.g. 'forward_cites', 'h_index')"
    )
    job.execute()
