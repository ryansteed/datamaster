import os
import logging
import sys
import json


class Config:

    try:
        ENV_NAME = json.load(open("settings.json", "r")).get("ENV_NAME")
    except FileNotFoundError as e:
        print("You're not running from an accepted directory. Using 'unknown' env.")
        ENV_NAME = 'unknown'

    DATA_PATH = os.path.abspath('../../data')
    EXT_DATA_PATH = os.path.abspath('../../data')
    LOG_PATH = os.path.dirname('logs/{}.log'.format(ENV_NAME))

    COLLECT_BCITES = False

    DOC_LIMIT = None

    # whether to use file caches first or make a new query
    USE_CACHED_QUERIES = True

    # how many divisions to use for progress bars
    PROGRESS_DIV = 100

    # How deeply to evaluate k
    K_DEPTH = 10

    # the discount rate
    DISCOUNT = 0.5

    # Whether or not to allow cited patents outside the query to be included in the citation network
    ALLOW_EXTERNAL = False

    FEATURES = [
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


log_dir = Config.LOG_PATH
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

logger = logging.getLogger(__package__)
logger.setLevel(logging.DEBUG)

handler = logging.FileHandler(os.path.join(log_dir,'{}.log'.format(Config.ENV_NAME)))
handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s> %(message)s')
formatter.datefmt = '%d%b%Y:%H:%M:%S'

handler.setFormatter(formatter)
logger.addHandler(handler)

handler_stdout = logging.StreamHandler(sys.stdout)
handler_stdout.setLevel(logging.DEBUG)
handler_stdout.setFormatter(formatter)
logger.addHandler(handler_stdout)
