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
    LOG_PATH = os.path.dirname('logs/{}.log'.format(ENV_NAME))

    COLLECT_BCITES = False

    DOC_LIMIT = None

    USE_CACHED_QUERIES = True

    PROGRESS_DIV = 100

    K_DEPTH = 10


log_dir = Config.LOG_PATH
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

logger = logging.getLogger(__package__)
logger.setLevel(logging.DEBUG)

handler = logging.FileHandler(os.path.join(log_dir,'{}.log'.format(Config.ENV_NAME)))
handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('[%(asctime)s] %(levelname)s> %(message)s')
formatter.datefmt = '%d%b%Y:%H:%M:%S'

handler.setFormatter(formatter)
logger.addHandler(handler)

handler_stdout = logging.StreamHandler(sys.stdout)
handler_stdout.setLevel(logging.DEBUG)
handler_stdout.setFormatter(formatter)
logger.addHandler(handler_stdout)

logger.info("==== NEW SESSION ====")
