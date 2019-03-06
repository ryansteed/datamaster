import pandas as pd
import os

from app.config import logger, Config


def main():
    logger.debug("Running")
    keys = ["engines", "radio", "robots", "transportation", "xray", "coherentlight"]
    df = pd.concat([
        pd.read_csv(
            os.path.join(Config.EXT_DATA_PATH, "colonial/FEATURE_both_None_{}_05Mar19.csv".format(key)),
            sep=","
        ) for key in keys
    ], keys=keys)
    df.head()


if __name__ == "__main__":
    main()
