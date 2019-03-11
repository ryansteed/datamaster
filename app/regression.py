import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from app.config import logger, Config


def main():
    logger.debug("Running")
    keys = [
        #"engines",
        "radio",
        "robots",
        "transportation",
        "xray",
        "coherent-light"
    ]
    df = pd.concat([
        FeatureExtractor(
            os.path.join(Config.EXT_DATA_PATH, "colonial/FEATURE_both_None_{}_05Mar19.csv".format(key))
        ).fit()
        for key in keys
    ], keys=keys)
    df.head()


class FeatureExtractor:
    max_list = ['inventor_total_num_patents', 'assignee_total_num_patents']
    one_hot = ["assignee_type", "cpc_category",
               "nber_category_id"]
    _types = {
        'knowledge_forward_cites': np.float64,
        'knowledge_h_index': np.float64,
        't': np.int64,
        'patent_date': str,
    }

    def __init__(self, path):
        self.path = path
        self.df = None
        self.load()

    @staticmethod
    def get_types():
        FeatureExtractor._types.update({key: str for key in FeatureExtractor.max_list + FeatureExtractor.one_hot})
        return FeatureExtractor._types

    def load(self):
        self.df = test = pd.read_csv(
            open(self.path, 'rb'),
            sep=",",
            header=0,
            usecols=list(FeatureExtractor.get_types().keys()),
            dtype=FeatureExtractor.get_types()
        )

    def fit(self):
        # transform date column
        self.df['patent_date'] = pd.to_datetime(self.df['patent_date'])
        # create centered date column (as numeric)
        self.df['patent_date_center'] = self.df['patent_date'] - min(self.df['patent_date'])
        # transform max columns
        for key in FeatureExtractor.max_list:
            self.make_list(key)
            self.df["max_{}".format(key)] = self.df[key].apply(lambda x: max(int(xx) for xx in x) if len(x) > 0 else 0)

        # one-hot encode list columns
        for key in FeatureExtractor.one_hot:
            self.make_list(key)
            self.one_hot_encode(key)
        return self.df

    def make_list(self, key):
        self.df[key] = self.df[key].apply(
            lambda x: [
                xx for xx in x.replace("[","").replace("]","").replace("'", "").split(" ") if xx not in ["", "None"]
            ]
        )

    def one_hot_encode(self, key):
        vals = self.df[key].values
        mlb = MultiLabelBinarizer()
        enc = mlb.fit_transform(vals)
        logger.debug(len(mlb.classes_))
        self.df["one-hot_{}".format(key)] = list(enc)


if __name__ == "__main__":
    main()
