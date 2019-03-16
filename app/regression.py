import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from statsmodels.api import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns

from app.config import logger, Config


def main():
    logger.debug("Running")
    keys = [
        "engines_12Mar19",
        "radio_05Mar19",
        "robots_05Mar19",
        "transportation_05Mar19",
        "xray_05Mar19",
        "coherent-light_05Mar19"
    ]
    df = pd.concat([
        FeatureExtractor(
            os.path.join(Config.EXT_DATA_PATH, "colonial/FEATURE_both_None_{}.csv".format(key))
        ).extract()
        for key in keys
    ], keys=keys)
    write(df_pretty_columns(df).describe().to_latex(), "description")
    df = FeatureTransformer(df).fit_transform()

    print(df.head())

    for metric in ["h_index", "forward_cites"]:
        write(regress(df, metric=metric), "all")
        for key in keys:
            print("\n\n## {} ##".format(key))
            write(regress(df.loc[key], metric=metric), "{}_{}".format(key, metric))


def write(content, name):
    with open("data/regression/{}.tex".format(name), 'w') as f:
        f.write(content)
        f.close()


def regress(df, metric="forward_cites"):
    features = ["max_{}".format(key) for key in FeatureExtractor.max_list]
    features += [key for key in df if key.startswith("one-hot")]
    features += ["patent_date_center", 'patent_num_claims', 'patent_processing_time']
    reg = Regressor(df, features=features, target="log(knowledge_{})".format(metric))
    return reg.summary()


def df_pretty_columns(df):
    copy = df.copy().drop("t", 1)
    keys = {
        "knowledge_forward_cites": "Knowledge Forward Cites",
        "knowledge_h_index": "Knowledge H Index",
        "patent_num_claims": "Number of Claims",
        "patent_processing_time": "Processing Time",
        "log(knowledge_h_index)": "Log(Knowledge H Index)",
        "log(knowledge_forward_cites)": "Log(Knowledge Forward Cites)",
        "max_inventor_total_num_patents": "Maximum Inventor Other Patents",
        "max_assignee_total_num_patents": "Maximum Assignee Other Patents",
    }
    keys.update({key: "NBER Category {}".format(key[-1]) for key in copy.columns if "nber_category_id" in key})
    cols = [keys[col] if col in keys else col for col in copy.columns]
    copy.columns = cols
    return copy


class Regressor:
    def __init__(self, data, features, target="log(knowledge_forward_cites)"):
        self.data = data
        self.target = target
        self.features = features

    def score(self):
        lm = LinearRegression()
        x, y = self.make_x_y()
        print(cross_validate(lm, x, y, cv=10, scoring=["neg_mean_squared_error", "explained_variance"]))

    def fit(self):
        lm = LinearRegression()
        x, y = self.make_x_y()
        lm.fit(x, y)
        print('Coefficients: \n', lm.coef_)
        # Explained variance score: 1 is perfect prediction

    def summary(self):
        x, y = self.make_x_y()
        # trim highly correlated vars to avoid multicollinearity
        x = Regressor.calculate_vif_(pd.DataFrame(x), thresh=5.0)
        # check correlation matrix
        Regressor.print_corr(x.corr())
        fit = OLS(y, x).fit()
        print(fit.summary())
        return fit.summary().as_latex()

    def make_x_y(self):
        # x = np.apply_along_axis(Regressor.flatten, 1, np.array(self.data[self.features].values))
        # x = np.nan_to_num(x)
        x = self.data[self.features].fillna(self.data[self.features].mean())
        y = self.data[self.target]
        return x, y

    @staticmethod
    def print_corr(corr):
        corr = corr.abs()
        # s = corr.unstack()
        # so = s.sort_values(ascending=False)
        sol = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool)).stack().sort_values(ascending=False)
        print("# Correlation Pairs #")
        print(sol[:10])

    @staticmethod
    def calculate_vif_(X, thresh=5.0):
        variables = list(range(X.shape[1]))
        dropped = True
        while dropped:
            dropped = False
            vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
                   for ix in range(X.iloc[:, variables].shape[1])]

            maxloc = vif.index(max(vif))
            if max(vif) > thresh:
                print("dropping a variable {}".format(X.columns[variables[maxloc]]))
                del variables[maxloc]
                dropped = True

        print('Remaining variables:')
        print(X.columns[variables])
        return X.iloc[:, variables]

    @staticmethod
    def flatten(x):
        return np.hstack(x)


class FeatureTransformer:
    def __init__(self, df):
        self.df = df

    def fit_transform(self):
        # create logarithm of knowledge columns
        for key in ["knowledge_h_index", "knowledge_forward_cites"]:
            self.make_logarithm(key)
        # transform date column
        self.df['patent_date'] = pd.to_datetime(self.df['patent_date'])
        # create centered date column (as numeric)
        self.df['patent_date_center'] = \
            (self.df['patent_date'] - min(self.df['patent_date'])).dt.total_seconds() / (24 * 60 * 60)
        # transform max columns
        for key in FeatureExtractor.max_list:
            self.make_list(key)
            self.df["max_{}".format(key)] = self.df[key].apply(lambda x: max(int(xx) for xx in x) if len(x) > 0 else 0)
        # one-hot encode list columns
        for key in FeatureExtractor.one_hot:
            self.make_list(key)
            self.one_hot_encode(key)
        return self.df

    def make_logarithm(self, key):
        self.df["log({})".format(key)] = np.log1p(self.df[key])

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
        codes = {c: [] for c in mlb.classes_}
        for code in list(enc):
            for i in range(len(code)):
                codes[mlb.classes_[i]].append(code[i])
        for k, c in codes.items():
            self.df["one-hot_{}_{}".format(key, k)] = c
        return self


class FeatureExtractor:
    max_list = ['inventor_total_num_patents', 'assignee_total_num_patents']
    one_hot = [
        # "assignee_type",
        # "cpc_category",
        "nber_category_id"
    ]
    _types = {
        'knowledge_forward_cites': np.float64,
        'knowledge_h_index': np.float64,
        't': np.float64,
        'patent_date': str,
        'patent_num_claims': np.float64,
        'patent_processing_time': np.float64
    }

    def __init__(self, path):
        self.path = path

    @staticmethod
    def get_types():
        FeatureExtractor._types.update({key: str for key in FeatureExtractor.max_list + FeatureExtractor.one_hot})
        return FeatureExtractor._types

    def extract(self):
        return pd.read_csv(
            open(self.path, 'rb'),
            sep=",",
            header=0,
            na_values="None",
            usecols=list(FeatureExtractor.get_types().keys()),
            dtype=FeatureExtractor.get_types()
        )


if __name__ == "__main__":
    main()
