import pandas as pd
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from statsmodels.api import OLS
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pickle
import enlighten

from app.config import logger, Config


def test_regression():
    keys = [
        "engines_12Mar19",
        "radio_05Mar19",
        "robots_05Mar19",
        "transportation_05Mar19",
        "xray_05Mar19",
        "coherent-light_05Mar19"
    ]
    df = get_stacked_df(keys)
    write(df_pretty_columns(df.drop("t", 1)).describe().to_latex(), "description")
    df = FeatureTransformer(df).fit_transform()

    for metric in ["forward_cites", "h_index"]:
        write(regress(df, metric=metric, exclude_nber=False), "all_{}".format(metric))
        for key in keys:
            print("\n\n## {} ##".format(key))
            write(regress(df.loc[key], metric=metric, exclude_nber=True), "{}_{}".format(key, metric))


def test_forecasting(bin_size=20, n=50):
    cache_name = os.path.abspath("data/regression/forecasting_cache.pkl")

    keys = [
        "coherent-light_19Mar19"
    ]
    bin_size_weeks = np.timedelta64(bin_size, 'W')

    try:
        df = pickle.load(open(cache_name, 'rb'))
    except FileNotFoundError:
        df = FeatureTransformer(get_stacked_df(keys, endpoint="TIME-DATA")).fit_transform()
        pickle.dump(df, open(cache_name, 'wb'))

    df_endog = df[["log(knowledge_forward_cites)", "t", "patent_date"]]
    features, protected = get_features(True, df)
    df_exog = df[features]
    logger.debug(df_endog.describe())
    logger.debug(df_endog.values.shape)

    # regress_varmax(df_endog, bin_size_weeks, n)

    regress_arima(df_endog, bin_size_weeks)


def regress_arima(df_endog, bin_size_weeks):
    cache_name = 'data/regression/arima.pkl'

    # average columns along "i" index
    start_date = df_endog["patent_date"].min()
    df_endog["t"] = (df_endog["t"] + ((df_endog["patent_date"] - start_date) / bin_size_weeks).astype(int))

    # try:
    #     df_endog, bin_size_stored = pickle.load(open(cache_name, 'rb'))
    #     if bin_size_weeks != bin_size_stored:
    #         raise FileNotFoundError
    # except FileNotFoundError:
    data = []
    ind = []
    mask = (df_endog["t"].shift(-1) == 0)
    for row in df_endog[mask][["log(knowledge_forward_cites)", "t", "patent_date"]].itertuples():
        index, k, t, date = row
        logger.debug(int(df_endog["t"].max()) - int(t))
        for i in range(int(df_endog["t"].max()) - int(t)):
            data.append((k, t + 1 + i, date))
            ind.append(index)
    to_add = pd.DataFrame(data, index=ind, columns=["log(knowledge_forward_cites)", "t", "patent_date"])
    df_endog = df_endog.append(to_add)
    # pickle.dump((df_endog, bin_size_weeks), open(cache_name, 'wb'))
    logger.debug(to_add)
    logger.debug(df_endog)

    # now convert back to date
    df_endog["t"] = df_endog["t"] * bin_size_weeks + start_date
    logger.debug(df_endog)
    df_endog = df_endog.groupby("t").mean()
    logger.debug(df_endog)

    # fig1 = df_endog.plot()
    # fig2 = autocorrelation_plot(df_endog)
    # result = seasonal_decompose(df_endog, model="linear")
    # fig = result.plot()
    # plt.show()

    aia_date = np.datetime64("2013-03-16")
    train = df_endog.loc[df_endog.index < aia_date]
    test = df_endog.loc[df_endog.index >= aia_date]

    model = ARIMA(train["log(knowledge_forward_cites)"], order=(2, 1, 0))
    fit = model.fit(maxiter=1000000, disp=True, transparams=True, trend='c')
    logger.debug(fit.summary())

    # residuals = pd.DataFrame(fit.resid)
    # autocorrelation_plot(residuals)
    # plt.show()
    # residuals.plot(kind='kde')
    # plt.show()
    # print(residuals.describe())

    # adapted from
    # http://www.statsmodels.org/devel/_modules/statsmodels/tsa/arima_model.html#ARIMAResults.plot_predict
    test["actual"] = test["log(knowledge_forward_cites)"]
    ax = test["actual"].plot()
    fit.plot_predict(start=train.index[-10], end=test.index[-1], ax=ax)
    plt.show()


def regress_varmax(df_endog, bin_size_weeks, n):
    """
    Trains a varmax model on time series for each patent up to n steps,
    working forwards from the publication date or working backwards from the current date. Also includes exogenous
    patent features.

    :param df_endog: the multiple endogenous time series, not yet transformed
    :param bin_size_weeks: the bin size in weeks
    :type bin_size_weeks: pd.Timedelta
    :param n: the number of steps required in each patent series - must make a square matrix!
    :return: None
    """
    df_endog = transform_endog(df_endog, bin_size_weeks, n, ascending=True)

    # remove columns with low variance
    order = 4
    df_endog = df_endog.loc[:, df_endog.apply(pd.Series.nunique, axis=0) > order]
    logger.debug(df_endog)
    logger.debug(df_endog.describe())

    logger.debug("Training VARMAX...")
    model = VARMAX(df_endog.values, order=(order, 0))
    res = model.fit(maxiter=1000, disp=True)
    logger.debug(res.summary())


def transform_endog(df_endog, bin_size_weeks, n, ascending=True):
    """

    :param df_endog:
    :param bin_size_weeks:
    :param n:
    :param past: whether or not the data is structured from inception up to n steps, or from current back n steps
    :return:
    """
    cache_name = os.path.abspath("data/regression/forecasting_cache2.pkl")

    try:
        df, bin_size_stored, n_stored, ascending_stored = pickle.load(open(cache_name, 'rb'))
        if bin_size_weeks != bin_size_stored or n != n_stored or ascending_stored != ascending:
            raise FileNotFoundError

    except (FileNotFoundError, ValueError):

        # adding a multindex that separates each patent - TODO collect patent number and group by that
        c = Counter()
        df_endog["i"] = df_endog.t.apply(lambda x: c.inc() if x == 0 else c.get())
        df_endog["t"] = df_endog.patent_date + df_endog.t * bin_size_weeks
        df_endog = df_endog.set_index(
            [df_endog.index.get_level_values(0), "i", "t"]
        ).drop(columns="patent_date")

        # iterate through patents ("i" index) with groupby
        # for each patent
        # 1. extend the maximum date to the maximum end-of-data date
        # 2. join the dataframes on the date index up to n weeks ago
        date_max = df_endog.index.get_level_values("t").max()

        manager = enlighten.get_manager()
        ticker = manager.counter(total=df_endog.index.get_level_values("i").nunique(), desc='Patent Samples Transformed',
                                 unit='patents')
        subs = []
        for date, subdf in df_endog.groupby(level="i"):
            local_max = subdf.index.get_level_values("t").max()
            num_new_vals = int((date_max - local_max) / bin_size_weeks) + 1

            subdf = subdf.reset_index().drop(["i", "level_0"], 1)
            vals = np.full((num_new_vals, subdf.shape[1]), subdf[-1:].values)
            index = np.array(
                [pd.Timestamp(np.datetime64(local_max + (i + 1) * bin_size_weeks)) for i in range(num_new_vals - 1)] + [
                    date_max]
            )
            vals[:, 0] = index
            df_append = pd.DataFrame(
                data=vals,
                columns=["t", "log(knowledge_forward_cites)"]
            )
            subdf = subdf.append(df_append).set_index("t").sort_index(level="t", ascending=ascending).reset_index(drop=True)
            if subdf.shape[0] >= n:
                subs.append(subdf.head(n))

            ticker.update()
        ticker.close()

        df_endog = pd.concat(subs, axis=1)
        df_endog.columns = range(df_endog.shape[1])
        logger.debug(df_endog.describe())

        pickle.dump((df_endog, bin_size_weeks, n), open(cache_name, 'wb'))

        df = df_endog

    logger.debug("Loaded transformed endogenous set")
    return df


class Counter:
    def __init__(self):
        self.i = 0

    def inc(self):
        self.i += 1
        return self.i

    def get(self):
        return self.i

    def reset(self):
        self.i = 0


def get_stacked_df(keys, endpoint="FEATURE"):
    return pd.concat([
        FeatureExtractor(
            os.path.join(Config.EXT_DATA_PATH, "colonial/{}_both_None_{}.csv".format(endpoint, key))
        ).extract()
        for key in keys
    ], keys=keys)


def write(content, name):
    with open("data/regression/{}.tex".format(name), 'w') as f:
        f.write(content)
        f.close()


def regress(df, metric="forward_cites", exclude_nber=False):
    features, protected = get_features(exclude_nber, df.columns)
    reg = Regressor(df, features=features, protected=protected, target="log(knowledge_{})".format(metric))
    return reg.summary()


def get_features(exclude_nber, columns):
    features = [
        "patent_date_center",
        'log(patent_num_claims)',
        # 'log(patent_processing_time)',
        "log(avg_inventor_total_num_patents)",
        # "interaction"
    ]

    protected = features.copy()

    features += [key for key in columns if key.startswith("one-hot")]

    exclude = [
        # "one-hot_assignee_type_5",
        # "one-hot_assignee_type_9"
    ]
    for e in exclude:
        df = df[df[e] == 0]
    features = [feature for feature in features if feature not in exclude]

    if exclude_nber:
        features = [feature for feature in features if "nber_category" not in feature]

    return features, protected


def df_pretty_columns(df):
    copy = df.copy()
    keys = {
        "knowledge_forward_cites": "Knowledge Forward Cites",
        "knowledge_h_index": "Knowledge H Index",
        "log(patent_num_claims)": "Log(Number of Claims)",
        "log(patent_processing_time)": "Log(Processing Time)",
        "log(knowledge_h_index)": "Log(Knowledge H Index)",
        "log(knowledge_forward_cites)": "Log(Knowledge Forward Cites)",
        "log(avg_inventor_total_num_patents)": "Log(Average Inventor Other Patents)",
        "log(avg_assignee_total_num_patents)": "Log(Average Assignee Other Patents)",
        "patent_date_center": "Centered Patent Date"
    }
    cols = [keys[col] if col in keys else col for col in copy.columns]
    copy.columns = cols
    return copy


class Regressor:
    def __init__(self, data, features, protected=None, target="log(knowledge_forward_cites)"):
        self.data = data
        self.target = target
        self.features = features
        self.protected = protected

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
        x = self.calculate_vif_(pd.DataFrame(x), thresh=5.0)
        # check correlation matrix
        Regressor.print_corr(x.corr())
        fit = OLS(y, df_pretty_columns(x)).fit(cov_type="HC0")
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

    def calculate_vif_(self, X, thresh=5.0):
        variables = list(range(X.shape[1]))
        copy = variables.copy()
        dropped = True
        while dropped:
            dropped = False
            vif = [variance_inflation_factor(X.iloc[:, copy].values, ix)
                   for ix in range(X.iloc[:, copy].shape[1])]
            avgloc = vif.index(max(vif))
            if max(vif) > thresh:
                if X.columns[copy[avgloc]] in self.protected:
                    print("Warning: {} has high VIF {} but is protected".format(X.columns[variables[avgloc]], max(vif)))
                    del copy[avgloc]
                else:
                    print("dropping a variable {}".format(X.columns[variables[avgloc]]))
                    del copy[avgloc]
                    del variables[avgloc]
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
        # print("Calculating durations with {}".format(min(self.df['patent_date'])))
        self.df['patent_date_center'] = \
            (self.df['patent_date'] - min(self.df['patent_date'])).dt.total_seconds() / (24 * 60 * 60)
        # print(self.df['patent_date_center'].describe())

        # transform avg columns
        for key in FeatureExtractor.avg_list:
            self.make_list(key)
            self.df["avg_{}".format(key)] = self.df[key].apply(lambda x: np.mean([int(xx) for xx in x]) if len(x) > 0 else 0)

        # one-hot encode list columns
        for key in FeatureExtractor.one_hot:
            self.make_list(key)

            # remove "1" from assignee id
            if key == "assignee_type":
                self.df[key] = self.df[key].apply(lambda x: [xx[-1] for xx in x])

            self.one_hot_encode(key)

        # logarithm the continuous dependents
        continuous_dependents = [
            "patent_num_claims",
            "avg_inventor_total_num_patents",
            "avg_assignee_total_num_patents",
            "patent_processing_time"
        ]
        self.df["interaction"] = np.ones(self.df.shape[0])
        for key in continuous_dependents:
            self.make_logarithm(key)
            self.df["interaction"] *= self.df[key]

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
        references = {
            "nber_category_id": "6",
            "assignee_type": "2"
        }
        if key == "nber_category_id":
            self.df[key] = self.df[key].apply(lambda x: [xx if xx != "7" else xx for xx in x])
        vals = self.df[key].values
        mlb = MultiLabelBinarizer()
        enc = mlb.fit_transform(vals)
        codes = {c: [] for c in mlb.classes_ if c != references[key]}
        print("REFERENCE CATEGORY: {}_{}".format(key, references[key]))
        for row in list(enc):
            # using range - 1 to only include n-1 dummies - nth dummy represented by the intercept
            for i in range(len(row)):
                if i != list(mlb.classes_).index(references[key]):
                    codes[mlb.classes_[i]].append(row[i])
        for k, c in codes.items():
            self.df["one-hot_{}_{}".format(key, k)] = c
        return self


class FeatureExtractor:
    avg_list = ['inventor_total_num_patents', 'assignee_total_num_patents']
    one_hot = [
        "assignee_type",
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
        FeatureExtractor._types.update({key: str for key in FeatureExtractor.avg_list + FeatureExtractor.one_hot})
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
