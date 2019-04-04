import pandas as pd
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from statsmodels.api import add_constant
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.outliers_influence import variance_inflation_factor
from linearmodels.panel import PooledOLS
from scipy import stats
import pickle
import enlighten
from collections import namedtuple

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


def test_forecasting(endpoint, bin_size=20, relative_series=False):
    cache_name = os.path.abspath("data/regression/forecasting_cache.pkl")

    keys = [
        "engines_25Mar19",
        "radio_25Mar19",
        "robots_25Mar19",
        "transportation_25Mar19",
        "xray_25Mar19",
        "coherent-light_25Mar19"
    ]

    bin_size_weeks = np.timedelta64(bin_size, 'W')

    try:
        df = pickle.load(open(cache_name, 'rb'))
    except FileNotFoundError:
        df = FeatureTransformer(get_stacked_df(keys, endpoint="TIME-DATA")).fit_transform()
        pickle.dump(df, open(cache_name, 'wb'))

    df_endog = df[["log(knowledge_forward_cites)", "t", "patent_date"]]
    features, protected = get_features(True, df)
    df_exog = df[features][df["t"] == 0]
    logger.debug(features)

    if endpoint == "arima":
        regress_arima(df_endog, bin_size_weeks, relative_series)

    if endpoint == "pooled":
        # regress_pooled(df_endog, df_exog, bin_size_weeks)
        entity_res = fit_write(None, "entity")
        plot_coeffs(entity_res)


def regress_pooled(df_endog, df_exog, bin_size_weeks):
    df_endog = PooledTransformer('pooled', load_from_cache=True).transform(df_endog, bin_size_weeks)
    df_endog.index = df_endog.index.rename(["source", "i"])
    df_exog = df_exog.reset_index().drop(columns=["level_1"])
    df_exog = df_exog.set_index(["level_0", df_exog.index.values+1])
    df_exog.index = df_exog.index.rename(["source", "i"])
    df = df_endog.reset_index().merge(df_exog.reset_index(), on=["i", "source"], how="left")
    df = df.set_index(["i", "t"], drop=False)\
        .drop(columns=["i"])
    model_pooled(df)


def plot_coeffs(pooled_res):
    time_coeffs = pd.concat(
        [
            pooled_res.params,
            pooled_res.std_errors,
            make_conf_int(pooled_res.params, pooled_res.std_errors, pooled_res.df_resid)
        ],
        axis=1
    )
    time_coeffs =  time_coeffs[time_coeffs.index.str.contains("t\.")]
    time_coeffs.index = (time_coeffs.index.str.strip("t.")).astype('datetime64[ns]')
    ax = time_coeffs.parameter.plot(label='_nolegend_')
    # ax = plt.scatter(time_coeffs.parameter.index, time_coeffs.parameter, label='_nolegend_')
    plt.axvline(
        x=np.datetime64("2011-09-16"),
        linestyle=':',
        color='orange',
        zorder=-1,
        label="AIA signed"
    )
    plt.axvline(
        x=np.datetime64("2013-03-16"),
        linestyle='--',
        color='orange',
        zorder=-1,
        label="AIA effective"
    )
    ax.legend()
    ax.set_xlabel("Time Dummy")
    ax.set_ylabel("Estimated Parameter")
    plt.savefig("data/regression/time_dummies.png")


def make_conf_int(params, std_errors, df_resid, level=.05):
    ci_quantiles = [(1 - level) / 2, 1 - (1 - level) / 2]
    q = stats.t.ppf(ci_quantiles, df_resid)
    q = q[None, :]
    ci = params[:, None] + std_errors[:, None] * q
    return pd.DataFrame(ci, index=params.index, columns=['lower', 'upper'])


def model_pooled(df):
    df["age"] = (df["t"] - df["patent_date"]) / np.timedelta64(1, 'Y')
    df["agesq"] = np.square(df.age)
    df["t"] = pd.Categorical(df.t)

    df = df.rename(index=str, columns={
        "log(knowledge_forward_cites)": "lknowledge_forward_cites"
    })
    df.index = df.index.set_levels([
        df.index.levels[0].astype(int),
        df.index.levels[1].astype('datetime64[ns]')
    ])

    exog_vars = [
        "t",
        "source",
        'log(patent_num_claims)',
        'log(avg_inventor_total_num_patents)',
        'log(patent_processing_time)',
        'one-hot_assignee_type_3',
        'one-hot_assignee_type_4',
        'one-hot_assignee_type_5',
        'one-hot_assignee_type_6',
        'one-hot_assignee_type_7',
        'one-hot_assignee_type_9',
        'age',
        'agesq'
    ]
    exog = add_constant(df[exog_vars])

    mod = PooledOLS(df.lknowledge_forward_cites, exog)
    # robust_res = fit_write(mod, "robust", cov_type='robust')
    fit_write(mod, "entity", cov_type='clustered', cluster_entity=True)
    fit_write(mod, "entity-time", cov_type='clustered', cluster_entity=True, cluster_time=True)


def fit_write(mod, filename, **kwargs):
    file = "data/regression/{}_res.pkl".format(filename)
    try:
        vars = pickle.load(open(file, 'rb'))
    except (FileNotFoundError, EOFError):
        logger.info("Fitting model {}".format(filename))
        pooled_res = mod.fit(**kwargs)
        vars = (
            pooled_res.summary,
            pooled_res.params,
            pooled_res.pvalues,
            pooled_res.resids,
            pooled_res.std_errors,
            pooled_res.df_resid,
            pooled_res.tstats
        )
        pickle.dump(vars, open(file, 'wb'), protocol=4)
        with open(file.strip(".pkl")+".txt", 'w') as f:
            f.write(str(pooled_res))
            f.close()
    Results = namedtuple('Results', 'summary params pvalues resids std_errors df_resid tstats')
    return Results(*vars)


def regress_arima(df_endog, bin_size_weeks, relative_series):
    df_endog = ARIMATransformer('arima', load_from_cache=True).transform(df_endog, bin_size_weeks)
    aia_date = np.datetime64("2013-03-16")
    if relative_series:
        arima_relative(df_endog, bin_size_weeks, aia_date)
    else:
        arima_absolute(df_endog, aia_date)


def arima_relative(df_endog, bin_size_weeks, aia_date):
    duration_since_pub = ((df_endog["t"] - df_endog["patent_date"]) / bin_size_weeks).astype(int)
    df_endog["t"] = duration_since_pub
    plt.clf()
    plt.close()

    df_before = df_endog[df_endog["patent_date"] < aia_date].groupby("t").count()
    df_after = df_endog[df_endog["patent_date"] >= aia_date].groupby("t").count()
    ax = df_before.plot()
    df_after.plot(ax=ax)
    plt.show()

    df_before = df_endog[df_endog["patent_date"] < aia_date].groupby("t").mean()
    df_after = df_endog[df_endog["patent_date"] >= aia_date].groupby("t").mean()
    df_before.columns = ["Before AIA"]
    df_after.columns = ["After AIA"]
    ax = df_before.plot()
    df_after.plot(ax=ax)
    ax.set_xlabel("Age")
    ax.set_ylabel("Average Log(TKC)")
    plt.savefig("data/regression/tkc_by_age.png")

    # df_endog["t"] = df_endog["t"] * bin_size_weeks + df_endog["patent_date"].min()
    # logger.warn(
    #    "Reminder that dummy date indexes were used to comply with statsmodels API. These dates are not real."
    # )
    # data = {
    #     "before": {},
    #     "after": {}
    # }
    # data["before"]["df"] = df_endog[df_endog["patent_date"] < aia_date].groupby("t").mean()
    # data["after"]["df"] = df_endog[df_endog["patent_date"] >= aia_date].groupby("t").mean()
    # for key, d in data.items():
    #     logger.debug("Fit for {}".format(key))
    #     # explore_series(d["df"])
    #     data[key]["model"] = ARIMA(d["df"]["log(knowledge_forward_cites)"], order=(4, 1, 0))
    #     data[key]["fit"] = data[key]["model"].fit(maxiter=1000000, disp=False, transparams=True, trend='c')
    #     logger.debug(data[key]["fit"].summary())
    # data["before"]["fit"].plot_predict(
    #     start=data["after"]["df"].index[1],
    #     end=data["after"]["df"].index[-1]+5*bin_size_weeks
    # )
    # ax = data["before"]["df"].plot()
    # data["after"]["fit"].plot_predict(
    #     start=data["after"]["df"].index[1],
    #     end=data["after"]["df"].index[-1]+5*bin_size_weeks,
    #     ax=ax
    # )
    # plt.show()


def arima_absolute(df_endog, aia_date, stratify=True):
    df_grouped = df_endog.groupby("t").mean()
    # df_count = df_endog.groupby("t").mean() / df_endog.groupby("t").count()
    # df_count.plot()
    # plt.show()

    # Note that the index before the last index is used
    # - the last index includes values averaged from during the AIA period
    aia_index = df_grouped.index[df_grouped.index < aia_date][-2]
    train = df_grouped.loc[:aia_index]
    test = df_grouped.loc[aia_index:]

    model = ARIMA(train["log(knowledge_forward_cites)"], order=(2, 1, 0))
    fit = model.fit(maxiter=1000000, disp=False, transparams=True, trend='c')
    logger.debug(fit.summary())

    # adapted from
    # http://www.statsmodels.org/devel/_modules/statsmodels/tsa/arima_model.html#ARIMAResults.plot_predict
    test["actual"] = test["log(knowledge_forward_cites)"]
    ax = test["actual"].plot()
    ax.set_xlabel("Year")
    ax.set_ylabel("Average Log(TKC)")
    fit.plot_predict(start=train.index[50], end=test.index[-1], ax=ax)
    plt.savefig("data/regression/absolute_forecast.png")

    analyze_res(fit)

    if stratify:
        logger.debug(np.unique(df_endog.index.get_level_values(0)))
        ax = None
        for source in np.unique(df_endog.index.get_level_values(0)):
            df_grouped_strat = df_endog.xs(source, level=0).groupby("t").mean()
            df_grouped_strat.columns = [source.split("_")[0]]
            if ax is None:
                ax = df_grouped_strat.plot()
            else:
                df_grouped_strat.plot(ax=ax)
        # df_grouped.plot(ax=ax)
        ax.set_xlabel("Year")
        ax.set_ylabel("Log(TKC)")
    plt.savefig("data/regression/strat_time_series.png")

    return fit


def analyze_res(fit):
    plt.cla()
    plt.clf()
    residuals = pd.Series(fit.resid)
    autocorrelation_plot(residuals)
    plt.savefig("data/regression/autocorrelation_plot.png")
    # residuals.plot(kind='kde')
    # plt.show()


def explore_series(df_endog):
    df_endog.plot()
    plt.show()
    autocorrelation_plot(df_endog)
    plt.show()
    result = seasonal_decompose(df_endog, model="linear")
    result.plot()
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
    df_endog = VARMAXTransformer("varmax").transform(df_endog, bin_size_weeks, n)

    # remove columns with low variance
    order = 4
    df_endog = df_endog.loc[:, df_endog.apply(pd.Series.nunique, axis=0) > order]
    logger.debug(df_endog)
    logger.debug(df_endog.describe())

    logger.debug("Training VARMAX...")
    model = VARMAX(df_endog.values, order=(order, 0))
    res = model.fit(maxiter=1000, disp=True)
    logger.debug(res.summary())


class ForecastingTransformer:
    def __init__(self, cache_name, load_from_cache=True):
        self.cache = "data/regression/{}.pkl".format(cache_name)
        self.load_from_cache = load_from_cache

    def transform(self, df_endog, *kwargs):
        try:
            if not self.load_from_cache:
                raise ValueError
            return self.load(*kwargs)
        except (FileNotFoundError, ValueError, EOFError):
            return self.dump(df_endog, *kwargs)

    def load(self, *kwargs):
        logger.debug("Attempting to load from cache {}".format(self.cache))
        df_endog, stored = pickle.load(open(self.cache, 'rb'))
        if stored != kwargs:
            logger.warn("Load failed due to param mismatch, refitting")
            raise ValueError
        return df_endog

    def dump(self, df_endog, *kwargs):
        logger.debug("Fitting data")
        df_endog = self.fit(df_endog, *kwargs)
        logger.debug("Dumping to cache {}".format(self.cache))
        logger.debug(kwargs)
        pickle.dump((df_endog, kwargs), open(self.cache, 'wb'))
        return df_endog

    def fit(self, *args):
        raise NotImplementedError

    @staticmethod
    def entity_id(df_endog):
        c = Counter()
        return df_endog.t.apply(lambda x: c.inc() if x == 0 else c.get())


class ARIMATransformer(ForecastingTransformer):
    def fit(self, df_endog, bin_size_weeks):
        df_endog = self.apply_cutoff(df_endog, bin_size_weeks)
        df_endog = self.add_duplicates(df_endog, bin_size_weeks)
        df_endog["t"] = df_endog["t"] * bin_size_weeks + df_endog["patent_date"].min()
        return df_endog

    @staticmethod
    def apply_cutoff(df_endog, bin_size_weeks, cutoff_date=np.datetime64("2018-11-27")):
        return df_endog[(df_endog["patent_date"] + df_endog["t"] * bin_size_weeks) < cutoff_date]

    @staticmethod
    def add_duplicates(df_endog, bin_size_weeks):
        start_date = df_endog["patent_date"].min()
        t_from_start = ((df_endog["patent_date"] - start_date) / bin_size_weeks).astype(int)
        df_endog["t"] = df_endog["t"] + t_from_start
        # iterate through rows where next t is zero - so iterating through last entry in each series
        data = []
        ind = []
        # a mask where true if row after is less than row before
        mask = (df_endog["t"].shift(-1) < df_endog["t"])
        manager = enlighten.get_manager()
        ticker = manager.counter(
            total=df_endog[mask].shape[0],
            desc='Patent Samples Transformed',
            unit='patents'
        )
        for row in df_endog[mask][["log(knowledge_forward_cites)", "t", "patent_date"]].itertuples():
            index, k, t, date = row
            # append the last k entry as many times as necessary to reach the present
            for i in range(int(df_endog["t"].max()) - int(t)):
                data.append((k, t + 1 + i, date))
                ind.append(index)
            ticker.update()
        ticker.close()

        to_add = pd.DataFrame(data, index=ind, columns=["log(knowledge_forward_cites)", "t", "patent_date"])
        df_endog = df_endog.append(to_add)

        return df_endog


class PooledTransformer(ARIMATransformer):
    def fit(self, df_endog, bin_size_weeks):
        df_endog["i"] = ForecastingTransformer.entity_id(df_endog)
        df_endog = df_endog.set_index([df_endog.index.get_level_values(0), "i"])
        return super().fit(df_endog, bin_size_weeks)


class VARMAXTransformer(ForecastingTransformer):
    def fit(self, df_endog, bin_size_weeks, n, ascending=True):
        df_endog["i"] = ForecastingTransformer.entity_id(df_endog)
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
        'log(patent_processing_time)',
        "log(avg_inventor_total_num_patents)",
        # "interaction"
    ]

    protected = features.copy()

    features += [key for key in columns if key.startswith("one-hot")]

    exclude = [
        # "one-hot_assignee_type_5",
        # "one-hot_assignee_type_9"
    ]
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
