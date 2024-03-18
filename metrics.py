import numpy as np
import pandas as pd 
import logging
from functools import lru_cache, wraps

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    roc_auc_score,
    log_loss,
)

logger = logging.getLogger(__name__)


DEFAULT_USER_COL = "userID"
DEFAULT_ITEM_COL = "itemID"
DEFAULT_PREDICTION_COL = "prediction"
DEFAULT_K = 10
DEFAULT_THRESHOLD = 10
DEFAULT_RATING_COL = "rating"

class ColumnMismatchError(Exception):
    """Exception raised when there is a mismatch in columns.

    This exception is raised when an operation involving columns
    encounters a mismatch or inconsistency.

    Attributes:
        message (str): Explanation of the error.
    """

    pass


class ColumnTypeMismatchError(Exception):
    """Exception raised when there is a mismatch in column types.

    This exception is raised when an operation involving column types
    encounters a mismatch or inconsistency.

    Attributes:
        message (str): Explanation of the error.
    """

    pass

class PandasHash:
    """Wrapper class to allow pandas objects (DataFrames or Series) to be hashable"""

    # reserve space just for a single pandas object
    __slots__ = "pandas_object"

    def __init__(self, pandas_object):
        """Initialize class

        Args:
            pandas_object (pandas.DataFrame|pandas.Series): pandas object
        """

        if not isinstance(pandas_object, (pd.DataFrame, pd.Series)):
            raise TypeError("Can only wrap pandas DataFrame or Series objects")
        self.pandas_object = pandas_object

    def __eq__(self, other):
        """Overwrite equality comparison

        Args:
            other (pandas.DataFrame|pandas.Series): pandas object to compare

        Returns:
            bool: whether other object is the same as this one
        """

        return hash(self) == hash(other)

    def __hash__(self):
        """Overwrite hash operator for use with pandas objects

        Returns:
            int: hashed value of object
        """

        hashable = tuple(self.pandas_object.values.tobytes())
        if isinstance(self.pandas_object, pd.DataFrame):
            hashable += tuple(self.pandas_object.columns)
        else:
            hashable += tuple(self.pandas_object.name)
        return hash(hashable)


def has_same_base_dtype(df_1, df_2, columns=None):
    """Check if specified columns have the same base dtypes across both DataFrames

    Args:
        df_1 (pandas.DataFrame): first DataFrame
        df_2 (pandas.DataFrame): second DataFrame
        columns (list(str)): columns to check, None checks all columns

    Returns:
        bool: True if DataFrames columns have the same base dtypes.
    """

    if columns is None:
        if any(set(df_1.columns).symmetric_difference(set(df_2.columns))):
            logger.error(
                "Cannot test all columns because they are not all shared across DataFrames"
            )
            return False
        columns = df_1.columns

    if not (
        has_columns(df=df_1, columns=columns) and has_columns(df=df_2, columns=columns)
    ):
        return False

    result = True
    for column in columns:
        if df_1[column].dtype.type.__base__ != df_2[column].dtype.type.__base__:
            logger.error("Columns {} do not have the same base datatype".format(column))
            result = False

    return result

def has_columns(df, columns):
    """Check if DataFrame has necessary columns

    Args:
        df (pandas.DataFrame): DataFrame
        columns (iterable(str)): columns to check for

    Returns:
        bool: True if DataFrame has specified columns.
    """
    if not isinstance(columns, set):
        columns = set(columns)
    return columns.issubset(df.columns)


def dcg_score(y_true, y_score, k=10):
    """Computing dcg score metric at k.

    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_score (np.ndarray): Predicted labels.

    Returns:
        np.ndarray: dcg scores.
    """
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10):
    """Computing ndcg score metric at k.

    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_score (np.ndarray): Predicted labels.

    Returns:
        numpy.ndarray: ndcg scores.
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best

def lru_cache_df(maxsize, typed=False):
    """Least-recently-used cache decorator for pandas Dataframes.

    Decorator to wrap a function with a memoizing callable that saves up to the maxsize most recent calls. It can
    save time when an expensive or I/O bound function is periodically called with the same arguments.

    Inspired in the `lru_cache function <https://docs.python.org/3/library/functools.html#functools.lru_cache>`_.

    Args:
        maxsize (int|None): max size of cache, if set to None cache is boundless
        typed (bool): arguments of different types are cached separately
    """

    def to_pandas_hash(val):
        """Return PandaHash object if input is a DataFrame otherwise return input unchanged"""
        return PandasHash(val) if isinstance(val, pd.DataFrame) else val

    def from_pandas_hash(val):
        """Extract DataFrame if input is PandaHash object otherwise return input unchanged"""
        return val.pandas_object if isinstance(val, PandasHash) else val

    def decorating_function(user_function):
        @wraps(user_function)
        def wrapper(*args, **kwargs):
            # convert DataFrames in args and kwargs to PandaHash objects
            args = tuple([to_pandas_hash(a) for a in args])
            kwargs = {k: to_pandas_hash(v) for k, v in kwargs.items()}
            return cached_wrapper(*args, **kwargs)

        @lru_cache(maxsize=maxsize, typed=typed)
        def cached_wrapper(*args, **kwargs):
            # get DataFrames from PandaHash objects in args and kwargs
            args = tuple([from_pandas_hash(a) for a in args])
            kwargs = {k: from_pandas_hash(v) for k, v in kwargs.items()}
            return user_function(*args, **kwargs)

        # retain lru_cache attributes
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear

        return wrapper

    return decorating_function


def _check_column_dtypes(func):
    """Checks columns of DataFrame inputs

    This includes the checks on:

    * whether the input columns exist in the input DataFrames
    * whether the data types of col_user as well as col_item are matched in the two input DataFrames.

    Args:
        func (function): function that will be wrapped

    Returns:
        function: Wrapper function for checking dtypes.
    """

    @wraps(func)
    def check_column_dtypes_wrapper(
        rating_true,
        rating_pred,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_prediction=DEFAULT_PREDICTION_COL,
        *args,
        **kwargs,
    ):
        """Check columns of DataFrame inputs

        Args:
            rating_true (pandas.DataFrame): True data
            rating_pred (pandas.DataFrame): Predicted data
            col_user (str): column name for user
            col_item (str): column name for item
            col_rating (str): column name for rating
            col_prediction (str): column name for prediction
        """
        # Some ranking metrics don't have the rating column, so we don't need to check.
        expected_true_columns = {col_user, col_item}
        if "col_rating" in kwargs:
            expected_true_columns.add(kwargs["col_rating"])
        if not has_columns(rating_true, expected_true_columns):
            raise ColumnMismatchError("Missing columns in true rating DataFrame")

        if not has_columns(rating_pred, {col_user, col_item, col_prediction}):
            raise ColumnMismatchError("Missing columns in predicted rating DataFrame")

        if not has_same_base_dtype(
            rating_true, rating_pred, columns=[col_user, col_item]
        ):
            raise ColumnTypeMismatchError(
                "Columns in provided DataFrames are not the same datatype"
            )

        return func(
            rating_true=rating_true,
            rating_pred=rating_pred,
            col_user=col_user,
            col_item=col_item,
            col_prediction=col_prediction,
            *args,
            **kwargs,
        )

    return check_column_dtypes_wrapper


def get_top_k_items(
    dataframe, col_user=DEFAULT_USER_COL, col_rating=DEFAULT_RATING_COL, k=DEFAULT_K
):
    """Get the input customer-item-rating tuple in the format of Pandas
    DataFrame, output a Pandas DataFrame in the dense format of top k items
    for each user.

    Note:
        If it is implicit rating, just append a column of constants to be
        ratings.

    Args:
        dataframe (pandas.DataFrame): DataFrame of rating data (in the format
        customerID-itemID-rating)
        col_user (str): column name for user
        col_rating (str): column name for rating
        k (int or None): number of items for each user; None means that the input has already been
        filtered out top k items and sorted by ratings and there is no need to do that again.

    Returns:
        pandas.DataFrame: DataFrame of top k items for each user, sorted by `col_user` and `rank`
    """
    # Sort dataframe by col_user and (top k) col_rating
    if k is None:
        top_k_items = dataframe
    else:
        top_k_items = (
            dataframe.sort_values([col_user, col_rating], ascending=[True, False])
            .groupby(col_user, as_index=False)
            .head(k)
            .reset_index(drop=True)
        )
    # Add ranks
    top_k_items["rank"] = top_k_items.groupby(col_user, sort=False).cumcount() + 1
    return top_k_items

@_check_column_dtypes
@lru_cache_df(maxsize=1)
def merge_ranking_true_pred(
    rating_true,
    rating_pred,
    col_user,
    col_item,
    col_prediction,
    relevancy_method,
    k=DEFAULT_K,
    threshold=DEFAULT_THRESHOLD,
    **_,
):
    """Filter truth and prediction data frames on common users

    Args:
        rating_true (pandas.DataFrame): True DataFrame
        rating_pred (pandas.DataFrame): Predicted DataFrame
        col_user (str): column name for user
        col_item (str): column name for item
        col_prediction (str): column name for prediction
        relevancy_method (str): method for determining relevancy ['top_k', 'by_threshold', None]. None means that the
            top k items are directly provided, so there is no need to compute the relevancy operation.
        k (int): number of top k items per user (optional)
        threshold (float): threshold of top items per user (optional)

    Returns:
        pandas.DataFrame, pandas.DataFrame, int: DataFrame of recommendation hits, sorted by `col_user` and `rank`
        DataFrame of hit counts vs actual relevant items per user number of unique user ids
    """

    # Make sure the prediction and true data frames have the same set of users
    common_users = set(rating_true[col_user]).intersection(set(rating_pred[col_user]))
    rating_true_common = rating_true[rating_true[col_user].isin(common_users)]
    rating_pred_common = rating_pred[rating_pred[col_user].isin(common_users)]
    n_users = len(common_users)

    # Return hit items in prediction data frame with ranking information. This is used for calculating NDCG and MAP.
    # Use first to generate unique ranking values for each item. This is to align with the implementation in
    # Spark evaluation metrics, where index of each recommended items (the indices are unique to items) is used
    # to calculate penalized precision of the ordered items.
    if relevancy_method == "top_k":
        top_k = k
    elif relevancy_method == "by_threshold":
        top_k = threshold
    elif relevancy_method is None:
        top_k = None
    else:
        raise NotImplementedError("Invalid relevancy_method")
    df_hit = get_top_k_items(
        dataframe=rating_pred_common,
        col_user=col_user,
        col_rating=col_prediction,
        k=top_k,
    )
    df_hit = pd.merge(df_hit, rating_true_common, on=[col_user, col_item])[
        [col_user, col_item, "rank"]
    ]

    # count the number of hits vs actual relevant items per user
    df_hit_count = pd.merge(
        df_hit.groupby(col_user, as_index=False)[col_user].agg({"hit": "count"}),
        rating_true_common.groupby(col_user, as_index=False)[col_user].agg(
            {"actual": "count"}
        ),
        on=col_user,
    )

    return df_hit, df_hit_count, n_users

@lru_cache_df(maxsize=1)
def _get_reciprocal_rank(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
    relevancy_method="top_k",
    k=DEFAULT_K,
    threshold=DEFAULT_THRESHOLD,
):
    df_hit, df_hit_count, n_users = merge_ranking_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_prediction=col_prediction,
        relevancy_method=relevancy_method,
        k=k,
        threshold=threshold,
    )

    if df_hit.shape[0] == 0:
        return None, n_users

    # calculate reciprocal rank of items for each user and sum them up
    df_hit_sorted = df_hit.copy()
    df_hit_sorted["rr"] = (
        df_hit_sorted.groupby(col_user).cumcount() + 1
    ) / df_hit_sorted["rank"]
    df_hit_sorted = df_hit_sorted.groupby(col_user).agg({"rr": "sum"}).reset_index()

    return pd.merge(df_hit_sorted, df_hit_count, on=col_user), n_users

def map(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
    relevancy_method="top_k",
    k=DEFAULT_K,
    threshold=DEFAULT_THRESHOLD,
    **_,
):
    """Mean Average Precision for top k prediction items

    The implementation of MAP is referenced from Spark MLlib evaluation metrics.
    https://spark.apache.org/docs/2.3.0/mllib-evaluation-metrics.html#ranking-systems

    A good reference can be found at:
    http://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf

    Note:
        The MAP is meant to calculate Avg. Precision for the relevant items, so it is normalized by the number of
        relevant items in the ground truth data, instead of k.

    Args:
        rating_true (pandas.DataFrame): True DataFrame
        rating_pred (pandas.DataFrame): Predicted DataFrame
        col_user (str): column name for user
        col_item (str): column name for item
        col_prediction (str): column name for prediction
        relevancy_method (str): method for determining relevancy ['top_k', 'by_threshold', None]. None means that the
            top k items are directly provided, so there is no need to compute the relevancy operation.
        k (int): number of top k items per user
        threshold (float): threshold of top items per user (optional)

    Returns:
        float: MAP (min=0, max=1)
    """
    df_merge, n_users = _get_reciprocal_rank(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_prediction=col_prediction,
        relevancy_method=relevancy_method,
        k=k,
        threshold=threshold,
    )

    if df_merge is None:
        return 0.0
    else:
        return (df_merge["rr"] / df_merge["actual"]).sum() / n_users


def map_at_k(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
    relevancy_method="top_k",
    k=DEFAULT_K,
    threshold=DEFAULT_THRESHOLD,
    **_,
):
    """Mean Average Precision at k

    The implementation of MAP@k is referenced from Spark MLlib evaluation metrics.
    https://github.com/apache/spark/blob/b938ff9f520fd4e4997938284ffa0aba9ea271fc/mllib/src/main/scala/org/apache/spark/mllib/evaluation/RankingMetrics.scala#L99

    Args:
        rating_true (pandas.DataFrame): True DataFrame
        rating_pred (pandas.DataFrame): Predicted DataFrame
        col_user (str): column name for user
        col_item (str): column name for item
        col_prediction (str): column name for prediction
        relevancy_method (str): method for determining relevancy ['top_k', 'by_threshold', None]. None means that the
            top k items are directly provided, so there is no need to compute the relevancy operation.
        k (int): number of top k items per user
        threshold (float): threshold of top items per user (optional)

    Returns:
        float: MAP@k (min=0, max=1)
    """
    df_merge, n_users = _get_reciprocal_rank(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_prediction=col_prediction,
        relevancy_method=relevancy_method,
        k=k,
        threshold=threshold,
    )

    if df_merge is None:
        return 0.0
    else:
        return (
            df_merge["rr"] / df_merge["actual"].apply(lambda x: min(x, k))
        ).sum() / n_users
    
def ndcg_at_k(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
    relevancy_method="top_k",
    k=DEFAULT_K,
    threshold=DEFAULT_THRESHOLD,
    score_type="binary",
    discfun_type="loge",
    **_,
):
    """Normalized Discounted Cumulative Gain (nDCG).

    Info: https://en.wikipedia.org/wiki/Discounted_cumulative_gain

    Args:
        rating_true (pandas.DataFrame): True DataFrame
        rating_pred (pandas.DataFrame): Predicted DataFrame
        col_user (str): column name for user
        col_item (str): column name for item
        col_rating (str): column name for rating
        col_prediction (str): column name for prediction
        relevancy_method (str): method for determining relevancy ['top_k', 'by_threshold', None]. None means that the
            top k items are directly provided, so there is no need to compute the relevancy operation.
        k (int): number of top k items per user
        threshold (float): threshold of top items per user (optional)
        score_type (str): type of relevance scores ['binary', 'raw', 'exp']. With the default option 'binary', the
            relevance score is reduced to either 1 (hit) or 0 (miss). Option 'raw' uses the raw relevance score.
            Option 'exp' uses (2 ** RAW_RELEVANCE - 1) as the relevance score
        discfun_type (str): type of discount function ['loge', 'log2'] used to calculate DCG.

    Returns:
        float: nDCG at k (min=0, max=1).
    """
    df_hit, _, _ = merge_ranking_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_prediction=col_prediction,
        relevancy_method=relevancy_method,
        k=k,
        threshold=threshold,
    )

    if df_hit.shape[0] == 0:
        return 0.0

    df_dcg = df_hit.merge(rating_pred, on=[col_user, col_item]).merge(
        rating_true, on=[col_user, col_item], how="outer", suffixes=("_left", None)
    )

    if score_type == "binary":
        df_dcg["rel"] = 1
    elif score_type == "raw":
        df_dcg["rel"] = df_dcg[col_rating]
    elif score_type == "exp":
        df_dcg["rel"] = 2 ** df_dcg[col_rating] - 1
    else:
        raise ValueError("score_type must be one of 'binary', 'raw', 'exp'")

    if discfun_type == "loge":
        discfun = np.log
    elif discfun_type == "log2":
        discfun = np.log2
    else:
        raise ValueError("discfun_type must be one of 'loge', 'log2'")

    # Calculate the actual discounted gain for each record
    df_dcg["dcg"] = df_dcg["rel"] / discfun(1 + df_dcg["rank"])

    # Calculate the ideal discounted gain for each record
    df_idcg = df_dcg.sort_values([col_user, col_rating], ascending=False)
    df_idcg["irank"] = df_idcg.groupby(col_user, as_index=False, sort=False)[
        col_rating
    ].rank("first", ascending=False)
    df_idcg["idcg"] = df_idcg["rel"] / discfun(1 + df_idcg["irank"])

    # Calculate the actual DCG for each user
    df_user = df_dcg.groupby(col_user, as_index=False, sort=False).agg({"dcg": "sum"})

    # Calculate the ideal DCG for each user
    df_user = df_user.merge(
        df_idcg.groupby(col_user, as_index=False, sort=False)
        .head(k)
        .groupby(col_user, as_index=False, sort=False)
        .agg({"idcg": "sum"}),
        on=col_user,
    )

    # DCG over IDCG is the normalized DCG
    df_user["ndcg"] = df_user["dcg"] / df_user["idcg"]
    return df_user["ndcg"].mean()