import warnings
import numpy as np
import pandas as pd
from dataclasses import MISSING, dataclass, field
from typing import Dict, List, Union

from darts.metrics import mae, mase, mse

from assa.utils.ts_utils import darts_metrics_adapter, forecast_bias

@dataclass
class FeatureConfig:

    date: List = field(
        default=MISSING,
        metadata={"help": "Column name of the data column"},
    )

    target: str = field(
        default=MISSING,
        metadata={"help": "Column name of the target column"},
    )

    original_target: str = field(
        default=None,
        metadata={
            "help": "Column name of the original target column in acse of transformed target. If None, it will be assigned same value as target"
        },
    )

    continuous_features: List[str] = field(
        default_factory=list,
        metadata={"help": "Column names of the numeric fields. Defaults to []"}
    )

    categorical_features: List[str] = field(
        default_factory=list,
        metadata={"help": "Column names of the boolean fields. Defaults to []"},
    )

    boolean_features: List[str] = field(
        default_factory=list,
        metadata={"help": "column names of the boolean fields. Defaults to []"},
    )

    index_cols: str = field(
        default_factory=list,
        metadata={
            "help": "Column names which needs to be set as index in the X and Y dataframes"
        },
    )

    exogenous_features: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Column names of the exogenous features. Must be a subset of categorical and continuous features"
        },
    )

    feature_list: List[str] = field(init=False)

    def __post_init__(self):
        assert(
            len(self.categorical_features) + len(self.continuous_features) > 0
        ), "There should be at-least one feature defined in categorical or continuous columns"
        self.feature_list = (
            self.categorical_features + self.continuous_features + self.boolean_features
        )
        assert (
            self.target not in self.feature_list
        ), f"'target' ({self.target}) should not be present in either categorical, continuous or boolean feature list"
        assert (
            self.date not in self.feature_list
        ), f"'date' ({self.date}) should not be present in either categorical, continuous or boolean feature list"
        extra_exog = set(self.exogenous_features) - set(self.feature_list)
        assert (
            len(extra_exog) == 0
        ), f"These exogenous features are not present in feature list: {extra_exog}"
        intersection = (
            set(self.continuous_features)
            .intersection(self.categorical_features + self.boolean_features)
            .union(
                set(self.categorical_features).intersection(
                    self.continuous_features + self.boolean_features
                )
            )
            .union(
                set(self.boolean_features).intersection(
                    self.continuous_features + self.categorical_features
                )
            )
        )
        assert (
            len(intersection) == 0
        ), f"There should not be any overlaps between the categorical continuous and boolean features. " \
           f"{intersection} are present in more than one definition."
        if self.original_target is None:
            self.original_target = self.target

def calculate_metrics(
        y: pd.Series, y_pred: pd.Series, name: str, y_train: pd.Series = None
):
    """
    Method to calculate the metrics given the actual and predicted series
    :param y (pd.Series): Actual target with datetime index
    :param y_pred (pd.Series): Predictions with datetime index
    :param name (str): Name or identification for the model
    :param y_train (pd.Series, optional): Actual train target to calculate MASE with datetime index. Defaults to None.
    :return (Dict): Dictionary with MAE, MSE, MASE, and Forecast Bias
    """
    return {
        "Algorithm": name,
        "MAE": darts_metrics_adapter(mae, actual_series=y, pred_series=y_pred),
        "MSE": darts_metrics_adapter(mse, actual_series=y, pred_series=y_pred)
        # "MASE": darts_metrics_adapter(
        #     mase, actual_series=y, pred_series=y_pred, insample=y_train
        # )
        if y_train is not None
        else None,
        "Forecast Bias": darts_metrics_adapter(
            forecast_bias, actual_series=y, pred_series=y_pred
        ),
    }
