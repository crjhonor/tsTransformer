import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from collections import namedtuple
try:
    import pymannkendall as mk
    MANN_KENDALL_INSTALLED = True
except ImportError:
    MANN_KENDALL_INSTALLED = False
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white

def _check_convert_y(y):
    assert not np.any(np.isnan(y)), "'y' should not have any nan values"
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.values.squeeze()
    assert y.ndim == 1
    return y

def _check_stationary_adfuller(y, confidence, **kwargs):
    y = _check_convert_y(y)
    res = namedtuple("ADF_test", ['stationary', 'results'])
    result = adfuller(y, **kwargs)
    if result[1]>confidence:
        return res(False, result)
    else:
        return res(True, result)

def _check_stationary_kpss(y, confidence, **kwargs):
    y = _check_convert_y(y)
    res = namedtuple("KPSS_test", ['stationary', 'results'])
    result = kpss(y, **kwargs)
    if result[1]<confidence:
        return res(False, result)
    else:
        return res(True, result)

def check_unit_root(y, confidence=0.05, test_params={}):
    test_params['regression'] = 'ct'
    return _check_stationary_adfuller(y, confidence, **test_params), \
        _check_stationary_kpss(y, confidence, **test_params)

def _check_kendall_tau(y, confidence=0.05):
    y = _check_convert_y(y)
    tau, p_value = stats.kendalltau(y, np.arange(len(y)))
    trend = True if p_value < confidence else False
    if tau>0:
        direction = "increasing"
    else:
        direction = "decreasing"
    return "Kendall_Tau_Test", tau, p_value, trend, direction

def _check_mann_kendall(y, confidence=0.05, seasonal_period=None, prewhiten=None):
    if not MANN_KENDALL_INSTALLED:
        raise ValueError("`pymannkendall` needs to be installed for the mann_kendal test. `pip install pymannkendall` to install")
    #https://www.tandfonline.com/doi/pdf/10.1623/hysj.52.4.611
    if prewhiten is None:
        if len(y)<50:
            prewhiten = True
        else:
            prewhiten = False
    else:
        if not prewhiten and len(y)<50:
            warnings.warn("For timeseries with < 50 samples, it is recommended to prewhiten the timeseries. Consider passing `prewhiten=True`")
        if prewhiten and len(y)>50:
            warnings.warn("For timeseries with > 50 samples, it is not recommended to prewhiten the timeseries. Consider passing `prewhiten=False`")
    y = _check_convert_y(y)
    if seasonal_period is None:
        if prewhiten:
            _res = mk.pre_whitening_modification_test(y, alpha=confidence)
        else:
            _res = mk.original_test(y, alpha=confidence)
    else:
        _res = mk.seasonal_test(y, alpha=confidence, period=seasonal_period)
    trend=True if _res.p<confidence else False
    if _res.slope>0:
        direction="increasing"
    else:
        direction="decreasing"
    return type(_res).__name__, _res.slope, _res.p, trend, direction

def check_trend(y, confidence=0.05, seasonal_period=None, mann_kendall=False, prewhiten=None):
    if mann_kendall:
        name, slope, p, trend, direction = _check_mann_kendall(y, confidence, seasonal_period, prewhiten)
    else:
        name, slope, p, trend, direction = _check_kendall_tau(y, confidence)
    det_trend_res = check_deterministic_trend(y, confidence)
    res = namedtuple(name, ["trend", "direction", "slope", "p_value", "deterministic", "deterministic_trend_results"])
    return res(trend, direction, slope, p, det_trend_res.deterministic_trend, det_trend_res)

def check_deterministic_trend(y, confidence=0.05):
    res = namedtuple("ADF_deterministic_Trend_Test", ["deterministic_trend", "adf_res", "adf_ct_res"])
    adf_res = _check_stationary_adfuller(y, confidence)
    adf_ct_res = _check_stationary_adfuller(y, confidence, regression="ct")
    if (not adf_res.stationary) and (adf_ct_res.stationary):
        deterministic_trend = True
    else:
        deterministic_trend = False
    return res(deterministic_trend, adf_res, adf_ct_res)

def check_heteroscedastisticity(y, confidence=0.05):
    y = _check_convert_y(y)
    res = namedtuple("White_Test", ["heteroscedastic", "lm_statistic", "lm_p_value"])
    #Fitting a linear trend regression
    x = np.arange(len(y))
    x = sm.add_constant(x)
    model = sm.OLS(y,x)
    results = model.fit()
    lm_stat, lm_p_value, f_stat, f_p_value = het_white(results.resid, x)
    if lm_p_value<confidence and f_p_value < confidence:
        hetero = True
    else:
        hetero = False
    return res(hetero, lm_stat, lm_p_value)