"""
Try using http request api to obtain trading data.
"""
import pandas as pd
import numpy as np
import requests
import os
import datetime as dt
import math

os.environ['HTTP_API_KEY'] = '2284526EBB2A4A839B7AD4FF9F1A5DCA'
apiKey = os.getenv('HTTP_API_KEY')
getURL = 'http://cn.yufenghy.cn:9009/hq/api/real/multiple?symbol='

class readingHifreq():
    def __init__(
            self
    ):
        self.headers = {
            'Authorization': apiKey
        }
    def reqGet(self, targets: list, origin_targets: list):
        assert type(targets) is list and len(targets) <= 10, \
            "targets should be a list and should contain less than 10 target in one time."
        targets_str = ','.join(targets)
        url = ''.join([getURL, targets_str])
        results = requests.get(url, headers=self.headers)
        results = results.json()['Data']
        # Prepare the return dataframe
        returnDataframe = pd.DataFrame(data=None, columns=['target', 'price', 'target_ind'])
        for num, itm in enumerate(targets):
            returnDataframe.loc[num] = [itm, results[num]['price'], origin_targets[num]]
        return returnDataframe

    def getBalance(self):
        url = 'http://cn.yufenghy.cn:9009/hq/api/balance'
        results = requests.get(url, headers=self.headers)
        results = results.json()
        print(results)

def _gen_logr(x0: float, x1: float) -> float:
    logr = (x1-x0)/x0
    logr = np.log(logr+1)
    return logr

def _rollback_logr(logr: float, x0: float=None, x1: float=None):
    assert x0 is None and x1 is None, "Both x0 or x1 is not provided."
    assert x0 is not None and x1 is not None, "Both x0 and x1 is provided."
    pct_c = math.e ** logr - 1
    if x0 is None:
        x0 = x1/(1+pct_c)
        return x0
    elif x1 is None:
        x1 = x0*(1+pct_c)
        return x1

def fillHifreq(
        df: pd.DataFrame,
        latestHifreq: pd.DataFrame,
        lastDay_fullprice: pd.DataFrame,
        lastDay_newOrder: pd.DataFrame,
        target_ind: str,
        isNighttrade: bool = False,
) -> pd.DataFrame:
    '''
    Fill the latest hifrequency data into the dataframe and return.
    :return:
    '''
    new_data_cols = df.columns
    returnDf = pd.DataFrame(data=np.zeros((1, len(df.columns))), columns=df.columns)
    if isNighttrade:
        daily_timestamp = dt.datetime.now() + dt.timedelta(days=1)
    else:
        daily_timestamp = dt.datetime.now()
    # Calculate new log returns.
    dfForcal = latestHifreq.copy()
    dfForcal = dfForcal.drop(columns='index')
    dfForcal['target_ind'] = dfForcal['target_ind'].apply(lambda x: ''.join([x, 'Close']))
    dfForcal = dfForcal.set_index('target_ind')
    dfForcal = dfForcal.join(lastDay_fullprice[1:], how='right')
    dfForcal = dfForcal.drop(columns='target')
    dfForcal = dfForcal.join(lastDay_newOrder, how="right", rsuffix='_newOrder')
    dfForcal.columns = ['hiFreq_price', 'last_fullprice', 'last_logr']
    dfForcal['hiFreq_price'] = dfForcal['hiFreq_price'].astype(np.float64)
    dfForcal['hiFreq_logr'] = dfForcal.apply(lambda x: np.log((x['hiFreq_price'] - x['last_fullprice'])/x['last_fullprice']+1), axis=1)

    # Dealling with missing values, just copy from the last day.
    dfForcal['hiFreq_logr'][dfForcal.hiFreq_logr.isna()] = dfForcal['last_logr'][dfForcal.hiFreq_logr.isna()]
    # Fill in the gaps
    returnDf.loc[0, 'target_ind'] = target_ind
    returnDf.loc[0, 'daily_timestamp'] = daily_timestamp.date()
    returnDf.loc[0, 'label'] = dfForcal.loc[target_ind, 'hiFreq_logr']
    returnDf.loc[0, 'label_T1'] = dfForcal.loc[target_ind, 'hiFreq_logr']
    # the trading features
    for l, e in enumerate(dfForcal['hiFreq_logr']):
        returnDf.loc[0, f'tf_{l}'] = e
    # copying last features.
    for i in range(56, 83):
        returnDf.iloc[0, i] = df.iloc[-1, i]
    returnDf.iloc[0, 83] = df.iloc[-1, 83] + 1
    for i in range(84, 89):
        returnDf.iloc[0, i] = df.iloc[-1, i]
    returnDf.loc[0, 'train'] = df['train'][-1]
    # the lagging feature dictionary
    the_lags = np.array(dfForcal.loc[target_ind, 'hiFreq_logr'])
    the_lags = np.append(the_lags, np.array(df.iloc[-1, 89:113]))
    for i in range(89, 114):
        returnDf.iloc[0, i] = the_lags[i-89]
    # concatenate the df and the returnDf to form the final returnDf
    returnDf.index = [daily_timestamp.date()]
    returnDf = pd.concat([df, returnDf])
    returnDf['time_idx'] = returnDf['time_idx'].astype(int)
    return returnDf

# targets = ['FTSE', 'INELU', 'USDX', 'USDCNH', 'GBPUSD', 'EURUSD', 'AUDUSD', 'USDJPY', 'AUDCAD']
# rh = readingHifreq()
# results = rh.reqGet(targets)