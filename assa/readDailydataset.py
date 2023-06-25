"""
# Part I:
# Prepare the dataset.==================================================================================================
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
import math
import datetime
import xlrd, xlwt
from tqdm import tqdm
from sklearn.impute import KNNImputer
import calendar
from assa.myPreprocessing import (
generate_logr,
generateDatefeature,
add_lags
)

source_data = Path("/home/crjLambda/PycharmProjects/tsTransformer/data")

dataDirName = "/home/crjLambda/PycharmProjects/profoundRNN/dataForlater"
emailReadfilename = Path(dataDirName, "II_seq2seq_moon2sun_cook_email_feature_forlater.json")
weiboReadfilename = Path(dataDirName, "II_seq2seq_moon2sun_cook_weibo_feature_forlater.json")
emailFeatures_df = pd.read_json(emailReadfilename)
weiboFeatures_df = pd.read_json(weiboReadfilename)
emailFeatures_df.index = emailFeatures_df['DATE'].apply(lambda x: x.date())
weiboFeatures_df.index = weiboFeatures_df['DATE'].apply(lambda x: x.date())


# Get labels and process both the features and labels for deep learning.
# Also read the indexes
dataDirName = "/home/crjLambda/PRO80/DailyTDs"
TD_indexes = pd.read_csv(Path(dataDirName, 'ref_TD.csv'))
TD_yields_indexes = pd.read_csv(Path(dataDirName, 'ref_yields.csv'))
TD_Currency_indexes = pd.read_csv(Path(dataDirName, 'ref_Currency.csv'))

# And generate wanted dataset
indexesAll = TD_indexes.join(TD_Currency_indexes, rsuffix='_Currency')
# indexesAll = indexesAll.join(TD_yields_indexes, rsuffix='_yields')
indexesAll_ind = indexesAll.iloc[0,]

# class to get labels and yields feature.----------

indexWanted_CU0 = ['CU0', 'P0', 'Y0', 'AG0', 'BU0', 'ZN0', 'C0', 'AL0', 'RM0', 'M0', 'CF0']
indexWanted_RB0 = ['RB0', 'HC0', 'I0', 'V0', 'BU0', 'JM0', 'UR0', 'FG0', 'MA0', 'SA0', 'SR0']
indexWanted_SCM = ["SCM", 'AU0', 'PG0', 'EB0', 'FU0', 'TA0', 'PP0', 'L0', 'V0', 'LUM', 'RU0']
indexList = list(np.unique(indexWanted_CU0 + indexWanted_RB0 + indexWanted_SCM))

# Adding bond yields as more features.
dataDirName = "/home/crjLambda/PycharmProjects/profoundRNN/data"

SEQUENCE_LEN = 25  # stand for 25 trading days which are very close to one month
SEQUENCE_PAD = 999 # the int used for <pad>

class readingYields:
    def __init__(self, yieldsWanted):
        self.yieldsWanted = yieldsWanted
        self.readFilename = Path(dataDirName, 'yields.xls')
        self.workSheet = self.readFiles()
        self.returnFeatures = self.generateFeatures()

    def readFiles(self):
        yieldsWorkbook = xlrd.open_workbook(self.readFilename)
        workSheet = yieldsWorkbook.sheet_by_index(0)
        return workSheet

    def generateFeatures(self):
        workSheet = self.workSheet
        yieldLambda = 1 # Try to fine tune.
        # # Loading the data.
        yieldsRead = pd.DataFrame(columns=['DATE']).join(pd.DataFrame(columns=[i for i in self.yieldsWanted]))
        for i in yieldsRead.columns:
            if(i == 'DATE'):
                yieldsRead[i] = [pd.Timestamp(dt.value) for dt in workSheet.col(0)[4:-7]]
            elif(i == 'US_10yry'):
                # locate the feature's col number
                for j in range(workSheet.ncols):
                    if (workSheet.row(0)[j].value == i):
                        tmp_x = j
                # Dealing with the data lagging for 1 day if there is any.
                tmp_y = [i.value for i in workSheet.col(tmp_x)[4:-7]]
                if (tmp_y[0] == ''):
                    tmp_y[0] = tmp_y[1]
                yieldsRead[i] = tmp_y
            else:
                # locate the feature's col number
                for j in range(workSheet.ncols):
                    if (workSheet.row(0)[j].value == i):
                        tmp_x = j
                yieldsRead[i] = [i.value for i in workSheet.col(tmp_x)[4:-7]]
        def f(x):
            if x == '':
                return np.nan
            else:
                return x
        yieldsRead = yieldsRead.applymap(f)
        yieldsRead = yieldsRead.dropna()
        # # Generate the yield features.
        returnFeatures = yieldsRead
        # returnFeatures = pd.DataFrame(columns=['DATE']).join(pd.DataFrame(columns=[i for i in self.yieldsWanted]))
        # for i in returnFeatures.columns:
        #     if (i=='DATE'):
        #         returnFeatures[i] = yieldsRead[i][:-1]
        #     else:
        #         close_t = np.array(yieldsRead[i][:-1])
        #         close_tsub1 = np.array(yieldsRead[i][1:])
        #         returnFeatures[i] = [np.log(close_t[j]/close_tsub1[j])*yieldLambda for j in range(len(close_t))]
        return returnFeatures

yieldsWanted = ['CN_10yry', 'US_10yry', 'CN_5yry', 'CN_2yry']
ry = readingYields(yieldsWanted)
featuresYieldsDL_df = ry.returnFeatures
featuresYieldsDL_df.index = featuresYieldsDL_df['DATE'].apply(lambda x: x.date())

# Getting the trading data features.
# Read dataset
dataDirName = "/home/crjLambda/PRO80/DEEPLEARN"
TD_all_dataset = pd.read_csv(Path(dataDirName, 'TD_All.csv'))
indX = TD_all_dataset.columns.values
indX[0] = 'DATE'
TD_all_dataset.columns = indX
# Dataset of Close
indX = ['DATE']
for ind in indexesAll_ind:
    indX.append(ind+'Close')
datasetClose = TD_all_dataset[indX]
# Dataset of hodrick prescott filter's trend product
indX = ['DATE']
for ind in indexesAll_ind:
    indX.append(ind+'_hpft')
datasetTrend = TD_all_dataset[indX]
# Dataset of hodrick prescott filter's cycle product
indX = ['DATE']
for ind in indexesAll_ind:
    indX.append(ind+'_hpfc')
datasetCycle = TD_all_dataset[indX]
# Close dataset
datasetClose_upperPart = datasetClose.iloc[:-10]
datasetClose_lowerPart = datasetClose.iloc[-10:]
# Drop NAs of the upperPart
datasetClose_upperPart_dropna = datasetClose_upperPart.dropna(axis=0)
# Impute the lowerPart NAs
imputed_data_DATE = datasetClose_lowerPart['DATE']
imputed_data_DATE = pd.DataFrame(imputed_data_DATE)
imputed_data_noDATE = datasetClose_lowerPart.drop(columns=['DATE'])
imr = KNNImputer(n_neighbors=2, weights='uniform')
imr = imr.fit(imputed_data_noDATE.values)
imputed_data = imr.transform(imputed_data_noDATE.values)
imputed_data_noDATE = pd.DataFrame(imputed_data, columns=imputed_data_noDATE.columns)
imputed_data_noDATE.index = imputed_data_DATE.index
datasetClose_lowerPart_imputedna = imputed_data_DATE.join(imputed_data_noDATE)
datasetClose = datasetClose_upperPart_dropna.append(datasetClose_lowerPart_imputedna)
X_predict_DATE = pd.to_datetime(datasetClose_lowerPart['DATE'].tail(1).values).date

datasetClose.index = pd.to_datetime(datasetClose['DATE'])
# dataClose_logr = generate_logr(datasetClose)
# dataClose_logr.index = pd.to_datetime(dataClose_logr['DATE']).apply(lambda x: x.date())

# Read Macro Data
class readingMacanamonthly:
    def __init__(self):
        self.dataDirName = "/home/crjLambda/PycharmProjects/profoundRNN/data"
        self.readFilename = Path(self.dataDirName, 'MacAnaMonthly_2.xls')
        self.workSheet = self.readFiles()
        self.returnFeatures = self.generateFeatures()

    def readFiles(self):
        Workbook = xlrd.open_workbook(self.readFilename)
        workSheet = Workbook.sheet_by_index(0)
        return workSheet

    def generateFeatures(self):
        # Transfer the worksheet reading by xlrd to DataFrame.
        workSheet = self.workSheet
        col_names = []
        for j in range(workSheet.ncols):
            if j == 0:
                col_names.append('DATE')
            else:
                col_names.append(workSheet.cell_value(0,j))

        data = np.ones((workSheet.nrows-4, workSheet.ncols))
        data = pd.DataFrame(data)
        for j in range(data.shape[1]):
            for i in range(data.shape[0]):
                try:
                    data.iloc[i, j] = workSheet.cell_value(i+4, j)
                except:
                    print("err occur:", i, j, workSheet.cell_value(i+4, j))
        data.columns = col_names

        def f(x):
            if x == '':
                return np.nan
            else:
                return x

        data = data.applymap(f)
        # copying last months' data if np.nan
        for i in range(data.shape[1]):
            if data.isna().iloc[0, i]:
                data.iloc[0, i] = data.iloc[1, i]

        data = data.dropna()
        # # Generate the yield features.
        returnFeatures = data.copy()
        # returnFeatures = pd.DataFrame(columns=['DATE']).join(pd.DataFrame(columns=[i for i in self.yieldsWanted]))
        # for i in returnFeatures.columns:
        #     if (i=='DATE'):
        #         returnFeatures[i] = yieldsRead[i][:-1]
        #     else:
        #         close_t = np.array(yieldsRead[i][:-1])
        #         close_tsub1 = np.array(yieldsRead[i][1:])
        #         returnFeatures[i] = [np.log(close_t[j]/close_tsub1[j])*yieldLambda for j in range(len(close_t))]
        return returnFeatures

mam = readingMacanamonthly()
mamDL_df = mam.returnFeatures
mamDL_df.index = mamDL_df['DATE'].apply(lambda x: pd.to_datetime(x).date())

def generateDataset():

    # Now I join the trading data, yields, sentiment results and macro analysis data into one dataset.
    # Generating features and labels.
    mamDL = mamDL_df.copy()
    mamDL.index = range(mamDL.shape[0])

    for i in tqdm(range(len(indexesAll_ind)), ncols=100, desc="Generating dataset", colour="blue"):

        # Creating the correlation matrix using rawDataset_logr
        target_inds, daily_timestamps, labels, tradingdata_features, yields_features, sentiment_features \
            = [], [], [], [], [], []

        rawDataset = datasetClose.copy()
        rawDataset_logr = generate_logr(rawDataset, isDATE=True)
        rawCols =rawDataset_logr.columns.delete([0]) # DATE column is excluded
        inds = [rawCols[i]]
        [inds.append(ind) for ind in rawCols[0:i]]
        [inds.append(ind) for ind in rawCols[i+1:]]
        beforeCorr = rawDataset_logr[inds]
        theCorr = beforeCorr.corr() # correlation matrix
        newOrder = theCorr.iloc[0, :].sort_values(axis=0, ascending=False)
        afterCorr = beforeCorr[newOrder.index.to_list()]
        tradingdata_order = afterCorr.shape[1]

        # Join with yields features
        afterCorr = afterCorr.join(featuresYieldsDL_df, rsuffix='_yield')
        afterCorr = afterCorr.drop(columns='DATE')
        yields_order = afterCorr.shape[1]

        # Align with sentiment data
        afterCorr = afterCorr.join(weiboFeatures_df, rsuffix='_weibo')
        afterCorr = afterCorr.drop(columns='DATE')
        afterCorr = afterCorr.join(emailFeatures_df, rsuffix='_email')
        afterCorr = afterCorr.drop(columns='DATE')
        afterCorr = afterCorr.dropna()
        sentiment_order = afterCorr.shape[1]

        # Generating the dataframe of a single target index
        for j in range(afterCorr.shape[0]):
            target_inds.append(rawCols[i])
            daily_timestamps.append(afterCorr.index[j].date())
            labels.append(afterCorr.iloc[j, 0])
            tradingdata_features.append(afterCorr.iloc[j, 0:tradingdata_order].to_list())
            yields_features.append(afterCorr.iloc[j, tradingdata_order:yields_order].to_list())
            sentiment_features.append(afterCorr.iloc[j, yields_order:sentiment_order].to_list())

        singleTargetdataset = pd.DataFrame({
            "target_ind":target_inds,
            "daily_timestamp":daily_timestamps,
            "label":labels,
            "tradingdata_features":tradingdata_features,
            "yields_features":yields_features,
            "sentiment_features":sentiment_features}
        )

        # Add macro analysis data columns to the single target dataset
        singleTargetdataset['DATE'] = ''
        for k in range(singleTargetdataset.shape[0]):
            singleTargetdataset.loc[k, 'DATE'] \
                = datetime.datetime.strftime(singleTargetdataset['daily_timestamp'][k], format="%Y-%m")
        singleTargetdataset = singleTargetdataset.merge(mamDL, on="DATE", how="left")
        # Dealing with missing value, just simple copy the lastest.
        singleTargetdataset = singleTargetdataset.interpolate(method="linear")

        # Creating a continuous time index for PyTorch Forecasting
        singleTargetdataset['time_idx'] = range(singleTargetdataset.shape[0])

        if i == 0:
            returnDataset = singleTargetdataset
        else:
            returnDataset = pd.concat([returnDataset, singleTargetdataset], axis=0)

    return returnDataset

raw_dataset_before_engineering = generateDataset()
# save it into data folder.
raw_dataset_before_engineering.to_parquet(source_data / "preprocessed/raw_dataset_before_engineering.parquet")

# Then we go into the steps of data engineering.
# At the very begining, I am not adding many but only temporal features, yet a few of them are added. It just include
# weekday and business day. Let's see how they perform and then add more.

raw_dataset_with_temporal_features = generateDatefeature(raw_dataset_before_engineering)
# save it into data folder.
raw_dataset_with_temporal_features.to_parquet(source_data / "preprocessed/raw_dataset_with_temporal_feature.parquet")

# Add lags of the labels in the raw dataset.
lags = list(range(SEQUENCE_LEN))
raw_dataset_with_temporal_features_lags, added_features = add_lags(
    raw_dataset_with_temporal_features, lags=lags, column="label", ts_id="target_ind"
)
# save it into data folder.
raw_dataset_with_temporal_features_lags.to_parquet\
    (source_data / "preprocessed/raw_dataset_with_temporal_feature_lags.parquet")

print('Done')