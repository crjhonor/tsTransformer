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
indexWanted_RB0 = ['RB0', 'HC0', 'I0', 'V0', 'BU0', 'PP0', 'L0', 'FG0', 'MA0', 'SA0', 'SR0']
indexWanted_SCM = ["SCM", 'AU0', 'PG0', 'EB0', 'FU0', 'TA0', 'PP0', 'L0', 'V0', 'MA0', 'RU0']
indexList = list(np.unique(indexWanted_CU0 + indexWanted_RB0 + indexWanted_SCM))

# Adding bond yields as more features.
dataDirName = "/home/crjLambda/PycharmProjects/profoundRNN/data"

SEQUENCE_LEN = 25  # stand for 25 trading days which are very close to one month

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
def generate_logr(dataset, isDATE=True):
    if isDATE:
        dataset_DATE = dataset['DATE']
        dataset_noDATE = dataset.drop(columns=['DATE'])
    else:
        dataset_noDATE = dataset

    dataset_noDATE_pct_change = dataset_noDATE.pct_change(periods=1)
    dataset_noDATE_pct_change = dataset_noDATE_pct_change.iloc[1:]
    dataset_noDATE_logr = dataset_noDATE_pct_change.applymap(lambda x: np.log(x + 1))

    if isDATE:
        dataset_DATE = pd.DataFrame(dataset_DATE.iloc[1:])
        returnDataset = dataset_DATE.join(dataset_noDATE_logr)
    else:
        returnDataset = dataset_noDATE_logr
    return returnDataset

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

def get_bussiness_days(current_date: datetime.date):
    last_day = calendar.monthrange(current_date.year, current_date.month)[1]
    rng = pd.date_range(current_date.replace(day=1), periods=last_day, freq='D')
    business_days = pd.bdate_range(rng[0], rng[-1])
    for n, b in enumerate(business_days.date == current_date):
        if b:
            business_day = n
    return business_day

def generateDatefeature(dataset):
    '''
    Day of the working days in particular month can be very important. Thus I am trying to generate these features.
    :param dataset: Input dataset wanted to generate the date feature
    :return: Return dataset with date feature
    '''
    returnDataset = dataset.copy()
    # Adding year, month and day
    returnDataset['day'] = dataset.index.day.values
    returnDataset['month'] = dataset.index.month.values
    returnDataset['year'] = dataset.index.year.values
    # Adding weekdays and businessdays
    weekdays = []
    businessdays = []
    for i in range(len(returnDataset)):
        weekday = calendar.weekday(returnDataset['year'].values[i], returnDataset['month'].values[i],
                         returnDataset['day'].values[i])
        businessday = get_bussiness_days(returnDataset.index[i].date())
        weekdays.append(weekday)
        businessdays.append(businessday)
    returnDataset['weekday'] = weekdays
    returnDataset['businessday'] = businessdays
    return returnDataset


def generateDataset():
    # Now I join the trading data, yields and sentiment results into one dataset.
    rawDataset = datasetClose.copy()
    rawDataset = rawDataset.join(featuresYieldsDL_df, rsuffix='_yield')
    rawDataset = rawDataset.drop(columns='DATE_yield')
    for i in tqdm(range(len(indexesAll_ind)), ncols=100, desc="Generating dataset", colour="blue"):
        ind = [indexesAll_ind[i]]
    # myDataset = myDataset.join(weiboFeatures_df, rsuffix='_weibo')
    # myDataset = myDataset.drop(columns='DATE_weibo')
    # myDataset = myDataset.join(emailFeatures_df, rsuffix='_email')
    # myDataset = myDataset.drop(columns='DATE_email')
    # Just drop all NaNs
    # myDataset = myDataset.dropna()
    # Generate date feature
    # myDataset = generateDatefeature(myDataset)
    # return myDataset

generateDataset()