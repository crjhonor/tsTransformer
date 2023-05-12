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
    returnDataset['year'] = dataset.index.year.values
    returnDataset['month'] = dataset.index.month.values
    returnDataset['day'] = dataset.index.day.values
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

def myPadding(sequence):
    lengths = [len(s) for s in sequence]
    max_lenbth = max(lengths)
    reSequence = []
    for s in sequence:
        if len(s) == max_lenbth:
            reSequence.append(s)
        else:
            num_padding = max_lenbth-len(s)
            if type(s[0]) == float:
                sequence_pad =SEQUENCE_PAD
                for i in range(num_padding):
                    s.append(sequence_pad)
            else:
                sequence_pad = list(np.repeat(SEQUENCE_PAD, len(s[0])))
                for i in range(num_padding):
                    s.append(sequence_pad)
            reSequence.append(s)
    return reSequence

def generateDataset():
    # Now I join the trading data, yields and sentiment results into one dataset.
    rawDataset = datasetClose.copy()
    rawDataset = rawDataset.join(featuresYieldsDL_df, rsuffix='_yield')
    rawDataset = rawDataset.drop(columns='DATE_yield')
    # Log return is required to perform correlation algorithm etc.
    rawDataset_logr = generate_logr(rawDataset, isDATE=True)

    # Generating features and labels.
    past_values, future_values, past_time_features, future_time_features = [], [], [], []
    for i in tqdm(range(len(indexesAll_ind)), ncols=100, desc="Generating dataset", colour="blue"):

        # Creating the correlation matrix using rawDataset_logr
        rawCols =rawDataset_logr.columns.delete([0]) # DATE column is excluded
        inds = [rawCols[i]]
        [inds.append(ind) for ind in rawCols[0:i]]
        [inds.append(ind) for ind in rawCols[i+1:]]
        beforeCorr = rawDataset_logr[inds]
        theCorr = beforeCorr.corr() # correlation matrix
        newOrder = theCorr.iloc[0, :].sort_values(axis=0, ascending=False)
        afterCorr = beforeCorr[newOrder.index.to_list()]

        # Align with sentiment data
        sentiment_order = afterCorr.shape[1]
        afterCorr = afterCorr.join(weiboFeatures_df, rsuffix='_weibo')
        afterCorr = afterCorr.drop(columns='DATE')
        afterCorr = afterCorr.join(emailFeatures_df, rsuffix='_email')
        afterCorr = afterCorr.drop(columns='DATE')
        afterCorr = afterCorr.dropna()

        # Generate Date Time features
        withDatefeature = generateDatefeature(afterCorr)
        date_order = afterCorr.shape[1]

        # Generate all the features we need.
        for l in range(withDatefeature.shape[0]):
            # past value and past time feature
            past_value = withDatefeature.iloc[l, :date_order]
            past_value = past_value.to_list()
            past_time_feature = withDatefeature.iloc[l, date_order:]
            past_time_feature = past_time_feature.squeeze(axis=0)
            past_time_feature = past_time_feature.astype(np.int32)
            past_time_feature = past_time_feature.to_list()
            # feature value and future time feature
            if (l + SEQUENCE_LEN) <= withDatefeature.shape[0]:
                future_value = withDatefeature.iloc[l:l+SEQUENCE_LEN, 0]
                future_time_feature = withDatefeature.iloc[l:l+SEQUENCE_LEN, date_order:]
            else:
                future_value = withDatefeature.iloc[l:, 0]
                future_time_feature = withDatefeature.iloc[l:, date_order:]
            future_value = future_value.to_list()
            future_time_feature = [future_time_feature.iloc[r, :].to_list() for r in range(future_time_feature.shape[0])]
            # Append theses features.
            past_values.append(past_value)
            past_time_features.append(past_time_feature)
            future_values.append(future_value)
            future_time_features.append(future_time_feature)

    print('past values length: {}'.format(len(past_values)))
    print('past time features length: {}'.format(len(past_time_features)))
    print('future values length: {}'.format(len(future_values)))
    print('future time features: {}'.format(len(future_time_features)))

    # One more problem before output to deep learning is that the future values may contain shorter length than SEQUENCE
    # _LEN than I have to padding them using 999 as the <pad> number.
    future_values = myPadding(future_values)
    future_time_features = myPadding(future_time_features)

    # Consider saving all the features into csv file. Before that converting them into DataFrame.
    returnDataframe = pd.DataFrame({
        'future_values': future_values,
        'past_values': past_values,
        'future_time_features': future_time_features,
        'past_time_features': past_time_features}
    )
    returnDataframe.to_csv('/home/crjLambda/PycharmProjects/profoundRNN/dataForlater/myDatasetfordeeplearning.csv',
                           index=False)

    # While attempt to use in baidu AI, this is to generate the necessary dataset formate.
    l_fv = len(future_values[0])
    baiduDataset_colnames = [''.join(['fv_', str(i)]) for i in range(l_fv)]
    l_pv = len(past_values[0])
    for i in range(l_pv):
        baiduDataset_colnames.append(''.join(['pv_', str(i)]))
    l_ftf_i = len(future_time_features[0])
    l_ftf_j = len(future_time_features[0][0])
    for i in range(l_ftf_i):
        for j in range(l_ftf_j):
            baiduDataset_colnames.append(''.join(['ftf_', str(i), '_', str(j)]))
    l_ptf = len(past_time_features[0])
    for i in range(l_ptf):
        baiduDataset_colnames.append(''.join(['ptf_', str(i)]))
    l_dataset = len(future_values)
    baiduDataset = pd.DataFrame(index=range(l_dataset), columns=baiduDataset_colnames)
    for l in tqdm(range(l_dataset), ncols=100, desc="filling baidu dataset", colour="yellow"):
        for i in range(l_fv):
            baiduDataset.iloc[l, i] = future_values[l][i]
        for i in range(l_pv):
            baiduDataset.iloc[l, i+l_fv] = past_values[l][i]
        for i in range(l_ftf_i):
            for j in range(l_ftf_j):
                baiduDataset.iloc[l, (l_fv+l_pv+i*l_ftf_j+j)] = future_time_features[l][i][j]
        for i in range(l_ptf):
            baiduDataset.iloc[l, (l_fv+l_pv+l_ftf_i*l_ftf_j+i)] = past_time_features[l][i]

    # I found biadu AI cannot deal with a dataset of so many columns and it needs a DATE column!!!
    # baiduDataset_slim = baiduDataset.iloc[:, :l_fv+l_pv]
    baiduDataset_slim = baiduDataset.iloc[:, :9] # try 9 columns first
    baiduDataset_slim['Date'] = [
        datetime.datetime(year=baiduDataset.loc[i, 'ptf_0'],
                          month=baiduDataset.loc[i, 'ptf_1'],
                          day=baiduDataset.loc[i, 'ptf_2']).date() for i in range(l_dataset)
    ]

    baiduDataset_slim.to_csv(
        '/home/crjLambda/PycharmProjects/profoundRNN/dataForlater/myDatasetforbaidu.csv',
        index=False
    )

    return baiduDataset_slim

baiduDataset = generateDataset()
# print('Done')