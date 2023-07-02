import math
from assa import readDailydataset as readD
import pandas as pd
import numpy as np
import math
from collections import namedtuple
import os
import datetime as dt
from pathlib import Path
from tqdm import tqdm
from assa.utils import http_request as hr
from assa.utils.http_request import fillHifreq
import matplotlib.pyplot as plt

# Obtain the raw dataset for prediction up to the last trading day.
targets = ['TA0']
rawDataset, lastDay_fullprice, lastDay_newOrder = readD.for_continous_prediction(targets)

"""
Creating dataloader and model, train and forecast.
"""
FeatureConfig = namedtuple(
    "FeatureConfig",
    [
        "target",
        "index_cols",
        "static_categoricals",
        "static_reals",
        "time_varying_known_categoricals",
        "time_varying_known_reals",
        "time_varying_unkown_reals",
        "group_ids"
    ],
)

# Define the different features
feat_config = FeatureConfig(
    target="label",
    index_cols=['target_ind', 'daily_timestamp'],
    static_categoricals=[
        "target_ind"
    ], # Categoricals which does not change with time
    static_reals=[], # Reals which does not change with time
    time_varying_known_categoricals=[  # Categories which change with time
        'CNPMI',
        'CAIXINPMI',
        'USPMI',
        'EUPMI',
        'UKPMI',
        'cncpi',
        'uscpi',
        'eucpi',
        'ukcpi',
        'CN_PPI',
        'US_PPI',
        'CNSPMI',
        'USPMI',
        'EUSPMI',
        'weekday',
        'businessday',
    ],
    time_varying_known_reals=[  # Reals which change with time
       'tf_0',
        'tf_1',
        'tf_2',
        'tf_3',
        'tf_4',
        'tf_5',
        'tf_6',
        'tf_7',
        'tf_8',
        'tf_9',
        'tf_10',
        'tf_11',
        'tf_12',
        'tf_13',
        'tf_14',
        'tf_15',
        'tf_16',
        'tf_17',
        'tf_18',
        'tf_19',
        'tf_20',
        'tf_21',
        'tf_22',
        'tf_23',
        'tf_24',
        'tf_25',
        'tf_26',
        'tf_27',
        'tf_28',
        'tf_29',
        'tf_30',
        'tf_31',
        'tf_32',
        'tf_33',
        'tf_34',
        'tf_35',
        'tf_36',
        'tf_37',
        'tf_38',
        'tf_39',
        'tf_40',
        'tf_41',
        'tf_42',
        'tf_43',
        'tf_44',
        'tf_45',
        'tf_46',
        'tf_47',
        'tf_48',
        'tf_49',
        'tf_50',
        'tf_51',
        'yf_52',
        'yf_53',
        'yf_54',
        'yf_55',
        'sf_56',
        'sf_57',
        'sf_58',
        'sf_59',
        'sf_60',
        'sf_61',
        'sf_62',
        'sf_63',
        'label_lag_0',
        'label_lag_1',
        'label_lag_2',
        'label_lag_3',
        'label_lag_4',
        'label_lag_5',
        'label_lag_6',
        'label_lag_7',
        'label_lag_8',
        'label_lag_9',
        'label_lag_10',
        'label_lag_11',
        'label_lag_12',
        'label_lag_13',
        'label_lag_14',
        'label_lag_15',
        'label_lag_16',
        'label_lag_17',
        'label_lag_18',
        'label_lag_19',
        'label_lag_20',
        'label_lag_21',
        'label_lag_22',
        'label_lag_23',
        'label_lag_24',
    ],
    time_varying_unkown_reals=[  # Reals which change with time, but we don`t have the future. Like the target
        'label'
    ],
    group_ids=[  # Feature or list of features which uniquely identifies each entity
        'target_ind'
    ]
)

# Take the latest month`s data to be the test dataset.
currDate = dt.datetime.now() - dt.timedelta(days=2)
test_mask = (pd.to_datetime(rawDataset.daily_timestamp.values).year == currDate.year) \
            & (pd.to_datetime(rawDataset.daily_timestamp.values).month == currDate.month)
train_df = rawDataset[~test_mask]
test_df = rawDataset[test_mask]
# Combining train and test with a flag
train_df['train'] = True
test_df['train'] = False

# Converting the categoricals to 'object' dtype
for itm in feat_config.static_categoricals + feat_config.time_varying_known_categoricals:
    train_df[itm] = train_df[itm].astype("object")
    test_df[itm] = test_df[itm].astype("object")

# def chunks(L, n):
#     """
#     Yield successive n-sized chunks from L.
#     :param L:
#     :param n:
#     :return:
#     """
#     returnList = []
#     for i in range(0, len(L), n):
#         returnList.append(L[i:i+n])
#     return returnList

# indexesAll_ind_list = indexesAll_ind.to_list()
# requests_list = chunks(indexesAll_ind_list, 10)

# To eliminate negatives, try to add 1.
# altered_cols = [feat_config.target] + feat_config.time_varying_known_reals
# altered_cols = altered_cols[0:57] + altered_cols[65:]
# train_df[altered_cols] = train_df[altered_cols].applymap(lambda x: x+0.1)
# test_df[altered_cols] = test_df[altered_cols].applymap(lambda x: x+0.1)

# Training Global Models
import pytorch_lightning as pl
pl.seed_everything(42)
import torch

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import RMSE, MAE

# Config
MAX_PREDICTION_LENGTH = 1
MAX_ENCODER_LENGTH = 16  # The dataset including sentiment data are too small for larger length.
BATCH_SIZE = 32

metric_record = []
individual_metrics = dict()

# Creating dataframes for train, val and test
# Adding 2 days of history to create the samples
history_cutoff = train_df.daily_timestamp.max() - pd.Timedelta(10, "D")
hist_df = train_df[train_df.daily_timestamp>history_cutoff]
print(f"History Min: {hist_df.daily_timestamp.min()} | Max: {hist_df.daily_timestamp.max()} "
      f"| Length: {len(hist_df.daily_timestamp.unique())}")

# Keeping 1 days aside as a validation set
cutoff = train_df.daily_timestamp.max() - pd.Timedelta(10, "D")
# Adding 2 days of history to create the samples
history_cutoff = train_df.daily_timestamp.max() - pd.Timedelta(30, "D")
val_history = train_df[(train_df.daily_timestamp>=history_cutoff)&(train_df.daily_timestamp<=cutoff)].reset_index(drop=True)
val_df = train_df[train_df.daily_timestamp>cutoff].reset_index(drop=True)
train_df = train_df[train_df.daily_timestamp<=cutoff].reset_index(drop=True)
print("Split Timestamps:")
print(f"Train Max: {train_df.daily_timestamp.max()} "
      f"| Val History Min and Max: {val_history.daily_timestamp.min(), val_history.daily_timestamp.max()} "
      f"| Val Min and Max: {val_df.daily_timestamp.min(), val_df.daily_timestamp.max()}")
print(f"Val History Size: {len(val_history.daily_timestamp.unique())} | Val Size: {len(val_df.daily_timestamp.unique())}")

pred_df = test_df[feat_config.index_cols+[feat_config.target]+['time_idx']].copy()
cols = feat_config.index_cols + [feat_config.target]
full_df = pd.concat(
    [
        train_df[cols],
        val_df[cols],
    ]
).set_index(feat_config.index_cols)

'''
Training TFT
'''
tag = 'TFT'

# Converting data into TimeSeriesDataset from PyTorch Forecasting

# Defining the training dataset
training = TimeSeriesDataSet(
    train_df,
    time_idx='time_idx',
    target=feat_config.target,
    group_ids=feat_config.group_ids,
    max_encoder_length=MAX_ENCODER_LENGTH,
    max_prediction_length=MAX_PREDICTION_LENGTH,
    static_categoricals=feat_config.static_categoricals,
    static_reals=feat_config.static_reals,
    time_varying_unknown_categoricals=feat_config.time_varying_known_categoricals,
    time_varying_known_reals=feat_config.time_varying_known_reals,
    time_varying_unknown_reals=[
        'label_T1',
    ],
    target_normalizer=GroupNormalizer(
        groups=feat_config.group_ids, transformation=None
    )
)
# Defining the validation dataset with the same parameters as training
validation = TimeSeriesDataSet.from_dataset(training, pd.concat([val_history, val_df]).reset_index(drop=True),
                                            stop_randomization=True)
# Making the dataloaders
# num_workers can be increased in linux to speed-up training
train_dataloader = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

# Defining the embedding sizes for each categorical variable
# Using a thumbrule to calculate  the embedding sizes
# Finding the cardinality using the categorical encoders in the dataset
cardinality = [len(training.categorical_encoders[c].classes_) for c in training.categoricals]
# Using the cardinality list to create embedding sizes
embedding_sizes = {
    col: (x, min(50, (x+1) // 2)) for col, x in zip(training.categoricals, cardinality)
}
"""
Model IV, the TemporalFusionTransformer Model
"""

# Importing the skeleton and helper models
from pytorch_forecasting.models import TemporalFusionTransformer

model_params = {
    "hidden_size": 128,
    "lstm_layers": 2,
    "attention_head_size": 4,
    "hidden_continuous_size": 128,
    "embedding_sizes": embedding_sizes
}

other_params = dict(
    learning_rate=1e-5,
    optimizer="adam",
    loss=RMSE(),
    logging_metrics=[RMSE(), MAE()],
)

model = TemporalFusionTransformer.from_dataset(
    training, **{**model_params, **other_params}
)

# Testing out the model
x, y = next(iter(train_dataloader))
_ = model(x)
print(type(_))
print(_.prediction.shape)

# Training the model
save_model_full = f'/home/crjLambda/PycharmProjects/tsTransformer/assa/model/saved_weights/{tag}.wt'
train_model = False

if train_model:
    trainer = pl.Trainer(
        auto_select_gpus=True,
        gpus=-1,
        min_epochs=1,
        max_epochs=100,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor='val_loss', patience=3),
            pl.callbacks.ModelCheckpoint(
                monitor='val_loss', save_last=True, mode='min', auto_insert_metric_name=True
            ),
        ],
        val_check_interval=1.0,
        log_every_n_steps=50,
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    # Loading the best model
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    torch.save(best_model.state_dict(), save_model_full)
else:
    best_model_path = save_model_full
    model.load_state_dict(torch.load(save_model_full))
    best_model = model
    print(f"Skipping Training and loading the model from {best_model_path}")

# Predicting on the test dataset and storing in a df

requests_list = [['CFFEXIF', 'CFFEXIH', 'SHFECU', 'CFFEXT', 'CFFEXTF', 'NYMEXCL', 'SHFEAU', 'SHFEAG', 'SHFEHC', 'CZCETA'],
                 ['DCEL', 'CZCECF', 'CZCEUR', 'DCEJM', # 'CBOTTY',
                    'SHFEAL', 'CZCEMA', # 'CBOTFV',
                    'DCEV', 'SHFERB', 'SHFEFU', 'SHFEBU'],
                 ['SHFECU', # 'LCPT',
                    'SHFERU', 'DCELH', 'CBOTZS', 'CZCEPK', 'DCEC', 'CZCESR', 'DCEPP', 'CZCEFG', "CZCESA"],
                 ['DCEP', 'DCEI', 'DCEM', 'DCEPG', 'DCEY', 'CZCERM', 'DCEEB', 'DCEEG', 'INESC', 'SPX'],
                 ['FTSE', 'INELU', 'USDX', 'USDCNH', 'GBPUSD', 'EURUSD', 'AUDUSD', 'USDJPY', 'AUDCAD']]
origin_requests_list = [['IF00C1', 'IH00C1', 'CU0', 'T00C1', 'TF00C1', 'CL00Y', 'AU0', 'AG0', 'HC0', 'TA0'],
                        ['L0', 'CF0', 'UR0', 'JM0', # 'TY00Y',
                            'AL0', 'MA0', # 'FV00Y',
                            'V0', 'RB0', 'FU0', 'BU0'],
                        ['ZN0', # 'LCPT',
                            'RU0', 'LH0', 'ZS00Y', 'PK0', 'C0', 'SR0', 'PP0', 'FG0', 'SA0'],
                        ['P0', 'I0', 'M0', 'PG0', 'Y0', 'RM0', 'EB0', 'EG0', 'SCM', 'SPX'],
                        ['FTSE', 'LUM', 'USDX', 'USDCNH', 'GBPUSD', 'EURUSD', 'AUDUSD', 'USDJPY', 'AUDCAD']]

rq = hr.readingHifreq()

def _pred_visualize(targets: list, Hifreq: float, Pred: float, Baseline_price: float):
    # Prepare the dataframe for plotting
    target = targets[0]
    Hifreq = Hifreq-Baseline_price
    Pred = Pred-Baseline_price
    df = pd.DataFrame({
        'name': ['Hifreq', 'Pred'],
        'data': [Hifreq, Pred]
    })
    fig = df.plot.barh(x='name', y='data')
    plt.grid(True)
    plt.xlabel('Price different from last close.')
    plt.ylabel('Prediction VS Hi Frequency Data')
    plt.title(f'{target} Latest')
    plt.show()

def pred_compare(targets: list):
    for i in range(len(requests_list)):
        itm = requests_list[i]
        itm_origin = origin_requests_list[i]
        results = rq.reqGet(itm, itm_origin)
        if i == 0:
            latestHifreq = results
        else:
            latestHifreq = pd.concat([latestHifreq, results])

    latestHifreq = latestHifreq.reset_index()

    target_ind = ''.join([targets[0], 'Close'])
    for_pred_df = test_df[test_df.target_ind == target_ind]
    for_pred_df = fillHifreq(for_pred_df, latestHifreq, lastDay_fullprice, lastDay_newOrder, target_ind, isNighttrade=False)

    # Converting the categoricals to 'object' dtype
    for itm in feat_config.static_categoricals + feat_config.time_varying_known_categoricals:
        for_pred_df[itm] = for_pred_df[itm].astype("object")

    # Define the test dataset with the same parameters as training
    test = TimeSeriesDataSet.from_dataset(training, pd.concat([hist_df, for_pred_df]).reset_index(drop=True),
                                          stop_randomization=True,
                                          allow_missing_timesteps=True)

    pred, index = best_model.predict(test, return_index=True, show_progress_bar=True)
    index[[f"{tag}_step_{i}" for i in range(pred.shape[-1])]] = pred
    pred_df = for_pred_df[feat_config.index_cols+[feat_config.target]+['time_idx']].copy()
    pred_df = (
        pred_df.reset_index(drop=True)
        .merge(
            index[["time_idx", "target_ind", f'{tag}_step_0']],
            on=['time_idx', 'target_ind'],
            how='left',
        ).dropna(subset=['target_ind', f'{tag}_step_0'])
        .set_index(feat_config.index_cols)
    )

    pred_df[f"{tag}_step_0"] = pred_df[f"{tag}_step_0"].astype(float)

    # Let's see the Hifreq data is whether stronger or weaker than the model's prediction.
    currentprice_Hifreq = np.array(latestHifreq['price'][latestHifreq['target_ind'] == targets[0]].astype(float))
    currentprice_Hifreq = round(currentprice_Hifreq[0])
    # Calculate the model's prediction into price
    pred_logr = pred_df[f"{tag}_step_0"][-1]
    lastDay_price = lastDay_fullprice.loc[''.join([targets[0], 'Close'])]
    pred_calc_price = round(math.e**pred_logr * lastDay_price)
    # Print output.
    currentTime = dt.datetime.now()
    print(f"Target: [{targets[0]}] at time {currentTime.strftime(format='%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    print(f"Current Price: [{currentprice_Hifreq}]")
    print(f"Calculated Price: [{pred_calc_price}]")
    if currentprice_Hifreq > pred_calc_price:
        print(f"It is [STRONG], you should be [LONGING].")
    elif currentprice_Hifreq < pred_calc_price:
        print(f"It is [WEAK], you should be [SHORTING].")
    print("="*50)

    # visualize the results
    _pred_visualize(targets, currentprice_Hifreq, pred_calc_price, lastDay_price)

pred_compare(targets)