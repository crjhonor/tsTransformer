"""
Further try within the pytorch-forecasting framework with more advanced models. In the previous attempts with the
standard RNN network, current day prediction had pretty good result, but one day ahead forecasting was quite bias and
didn't achieve good result at all.
"""
import matplotlib.pyplot as plt
import plotly.io as pio
pio.templates.default = "plotly_white"
import plotly.express as px
import pandas as pd
from statsmodels.tsa.stattools import acf
import numpy as np
from collections import namedtuple
import os
import datetime as dt
from pathlib import Path
from tqdm import tqdm
import torch

from assa import readDailydataset as readD
rawDataset = readD.for_training()
from assa.stationary_utils import check_unit_root, check_trend, check_heteroscedastisticity
from assa.forecasting.forecasting import calculate_metrics


# Let's try to have some visualizing of the label timeseries data
plot_df = pd.DataFrame({"Time":range(rawDataset.shape[0]), "Timeseries 1":rawDataset['label']})
# fig = px.line(plot_df, x="Time", y="Timeseries 1",
#               title="The Raw Plot of the 'label', which is the log return of daily trading close.")
# fig.update_yaxes(matches=None)
# fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font=dict(size=16)))
# fig.show()

# Unit root test.
y_unit_root = plot_df['Timeseries 1']
adfuller_res, kpss_res = check_unit_root(y_unit_root, confidence=0.05)
print(f"ADF Stationary: {adfuller_res.stationary} | p-value: {adfuller_res.results[1]}")
print(f"KPSS Stationary: {kpss_res.stationary} | p-value: {kpss_res.results[1]}")

# Check trend.
kendall_tau_res = check_trend(y_unit_root, confidence=0.05)
mann_kendall_res = check_trend(y_unit_root, confidence=0.05, mann_kendall=True)
print(f"Kendalls Tau: Trend: {kendall_tau_res.trend} | Direction: {kendall_tau_res.direction} | {kendall_tau_res.deterministic}")
print(f"Mann-Kendalls: Trend: {mann_kendall_res.trend} | Direction: {mann_kendall_res.direction} | {mann_kendall_res.deterministic}")

# Check seasonality.
r = acf(y_unit_root, nlags=60, fft=False)
r = r[1:]
plot_df = pd.DataFrame(dict(x=np.arange(len(r))+1, y=r))
plot_df['seasonal_lag'] = False
plot_df.loc[plot_df["x"].isin([25,50]), "seasonal_lag"] = True
# fig = px.bar(plot_df, x="x", y="y", pattern_shape="seasonal_lag", color="seasonal_lag", title="Auto-Correlation Plot")
# fig.add_annotation(x=25, y=r[24], text="Lag 25")
# fig.add_annotation(x=50, y=r[49], text="Lag 50")
# fig.update_layout(
#             showlegend = False,
#             autosize=False,
#             width=900,
#             height=500,
#             title={
#             'x':0.5,
#             'xanchor': 'center',
#             'yanchor': 'top'},
#             titlefont={
#                 "size": 20
#             },
#             yaxis=dict(
#                 title_text="Auto Correlation",
#                 titlefont=dict(size=15),
#                 tickfont=dict(size=15),
#             ),
#             xaxis=dict(
#                 title_text="Lags",
#                 titlefont=dict(size=15),
#                 tickfont=dict(size=15),
#             )
#         )
# fig.update_annotations(font_size=15)
# fig.show()

# Heteroscedasticity
hetero_res = check_heteroscedastisticity(y_unit_root, confidence=0.05)
print(f"White Test for Heteroscedasticity: {hetero_res.heteroscedastic} with a p-value of {hetero_res.lm_p_value}")

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

# Creating a continuous time index for PyTorch Forecasting
# # For accessing the performance of different models, it is reasonable to drop the latest data with a sacrifice of
# most uptodate feature injections to become the test dataset. Then lately, when I am using the data to daily practice,
# I will ignore the test dataset.
# Take the latest 20 days data to be the test dataset.
currDate = rawDataset.daily_timestamp.max() - dt.timedelta(days=20)
test_mask = (rawDataset.daily_timestamp >= currDate)
train_df = rawDataset[~test_mask]
test_df = rawDataset[test_mask]
# Combining train and test with a flag
train_df['train'] = True
test_df['train'] = False

# Converting the categoricals to 'object' dtype
for itm in feat_config.static_categoricals + feat_config.time_varying_known_categoricals:
    train_df[itm] = train_df[itm].astype("object")
    test_df[itm] = test_df[itm].astype("object")

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
        'label',
    ],
    target_normalizer=GroupNormalizer(
        groups=feat_config.group_ids, transformation=None
    )
)
# Defining the validation dataset with the same parameters as training
validation = TimeSeriesDataSet.from_dataset(training, pd.concat([val_history, val_df]).reset_index(drop=True),
                                            stop_randomization=True)
# Define the test dataset with the same parameters as training
test = TimeSeriesDataSet.from_dataset(training, pd.concat([hist_df, test_df]).reset_index(drop=True),
                                      stop_randomization=True,
                                      allow_missing_timesteps=True)
# Making the dataloaders
# num_workers can be increased in linux to speed-up training
train_dataloader = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

# Testing the dataloader
x, y = next(iter(train_dataloader))
print("\nsizes of x =")
for key, value in x.items():
    print(f"\t{key} = {value.size()}")
print("\nsize of y =")
print(f"\ty = {y[0].size()}")

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
save_model_full = Path(os.getcwd()).parent / f'assa/model/saved_weights/{tag}.wt'
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
pred, index = best_model.predict(test, return_index=True, show_progress_bar=True)
index[[f"{tag}_step_{i}" for i in range(pred.shape[-1])]] = pred
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

# Evaluating the forecast
def evaluate_forecast(pred_df, train_data, fc_column, name, target_name='label'):
    metric_l = []
    for _id in tqdm(pred_df.index.get_level_values(0).unique(), desc="Calculating metrics..."):
        target = pred_df.xs(_id)[[target_name]]
        _y_pred = pred_df.xs(_id)[[fc_column]]
        history = train_data.xs(_id)[[target_name]]
        metric_l.append(
            calculate_metrics(target, _y_pred, name=name, y_train=history)
        )
    eval_metrics_df = pd.DataFrame(metric_l)
    agg_metrics = {
        "Algorithm": name,
        "MAE": np.nanmean(np.abs(pred_df[fc_column]-pred_df[target_name])),
        "MSE": np.nanmean(np.power(pred_df[fc_column]-pred_df[target_name], 2)),
        # "meanMASE": eval_metrics_df.loc[: "MASE"].mean(),
        "Forecast Bias": 100*(np.nansum(pred_df[fc_column])-np.nansum(pred_df[target_name]))/np.nansum([pred_df[target_name]])
    }
    return agg_metrics, eval_metrics_df

agg_metrics, eval_metrics_df = evaluate_forecast(
    pred_df=pred_df,
    train_data=full_df,
    fc_column=f"{tag}_step_0",
    name=f"{tag}_step_0",
    target_name='label'
)

metric_record.append(agg_metrics)
individual_metrics[tag] = eval_metrics_df
performance_show = pd.DataFrame(metric_record)
performance_show.to_csv(Path(os.getcwd()).parent / f'assa/model/performance_metrics/{tag}_metric_record.csv')
pred_df.to_csv(Path(os.getcwd()).parent / f'outputs/pred_dfs/{tag}_metric_label.csv')

print(pred_df.xs('V0Close'))
print(performance_show)
print("Done")