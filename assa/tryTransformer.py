from huggingface_hub import hf_hub_download
import torch
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel

# Initializing a Time Series Transformer configuration with 12 time steps for prediction
configurationInit = TimeSeriesTransformerConfig(prediction_length=12)

# Randomly initializing a model (with random weights) from the configuration
model = TimeSeriesTransformerModel(configurationInit)

# Accessing the model configuration
configuration = model.config

file = hf_hub_download(
    repo_id="kashif/tourism-monthly-batch", filename='train-batch.pt', repo_type='dataset'
)
batch = torch.load(file)

model = TimeSeriesTransformerModel.from_pretrained('huggingface/time-series-transformer-tourism-monthly')

# During training, one provide both past and future values
# as well as possible additional features
outputs = model(
    past_values = batch['past_values'],
    past_time_features = batch['past_time_features'],
    past_observed_mask = batch['past_observed_mask'],
    static_categorical_features = batch['static_categorical_features'],
    static_real_features = batch['static_real_features'],
    future_values = batch['future_values'],
    future_time_features = batch['future_time_features'],
)

last_hidden_state = outputs.last_hidden_state

print("Done")