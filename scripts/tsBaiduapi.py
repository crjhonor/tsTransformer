"""
The following is using baidu AI to predict from the model I've already trained online.
"""
import json
from aip import easydl
from assa import myDailydataset as MD

""" MY APPID AK SK """
APP_ID = '33222388'
API_KEY = 'tKMXrzkmhcO9IPt3isS8AOXr'
SECRET_KEY = 'SZyAlQiFpZuPCke1bZmqcyueIoPVkZ9v'
URL = 'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/time_series/mytry'

client = easydl.EasyDL(APP_ID, API_KEY, SECRET_KEY)


""" Preparing predicting feature """
myDataset = MD.baiduDataset
sequence_len = MD.SEQUENCE_LEN
sequence_pad = MD.SEQUENCE_PAD
X_predict = myDataset.iloc[-sequence_len, :].to_json()

HEADERS = {
    'Content-Type': 'application/json'
}

DATA = json.dumps({
    "include_req": False,
    "data": {
        "fv_0": ["-0.0012534129892353368"],
        "fv_1": ["-0.003988662203504309"],
        "fv_2": ["-0.005236677407745476"],
        "fv_3": ["0.004228800546318335"],
        "fv_4": ["-0.0066533113218081846"],
        "fv_5": ["0.0015104993582954554"],
        "fv_6": ["-0.004283843897840647"],
        "fv_7": ["0.008746861089329962"],
        "fv_8": ["-0.004363867127786866"],
        "Date": ["2023-03-23"]
    }
})

results = client.post(url=URL, data=DATA, headers=HEADERS)

print("Done")