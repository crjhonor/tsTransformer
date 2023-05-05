"""
The following is using baidu AI to predict from the model I've already trained online.
"""
from aip import easydl

""" MY APPID AK SK """
APP_ID = '33154825'
API_KEY = 'YPUx1kjyU8FLZrvbH13N0nwT'
SECRET_KEY = 'N3WTrY11yCoHsiVXu2dvw6dWQRt3FcVm'

client = easydl.EasyDL(APP_ID, API_KEY, SECRET_KEY)

print("Done")