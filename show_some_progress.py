"""
With the idea of predicting according the Hi Frequency data, I can predict in day results and compare with the latest
trading data to take actions.
And, using the tkinter framework, it can be more and more visual able about this some progress.
"""

"""
The tkinter framework.
"""
import time
import datetime as dt
import numpy as np
import pandas as pd
import math
import tkinter as tk
from tkinter import ttk
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
matplotlib.style.use('ggplot')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
FigureCanvasTkAgg,
NavigationToolbar2Tk
)
from scripts import TFT_continuous_prediction as TFTCP

targets = ['Y0', 'TA0', 'SA0', 'RU0',
           'SR0', 'V0', 'FG0', 'CF0']

# Define extra tkinter class
class BoundText(tk.Text):
    def __init__(self, *args, textvariable=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._variable = textvariable
        if self._variable:
            self.insert('1.0', self._variable.get())
            self._variable.trace_add('write', self._set_content)
            self.bind('<<Modified>>', self._set_var)

    def _set_content(self, *_):
        self.delete('1.0', tk.END)
        self.insert('1.0', self._variable.get())

    def _set_var(self, *_):
        if self.edit_modified():
            content = self.get('1.0', 'end-1chars')
            self._variable.set(content)
            self.edit_modified(False)

class LabelInput(tk.Frame):
    def __init__(self, parent, label, var, input_class=tk.Entry,
                 input_args=None, label_args=None, **kwargs):
        super().__init__(parent, **kwargs)
        input_args = input_args or {}
        label_args = label_args or {}
        self.variable = var
        self.variable.label_widget = self

        if input_class in (ttk.Checkbutton, ttk.Button):
            input_args['text'] = label
        else:
            self.label = ttk.Label(self, text=label, **label_args)
            self.label.grid(row=0, column=0, sticky=(tk.W+tk.E))

        if input_class in (
            ttk.Checkbutton, ttk.Button, ttk.Radiobutton
        ):
            input_args['variable'] = self.variable
        else:
            input_args['textvariable'] = self.variable

        # setup the input
        if input_class == ttk.Radiobutton:
            # for Radiobutton, create one input per value
            self.input = tk.Frame(self)
            for v in input_args.pop('values', []):
                button = ttk.Radiobutton(
                    self.input, value=v, text=v, **input_args
                )
                button.pack(side=tk.LEFT, ipadx=10, ipady=2, expand=True, fill='x')
        else:
            self.input = input_class(self, **input_args)

        self.input.grid(row=1, column=0, sticky=(tk.W+tk.E))
        self.columnconfigure(0, weight=1)

    def grid(self, sticky=(tk.E + tk.W), **kwargs):
        super().grid(sticky=sticky, **kwargs)

class processingWindow(ttk.Frame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vars = {
            'time interval': tk.IntVar(),
            'stop flag': tk.BooleanVar(),
        }

        self.columnconfigure(1, weight=1)

        self._vars['time interval'].set(value=5)
        self._vars['stop flag'].set(value=True)

        top_frame = self._add_frame(label='LATEST RESULTS')
        top_frame.grid(row=0, column=0, sticky=(tk.E + tk.W))
        # Visualizing outputs
        ttk.Label(top_frame, text='Prediction Versus Hi Frequency Data, Showing Weakness or Strength.'
                  ).grid(row=0, column=0, sticky=(tk.W+tk.E))
        self.figure = Figure(figsize=(16, 8), dpi=100, constrained_layout=True)
        self.canvas_tkagg = FigureCanvasTkAgg(self.figure, top_frame)
        self.canvas_tkagg.get_tk_widget().grid(row=1, column=0, sticky=(tk.W+tk.E))

        middle_frame = self._add_frame(label='OPTIONS')
        middle_frame.grid(row=1, column=0, sticky=(tk.E + tk.W))

        LabelInput(middle_frame, "TIME INTERVAL(minute)", input_class=ttk.Spinbox,
                   var=self._vars['time interval'],
                   input_args={'from': 1, 'to': 60, 'increment': 1}
        ).grid(row=0, column=0, pady=5)

        bottom_frame = self._add_frame(label='MULTIPLE ACTIONS')
        bottom_frame.grid(row=2, column=0, sticky=(tk.E + tk.W))
        continuous_prediction_btn = ttk.Button(
            bottom_frame,
            text='START',
            command=lambda: self._continuous_prediction()
        )
        continuous_prediction_btn.grid(row=0, column=0, sticky=(tk.E + tk.W), pady=5, padx=5)
        LabelInput(bottom_frame, 'STOP CURRENT ACTION', input_class=ttk.Checkbutton,
                   var=self._vars['stop flag'],
                   input_args={'onvalue': True, 'offvalue': False}
                   ).grid(row=0, column=1, sticky=(tk.W + tk.E), pady=5, padx=5)
        get_api_balance_btn = ttk.Button(
            bottom_frame,
            text='API balance',
            command=lambda: self._get_api_balance()
        )
        get_api_balance_btn.grid(row=1, column=0, sticky=(tk.E + tk.W), pady=5, padx=5)

    def _add_frame(self, label, cols=3):
        frame = ttk.LabelFrame(self, text=label)
        frame.grid(sticky=(tk.W + tk.E))
        for i in range(cols):
            frame.columnconfigure(i, weight=1)
        return frame

    def get(self):
        data = dict()
        for key, variable in self._vars.items():
            data[key] = ''
        return data

    def _pred_visualize(self, targets: list, info_for_plots: pd.DataFrame):
        self.figure.clf()
        # Prepare the dataframe for plotting
        info_for_plots['Hifreq'] = info_for_plots.apply(lambda x: x['currentprice_Hifreq'] - x['lastDay_price'], axis=1)
        info_for_plots['Pred'] = info_for_plots.apply(lambda x: x['pred_calc_price'] - x['lastDay_price'], axis=1)
        # Plotting using matplotlib
        plt.style.use('default')
        self.axes = self.figure.subplots(nrows=2, ncols=math.ceil(len(targets) / 2))
        for num, target in enumerate(targets):
            x1 = num // math.ceil((len(targets) / 2))
            y1 = num % math.ceil((len(targets) / 2))
            if info_for_plots.loc[target, 'Hifreq'] >= 0:
                col_H = 'red'
            else:
                col_H = 'green'
            if info_for_plots.loc[target, 'Pred'] >= 0:
                col_P = 'red'
            else:
                col_P = 'green'
            self.axes[x1, y1].barh(['Hifreq', 'Pred'],
                                   [info_for_plots.loc[target, 'Hifreq'],
                                    info_for_plots.loc[target, 'Pred']])
            self.axes[x1, y1].get_children()[0].set_color(col_H)
            self.axes[x1, y1].get_children()[1].set_color(col_P)
            self.axes[x1, y1].text(y=['Hifreq'], x=info_for_plots.loc[target, 'Hifreq'],
                                   s=str(info_for_plots.loc[target, 'currentprice_Hifreq']))
            self.axes[x1, y1].text(y=['Pred'], x=info_for_plots.loc[target, 'Pred'],
                                   s=str(info_for_plots.loc[target, 'pred_calc_price']))
            self.axes[x1, y1].set_xlabel('Price different from last close.')
            self.axes[x1, y1].set_ylabel('Prediction VS Hi Frequency Data')
            self.axes[x1, y1].set_title(f'{target} Latest')
            self.axes[x1, y1].grid(True)
        # Put them on the screen
        self.canvas_tkagg.draw()

    def _continuous_prediction(self):
        # Setting up the time interval running framework
        time_interval = self._vars['time interval'].get()
        while True:
            # Update the current time.
            current_time = dt.datetime.now().strftime(format="%Y-%m-%d %H:%M:%S")
            print(time_interval)
            print(current_time)
            self.master._to_readtime(f"Data Read AT: {current_time} ")

            # Reading the data
            info_for_plots = TFTCP.pred_compare(targets)
            self._pred_visualize(targets, info_for_plots)
            self.master.update()
            # If clicking the stop flag, stop the running.
            if self._vars['stop flag'].get():
                self._vars['stop flag'].set(value=False)
                self.master.update()
                break
            time.sleep(60 * time_interval)

    def _get_api_balance(self):
        results = TFTCP.get_api_balance()
        text = 'API Balance: ' + results['Msg']
        self.master._to_status(text)

class Application(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title('SHOW SOME PROGRESS')
        self.columnconfigure(0, weight=1)
        ttk.Label(
            self,
            text='LATEST PREDICTION OF INTERESTED HEDGING PAIRS',
            font=('TkDefault', 16)
        ).grid(row=0, padx=10)

        self.readtime = tk.StringVar()
        self.readtime.set('Data Read AT: ')
        ttk.Label(
            self, textvariable=self.readtime
        ).grid(row=1, padx=10, sticky=(tk.W+tk.E))

        self.processingWindow = processingWindow()
        self.processingWindow.grid(row=2, padx=10, sticky=(tk.W+tk.E))

        self.status = tk.StringVar()
        self.status.set('Status: ')
        ttk.Label(
            self, textvariable=self.status
        ).grid(row=99, padx=10, sticky=(tk.W+tk.E))

    def _to_readtime(self, text):
        self.readtime.set(text)

    def _to_status(self, text):
        self.status.set(text)

if __name__ == '__main__':
    App = Application()
    App.mainloop()
