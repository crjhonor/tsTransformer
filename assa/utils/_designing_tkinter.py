"""
Then let me design the tkinter for pretty face.
"""
import numpy as np
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

# More definition of commodity indexes
indexWanted_CU0 = ['CU0', 'P0', 'Y0', 'AG0', 'BU0', 'ZN0', 'C0', 'AL0', 'RM0', 'M0', 'CF0']
indexWanted_RB0 = ['RB0', 'HC0', 'I0', 'V0', 'BU0', 'PP0', 'L0', 'FG0', 'MA0', 'SA0', 'SR0']
indexWanted_SCM = ['SCM', 'AU0', 'PG0', 'EB0', 'FU0', 'TA0', 'PP0', 'L0', 'V0', 'MA0', 'RU0']

# Include all the interested commodity indexes and making the target index as the first one.
indexList = list(np.unique(indexWanted_CU0 + indexWanted_RB0 + indexWanted_SCM))

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

"""
Widget and frame creation.
"""

class processingWindow(ttk.Frame):
    def __init__(self, *args, indexes=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.indexes = indexes
        self._vars = {
            'TARGET': tk.StringVar()
        }

        self.columnconfigure(1, weight=1)

        self._vars['TARGET'].set('RU0')

        top_frame = self._add_frame(label='Results Windows')
        top_frame.grid(row=0, column=0, sticky=(tk.E + tk.W))
        # Visualizing outputs
        ttk.Label(top_frame, text='Visualizing outputs').grid(row=0, column=0, sticky=(tk.W+tk.E))
        self.figure = Figure(figsize=(16, 8), dpi=100)
        self.canvas_tkagg = FigureCanvasTkAgg(self.figure, top_frame)
        self.canvas_tkagg.get_tk_widget().grid(row=1, column=0, sticky=(tk.W+tk.E))

        middle_frame = self._add_frame(label='Commodity Index Action')
        middle_frame.grid(row=1, column=0, sticky=(tk.E + tk.W))

        LabelInput(middle_frame, "SELECT THE TARGET", input_class=ttk.Combobox,
                   var=self._vars['TARGET'],
                   input_args={'values': self.indexes}
        ).grid(row=0, column=0, pady=5)
        target_predict_btn = ttk.Button(
            middle_frame,
            text='Target Predict',
            command=lambda: self._single_predict(ind=self._vars['TARGET'].get())
        )
        target_predict_btn.grid(row=0, column=1, sticky=(tk.W + tk.S + tk.E), pady=5)

        bottom_frame = self._add_frame(label='Multiple Action')
        bottom_frame.grid(row=2, column=0, sticky=(tk.E + tk.W))
        all_predict_btn = ttk.Button(
            bottom_frame,
            text='Predict All',
            command=lambda: self._all_predict(ind=self.indexes)
        )
        all_predict_btn.grid(row=0, column=0, sticky=(tk.E + tk.W))

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

    def _generate_xy(self, output, classesTable):
        nor_eta = classesTable.shape[0]
        y = output - nor_eta / 2
        x = np.arange(0, len(output))
        radiant = np.linspace(1, 0, len(output))
        y_p_radiant = np.round([a * b for a, b in zip(y, radiant)], 0)
        y_radiant = np.cumsum(y_p_radiant)
        return x, y, y_radiant

    def _single_predict(self, ind):
        # Setting the master's status
        self.master._to_status(" ".join(['Prediction of single target to', ind]))

        # Visualize results output
        self.figure.clf()
        self.fig1 = self.figure.add_subplot(2, 3, (1, 2))
        x = np.linspace(1, 10, 25)
        line1 = self.fig1.plot(x, color='black')
        self.fig1.set_xlabel('FUTURE TIME')
        self.fig1.set_ylabel('CLASSES')
        self.fig1.set_title('PREDICTION HEAT MAP')
        self.fig1.legend(['Prediction', 'Heating'], loc='lower center')
        self.fig2 = self.figure.add_subplot(2, 3, 3)
        text2 = self.fig2.text(x=0, y=0.5,
                               ha='left', va='center', color='black',
                               bbox=dict(facecolor='red', alpha=0.5),
                               fontsize=18,
                               s='Hello world!\nIt is a nice world.')
        self.fig2.axis('off')
        self.canvas_tkagg.draw()

    def _all_predict(self, ind):
        # Setting the master's status
        self.master._to_status(" ".join(['Prediction of single target to', " ".join(ind)]))

        # Visualize results output
        self.figure.clf()
        self.fig = self.figure.add_subplot(2, 3, (1, 6))
        x = np.linspace(1, 10, 25)
        line1 = self.fig.plot(x, color='black')
        self.fig.set_xlabel('TIME')
        self.fig.set_ylabel('CLASSES')
        self.fig.set_title('PREDICTION HEAT MAP')
        self.fig.legend(['Prediction', 'Heating'], loc='lower center')
        self.canvas_tkagg.draw()

class Application(tk.Tk):
    def __init__(self, *args, indexes=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.indexes = indexes
        self.title('III_seq2seq_longertime')
        self.columnconfigure(0, weight=1)
        ttk.Label(
            self,
            text='SEQUENCE TO SEQUENCE LONGER TIME ANALYSIS TO MONTHLY DATA',
            font=('TkDefault', 16)
        ).grid(row=0, padx=10)

        self.processingWindow = processingWindow(self,
                                                 indexes=indexes)
        self.processingWindow.grid(row=1, padx=10, sticky=(tk.W+tk.E))

        self.status = tk.StringVar()
        ttk.Label(
            self, textvariable=self.status
        ).grid(row=99, padx=10, sticky=tk.W+tk.E)

    def _to_status(self, text):
        self.status.set(text)

App = Application(indexes=indexList)
App.mainloop()