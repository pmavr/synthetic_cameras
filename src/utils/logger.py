import os
from datetime import datetime
import numpy as np
from tabulate import tabulate
np.set_printoptions(suppress=True)
from src.utils import utils


class Logger:

    def __init__(self, log_filepath):
        self.filepath = log_filepath
        self.filename = 'experiments_log.txt'
        self.file = None

    def file_exists(self):
        return os.path.isfile(f'{self.filepath}{self.filename}')

    def create_file(self):
        self.file = open(f"{self.filepath}{self.filename}", "x")

    def open_existing_file(self):
        self.file = open(f"{self.filepath}{self.filename}", "a")

    def write_to_file(self, experiment):
        line = str(experiment)
        self.file.write(line)

    def close_file(self):
        self.file.close()

    def log_experiment(self, experiment):
        if self.file_exists():
            self.open_existing_file()
        else:
            self.create_file()

        self.write_to_file(experiment)
        self.close_file()


class Experiment:
    def __init__(self, model_filename, model_params, data_params, history):
        self.model_filename = model_filename
        self.model_params = model_params
        self.data_params = data_params
        self.history = history

    def __str__(self):
        line = f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        line += f'Model filename: {self.model_filename}\n'
        line += f'Model params---------\n\t\t{str(self.model_params)}\n'
        line += f'Data params---------\n\t\t{str(self.data_params)}\n'
        line += f'-------------------------\n'
        headers = ['Epoch'] + [metric for metric in self.history]
        rows = []
        epochs = len(self.history[next(iter(self.history))])
        for e in range(epochs):
            rows.append([e]+[np.round(self.history[metric][e], 3).tolist() for metric in self.history])
        line += tabulate(rows, headers=headers)
        line += '\n\n\n\n'
        return line

