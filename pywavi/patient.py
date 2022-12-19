from key import FLANKER_CORRECT_DICT, FLANKER_EVENT_KEY, ALL_REGIONS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os
import glob

class Chunk:
    def __init__(self, data) -> None:
        self.chunk_data = data # pd.df
        self.chunk_name = str(data["Event"][0])
    
    def average_chunk(self):
        """Averages all regions from EEG cap into one vector"""
        return self.chunk_data[["FP1", "FP2", "F3", "F4", "F7", "F8", "C3", "C4", "P3", "P4", "O1", "O2", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"]].mean(axis=1)

class Patient:
    def __init__(self, path, identifier=None, condition=None):
        self.csv_path = path
        self.condition = condition
        self.patient_identifier = identifier
        self.name = os.path.basename(self.csv_path)[:4]
        self.eeg_df = pd.read_csv(path)[["Event","FP1", "FP2", "F3", "F4", "F7", "F8", "C3", "C4", "P3", "P4", "O1", "O2", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"]]

    def linear_predictor_coefficient(self):
        return
    
    def fast_fourier(self, seq):
        '''param: seq pandas.Series of a specific region of the brain'''
        name = seq.name
        data = np.array(seq)
        y = fft(np.array(data))
        xf = fftfreq(data.shape[0], 0.01)
        plt.plot(xf[:data.shape[0]], y[:data.shape[0]])
        plt.show()
        return

    def wavelet(self):
        return