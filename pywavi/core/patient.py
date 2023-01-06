from key import FLANKER_CORRECT_DICT, FLANKER_EVENT_KEY, ALL_REGIONS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os
import glob

class Chunk:
    """
    A class used to represent an Animal

    ...

    Attributes
    ----------
    chunk_data : pandas dataframe
        a pandas dataframe containing the eeg data from an event occuring within a test
    chunk_name : str
        the name of the event specifically coded for the test that is happening please see util for category translation

    Methods
    -------
    average_chunk()
        Averages the all individual node waves into one general wave returns numpy array
    """

    def __init__(self, data) -> None:
        self.chunk_data = data # pd.df
        self.chunk_name = str(data["Event"][0])
    
    def average_chunk(self):
        """Averages all regions from EEG cap into one vector"""
        return self.chunk_data[["FP1", "FP2", "F3", "F4", "F7", "F8", "C3", "C4", "P3", "P4", "O1", "O2", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"]].mean(axis=1)

class Patient:
    """
    A class used to represent an Animal

    ...

    Attributes
    ----------
    csv_path : str
        a path pointing to the parsed WAVI dataframe
    condition : str
        the name of the condition that the patient has (OPTIONAL)
    patient_identifier : str
        identifier of the patient
    eeg_df : pandas.DataFrame
        dataframe from the csv_path entire dataset

    Methods
    -------
    fast_fourier(node_name)
        node_name : str
            performs a fast fourier transorm on the node_name
    
    visualize_region(node_name)
        node_name : str
            creates a viz for a specific region of the brain
    """

    def __init__(self, path, identifier=None, condition=None):
        self.csv_path = path
        self.condition = condition
        self.patient_identifier = identifier
        self.name = os.path.basename(self.csv_path)[:4] # this may be redundant 
        self.eeg_df = pd.read_csv(path)[["Event","FP1", "FP2", "F3", "F4", "F7", "F8", "C3", "C4", "P3", "P4", "O1", "O2", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"]]

    def fast_fourier(self, node_name:str):
        '''param: node_name:str is the initialism of the region of the brain that you would like to '''
        data = np.array(self.eeg_df[node_name])
        y = fft(np.array(data))
        xf = fftfreq(data.shape[0], 0.01)
        plt.plot(xf[:data.shape[0]], y[:data.shape[0]])
        plt.show()
        return
    
    def visualize_region(self, node_name):
        if node_name == "all":
            data = self.eeg_df.iloc[:, 1:].mean(axis=1)
        else:
            data = self.eeg_df[node_name]
        
        fig, ax = plt.subplot()
        X = list(range(len(data)))
        Y = data.to_numpy().reshape(1,-1)
        ax.plot(X, Y, '-gD', markevery=[self.three], label='Patient Impact')
        ax.set_title("Wave of {}".format(node_name))
        plt.legend()
        plt.show()
        return