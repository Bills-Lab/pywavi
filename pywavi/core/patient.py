from key import FLANKER_CORRECT_DICT, FLANKER_EVENT_KEY, ALL_REGIONS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os
import glob

class IncorrectInterval(Exception):
    "Raised: Interval is not of length 2. Must be length two (a,b) for interval that preceeds the initial condition of the test please format (-100, 500) to signal 100 indexes before stimulus and 500 indexes after stimulus"
    pass

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
        self.impact = None
    
    def combine_nodes(self, method='avg'):
        if method == 'avg':
            return np.array(self.data_df.drop(["Event"], axis=1).mean(axis=1))

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

    def __init__(self, path:str, interval:tuple, identifier=None, condition=None):
        self.csv_path = path
        self.condition = condition
        self.patient_identifier = os.path.basename(self.csv_path)[:4]
        self.eeg_df = pd.read_csv(path)[["Event","FP1", "FP2", "F3", "F4", "F7", "F8", "C3", "C4", "P3", "P4", "O1", "O2", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"]]
        self.test_name = None
        try:
            self.a, self.b = tuple(interval)
        except IncorrectInterval:
            raise("length of interval is {}".format(len(interval)))