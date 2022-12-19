'''
Race Peterson

These classes digest a patient WAVI eeg file.

Combining nodes and artifacts take some more nuanced details, so a professor is needed

0 - nothing
1 - normal beat
2 - oddball
3 - user input
'''
import pandas as pd
from key import ALL_REGIONS
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from patient import Chunk, Patient
from scipy.integrate import simpson
import os
import math

class PChunk(Chunk):
    def __init__(self, data) -> None:
        super().__init__(data)
        self.three = self.find_impact()

    def find_impact(self):
        for i, row in self.data_df.iterrows():
                if row["Event"] == 3:
                    return i

    def combine_nodes(self, method='avg'):
        '''
        This function takes a chunk and combines it.
        This method will have multiple options
        param: method
            -'avg'
                takes the average across all of the nodes at any given time
            -'ffs'
                Fourier fuse
            - 'linear predicter analysis LPC'

        '''
        if method == 'avg':
            return np.array(self.data_df.drop(["Event"], axis=1).mean(axis=1))
    
    def return_node(self):
        return
    
    def vis_event_diff(self):
        if self.three:
            if self.three > 0:
                fig, ax = plt.subplots()
                X = list(range(len(self.data_df)))
                Y = self.combine_nodes()
                ax.plot(X, Y, '-gD', markevery=[self.three], label='Patient Impact')
                ax.set_title("Wave in {} event".format(self.name))
                plt.legend()
                plt.show()

class PThreeHundred(Patient):
    def __init__(self, path, identifier=None, condition=None) -> None:
        super().__init__(path, identifier, condition)
        self.csv_path = path
        self.name = os.path.basename(self.csv_path)[:4]
        self.eeg_df = pd.read_csv(path)[["Event","FP1", "FP2", "F3", "F4", "F7", "F8", "C3", "C4", "P3", "P4", "O1", "O2", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"]]
        self.get_index()
        self.all_chunks = self.chunk_csv()

    def get_index(self):
        self.event_index = [index for index, row in self.eeg_df.iterrows() if row["Event"] != 0]

    def smoothing(self):
        '''There may need to be some smothing done to the data set in the event of a node disconnecting'''
        pass

    def chunk_csv(self):
        all_chunks = []
        for i in range(len(self.event_index) - 1):
            all_chunks.append(Chunk(self.eeg_df.iloc[self.event_index[i]:self.event_index[i] + 500])) # Originally 100
        return all_chunks
    
    def combine_events(self):
        two_events = [chunk.combine_nodes() for chunk in self.all_chunks if chunk.name == 2]
        return np.sum(two_events, axis=0) / len(two_events)
    
    def get_node_avg_of_chunk(self, heading:str):
        all_node = [chunk.data_df[heading].to_numpy() for chunk in self.all_chunks]
        return np.sum(all_node, axis=0) / len(all_node)

    def get_node_and_integrate(self, heading:str):
        all_node = [abs(simpson(chunk.data_df[heading].to_numpy())) for chunk in self.all_chunks]
        return sum(all_node) / len(all_node)
        # return all_node

    def combine_markers(self):
        threes = [int(chunk.three) for chunk in self.all_chunks if chunk.three and chunk.three > 0]
        if threes:
            return np.sum(threes)/len(threes)
        else:
            return 100 # this is becasue within the second there was not a response or "missed oddball"

    def return_node_event(self):
        return

    def vis_patient(self):
        fig, ax = plt.subplots()
        X = list(range(100))
        Y = self.combine_events()
        marker = math.ceil(self.combine_markers())
        print("MARKER: ", marker)
        ax.plot(X, Y, '-gD', markevery=[marker], label='Average Patient Impact')
        ax.set_title("Patient Wave Average")
        plt.legend()
        plt.show()

    def markers(self):
        print(pd.Series([int(chunk.three) for chunk in self.all_chunks if chunk.three and chunk.three > 0]).describe())

    def event(self):
        print(pd.Series([chunk.combine_nodes() for chunk in self.all_chunks if chunk.name == 2]).describe())

    def __getitem__(self, index):
        return self.all_chunks[index]