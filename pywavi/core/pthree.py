'''
Race Peterson Noorda COM

These classes help organize data coming in from a P300 exam.

A P300 exam measure reaction time. The patient puts on headphones and listens to a series of beeps
Every beep sounds the same but there is an "oddball" beep. When the oddball beep sounds the patient is instructed to click as fast as they can

The data of the event is signified by the following
0 - nothing
1 - normal beep
2 - oddball beep
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
    """
    A class to collect all the events during a p300 test

    ...

    Attributes
    ----------
    three : int
        the index in the data of where the user impact was

    Methods
    -------
    find_impact()
        searches through the dataset to find where the human impact is
    combine_nodes()
        averges all nodes together into one wave
    vis_event_diff()
        simple visualization of the chunk that shows the marker of where they made impact
    """
    def __init__(self, data) -> None:
        super().__init__(data)
        self.three = self.find_impact()

    def find_impact(self):
        for i, row in self.data_df.iterrows():
                if row["Event"] == 3:
                    return i

    def combine_nodes(self, method='avg'):
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
    """
    A class that represents the p300 test. Abstraacted from the Patient class
    
    ...

    Attributes
    ----------
    event_index : a list of indexes that represent the location of events
    all_chunks : a collection of all the PChunk objects for a specific test

    Methods
    -------
    get_index()
        returns a list of all the indexes where a event occurs
    smoothing()
        NOT IMPLEMENTED
    chunk_csv()
        creates a collection of chunks
    combine_events()
        takes all oddball events in the p300 test and averages them together
    get_node_avg_of_chunk(node_name)
        gets all chunks of the same event name and averages them together
    get_node_and_integrate(node_name)
        takes simpson integral of a node wave
    combine_markers()
        averages the impact time of the patient 

    
    """
    def __init__(self, path, identifier=None, condition=None):
        super().__init__(path, identifier, condition)
        self.csv_path = path # not neccessary
        self.name = os.path.basename(self.csv_path)[:4]
        self.event_index = self.get_index()
        self.all_chunks = self.chunk_csv()

    def get_index(self):
        return [index for index, row in self.eeg_df.iterrows() if row["Event"] != 0]

    def smoothing(self):
        '''There may need to be some smothing done to the data set in the event of a node disconnecting'''
        pass

    def chunk_csv(self):
        all_chunks = []
        for i in range(len(self.event_index) - 1):
            all_chunks.append(PChunk(self.eeg_df.iloc[self.event_index[i]:self.event_index[i] + 500])) # Originally 100
        return all_chunks
    
    def combine_events(self):
        two_events = [chunk.combine_nodes() for chunk in self.all_chunks if chunk.name == 2]
        return np.sum(two_events, axis=0) / len(two_events)
    
    def get_node_avg_of_chunk(self, node_name:str):
        all_node = [chunk.data_df[node_name].to_numpy() for chunk in self.all_chunks]
        return np.sum(all_node, axis=0) / len(all_node)

    def get_node_and_integrate(self, node_name:str):
        all_node = [abs(simpson(chunk.data_df[node_name].to_numpy())) for chunk in self.all_chunks]
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