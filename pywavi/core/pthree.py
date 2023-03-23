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


class PChunk(Chunk):
    """
    A class to collect all the events during a p300 test

    ...

    Attributes
    ----------
    impact : int
        the index in the data of where the user impact was

    Methods
    -------
    find_impact()
        searches through the dataset to find where the human impact are and the stimulus

    vis_event_diff()
        simple visualization of the chunk that shows the marker of where they made impact
    """
    def __init__(self, data) -> None:
        super().__init__(data)
        self.stimulus, self.reaction = self.find_impact() # these are indicies not times
        self.name_test()
    
    def name_test(self):
        self.test_name = "P300"

    def find_impact(self):
        events = np.where(self.chunk_data["Event"])[0]
        stimulus, reaction = events[0], events[1]
        return stimulus, reaction

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
        self.event_index = self.get_index()
        self.all_chunks = self.chunk_csv()

    def get_index(self):
        ''' returns the pandas dataframe index of an event 1 (normal tone) or 2 (oddball)'''
        return [index for index, row in self.eeg_df.iterrows() if int(row["Event"]) == 1 or int(row["Event"]) == 2]

    def chunk_csv(self):
        '''chunks the dataset based on the 1 and 2 tones'''
        all_chunks = []
        for i in range(len(self.event_index) - 1):
            all_chunks.append(PChunk(self.eeg_df.iloc[self.event_index[i] + self.a :self.event_index[i] + self.b])) # Originally 100
        return all_chunks
    
    def combine_oddball_events(self):
        ''' This function combines all of the oddball evenst into a singal wave'''
        two_events = [chunk.combine_nodes() for chunk in self.all_chunks if chunk.name == 2]
        return np.sum(two_events, axis=0) / len(two_events)

    def combine_impacts(self):
        """This function comes up with an average occurence of the impaoct event after an oddball tone"""
        threes = [int(chunk.three) for chunk in self.all_chunks if chunk.three and chunk.three > 0]
        if threes:
            return np.sum(threes)/len(threes)
        else:
            return 100 # this is becasue within the second there was not a response or "missed oddball"
    
    def __getitem__(self, index):
        return self.all_chunks[index]