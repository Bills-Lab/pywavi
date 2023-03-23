'''
Race Peterson Noorda COM

The flanker test like the P300 test is a reactionary test. It differs though because there are multiple inputs to keep track of
the patient is presented with a scenario:

>>><>>>
They must then hit the left or right mouse button based on the oreintation of the middle arrow.

The dictionary that shows the correct mouse button for the event is stored seperately in the key.py file and is imported into this script for cleanliness
'''

import pandas as pd
from key import FLANKER_CORRECT_DICT, FLANKER_EVENT_KEY, ALL_REGIONS
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sklearn.decomposition
import numpy.linalg as LA
from scipy.fft import fft, fftfreq
from patient import Chunk, Patient

class FlankerChunk(Chunk):
    """
    A class used to represent an Animal

    ...

    Attributes
    ----------
    reaction : int
        a pandas dataframe containing the eeg data from an event occuring within a test
    correct : str
        checks to see if the patient got the flanker scenario correct or not
    Methods
    -------
    find_reaction()
        populates the self.reaction by finding the correct index in data
    accuracy_check()
        populates the self.correct by checking with the FLANKER_CORRECT_DICT
    """
    def __init__(self, data) -> None:
        super().__init__(data)
        self.stimulus, self.reaction = self.find_reaction() # these are indices values
        self.correct = self.accuracy_check()

    def find_reaction(self):
        """
        finds the reaction and the event stimulus
        """
        return np.where(self.chunk_data["Event"] > 0)[0][0] , np.where(self.chunk_data["Event"] > 0)[0][1]

    def accuracy_check(self):
        """
        labels the chunk based on the input of the user. True if the patient gets it correct.
        """
        if str(self.chunk_data["Event"][self.stimulus]) in FLANKER_CORRECT_DICT[str(self.chunk_data["Event"][self.reaction])]:
            return True
        else: return False

class Flanker(Patient):
    """
    Flanker class has a lot of attention on it.

    Attributes
    ----------
    _index : int
    transformed_to : bool
    chunk_collection : list
    pca_components_entire_brain

    Methods
    -------
    chunk_events()
        appends FlankerChunk into the self.chunk_collection list

    get_correct_events()
        creates a list of only the chunks that were correctly answered by the patient

    get_incorrect_events()
        creates a list of only the chunks that were incorrectly answered by the patient

    region_breakdown
    meta_collections
    accuracy_total
    pca_single_region
    pca_brain
    basis_change
    """
    def __init__(self, path, interval, identifier=None, condition=None):
        super().__init__(path, interval, identifier, condition)
        self._index = 0
        self.transformed_to = False
        self.chunk_collection = []
        self.chunk_events(self.eeg_df[["Event","FP1", "FP2", "F3", "F4", "F7", "F8", "C3", "C4", "P3", "P4", "O1", "O2", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"]])
        self.pca_components_entire_brain = False

    def chunk_events(self, data):
        '''creates FlankerChunk objects from a flanker test csv'''
        for i,row in data.iterrows():
            if int(row["Event"]) != 0:
                if int(row["Event"]) > 121:
                    if i + self.a > 0 and i + self.b <= data.shape[0]:
                        self.chunk_collection.append(FlankerChunk(data.iloc[i + self.a:i+self.b, :].reset_index(drop=True))) # NOTE: currently set for 2 seconds from event stimulus

    # TODO: These functions can be combined with out iterating through twice
    def get_correct_events(self):
        """Flanker tests have a right and wrong answer and this """
        return [event for event in self.chunk_collection if event.correct]
    def get_incorrect_events(self):
        """Flanker tests have a right and wrong answer and this """
        return [event for event in self.chunk_collection if not event.correct]

    def region_breakdown(self, event:str) -> list: # TODO: I don't remember what this does which probably means it can go
        """ gets a chunk based on name and takes its data to numpy array"""
        a = [chunk.chunk_data for chunk in self.chunk_collection if chunk.chunk_name == event]
        region_dict = {}
        for r in ALL_REGIONS:
            region_temp = []
            for df_ in a:
                region_temp.append(df_[r].to_numpy())
            region_dict[r] = np.hstack(region_temp)
            if region_dict[r].shape[0] % 500 != 0:
                remainder = region_dict[r].shape[0] % 500
                dim_ = region_dict[r].shape[0] - remainder
                region_dict[r] = region_dict[r][:dim_][:]
            region_dict[r] = np.reshape(region_dict[r], (500, -1))
        return region_dict

    def meta_collections(self, operation="avg"):
        """this creates a higher abstraction of collections, 
        for instance chunks that represent all '123' events averaged together will be one singular chunks
        
        avg -- average of all regions into one total chunk
        
        """
        groups = defaultdict(list)
        for chunk in self.chunk_collection:
            groups[chunk.chunk_name].append(chunk)
        if operation == "avg":
            average_events = {}
            for k in groups.keys():
                average_events[k] = pd.concat([chunk.average_chunk() for chunk in groups[k]], axis=1).mean(axis=1)
            return average_events
    
    def accuracy_total(self):
        self.cul_correct = 0
        for chunk in self.chunk_collection:
            if chunk.correct:
                self.cul_correct += 1
        return self.cul_correct / len(self.chunk_collection)
        
    def __getitem__(self,index):
        return self.chunk_collection[index]
