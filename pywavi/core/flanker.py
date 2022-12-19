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
    def __init__(self, data) -> None:
        super().__init__(data)
        self.reaction = self.find_reaction()
        self.correct = self.accuracy_check()

    def find_reaction(self):
        """
        finds the reaction of the event stimulus
        """
        for i, row in self.chunk_data.iloc[1:].iterrows():
            if row["Event"] != 0:
                return row["Event"]
        return 0

    def accuracy_check(self):
        """
        labels the chunk based on the input of the user. True if the patient gets it correct.
        """
        if self.reaction == 0:
            return False
        elif self.chunk_name in FLANKER_CORRECT_DICT[str(int(self.reaction))]:
            return True
        else:
            return False


class Flanker(Patient):
    def __init__(self, path, identifier=None, condition=None):
        super().__init__(path, identifier, condition)
        self._index = 0
        self.transformed_to = False
        self.chunk_collection = []
        self.chunk_events(self.eeg_df[["Event","FP1", "FP2", "F3", "F4", "F7", "F8", "C3", "C4", "P3", "P4", "O1", "O2", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"]])
        self.pca_components_entire_brain = False

    def chunk_events(self, data):
        for i,row in data.iterrows():
            if int(row["Event"]) != 0:
                if int(row["Event"]) > 121:
                    self.chunk_collection.append(FlankerChunk(data.iloc[i:i+500, :].reset_index(drop=True))) # NOTE: currently set for 2 seconds from event stimulus

    def get_correct_events(self):
        return [event for event in self.chunk_collection if event.correct]
    
    def get_incorrect_events(self):
        return [event for event in self.chunk_collection if not event.correct]

    def region_breakdown(self, event:str) -> list:
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
    
    def pca_single_region(self, region:str, components=6):
        '''
        TODO: get all eigenvectors based on a tolerance for each region
        UNDERCONSTRUCTION:
        '''
        a = np.concatenate([self.region_breakdown(e)[region][:,:10] for e in FLANKER_EVENT_KEY], axis=1) # This average is called "The Event Related Potential"
        pca = sklearn.decomposition.PCA(n_components=components)
        pca.fit(a.T)

        return pca.components_
    
    def pca_brain(self, components=6):
        brain = np.concatenate([self.pca_single_region(region) for region in ALL_REGIONS], axis=0)
        pca = sklearn.decomposition.PCA(n_components=components)
        pca.fit(brain)
        self.pca_components_entire_brain = pca.components_
        return pca.components_
    
    def basis_change(self, input_space):
        """
        Performs the basis change transformation input space being 
        the argument Flanker and the output space being the self
        let X be the input basis vectors and let Y be the Output Space basis vectors
        then X B = Y
        B = inv(X) Y
        makes B a basis transformation from X to Y
        """
        # if self.pca_components_entire_brain == False:
        #     raise("PCA components for the entire brain do not exis yet please run self.pca_brain and try again")
        # if input_space.pca_components_entire_brain == False:
        #     print("Input Flanker has not made PCA components doing so now...")
        #     input_space.pca_brain()
        #     print("Complete")
        # LA.inv(input_space.pca_components_entire_brain)
        # transformation = np.matmul(LA.inv(input_space.pca_components_entire_brain), self.pca_components_entire_brain)
        # print(transformation)
        transformation = LA.lstsq(self.pca_components_entire_brain, input_space.pca_components_entire_brain)
        print(transformation[0])
        print(transformation[0].shape)
        print(self.pca_components_entire_brain)
        print("===================================")
        print(np.matmul(input_space.pca_components_entire_brain, transformation[0]))
        return transformation[0]
