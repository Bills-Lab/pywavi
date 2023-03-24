import numpy as np
from scipy.integrate import simpson
from patient import Patient
from key import RATE, FREQ
import matplotlib.pyplot as plt

def fft_denoise_wave(chunk_data, region, frequency_threshold):
    fhat = np.fft.rfft(chunk_data, len(chunk_data))
    PSD = fhat * np.conj(fhat)/len(chunk_data)
    plt.figure()
    plt.plot(PSD[:len(chunk_data)//2])
    indices = PSD > 300
    psdclean = PSD * indices
    fhat_clean = indices * fhat
    f = np.fft.irfft(fhat_clean)
    plt.figure()
    plt.plot(f)

def fft_denoise_entire_patient(patient, region, freqency_threshold):
    """
    region: a region from the key.py of the ALL_REGION
    chunk: a instance of the PChunk or FlankerChunk
    freqency_threshold: a threshold to leave out frequencies can be tuple for a low high range
    """
    for chunk_data in patient.chunk_collection:
        fft_denoise_wave(chunk_data, region=region, frequency_threshold=freqency_threshold)
    return

def get_node_and_integrate(self, node_name:str): # analysis
    all_node = [abs(simpson(chunk.data_df[node_name].to_numpy())) for chunk in self.all_chunks]
    return sum(all_node) / len(all_node)
    # return all_node

def get_node_avg_of_chunk(self, node_name:str):
    """returns the the average of every chunk for a specific node"""
    all_node = [chunk.data_df[node_name].to_numpy() for chunk in self.all_chunks]
    return np.sum(all_node, axis=0) / len(all_node)


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