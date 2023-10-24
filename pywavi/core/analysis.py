import numpy as np
from scipy.integrate import simpson
from patient import Patient
from key import RATE, FREQ, ALL_REGIONS
import matplotlib.pyplot as plt

def fft_denoise_wave(data, frequency_threshold:int):
    # chunk_data = data.to_numpy()
    chunk_data = data
    fhat = np.fft.rfft(chunk_data, len(chunk_data))
    PSD = fhat * np.conj(fhat)/len(chunk_data)
    indices = PSD > frequency_threshold
    fhat_clean = indices * fhat
    return np.fft.irfft(fhat_clean)

def get_regional_average(self, patient, region:str):
    """returns the the average of every chunk for a specific region"""
    return np.mean(np.vstack([chunk.chunk_data[region].to_numpy() for chunk in patient.chunk_collection]),axis=0)


def fft_denoise_average_of_patient(patient, region:str, frequency_threshold:int):
    regional_data = get_regional_average(patient, region)
    print(regional_data.shape)
    average_reaction = sum([chunk.reaction for chunk in patient.chunk_collection])/len(patient.chunk_collection)
    print(average_reaction)
    denoised_regional_data = fft_denoise_wave(regional_data, frequency_threshold)
    plt.figure()
    plt.title("Average of {} region".format(region))
    plt.plot(denoised_regional_data, '-gD', markevery=[100, int(average_reaction)])
    plt.savefig("Average of {} region.png".format(region))
    return denoised_regional_data


def fft_denoise_entire_patient(patient, region:str, freqency_threshold):
    """
    region: a region from the key.py of the ALL_REGION
    chunk: a instance of the PChunk or FlankerChunk
    freqency_threshold: a threshold to leave out frequencies
    """
    print("LEN OF CHUNK _COLLECTION", len(patient.chunk_collection))
    every_sample = []
    for i, chunk in enumerate(patient.chunk_collection):
        f = fft_denoise_wave(chunk.chunk_data[region], frequency_threshold=freqency_threshold)
        every_sample.append(f)
        if not chunk.reaction:
            plt.subplot(10,11,i + 1)
            plt.plot(f, '-gD', markevery=[chunk.stimulus])
        else:
            plt.subplot(10,11,i + 1)
            plt.plot(f, '-gD', markevery=[chunk.stimulus, chunk.reaction])
    plt.savefig("Every {} wave.png".format(region))
    return every_sample

def get_node_and_integrate(self, patient, region:str): # analysis
    all_node = [abs(simpson(chunk.data_df[region].to_numpy())) for chunk in patient.all_chunks]
    return sum(all_node) / len(all_node)
    # return all_node

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