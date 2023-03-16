import numpy as np
from scipy.integrate import simpson


def fft_denoise_wave(region, chunk, freqency_threshold):
    """
    region: a region from the key.py of the ALL_REGION
    chunk: a instance of the PChunk or FlankerChunk
    freqency_threshold: a threshold to leave out frequencies can be tuple for a low high range
    """

    return

def get_node_and_integrate(self, node_name:str): # analysis
    all_node = [abs(simpson(chunk.data_df[node_name].to_numpy())) for chunk in self.all_chunks]
    return sum(all_node) / len(all_node)
    # return all_node

def get_node_avg_of_chunk(self, node_name:str):
    """returns the the average of every chunk for a specific node"""
    all_node = [chunk.data_df[node_name].to_numpy() for chunk in self.all_chunks]
    return np.sum(all_node, axis=0) / len(all_node)