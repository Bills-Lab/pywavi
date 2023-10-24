import numpy as np
from scipy.signal import butter, lfilter, iirnotch
from .. import SAMPLING_RATE


# FILTERING FUNCTIONS
def butter_highpass(cutoff=1, order=5):
    '''This function will apply a high pass filter to the data.'''
    nyq = 0.5 * SAMPLING_RATE  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype='high', analog=False)

def butter_highpass_filter(data, cutoff=1, order=5):
    '''This function will apply a high pass filter to the data.'''
    b,a = butter_highpass(cutoff, order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass(cutoff=50, order=5):
    nyq = 0.5 * SAMPLING_RATE  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff=50, order=5):
    b, a = butter_lowpass(cutoff, order=order)
    y = lfilter(b, a, data)
    return y

def apply_notch_filter(signal, notch_freq=60, Q=32, fs=SAMPLING_RATE):
    """
    Apply a notch filter to a signal.

    Parameters:
    - signal: Input signal to be filtered
    - notch_freq: Frequency to be notched out which is set at 60HZ in the US (50HZ in europe & asia)
    - Q: Quality factor (a higher value means a narrower notch)
    - fs: Sampling frequency of the signal

    Returns:
    - Filtered signal
    """
    b, a = iirnotch(notch_freq, Q, fs)

    return lfilter(b, a, signal)

# ARTIFACT DETECTION FUNCTIONS
def detect_artifacts():
    """This function will detect artifacts in the data."""
    return
