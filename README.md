# pywavi

Lightweight, live, modular. Simple csv based EEG analysis tool specifically made for the WAVI eeg headset but analysis tools and structures are open to all! WAVI dataset info for porting to MNE available.

## Lightweight
All of or processes are done with numpy arrays and made availble to you for custom functions.

## Live
We have published docker images for easy use and deployment. We also provide a structure to do live analysis on incoming data from the WAVI headset.

## Modular
All of our processes are broken down into modules that can be used in any order and in any combination. However, we provide a Chain class that allows you to easily chain together modules in a pipeline.

## WAVI in mind
Parsers that are specific to the WAVI headset are provided to make it easy to get started with your own analysis. However, the WAVI datasets are also available to port to MNE format.

*need example of porting from wave dataset and then sending information to mne*


import pandas as pd

# Load your EEG data from CSV
eeg_data = pd.read_csv('your_eeg_data.csv')
import mne

# Create an mne.Info object with your EEG data info
info = mne.create_info(ch_names=['EEG1', 'EEG2', ...], sfreq=your_sampling_rate)
raw = mne.io.RawArray(eeg_data.T, info)
