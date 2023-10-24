import glob
import pandas as pd
from .. import WaviDataset
import os

def parse_raw(path):
    """
    Parses raw data from a WAVI EEG recording and persists it to a CSV file.

    Parameters
    ----------
    path : str
        The path to the directory containing the WAVI EEG recording. A single recording (patient) should be a directory of its own that contains the .eeg, .art, .mag, and .evt files.
    """

    def combine_eeg_files(folder):
        files = os.listdir(folder)
        
        # Filter files based on their extensions
        eeg_files = [f for f in files if f.endswith('.eeg')]
        art_files = [f for f in files if f.endswith('.art')]
        mag_files = [f for f in files if f.endswith('.mag')]
        evt_files = [f for f in files if f.endswith('.evt')]

        if len(eeg_files) != 1:
            print(f"\tSubfolder {folder} contains {len(eeg_files)} '.eeg' files.")
            print("\tThis folder will be skipped.")
            return

        eeg_file, art_file, mag_file, evt_file = [os.path.join(folder, f) for f in (eeg_files[0], art_files[0] if art_files else None, mag_files[0] if mag_files else None, evt_files[0] if evt_files else None)]
        
        try:
            output_file_name = os.path.join(folder, '_'.join(folder.split(os.sep)[-2:]) + '_WAVI_eeg.csv')
        except IndexError:
            output_file_name = os.path.join(folder, folder.split(os.sep)[-1] + '_WAVI_eeg.csv')

        eeg_df = pd.read_csv(eeg_file, header=None, delim_whitespace=True)

        if mag_file:
            mag_df = pd.read_csv(mag_file, delim_whitespace=True)
            probe_labels = mag_df["LOC"].values
            eeg_df = eeg_df.iloc[:, :len(probe_labels)]
            eeg_df.columns = probe_labels

        time_msec = 4 * eeg_df.index.to_numpy()
        eeg_df['time_msec'] = time_msec
        eeg_df['time_sec'] = time_msec / 1000
        eeg_df['time_min'] = eeg_df['time_sec'] / 60

        data_to_combine = [eeg_df]

        if art_file:
            art_df = pd.read_csv(art_file, header=None, sep=' ')
            if mag_file:
                art_df = art_df.iloc[:, :len(probe_labels)]
                art_df.columns = [f"{name}_Artifact" for name in probe_labels]
            data_to_combine.append(art_df)
        
        if evt_file:
            evt_df = pd.read_csv(evt_file, header=None)
            evt_df.columns = ['Event']
            data_to_combine.append(evt_df)

        combined_data = pd.concat(data_to_combine, axis=1)
        print(combined_data.columns)
        return output_file_name, combined_data.values

    usable_folders = set()

    for root, dirs, files in os.walk(path):
        if any(f.endswith('.eeg') for f in files):
            usable_folders.add(root)

    if not usable_folders:
        raise IndexError("\tNo subdirectories from your specified input directory contain '.eeg' files.")
    
    return [WaviDataset(*combine_eeg_files(folder)) for folder in usable_folders]
