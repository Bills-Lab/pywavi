import glob
import pandas as pd

def parse_raw(self, path):
    """This function parses raw files from the EEG device and stores them in CSV files
        
        This function does not store the data in memory rather it persists it to a directory for further use
    """

    def get_time(df):
        time_msec = 4*df.name
        time_sec = time_msec/1000
        time_min = time_sec/60
        df['time_msec'] = time_msec
        df['time_sec'] = time_sec
        df['time_min'] = time_min
        return df

    def combine_eeg_files(folder):
        # files = os.listdir(folder)
        eeg_files = glob.glob(folder+'/*eeg')
        if len(eeg_files) > 1:
            print("\tSubfolder "+folder+' contains more than one ".eeg" file.')
            print("\tThis folder will be skipped.")
            #continue
        eeg_file = eeg_files[0]
        art_file = None
        art_files = glob.glob(folder+'/*art')
        if len(art_files) == 1:
            art_file = art_files[0]
        mag_file = None
        mag_files = glob.glob(folder+'/*mag')
        if len(mag_files) == 1:
            mag_file = mag_files[0]
        evt_file = None
        evt_files = glob.glob(folder+'/*evt')
        if len(evt_files) == 1:
            evt_file = evt_files[0]

        try:
            output_file_name = folder+'/'+folder.split('/')[-2]+'_'+folder.split('/')[-1]+'_WAVI_eeg.csv'
        except IndexError:
            output_file_name = folder+'/'+folder.split('/')[-1]+'_WAVI_eeg.csv'

        eeg_df = pd.read_csv(eeg_file, header = None, delim_whitespace=True)
        if mag_file:
            mag_df = pd.read_csv(mag_file, delim_whitespace=True)
            probe_labels = mag_df["LOC"].values
            eeg_df = eeg_df[eeg_df.columns[0:len(probe_labels)]]
            eeg_df.columns = probe_labels
        # Now to get the time stamp for each entry.
        eeg_df = eeg_df.apply(get_time, axis = 1)

        if art_file:
            art_df = pd.read_csv(art_file, header = None, sep=' ')
            if mag_file:
                art_df = art_df[art_df.columns[0:len(probe_labels)]]
                art_labels = []
                for name in probe_labels:
                    art_labels.append(name+'_Artifact')
                art_df.columns = art_labels
        if evt_file:
            evt_df = pd.read_csv(evt_file, header = None)
            evt_df.columns=['Event']

        dataframes_to_combine = []
        if evt_file:
            dataframes_to_combine.append(evt_df)
        dataframes_to_combine.append(eeg_df)
        if art_file:
            dataframes_to_combine.append(art_df)
        combined_data = pd.concat(dataframes_to_combine, axis = 1)
        combined_data.to_csv(output_file_name)

    input_directory = path
    all_folders = glob.glob(input_directory+'/**/**')
    usable_folders = []
    for subfolder in all_folders:
        if subfolder.endswith('eeg'):
            usable_folders.append('/'.join(subfolder.split('/')[:-1]))
        if len(glob.glob(subfolder+'/*.eeg')) > 0:
            usable_folders.append(subfolder)
    if len(usable_folders) == 0:
        raise IndexError("\tNo subdirectories from your specified input directory contain '.eeg' files.")
    
    for subfolder in usable_folders:
        combine_eeg_files(subfolder)