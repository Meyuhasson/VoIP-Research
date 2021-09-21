import numpy as np

import audio_utils

np.set_printoptions(suppress=True) #prevent numpy exponential

#the output path for the csv with features extracted
OUTPUT_PATH = r"C:\Users\edenm\Documents\GitHub\VoIP-Research\Data\features_extracted\extracted_datasets.csv"

#the path consist files with wav files isnide them - only benign
BENIGN_RECORDING_PATH = r"C:\Users\edenm\Desktop\thesis\proccesed and tagged external datasets\RTP Flood records\benign recording"

#the path consist files with wav files isnide them - only malicious
MALICIOUS_RECORDING_PATH = r"C:\Users\edenm\Desktop\thesis\proccesed and tagged external datasets\RTP Flood records\malicious recordings"



if __name__ == '__main__':

    list_of_benign_wav_files = audio_utils.get_list_of_files_fromType(BENIGN_RECORDING_PATH, "wav")
    list_of_malicious_wav_files = audio_utils.get_list_of_files_fromType(MALICIOUS_RECORDING_PATH, "wav")
    audio_utils.write_wav_features_to_csv(list_of_benign_wav_files, list_of_malicious_wav_files, OUTPUT_PATH)

