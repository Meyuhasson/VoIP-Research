import numpy as np
import feature_processing
import audio_utils

FEATURES_PATH = r"C:\Users\edenm\Documents\GitHub\VoIP-Research\Data\features_extracted\extracted_datasets.csv"
OUTPUT_PROCESSED_FEATURES_PATH = r"C:\Users\edenm\Documents\GitHub\VoIP-Research\Data\features_extracted\processed_features_for_train.csv"

np.set_printoptions(suppress=True) #prevent numpy exponential

#the output path for the csv with features extracted
OUTPUT_PATH = r"C:\Users\edenm\Documents\GitHub\VoIP-Research\Data\features_extracted\extracted_datasets.csv"

#the path consist files with wav files isnide them - only benign
BENIGN_RECORDING_PATH = r"C:\Users\edenm\Documents\GitHub\VoIP-Research\Data\proccesed and tagged external datasets\benign recording"

#the path consist files with wav files isnide them - only malicious
MALICIOUS_RECORDING_PATH = r"C:\Users\edenm\Documents\GitHub\VoIP-Research\Data\proccesed and tagged external datasets\malicious recordings"



if __name__ == '__main__':

    list_of_benign_wav_files = audio_utils.get_list_of_files_fromType(BENIGN_RECORDING_PATH, "wav")
    list_of_malicious_wav_files = audio_utils.get_list_of_files_fromType(MALICIOUS_RECORDING_PATH, "wav")
    audio_utils.write_wav_features_to_csv(list_of_benign_wav_files, list_of_malicious_wav_files, OUTPUT_PATH)
    feature_processing.feature_processing(FEATURES_PATH, OUTPUT_PROCESSED_FEATURES_PATH)
