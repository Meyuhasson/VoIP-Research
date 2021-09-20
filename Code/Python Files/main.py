import pandas as pd
import numpy as np
import os
from IPython.display import display
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import sklearn
import matplotlib.pyplot as plt
import IPython.display as ipd
from tqdm import tqdm
import librosa.display
import audio_utils

np.set_printoptions(suppress=True) #prevent numpy exponential

#the output path for the csv with features extracted
OUTPUT_PATH = r"C:\Users\edenm\Desktop\thesis\features_extracted\extracted_datasets.csv"

#the path consist files with wav files isnide them - only benign
BENIGN_RECORDING_PATH = r"C:\Users\edenm\Desktop\thesis\proccesed and tagged external datasets\RTP Flood records\benign recording"

#the path consist files with wav files isnide them - only malicious
MALICIOUS_RECORDING_PATH = r"C:\Users\edenm\Desktop\thesis\proccesed and tagged external datasets\RTP Flood records\malicious recordings"

#the time slice in second, which we analyse in the function.
TIME_SLICE = 1

FIRST_RUN_INDICATOR = True

if __name__ == '__main__':

    list_of_benign_wav_files = audio_utils.get_list_of_wav_files(BENIGN_RECORDING_PATH)
    list_of_malicious_wav_files = audio_utils.get_list_of_wav_files(MALICIOUS_RECORDING_PATH)
    i = 0

    for wavFilePath in tqdm(list_of_benign_wav_files):

        # path for load the audio file(wav)
        WAVEFILE_PATH = wavFilePath

        #reading data and sample rate, sample rate is by deafult value 22050, and it's defines the audio speed
        data, sr = librosa.load(WAVEFILE_PATH)

        #audio_utils.amplitude_plot_byTime(TIME_SLICE, data, sr)
        #audio_utils.stft_plot(data, sr)

        duration = audio_utils.getTime(data)
        original_sr = audio_utils.sampleRate_value(WAVEFILE_PATH)
        extermom_point_diff_dict = audio_utils.Extermom_Points_Change(TIME_SLICE, data)
        suspicious_diff = audio_utils.Suspicious_Change_In_Extermom(extermom_point_diff_dict)
        min = audio_utils.minimum_amplitude(data)
        max = audio_utils.maximum_amplitude(data)
        max_magnitude,min_magnitude, max_phase, min_phase = audio_utils.stft(data, sr)
        print ("the duration of the audio file is: "+ str(duration))
        print ("the original sample rate of the audio file is: "+ str(original_sr))
        print ("is the audio file extermom points are chaged Suspiciously :" + str(suspicious_diff))
        print ("min point of the audio file: " + str(min))
        print ("max point of the audio file: " + str(max))
        print ("5 max values in magnitude complex matrix are: " + str(max_magnitude))
        print ("5 min values in magnitude complex matrix are: " + str(min_magnitude))
        print("5 max values in phase complex matrix are: " + str(max_phase))
        print("5 min values in phase complex matrix are: " + str(min_phase))
        df = pd.DataFrame()
        df._set_value(i, 'duration', duration)
        df._set_value(i, 'original_sr', original_sr)
        df._set_value(i, 'suspicious_diff', suspicious_diff)
        df._set_value(i, 'min amplitude', min)
        df._set_value(i, 'max amplitude', max)
        df._set_value(i, 'max_magnitude1', max_magnitude[0])
        df._set_value(i, 'max_magnitude2', max_magnitude[1])
        df._set_value(i, 'max_magnitude3', max_magnitude[2])
        df._set_value(i, 'min_magnitude1', min_magnitude[0])
        df._set_value(i, 'min_magnitude2', min_magnitude[1])
        df._set_value(i, 'min_magnitude3', min_magnitude[2])
        df._set_value(i, 'max_phase1', max_phase[0])
        df._set_value(i, 'max_phase2', max_phase[1])
        df._set_value(i, 'max_phase3', max_phase[2])
        df._set_value(i, 'min_phase1', min_phase[0])
        df._set_value(i, 'min_phase2', min_phase[1])
        df._set_value(i, 'min_phase3', min_phase[2])
        df._set_value(i, 'isMalicious', False)

        if (FIRST_RUN_INDICATOR == True):
            df.to_csv(OUTPUT_PATH, encoding='utf-8')
            FIRST_RUN_INDICATOR = False
            i+= 1
        else:
            df.to_csv(OUTPUT_PATH, mode='a', header=False)
            i+= 1

    for wavFilePath in tqdm(list_of_malicious_wav_files):
        # path for load the audio file(wav)
        WAVEFILE_PATH = wavFilePath

        # reading data and sample rate, sample rate is by deafult value 22050, and it's defines the audio speed
        data, sr = librosa.load(WAVEFILE_PATH)

        # audio_utils.amplitude_plot_byTime(TIME_SLICE, data, sr)
        # audio_utils.stft_plot(data, sr)

        duration = audio_utils.getTime(data)
        original_sr = audio_utils.sampleRate_value(WAVEFILE_PATH)
        extermom_point_diff_dict = audio_utils.Extermom_Points_Change(TIME_SLICE, data)
        suspicious_diff = audio_utils.Suspicious_Change_In_Extermom(extermom_point_diff_dict)
        min = audio_utils.minimum_amplitude(data)
        max = audio_utils.maximum_amplitude(data)
        max_magnitude, min_magnitude, max_phase, min_phase = audio_utils.stft(data, sr)
        print("the duration of the audio file is: " + str(duration))
        print("the original sample rate of the audio file is: " + str(original_sr))
        print("is the audio file extermom points are chaged Suspiciously :" + str(suspicious_diff))
        print("min point of the audio file: " + str(min))
        print("max point of the audio file: " + str(max))
        print("5 max values in magnitude complex matrix are: " + str(max_magnitude))
        print("5 min values in magnitude complex matrix are: " + str(min_magnitude))
        print("5 max values in phase complex matrix are: " + str(max_phase))
        print("5 min values in phase complex matrix are: " + str(min_phase))
        df = pd.DataFrame()
        df._set_value(i, 'duration', duration)
        df._set_value(i, 'original_sr', original_sr)
        df._set_value(i, 'suspicious_diff', suspicious_diff)
        df._set_value(i, 'min amplitude', min)
        df._set_value(i, 'max amplitude', max)
        df._set_value(i, 'max_magnitude1', max_magnitude[0])
        df._set_value(i, 'max_magnitude2', max_magnitude[1])
        df._set_value(i, 'max_magnitude3', max_magnitude[2])
        df._set_value(i, 'min_magnitude1', min_magnitude[0])
        df._set_value(i, 'min_magnitude2', min_magnitude[1])
        df._set_value(i, 'min_magnitude3', min_magnitude[2])
        df._set_value(i, 'max_phase1', max_phase[0])
        df._set_value(i, 'max_phase2', max_phase[1])
        df._set_value(i, 'max_phase3', max_phase[2])
        df._set_value(i, 'min_phase1', min_phase[0])
        df._set_value(i, 'min_phase2', min_phase[1])
        df._set_value(i, 'min_phase3', min_phase[2])
        df._set_value(i, 'isMalicious', True)

        df.to_csv(OUTPUT_PATH, mode='a', header=False)
        i+=1