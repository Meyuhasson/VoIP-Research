import pandas as pd
import numpy as np
from IPython.display import display
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import sklearn
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa.display
import os

def get_list_of_wav_files(BENIGN_RECORDING_PATH):
    list_of_wav_files = []
    for dir in os.listdir(BENIGN_RECORDING_PATH):
        for file in os.listdir(BENIGN_RECORDING_PATH + '\\' + dir):
            if file.endswith(".wav"):
                list_of_wav_files.append(os.path.join(BENIGN_RECORDING_PATH + "\\" + dir, file))
    return list_of_wav_files

def getTime(inner_data):
    return librosa.get_duration(inner_data)
def sampleRate_value(original_AudioFile_Path):
    return librosa.core.get_samplerate(original_AudioFile_Path)
#return plots of amlitude sliced by time
def amplitude_plot_byTime(time_slice,inner_data,sr):
    rounded_time_Audio = int(round(librosa.get_duration(inner_data)))
    i = time_slice
    while i <= (rounded_time_Audio):
        data_temp = inner_data[int(((i-time_slice)/rounded_time_Audio) * int(inner_data.size)) : int((i/rounded_time_Audio) * int(inner_data.size))]
        #plt.figure(figsize=(7, 3))
        librosa.display.waveplot(data_temp, sr=sr)
        plt.show()
        i += time_slice
# return dict with the diffrences between all the extremom points
# between time slices by "time_slice" input in a audio file
def Extermom_Points_Change(time_slice, inner_data):
    minmax = {}
    minmax_diff = {}
    rounded_time_Audio = int(round(librosa.get_duration(inner_data)))
    i = time_slice

    # every slice of the audio data, which stored in minmax, can sound piece of audio data,
    # and do all the other functions of regular audio data loaded can do
    # the slices are also ordered as series of the original load data.
    while i <= (rounded_time_Audio):
        data_temp = inner_data[int(((i-time_slice)/rounded_time_Audio) * int(inner_data.size)) : int((i/rounded_time_Audio) * int(inner_data.size))]
        minmax[i] = data_temp.min()
        i += time_slice
    i=1
    for index in range(len(minmax)-1):
        minmax_diff[i] = abs(minmax[i] - minmax[i+1])
        i+=1
    return minmax_diff
# get dict of Extermom_Points_Change function and return bool if suspicious due to the ratio betweeen the dict values,
# RATIO_TRASHOLD sets the trashold to the ratio, if higher than trashold then returns true for suspiciuos.
def Suspicious_Change_In_Extermom(point_change_dict):
    RATIO_TRASHOLD = 11
    suspicious = False
    i = 1
    for index in range(len(point_change_dict) - 1):
        if((abs(point_change_dict[i] / point_change_dict[i + 1])) > RATIO_TRASHOLD):
            suspicious = True
        if ((abs(point_change_dict[i+1] / point_change_dict[i])) > RATIO_TRASHOLD):
            suspicious = True
        i += 1
    return suspicious
def minimum_amplitude(inner_df):
    min = inner_df.min()
    return min
def maximum_amplitude(inner_df):
    max = inner_df.max()
    return max
# plot the magnitude of frequency bin f at frame t
# and plot the phase of frequency bin f at frame t
def stft_plot(inner_data, sr):
    stft = librosa.core.stft(inner_data, sr)
    magnitude_of_freq = np.abs(stft)
    phase_of_freq = np.angle(stft)
    librosa.display.specshow(librosa.amplitude_to_db(magnitude_of_freq, ref=np.max), y_axis='log', x_axis='time')
    plt.title('Power spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()
    librosa.display.specshow(librosa.amplitude_to_db(phase_of_freq, ref=np.max), y_axis='log', x_axis='time')
    plt.title('Power spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()
# return the 5 max, and 5 min values in magnitude table of frequency bin f at frame t
# and the 5 max, and 5 min values in phase table of frequency bin f at frame t
def stft(inner_data, sr):
    stft = librosa.core.stft(inner_data, sr)
    magnitude_of_freq_max = np.abs(stft)
    magnitude_of_freq_min = np.abs(stft)
    phase_of_freq_min = np.angle(stft)
    phase_of_freq_max = np.angle(stft)
    max_magnitude = np.zeros(5)
    min_magnitude = np.zeros(5)
    max_phase = np.zeros(5)
    min_phase = np.zeros(5)

    for i in range(5):
        max_magnitude[i] = magnitude_of_freq_max[np.unravel_index(magnitude_of_freq_max.argmax(), magnitude_of_freq_max.shape)]
        magnitude_of_freq_max[np.unravel_index(magnitude_of_freq_max.argmax(), magnitude_of_freq_max.shape)] = magnitude_of_freq_min[np.unravel_index(magnitude_of_freq_min.argmin(), magnitude_of_freq_min.shape)]
    for i in range(5):
        min_magnitude[i] = magnitude_of_freq_min[np.unravel_index(magnitude_of_freq_min.argmin(), magnitude_of_freq_min.shape)]
        magnitude_of_freq_min[np.unravel_index(magnitude_of_freq_min.argmin(), magnitude_of_freq_min.shape)] = magnitude_of_freq_max[np.unravel_index(magnitude_of_freq_max.argmax(), magnitude_of_freq_max.shape)]
    for i in range(5):
        max_phase[i] = phase_of_freq_max[np.unravel_index(phase_of_freq_max.argmax(), phase_of_freq_max.shape)]
        phase_of_freq_max[np.unravel_index(phase_of_freq_max.argmax(), phase_of_freq_max.shape)] = phase_of_freq_min[np.unravel_index(phase_of_freq_min.argmin(), phase_of_freq_min.shape)]
    for i in range(5):
        min_phase[i] = phase_of_freq_min[np.unravel_index(phase_of_freq_min.argmin(), phase_of_freq_min.shape)]
        phase_of_freq_min[np.unravel_index(phase_of_freq_min.argmin(), phase_of_freq_min.shape)] = phase_of_freq_max[np.unravel_index(phase_of_freq_max.argmax(), phase_of_freq_max.shape)]

    return max_magnitude,min_magnitude, max_phase, min_phase