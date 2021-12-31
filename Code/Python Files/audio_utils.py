import pandas as pd
import numpy as np
import librosa.display
from IPython.display import display
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import sklearn
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa.display
import os
from tqdm import tqdm


# the time slice in second, which we analyse in the function.
TIME_SLICE = 1

def write_wav_features_to_csv(list_of_benign_wavFiles_paths, list_of_malicious_wavFiles_paths, output_path):

    FIRST_RUN_INDICATOR = True
    i = 0
    for wavFilePath in tqdm(list_of_benign_wavFiles_paths):

        # path for load the audio file(wav)
        WAVEFILE_PATH = wavFilePath

        #reading data and sample rate, sample rate is by deafult value 22050, and it's defines the audio speed
        data, sr = librosa.load(WAVEFILE_PATH)

        #audio_utils.amplitude_plot_byTime(TIME_SLICE, data, sr)
        #audio_utils.stft_plot(data, sr)

        duration = getTime(data)
        original_sr = sampleRate_value(WAVEFILE_PATH)
        extermom_point_diff_dict = Extermom_Points_Change(TIME_SLICE, data)
        suspicious_diff = Suspicious_Change_In_Extermom(extermom_point_diff_dict)
        min = minimum_amplitude(data)
        max = maximum_amplitude(data)
        max_magnitude,min_magnitude, max_phase, min_phase = stft(data, sr)
        dict_of_RTPSPLIT_features = get_rtp_features_of_RTPSPLIT(wavFilePath.replace("_.wav", ".txt"))
        if (dict_of_RTPSPLIT_features != None):

            RTP_ssrc = dict_of_RTPSPLIT_features["RTP ssrc"]
            RTP_payload_type = dict_of_RTPSPLIT_features["RTP payload type"]
            Flushed_packets = dict_of_RTPSPLIT_features["Flushed packets"]
            Lost_packets = dict_of_RTPSPLIT_features["Lost packets"]
            RTP_payload_length = dict_of_RTPSPLIT_features["RTP payload length"]
        else:
            RTP_ssrc = None
            RTP_payload_type = None
            Flushed_packets = None
            Lost_packets = None
            RTP_payload_length = None
        Audio_File_Path = WAVEFILE_PATH
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
        df._set_value(i, 'RTP_ssrc', RTP_ssrc)
        df._set_value(i, 'RTP_payload_length', RTP_payload_length)
        df._set_value(i, 'RTP_payload_type', RTP_payload_type)
        df._set_value(i, 'Lost_packets', Lost_packets)
        df._set_value(i, 'Flushed_packets', Flushed_packets)
        df._set_value(i, 'Audio_File_Path', Audio_File_Path)
        df._set_value(i, 'isMalicious', False)

        if (FIRST_RUN_INDICATOR == True):
            df.to_csv(output_path, encoding='utf-8')
            FIRST_RUN_INDICATOR = False
            i+= 1
        else:
            df.to_csv(output_path, mode='a', header=False)
            i+= 1

    for wavFilePath in tqdm(list_of_malicious_wavFiles_paths):
        # path for load the audio file(wav)
        WAVEFILE_PATH = wavFilePath

        # reading data and sample rate, sample rate is by deafult value 22050, and it's defines the audio speed
        data, sr = librosa.load(WAVEFILE_PATH)

        # audio_utils.amplitude_plot_byTime(TIME_SLICE, data, sr)
        # audio_utils.stft_plot(data, sr)

        duration = getTime(data)
        original_sr = sampleRate_value(WAVEFILE_PATH)
        extermom_point_diff_dict = Extermom_Points_Change(TIME_SLICE, data)
        suspicious_diff = Suspicious_Change_In_Extermom(extermom_point_diff_dict)
        min = minimum_amplitude(data)
        max = maximum_amplitude(data)
        max_magnitude, min_magnitude, max_phase, min_phase = stft(data, sr)
        dict_of_RTPSPLIT_features = get_rtp_features_of_RTPSPLIT(wavFilePath.replace("_.wav",".txt"))
        if (dict_of_RTPSPLIT_features != None):

            RTP_ssrc = dict_of_RTPSPLIT_features["RTP ssrc"]
            RTP_payload_type = dict_of_RTPSPLIT_features["RTP payload type"]
            Flushed_packets = dict_of_RTPSPLIT_features["Flushed packets"]
            Lost_packets = dict_of_RTPSPLIT_features["Lost packets"]
            RTP_payload_length = dict_of_RTPSPLIT_features["RTP payload length"]
        else:
            RTP_ssrc = None
            RTP_payload_type = None
            Flushed_packets = None
            Lost_packets = None
            RTP_payload_length = None
        Audio_File_Path = WAVEFILE_PATH
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
        df._set_value(i, 'RTP_ssrc', RTP_ssrc)
        df._set_value(i, 'RTP_payload_length', RTP_payload_length)
        df._set_value(i, 'RTP_payload_type', RTP_payload_type)
        df._set_value(i, 'Lost_packets', Lost_packets)
        df._set_value(i, 'Flushed_packets', Flushed_packets)
        df._set_value(i, 'Audio_File_Path', Audio_File_Path)
        df._set_value(i, 'isMalicious', True)

        df.to_csv(output_path, mode='a', header=False)
        i+=1

#get a directory path and the desired file type as string and return a list of files from that type.
def get_list_of_files_fromType(RECORDING_PATH, type):
    list_of_wav_files = []
    for dir in os.listdir(RECORDING_PATH):
        for file in os.listdir(RECORDING_PATH + '\\' + dir):
            if(os.path.isdir(RECORDING_PATH + '\\' + dir + '\\' + file)):
                for filely in os.listdir(RECORDING_PATH + '\\' + dir + '\\' + file):
                    if filely.endswith("." + type):
                        list_of_wav_files.append(os.path.join(RECORDING_PATH + "\\" + dir + '\\' + file, filely))
            else:
                if file.endswith("."+type):
                    list_of_wav_files.append(os.path.join(RECORDING_PATH + "\\" + dir, file))
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

#get txt file path with rtpsplit output and return the features out of it
def get_rtp_features_of_RTPSPLIT(file_path):
    dict_of_rtp_features = {}
    try:
        with open(file_path.replace("wav", "txt")) as fh:
            for line in fh:
                if line.strip():  # Ignore blank lines
                    key, value = [x.strip() for x in line.strip().split(':', 1)]
                    dict_of_rtp_features[key] = value
        return dict_of_rtp_features
    except:
        return None