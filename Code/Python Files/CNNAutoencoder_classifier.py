import pandas as pd
import audio_utils
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.models import Model
import numpy as np
from matplotlib import pyplot as plt
from numpy import sqrt, array, random, argsort
from tensorflow import keras
from ast import literal_eval



#the path consist files with wav files isnide them - only benign
BENIGN_RECORDING_PATH = r"C:\Users\edenm\Documents\GitHub\VoIP-Research\Data\external & internal datasets - after proccesing and tagging\benign recording"

#the path consist files with wav files isnide them - only malicious
MALICIOUS_RECORDING_PATH = r"C:\Users\edenm\Documents\GitHub\VoIP-Research\Data\external & internal datasets - after proccesing and tagging\malicious recordings"

PROCESSED_FEATURES_PATH = r"C:\Users\edenm\Documents\GitHub\VoIP-Research\Data\features_extracted\processed_features_for_train.csv"

# I dropped suspicious_diff feature, just because it's boolean not numeric, somtime it will be good practic to find a way to transfer it also to the network.
list_of_benign_files = audio_utils.get_list_of_files_fromType(BENIGN_RECORDING_PATH, "raw")
list_of_malicious_files = audio_utils.get_list_of_files_fromType(MALICIOUS_RECORDING_PATH, "raw")

import ast
import decimal

list_of_benign_raw_data = []
for file in list_of_benign_files:
    f = open(file, "rb")
    #read and scale by 255 divition
    list_of_benign_raw_data.append(np.array(list(f.read(28*28))).reshape(28,28,1))

list_of_malicious_raw_data = []
for file in list_of_malicious_files:
    f = open(file, "rb")
    #read and scale by 255 divition
    list_of_malicious_raw_data.append(np.array(list(f.read(28*28))).reshape(28,28,1))

list_of_benign_raw_data = np.array(list_of_benign_raw_data)/255
list_of_malicious_raw_data = np.array(list_of_malicious_raw_data)/255

AE_model = Sequential()
AE_model.add(Input(shape=(28, 28, 1)))
AE_model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu', padding='same'))
AE_model.add(MaxPooling2D(pool_size=(2, 2), strides=None))
AE_model.add(Conv2D(filters = 8, kernel_size = (3, 3), activation='relu', padding='same'))
AE_model.add(MaxPooling2D(pool_size=(2, 2), strides=None))
AE_model.add(Conv2D(filters =8,  kernel_size =(3, 3), activation='relu', padding='same'))
AE_model.add(UpSampling2D((2, 2)))
AE_model.add(Conv2D(filters =16,  kernel_size =(3, 3), activation='relu', padding='same'))
AE_model.add(UpSampling2D((2, 2)))
AE_model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
AE_model.compile(loss='binary_crossentropy', optimizer='adam')


# summarize layers
print(AE_model.summary())

AE_model.compile("Adam", "mse")

AE_model.fit(list_of_benign_raw_data[:-25], list_of_benign_raw_data[:-25], batch_size=6, epochs=10)

AE_output = AE_model.predict(np.concatenate((list_of_benign_raw_data[-25:] , list_of_malicious_raw_data)))

AE_deferences = sqrt((AE_output - np.concatenate((list_of_benign_raw_data[-25:] , list_of_malicious_raw_data)))**2)
AE_df = pd.DataFrame(AE_deferences.reshape(31,28*28))
AE_df["grades"] = AE_df.apply(lambda row: np.linalg.norm(0 - row), axis=1)
AE_df = AE_df.sort_values("grades")
print(AE_df)
