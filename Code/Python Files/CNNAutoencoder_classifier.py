import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.models import Model
import numpy as np
from matplotlib import pyplot as plt
from numpy import sqrt, array, random, argsort
from tensorflow import keras



PROCESSED_FEATURES_PATH = r"C:\Users\edenm\Documents\GitHub\VoIP-Research\Data\features_extracted\processed_features_for_train.csv"

# I dropped suspicious_diff feature, just because it's boolean not numeric, somtime it will be good practic to find a way to transfer it also to the network.
data = pd.read_csv(PROCESSED_FEATURES_PATH).drop([r"Unnamed: 0"], axis=1)
X = data.drop("isMalicious", axis=1)

#drop irrelevant for anomalous data features
X = X.drop(["Lost_packets_count", "RTP_payload_length", "original_sr", r"RTP_payload_type", "suspicious_diff", "Lost_packets_precentage", "min_magnitude1", "min_magnitude2", "min_magnitude3"], axis=1)

y = data.drop(data.columns.difference(["isMalicious"]), axis=1)
#scaler = StandardScaler()

X = X.apply(lambda row: np.array(row).reshape(4,2,1), axis=1)
X = np.asarray(X).astype('float32')
#scaler = MinMaxScaler()
#X_scaled = scaler.fit_transform(X)
#X = np.true_divide(X, X.max())

AE_model = Sequential()
AE_model.add(Input(shape=(4, 2, 1)))
AE_model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu', padding='same'))
AE_model.add(MaxPooling2D(pool_size=(2, 2), strides=None))
AE_model.add(Conv2D(filters =16,  kernel_size =(3, 3), activation='relu', padding='same'))
AE_model.add(UpSampling2D((2, 2)))
AE_model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
AE_model.compile(loss='binary_crossentropy', optimizer='adam')

# summarize layers
print(AE_model.summary())

AE_model.compile("Adam", "mse")

AE_model.fit(np.array(X[:-25]), np.array(X[:-25]), batch_size=6, epochs=150)

AE_output = AE_model.predict(np.array(X[-25:]))

AE_deferences = sqrt((AE_output - X)**2)
AE_df = pd.DataFrame(AE_deferences, columns=X.columns)
AE_df["grades"] = AE_df.apply(lambda row: np.linalg.norm(0 - row), axis=1)
AE_df = AE_df.sort_values("grades")
print(AE_df)
plt.hist(AE_df)
plt.show()
