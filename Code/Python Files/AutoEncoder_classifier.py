import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import numpy as np
from matplotlib import pyplot as plt
from numpy import sqrt, array, random, argsort
from tensorflow import keras



PROCESSED_FEATURES_PATH = r"C:\Users\edenm\Documents\GitHub\VoIP-Research\Data\features_extracted\processed_features_for_train.csv"

# I dropped suspicious_diff feature, just because it's boolean not numeric, somtime it will be good practic to find a way to transfer it also to the network.
data = pd.read_csv(PROCESSED_FEATURES_PATH).drop([r"Unnamed: 0"], axis=1)
X = data.drop("isMalicious", axis=1)

#drop irrelevant for anomalous data features
X = X.drop(["Lost_packets_count", "RTP_payload_length", "original_sr", "suspicious_diff", "Lost_packets_precentage", "min_magnitude1", "min_magnitude2", "min_magnitude3"], axis=1)

y = data.drop(data.columns.difference(["isMalicious"]), axis=1)
#scaler = StandardScaler()


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)




input_ = keras.layers.Input(shape=8,)
dense1 = keras.layers.Dense(7, activation='relu')(input_)
dense2 = keras.layers.Dense(6, activation='relu')(dense1)
dense3 = keras.layers.Dense(5, activation='relu')(dense2)
dense4 = keras.layers.Dense(4, activation='relu')(dense3)
dense5 = keras.layers.Dense(5, activation='relu')(dense4)
dense6 = keras.layers.Dense(6, activation='relu')(dense5)
dense7 = keras.layers.Dense(7, activation='relu')(dense6)
output = keras.layers.Dense(8, activation='sigmoid')(dense7)
model = keras.Model(input_, output)

# summarize layers
print(model.summary())

model.compile("Adam", "mse")

model.fit(np.array(X_scaled[:-100]), np.array(X_scaled[:-100]), batch_size=6, epochs=150)
#model.fit(np.array(X[:-6]).astype("float32"), np.array(X[:-6]).astype("float32"), batch_size=12, epochs=10000)

#AE_output = model.predict(np.array(X))
AE_output = model.predict(np.array(X_scaled[-100:]))
#AE_deferences = sqrt((AE_output - X)**2)
AE_deferences = sqrt((AE_output - X_scaled[-100:])**2)
AE_df = pd.DataFrame(AE_deferences, columns=X.columns)
AE_df["grades"] = AE_df.apply(lambda row: np.linalg.norm(0 - row), axis=1)
AE_df = AE_df.sort_values("grades")
print(AE_df)
plt.hist(AE_df)
plt.show()
