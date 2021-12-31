from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler



PROCESSED_FEATURES_PATH = r"C:\Users\edenm\Documents\GitHub\VoIP-Research\Data\features_extracted\processed_features_for_train.csv"

# I dropped suspicious_diff feature, just because it's boolean not numeric, somtime it will be good practic to find a way to transfer it also to the network.
data = pd.read_csv(PROCESSED_FEATURES_PATH).drop([r"Unnamed: 0"], axis=1)
X = data.drop("isMalicious", axis=1)
y = data.drop(data.columns.difference(["isMalicious"]), axis=1)

#drop irrelevant for anomalous data features
X = X.drop(["Lost_packets_count", "RTP_payload_length", "original_sr", "suspicious_diff", "Lost_packets_precentage", "min_magnitude1", "min_magnitude2", "min_magnitude3"], axis=1)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

model = LocalOutlierFactor()
model.fit_predict(X)
pred = model.negative_outlier_factor_
print("hey")
