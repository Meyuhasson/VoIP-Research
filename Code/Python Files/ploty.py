import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import DS_utils
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA



PROCESSED_FEATURES_PATH = r"C:\Users\edenm\Documents\GitHub\VoIP-Research\Data\features_extracted\‏‏processed_features_for_train_no_imputing.csv"

# I dropped suspicious_diff feature, just because it's boolean not numeric, somtime it will be good practic to find a way to transfer it also to the network.
data = pd.read_csv(PROCESSED_FEATURES_PATH).drop([r"Unnamed: 0"], axis=1)
X = data.drop("isMalicious", axis=1)

#drop irrelevant for anomalous data features
X = X.drop(["Lost_packets_count", "RTP_payload_length", "original_sr", "suspicious_diff", "Lost_packets_precentage", "min_magnitude1", "min_magnitude2", "min_magnitude3"], axis=1)

y = data.drop(data.columns.difference(["isMalicious"]), axis=1)

#scaler = MinMaxScaler()
#X_scaled = scaler.fit_transform(X)

pca_model = PCA(n_components=2)
data_transformed = pca_model.fit_transform(X)

y = y.replace(False,0)
y = y.replace(True,1)
colors = dict(zip([0,1]+[-1], [0.1, 0.5, 0.9]))
plt.scatter(data_transformed[:,0], data_transformed[:,1], c = list(map(lambda x: colors[x], y["isMalicious"].tolist())))
plt.show()