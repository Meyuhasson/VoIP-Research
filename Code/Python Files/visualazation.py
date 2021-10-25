import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import DS_utils

PROCESSED_FEATURES_PATH = r"C:\Users\edenm\Documents\GitHub\VoIP-Research\Data\features_extracted\processed_features_for_train.csv"

# I dropped suspicious_diff feature, just because it's boolean not numeric, somtime it will be good practic to find a way to transfer it also to the network.
data = pd.read_csv(PROCESSED_FEATURES_PATH).drop([r"Unnamed: 0"], axis=1)
X = data.drop("isMalicious", axis=1)
y = data.drop(data.columns.difference(["isMalicious"]), axis=1)


#drop irrelevant for anomalous data features
data_droped = data.drop(["isMalicious", "RTP_payload_length", "original_sr", r"RTP_payload_type", "suspicious_diff", "Lost_packets_precentage", "Lost_packets_count", "min_magnitude1", "min_magnitude2", "min_magnitude3"], axis=1)

scaler = MinMaxScaler()
minmax_scaled = scaler.fit_transform(data_droped)

scaler = StandardScaler()
standart_scaled = scaler.fit_transform(data_droped)

pca_model = PCA(n_components=2)
data_transformed = pca_model.fit_transform(data_droped)
data_transformed = pca_model.inverse_transform(data_transformed)

#plots
DS_utils.plot_points_scatter(minmax_scaled, data, "minmax scaled")
DS_utils.plot_points_scatter(data_droped, data, "original data")
DS_utils.plot_points_scatter(data_transformed, data, "after_PCA_transformed_and_inverse")
DS_utils.plot_points_scatter(standart_scaled, data, "standartscaler scaled")



