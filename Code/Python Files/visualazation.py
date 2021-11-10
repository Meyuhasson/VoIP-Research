import pandas as pd
import numpy as np
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

#inversed data means exactly the data we think it not relevant.
data_droped_inversed = data.drop(data.columns.difference(["isMalicious", "RTP_payload_length", "original_sr", "suspicious_diff", "Lost_packets_precentage", "Lost_packets_count", "min_magnitude1", "min_magnitude2", "min_magnitude3"]), axis=1)

df_temp_normal = pd.DataFrame()
df_temp = pd.DataFrame()
for i in data_droped.columns:

    # generate 1000 random samples between the range of the specific column's benign values
    uniform_samples = np.random.uniform(min(X[:-6][i].values), max(X[:-6][i].values), 1000)
    avg_std = (X[:-6][i].max() - X[:-6][i].mean() + X[:-6][i].mean() -X[:-6][i].min())/2
    normal_samples = np.random.normal(loc=X[:-6][i].mean(), scale= avg_std, size= 1000)

    #creating the new generated df
    df_temp[i] = uniform_samples
    df_temp_normal[i] = normal_samples

#append the new generated data df to the original
df_temp = data_droped.append(df_temp)
df_temp_normal = data_droped.append((df_temp_normal))

#cliping all columns with generated data by min, max of the original data
#for col in data_droped.columns.values:
#    df_temp[col] = df_temp[col].clip(lower = data_droped[:-6][col].min(), upper= data_droped[:-6][col].max())

#for col in data_droped.columns.values:
#    df_temp_normal[col] = df_temp_normal[col].clip(lower = data_droped[:-6][col].min(), upper= data_droped[:-6][col].max())

scaler = MinMaxScaler()
minmax_scaled = scaler.fit_transform(data_droped)
minmax_scaled_inversed = scaler.fit_transform(data_droped_inversed)

scaler = StandardScaler()
standart_scaled = scaler.fit_transform(data_droped)
standart_scaled_inversed = scaler.fit_transform(data_droped_inversed)

pca_model = PCA(n_components=2)
data_transformed = pca_model.fit_transform(data_droped)
data_transformed = pca_model.inverse_transform(data_transformed)
data_transformed_inversed = pca_model.fit_transform(data_droped_inversed)
data_transformed_inversed = pca_model.inverse_transform(data_transformed_inversed)

#plots
DS_utils.plot_points_scatter(minmax_scaled, data, "minmax scaled")
DS_utils.plot_points_scatter(minmax_scaled_inversed, data, "minmax scaled inversed")
DS_utils.plot_points_scatter(data_droped, data, "original data")
DS_utils.plot_points_scatter(data_droped_inversed, data, "original inversed data")
DS_utils.plot_points_scatter(standart_scaled, data, "standartscaler scaled")
DS_utils.plot_points_scatter(standart_scaled_inversed, data, "standartscaler scaled inversed")
DS_utils.plot_points_scatter(data_transformed, data, "after_PCA_transformed_and_inverse")
DS_utils.plot_points_scatter(data_transformed_inversed, data, "after_PCA_transformed_and_inverse of inversed data")
DS_utils.plot_points_scatter(df_temp, data, "generated and original uniform distributed data")
DS_utils.plot_points_scatter(df_temp_normal, data, "generated and original normal distributed data")

