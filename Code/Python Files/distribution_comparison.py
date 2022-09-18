import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import ks_2samp

data = pd.read_csv("/Users/meyuhasson/dev/VoIP-Research/Data/features_extracted/processed_features_for_train.csv")
data_imp = pd.read_csv("/Users/meyuhasson/dev/VoIP-Research/Data/features_extracted/‏‏processed_features_for_train_no_imputing.csv")
data = data.drop(["isMalicious", "RTP_payload_length", "original_sr", "suspicious_diff", "Lost_packets_precentage", "Lost_packets_count", "min_magnitude1", "min_magnitude2", "min_magnitude3"], axis=1)
data_imp = data_imp.drop(["isMalicious", "RTP_payload_length", "original_sr", "suspicious_diff", "Lost_packets_precentage", "Lost_packets_count", "min_magnitude1", "min_magnitude2", "min_magnitude3"], axis=1)

ks_2samp(data["Flushed_packets"], data_imp["Flushed_packets"])
