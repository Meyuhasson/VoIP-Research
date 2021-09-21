import pandas as pd

FEATURES_PATH = r"C:\Users\edenm\Documents\GitHub\VoIP-Research\Data\features_extracted\extracted_datasets.csv"
OUTPUT_PROCESSED_FEATURES_PATH = r"C:\Users\edenm\Documents\GitHub\VoIP-Research\Data\features_extracted\processed_features_for_train.csv"
data = pd.read_csv(FEATURES_PATH)

#remove unimportant columns
data = data.drop(["max_phase1", "max_phase2", "max_phase3", "min_phase1", "min_phase2", "min_phase3"], axis=1)

#remove the text after the bytes value in RTP_payload_length column
data["RTP_payload_length"] = data.apply(lambda row: int(row["RTP_payload_length"].split(" ")[0]) ,axis=1)

#remove the text after the bytes value in RTP_payload_length column
data["Lost_packets_count"] = data.apply(lambda row: int(row["Lost_packets"].split(" ")[0]) ,axis=1)
data["Lost_packets_precentage"] = data.apply(lambda row: float(row["Lost_packets"].split(" ")[1].replace('%)','').replace('(', '')) ,axis=1)
data = data.drop(["Lost_packets"], axis=1)
data.to_csv(OUTPUT_PROCESSED_FEATURES_PATH, index=False)



