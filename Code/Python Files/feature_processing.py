import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def feature_processing(FEATURES_PATH, OUTPUT_PROCESSED_FEATURES_PATH):
    data = pd.read_csv(FEATURES_PATH)

    #remove unimportant columns
    data = data.drop(["max_phase1", "max_phase2", "max_phase3", "min_phase1", "min_phase2", "min_phase3"], axis=1)

    #*** all the followong section deals with the situations of audio data which didnt came from PCAP, ***
    #*** because of that it doesn't have the PCAP analysis columns, then i impute them. ***

    #remove the text after the bytes value in RTP_payload_length column
    data["RTP_payload_length"] = data.apply(lambda row: int(row["RTP_payload_length"].split(" ")[0]) if str(row["Lost_packets"])!="nan" else None ,axis=1)

    #remove the text after the bytes value in Lost_packets_count column
    data["Lost_packets_count"] = data.apply(lambda row: float(str(row["Lost_packets"]).split(" ")[0]) if str(row["Lost_packets"])!="nan" else None ,axis=1)
    data["Lost_packets_precentage"]= data.apply(lambda row: float(str(row["Lost_packets"]).split(" ")[1].replace('%)','').replace('(', '')) if str(row["Lost_packets"])!="nan" else None ,axis=1)
    data = data.drop(["Lost_packets", "Audio_File_Path", "RTP_payload_type"], axis=1)
    imp_mean = IterativeImputer(random_state=0)
    imp_mean.fit(data[:150])
    data = pd.DataFrame(imp_mean.transform(data), columns=data.columns)
    data["isMalicious"] = data.apply(lambda row: True if int(row["isMalicious"]) == 1 else False,axis=1)
    data["suspicious_diff"] = data.apply(lambda row: True if int(row["suspicious_diff"]) == 1 else False,axis=1)
    data.to_csv(OUTPUT_PROCESSED_FEATURES_PATH, index=False)



