from sklearn.ensemble import IsolationForest
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV


PROCESSED_FEATURES_PATH = r"C:\Users\edenm\Documents\GitHub\VoIP-Research\Data\features_extracted\processed_features_for_train.csv"

# I dropped suspicious_diff feature, just because it's boolean not numeric, somtime it will be good practic to find a way to transfer it also to the network.
data = pd.read_csv(PROCESSED_FEATURES_PATH).drop([r"Unnamed: 0"], axis=1)
X = data.drop("isMalicious", axis=1)
y = data.drop(data.columns.difference(["isMalicious"]), axis=1)

#drop irrelevant for anomalous data features
X = X.drop(["Lost_packets_count", "RTP_payload_length", "original_sr", "suspicious_diff", "Lost_packets_precentage", "min_magnitude1", "min_magnitude2", "min_magnitude3"], axis=1)

#scaler = MinMaxScaler()
#X_scaled = scaler.fit_transform(X)

parameters = {"n_estimators":[100, 90, 80, 70, 60, 55 ,50 ,45, 40, 35, 30, 20, 10, 5], "bootstrap":[True, False], "max_features":[1,2,3,4,5,6,7,8], "contamination" :[0.4, 0.35, 0.3, 0.25, 0.2, 0.15 ,0.1, 0.07 ,0.05, 0.03 ,0.01]}
#clf = GridSearchCV(isof, parameters, scoring="accuracy", cv=5, n_jobs = -1)

run = 0
min = 100
best_params = {}
for i in tqdm(parameters["n_estimators"]):
    for j in parameters["contamination"]:
        for z in parameters["max_features"]:
            for w in parameters["bootstrap"]:
                isof = IsolationForest(n_jobs=-1, n_estimators = i, contamination = j, max_features = z, bootstrap = w)
                #isof.fit(X_scaled)
                isof.fit(X)
                #X["isof_output" + " " + str(run)] = isof.predict(X_scaled)
                X["isof_output" + " " + str(run)] = isof.predict(X)
                if (min > (X["isof_output" + " " + str(run)] == -1).sum()):
                    if ((X["isof_output" + " " + str(run)] == -1).sum()>5):
                        min = (X["isof_output" + " " + str(run)] == -1).sum()
                        best_params["n_estimators"] = i
                        best_params["contamination"] = j
                        best_params["max_features"] = z
                        best_params["bootstrap"] = w
                run += 1

print(best_params)
print(X)

'''
isof = IsolationForest(n_estimators=100, contamination=0.05, max_features=2, bootstrap=False)
isof.fit(X_scaled[:-6])
print(isof.predict(X_scaled))

'''