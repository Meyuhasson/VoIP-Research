import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import graphviz

PROCESSED_FEATURES_PATH = r"C:\Users\edenm\Documents\GitHub\VoIP-Research\Data\features_extracted\processed_features_for_train.csv"

data = pd.read_csv(PROCESSED_FEATURES_PATH).drop([r"Unnamed: 0", r"RTP_payload_type"], axis=1)
X = data.drop("isMalicious", axis=1)
y = data.drop(data.columns.difference(["isMalicious"]), axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = tree.DecisionTreeClassifier()
model.fit(X, y)
print(model.score(X_test, model.predict(X_test)))
