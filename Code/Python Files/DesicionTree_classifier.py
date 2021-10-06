import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


PROCESSED_FEATURES_PATH = r"C:\Users\edenm\Documents\GitHub\VoIP-Research\Data\features_extracted\processed_features_for_train.csv"

data = pd.read_csv(PROCESSED_FEATURES_PATH).drop([r"Unnamed: 0", r"RTP_payload_type"], axis=1)
X = data.drop("isMalicious", axis=1)
y = data.drop(data.columns.difference(["isMalicious"]), axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = tree.DecisionTreeClassifier(criterion = "entropy")
model = clf.fit(X, y)
print(model.score(X_test, model.predict(X_test)))
text_representation = tree.export_text(model)
print(text_representation)
fig = plt.figure(figsize=(10,5))
_ = tree.plot_tree(clf,
                   feature_names=list(X_train.columns.values),
                   class_names= ["True", "False"],
                   filled=True)
plt.show()


"""
to be continue:
---------------

- to print the tree.
- to plot the correlation matrix due to the target (malicious/benign).

"""