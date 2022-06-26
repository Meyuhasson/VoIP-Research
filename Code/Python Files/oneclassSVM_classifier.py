from sklearn.svm import OneClassSVM
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import DS_utils
from sklearn.decomposition import PCA
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

model = OneClassSVM(kernel='linear', nu =0.9, gamma='scale')
#train over benign data
model.fit(X_scaled[:-100])

#pred the test case with benign and malicious data
decision_function_score = model.decision_function(X_scaled[-100:])
pred = model.predict(X_scaled[-100:])



'''
x_scaled = MinMaxScaler().fit_transform(X)

# reduce the data to 2 dimensions using t-SNE
x_reduced = TSNE(n_components=2, random_state=0).fit_transform(x_scaled)
# fit the model to the reduced data
svm = OneClassSVM()
svm.fit(x_reduced[:-50])

# extract the model predictions
x_predicted = svm.predict(x_reduced[-50:])


# define the meshgrid
x_min, x_max = x_reduced[:, 0].min() - 5, x_reduced[:, 0].max() + 5
y_min, y_max = x_reduced[:, 1].min() - 5, x_reduced[:, 1].max() + 5

x_ = np.linspace(x_min, x_max, 500)
y_ = np.linspace(y_min, y_max, 500)

xx, yy = np.meshgrid(x_, y_)

# evaluate the decision function on the meshgrid
z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

# plot the decision function and the reduced data
plt.contourf(xx, yy, z, cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, z, levels=[0], linewidths=2, colors='darkred')
b = plt.scatter(x_reduced[-50:][x_predicted == 1, 0], x_reduced[-50:][x_predicted == 1, 1], c='white', edgecolors='k')
c = plt.scatter(x_reduced[-50:][x_predicted == -1, 0], x_reduced[-50:][x_predicted == -1, 1], c='gold', edgecolors='k')
plt.legend([a.collections[0], b, c], ['learned frontier', 'regular observations', 'abnormal observations'], bbox_to_anchor=(1.05, 1))
plt.axis('tight')
plt.show()

a = svm.decision_function(x_reduced[-50:])
print(svm.decision_function(x_reduced[-50:]))
'''