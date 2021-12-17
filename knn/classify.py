import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("knn_data.csv")
labels = data['spam']
label_encoder = preprocessing.LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

features = data[["money", 'free', 'for', 'gambling', 'fun', 'machine', 'learning']]
# print(features)

k = int(input("How many neighbors do you want to consider?"))
# Model built using Euclidean distance
model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')

# Model built using weights
# model = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')

# Model built using cosine similarity
# model = KNeighborsClassifier(n_neighbors=k, metric='cosine')

model.fit(features, labels)

test = [0, 1, 1, 0, 0, 1, 1]
predicted = model.predict([test])
print(predicted)

