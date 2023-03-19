import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from matplotlib.colors import ListedColormap

df_digital = pd.read_csv(filepath_or_buffer='DigitalRecognizerData/train.csv')
digital = np.array(df_digital)
targets = digital[:, 0]
k_range = range(1, 20)
k_error = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, algorithm='auto')
    scores = cross_val_score(knn, digital[:, 1:], targets, cv=6, scoring='accuracy')
    k_error.append(1 - scores.mean())
plt.plot(k_range, k_error)
plt.xlabel('Value of K for KNN')
plt.ylabel('Error')
plt.show()
knn = KNeighborsClassifier(n_neighbors=3, algorithm='auto')
knn.fit(digital[:, 1:], targets)
score = knn.score(digital[:, 1:], targets, sample_weight=None)
print(score)
# 0.9837619047619047
