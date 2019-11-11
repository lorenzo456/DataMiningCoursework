"""
Class to run K-means algorithm, changing the label from 0 and 255 to 0 and 1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import KMeans




def main():

    dataset = pd.read_csv("./datasets/dataset_c&f.csv", delimiter=",", dtype=np.uint8)
    dataset = np.asmatrix(dataset)

    x = dataset[:, 0:1599]
    y = dataset[:, -1]

    x[x > 1] = 1

    kmeans = KMeans()
    kmeans.fit(x)

    f = np.unique(kmeans.labels_)
    print(f"Total number of clusters found: {len(f)}")

    print("Confusion Matrix")
    print(confusion_matrix(y, kmeans.labels_))
    print("\n")

    plt.show()
    print("Classification report")
    print(classification_report(y, kmeans.labels_))

    kmeans1 = KMeans(n_clusters=10)
    kmeans1.fit(x)
    # f = np.unique(kmeans.labels_)
    print(f"Total number of clusters found: {len(np.unique(kmeans1.labels_))}")

    print("Confusion matrix")
    print(confusion_matrix(y,kmeans1.labels_))
    print("\n")

    print("Classification report")
    print(classification_report(y, kmeans1.labels_))


if __name__ == "__main__":
    main()
