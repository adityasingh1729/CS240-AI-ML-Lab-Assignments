import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy.random import uniform
from sklearn.datasets import make_blobs
import seaborn as sns
import random

class KMeans:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = []
        

    def initialise(self, X_train):
        """
        Initialize the self.centroids class variable, using the "k-means++" method, 
        Pick a random data point as the first centroid,
        Pick the next centroids with probability directly proportional to their distance from the closest centroid
        Function returns self.centroids as an np.array
        USE np.random for any random number generation that you may require 
        (Generate no more than K random numbers). 
        Do NOT use the random module at ALL!
        """
        for i in range(self.max_iter):
            parts = [[] for _ in range(self.n_clusters)]

            for x in X_train:
                distances = [np.linalg.norm(x - centroid) for centroid in self.centroids]
                part = distances.index(min(distances))
                parts[part].append(x)

            previous = self.centroids.copy()

            for classification in range(self.n_clusters):
                self.centroids[classification] = np.average(parts[classification], axis=0)

            if np.all(np.array(previous) == np.array(self.centroids)):
                break

        return np.array(self.centroids)

    def fit(self, X_train):
        """
        Updates the self.centroids class variable using the two-step iterative algorithm on the X_train dataset.
        X_train has dimensions (N,d) where N is the number of samples and each point belongs to d dimensions
        Ensure that the total number of iterations does not exceed self.max_iter
        Function returns self.centroids as an np array
        """
        if X_train is None or len(X_train) == 0:
            raise ValueError("X_train cannot be None or empty.")

        for _ in range(self.max_iter):
            # Step 1: Assign points to the nearest centroid
            classifications = np.argmin(np.linalg.norm(X_train[:, np.newaxis, :] - self.centroids, axis=2), axis=1)

            # Step 2: Update centroids based on the mean of the points assigned to each cluster
            new_centroids = np.array([X_train[classifications == k].mean(axis=0) for k in range(self.n_clusters)])

            # Check for convergence
            if np.allclose(new_centroids, self.centroids):
                break

            self.centroids = new_centroids

        return self.centroids

    def evaluate(self, X):
        """
        Given N data samples in X, find the cluster that each point belongs to 
        using the self.centroids class variable as the centroids.
        Return two np arrays, the first being self.centroids 
        and the second is an array having length equal to the number of data points 
        and each entry being between 0 and K-1 (both inclusive) where K is number of clusters.
        """
        if X is None or len(X) == 0:
            raise ValueError("X cannot be None or empty.")

        classifications = np.argmin(np.linalg.norm(X[:, np.newaxis, :] - self.centroids, axis=2), axis=1)
        return self.centroids, classifications

def evaluate_loss(X, centroids, classification):
    if X is None or len(X) == 0:
        raise ValueError("X cannot be None or empty.")

    loss = 0
    for idx, point in enumerate(X):
        loss += np.linalg.norm(point - centroids[classification[idx]])
    return loss

if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed + 1)

    random_state = random.randint(10, 1000)
    centers = 5

    X_train, true_labels = make_blobs(n_samples=100, centers=centers, random_state=seed)
    X_train = StandardScaler().fit_transform(X_train)

    # Fit centroids to dataset
    kmeans = KMeans(n_clusters=centers)
    kmeans.initialise(X_train)
    kmeans.fit(X_train)
    print(kmeans.evaluate(X_train))
    class_centers, classification = kmeans.evaluate(X_train)

    # Calculate and print the loss
    print(evaluate_loss(X_train, class_centers, classification))

    # View results
    sns.scatterplot(x=[X[0] for X in X_train],
                    y=[X[1] for X in X_train],
                    hue=true_labels,
                    style=classification,
                    palette="deep",
                    legend=None
                    )
    plt.plot([x for x, _ in kmeans.centroids],
             [y for _, y in kmeans.centroids],
             'k+',
             markersize=10,
             )
    plt.savefig("hello.png")
    plt.show()


