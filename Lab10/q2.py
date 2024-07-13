import numpy as np
import pickle as pkl

class LDA:
    def __init__(self, k):
        self.n_components = k
        self.linear_discriminants = None

    def fit(self, X, y):
        """
        X: (n,d,d) array consisting of input features
        y: (n,) array consisting of labels
        return: Linear Discriminant np.array of size (d*d,k)
        """
        n, d, _ = X.shape
        X = X.reshape(n, d*d)
        A = np.zeros((d*d, d*d))
        B = np.zeros((d*d, d*d))
        for x in np.unique(y):
            C = X[y == x]
            averages = np.mean(C, axis=0)
            A += (C - averages).T @ (C - averages)
            n_c = C.shape[0]
            mean_diff = (averages - np.mean(X, axis=0)).reshape(d*d, 1)
            B += n_c * (mean_diff @ mean_diff.T)

        eigenvalues, eigenvectors = np.linalg.eig(np.matmul(np.linalg.pinv(A), B))
        
        index = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvectors = eigenvectors[:, index]
        
        self.linear_discriminants = eigenvectors[:, :self.n_components] 
        return(self.linear_discriminants)

    
    def transform(self, X, w):
        """
        X: (n,d,d) array consisting of input features
        w: Linear Discriminant array of size (d*d,k)
        return: np-array of the projected features of size (n,k)
        """
        n, d, _ = X.shape
        X_flat = X.reshape(n, -1)  # Flatten each 2D feature matrix into a 1D vector
        projected = X_flat @ w
        return projected.reshape(n, self.n_components)

if __name__ == '__main__':
    mnist_data = 'mnist.pkl'
    with open(mnist_data, 'rb') as f:
        data = pkl.load(f)
    X = data[0]
    y = data[1]
    k = 10
    lda = LDA(k)
    w = lda.fit(X, y)
    X_lda = lda.transform(X, w)
