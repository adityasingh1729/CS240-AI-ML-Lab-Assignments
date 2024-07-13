import pickle as pkl
import numpy as np

def pca(X: np.array, k: int) -> np.array:
    """
    X is an (N,a,b) array comprising N images, each of size (a,b).
    Return (a*b,k) np array comprising the k normalised basis vectors comprising the k-dimensional subspace for all images
    where the first column must be the most relevant principal component and so on
    """
    # Reshape the images to (N, a*b) where each image is flattened into a vector
    X_flat = X.reshape(X.shape[0], -1)
    
    # Compute the mean image
    mean_image = np.mean(X_flat, axis=0)
    
    # Center the data by subtracting the mean
    centered_X = X_flat - mean_image
    
    # Compute the covariance matrix
    cov_matrix = np.cov(centered_X, rowvar=False)
    
    # Perform eigen decomposition on the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvectors based on eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # Select the top k eigenvectors as basis vectors
    basis = sorted_eigenvectors[:, :k]
    
    return basis

def projection(X: np.array, basis: np.array) -> np.array:
    """
    X is an (N,a,b) array comprising N images, each of size (a,b).
    basis is an (a*b,k) array comprising of k normalised vectors
    Return (N,k) np array comprising the k dimensional projections of the N images on the normalised basis vectors
    """
    # Reshape the images to (N, a*b) where each image is flattened into a vector
    X_flat = X.reshape(X.shape[0], -1)
    
    # Project the centered data onto the basis vectors
    projections = np.dot(X_flat, basis)
    
    return projections

if __name__ == '__main__':
    mnist_data = 'mnist.pkl'
    with open(mnist_data, 'rb') as f:
        data = pkl.load(f)
    # Now you can work with the loaded object
    X = data[0]
    y = data[1]
    k = 10
    basis = pca(X, k)
    print(projection(X, basis))
