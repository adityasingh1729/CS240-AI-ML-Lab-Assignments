import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def PCA(init_array: pd.DataFrame):

    sorted_eigenvalues = None
    final_data = None
    dimensions = 2

    # TODO: transform init_array to final_data using PCA
    init_array = init_array.values
    # np.matrix.std(init_array)
    mean_values = np.mean(init_array, axis=0)
    std_values = np.std(init_array, axis=0)
    init_array = (init_array - mean_values) / std_values
    covMatrix = np.cov(init_array, rowvar=False)
    eigVals, eigVecs = np.linalg.eig(covMatrix)
    sorted_indices = np.argsort(np.abs(eigVals))[::-1]
    sorted_eigenvalues = eigVals[sorted_indices]
    eigVecs = eigVecs[:, sorted_indices]
    K = 2
    subEigVecs = eigVecs[:, 0:K]
    final_data = np.dot(init_array, subEigVecs)
    # END TODO

    return sorted_eigenvalues, final_data


if __name__ == '__main__':
    init_array = pd.read_csv("pca_data.csv", header = None)
    sorted_eigenvalues, final_data = PCA(init_array)
    np.savetxt("transform.csv", final_data, delimiter = ',')
    for eig in sorted_eigenvalues:
        print(eig)

    # TODO: plot and save a scatter plot of final_data to out.png
    plt.scatter(final_data[:, 0] ,final_data[:, 1])
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.savefig("out.png")
    # END TODO
