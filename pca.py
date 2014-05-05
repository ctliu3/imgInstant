import numpy as np
from scipy.sparse.linalg import eigs

def pca(X, k):
  """ Compute the principal components analysis of matrix X, only the first k
      eigenvalues and eigenvectors will be returned.

  Parmaters
  ---------
  X: matrix
     A feature matrix with shape (n, d)

  k: matrix
     The number of principal components need to be returned 

  Returns
  -------
  eigvec: k eigenvectors
          An matrix with shape (d, k)

  eigval: k eigenvalues 
          An matrix with shape (1, k)
  """
  n, dim = X.shape

  # center the data
  X_mean = np.mean(X, axis = 0)
  X = X - X_mean
  # get the covariance matrix
  covariance_matrix = np.dot(X.T, X) / (n - 1)
  eigval, eigvec = eigs(covariance_matrix, k)
  return np.array(eigvec), np.array(eigval)

X = [
    [2.5, 2.4, 4.5],
    [0.5, 0.7, 4.5],
    [2.2, 2.9, 9.3],
    [1.9, 2.2, 2.3],
    [3.1, 3.0, 2.1],
    [2.3, 2.7, 5.3],
    [2.0, 1.6, 2.9],
    [1.0, 1.1, 9.4],
    [1.5, 1.6, 8.1],
    [1.1, 0.9, 4.2],
    ];
X = np.array(X);
#pca(X)
