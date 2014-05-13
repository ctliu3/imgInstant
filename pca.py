import numpy as np
from scipy.sparse.linalg import eigs

# TODO
# It's seems that k should be only can the situations that when k less or
# equal than rank(cov) -1.
# Find other alternatives or delve more details into eigs()

def pca(X, k):
  """ Compute the principal components analysis of matrix X, only the first k
      eigenvalues and eigenvectors will be returned.

  Args:
    X: Feature matrix (n x d). 
    k: Single integer. The number of principal components need to be returned 
  Returns:
    eigvec: k eigenvectors. An matrix with shape (d, k)
    eigval: k eigenvalues. An matrix with shape (1, k)
    Note: the eigvec and eigval is corresponding, however, the eigval are not
          sorted, which likes the `eigs()` version of matlab
  """
  n, dim = X.shape

  # Center the data
  X_mean = np.mean(X, axis = 0)
  X = X - X_mean
  # Get the covariance matrix
  covariance_matrix = np.dot(X.T, X) / (n - 1)
  eigval, eigvec = eigs(covariance_matrix, k)
  return np.array(eigvec), np.array(eigval)
